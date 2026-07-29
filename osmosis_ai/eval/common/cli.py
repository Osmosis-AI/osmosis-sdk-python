"""Workflow + grader resolution helpers shared by remote submit preflight.

Historically these utilities also powered the local `osmosis eval run` and
`osmosis test` flows; with eval moving to remote execution the only
remaining caller is `platform.cli.workspace_directory_contract.validate_rollout_backend`,
which uses them to load and inspect a workspace rollout before submitting a
remote training run or evaluation run.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osmosis_ai.cli.console import Console
from osmosis_ai.cli.errors import CLIError


def _resolve_rollout_entrypoint(
    rollout: str,
    entrypoint: str,
    *,
    workspace_directory: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve and validate the rollout root and entrypoint file path."""
    workspace_directory = (workspace_directory or Path.cwd()).resolve()
    rollouts_root = (workspace_directory / "rollouts").resolve()
    rollout_path = Path(rollout)
    rollout_dir = (
        rollout_path if rollout_path.is_absolute() else rollouts_root / rollout_path
    ).resolve()
    try:
        rollout_dir.relative_to(rollouts_root)
    except ValueError as exc:
        raise CLIError(f"Rollout must stay within rollouts/, got: {rollout}") from exc

    if not rollout_dir.is_dir():
        raise CLIError(
            f"Rollout directory not found: rollouts/{rollout}/\n"
            f"  Expected at: {rollout_dir}"
        )

    entrypoint_rel = Path(entrypoint)
    if entrypoint_rel.is_absolute():
        raise CLIError(
            f"Entrypoint must be a path relative to rollouts/{rollout}/, got: {entrypoint}"
        )
    if entrypoint_rel.suffix != ".py":
        raise CLIError(
            f"Entrypoint must point to a Python file ending in .py, got: {entrypoint}"
        )

    entrypoint_path = (rollout_dir / entrypoint_rel).resolve()
    try:
        entrypoint_path.relative_to(rollout_dir)
    except ValueError as exc:
        raise CLIError(
            f"Entrypoint must stay within rollouts/{rollout}/, got: {entrypoint}"
        ) from exc

    if not entrypoint_path.is_file():
        raise CLIError(
            f"Entrypoint file not found in rollouts/{rollout}/: {entrypoint}\n"
            f"  Expected at: {entrypoint_path}"
        )

    return rollout_dir, entrypoint_path


def _synthetic_rollout_package_name(rollout_dir: Path) -> str:
    digest = hashlib.sha256(str(rollout_dir).encode("utf-8")).hexdigest()[:16]
    return f"_osmosis_rollout_{digest}"


def _clear_rollout_module_cache(package_name: str) -> None:
    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            sys.modules.pop(module_name, None)


def _load_package_module(package_name: str, package_dir: Path) -> types.ModuleType:
    init_py = package_dir / "__init__.py"
    if init_py.is_file():
        spec = importlib.util.spec_from_file_location(
            package_name,
            init_py,
            submodule_search_locations=[str(package_dir)],
        )
        if spec is None or spec.loader is None:
            raise CLIError(f"Failed to load rollout package: {package_dir}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)
        return module

    module = types.ModuleType(package_name)
    module.__file__ = str(init_py)
    module.__package__ = package_name
    module.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
    spec = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(package_dir)]
    module.__spec__ = spec
    sys.modules[package_name] = module
    return module


def _ensure_parent_packages(
    package_name: str,
    rollout_dir: Path,
    entrypoint_path: Path,
) -> None:
    parts = entrypoint_path.relative_to(rollout_dir).with_suffix("").parts[:-1]
    current_dir = rollout_dir
    current_package = package_name
    for part in parts:
        current_dir = current_dir / part
        current_package = f"{current_package}.{part}"
        _load_package_module(current_package, current_dir)


def _ensure_rollout_dir_on_path(rollout_dir: Path) -> None:
    """Add the rollout directory to ``sys.path`` so sibling packages resolve.

    The synthetic-package wrapper isolates the entrypoint module itself, but
    real-world entrypoints commonly do absolute imports of sibling packages
    that live next to them (e.g. ``from multiply_openai_agents.grader import
    ...`` next to ``local_rollout_server_openai_agents_example.py``). Those
    are top-level imports, so the rollout directory must be searchable via
    ``sys.path`` for them to resolve.
    """
    rollout_dir_str = str(rollout_dir)
    if rollout_dir_str not in sys.path:
        sys.path.insert(0, rollout_dir_str)


def _load_rollout_module(
    rollout: str,
    entrypoint: str,
    *,
    workspace_directory: Path | None = None,
) -> types.ModuleType:
    """Load an entrypoint as an isolated synthetic package subtree."""
    rollout_dir, entrypoint_path = _resolve_rollout_entrypoint(
        rollout,
        entrypoint,
        workspace_directory=workspace_directory,
    )
    _ensure_rollout_dir_on_path(rollout_dir)
    package_name = _synthetic_rollout_package_name(rollout_dir)
    _clear_rollout_module_cache(package_name)
    _load_package_module(package_name, rollout_dir)
    _ensure_parent_packages(package_name, rollout_dir, entrypoint_path)

    relative_parts = entrypoint_path.relative_to(rollout_dir).with_suffix("").parts
    module_name = ".".join((package_name, *relative_parts))
    spec = importlib.util.spec_from_file_location(module_name, entrypoint_path)
    if spec is None or spec.loader is None:
        raise CLIError(f"Failed to load entrypoint module: {entrypoint}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _group_by_object_id(
    pairs: list[tuple[str, Any]],
) -> dict[int, list[tuple[str, Any]]]:
    """Group (binding name, object) pairs by object identity."""
    groups: dict[int, list[tuple[str, Any]]] = {}
    for name, obj in pairs:
        groups.setdefault(id(obj), []).append((name, obj))
    return groups


def _format_ambiguous_binding_names(pairs: list[tuple[str, Any]]) -> str:
    """Sorted unique binding names for an ambiguous candidate set."""
    return ", ".join(sorted({n for n, _ in pairs}))


def _pick_representative(pairs_for_one_object: list[tuple[str, Any]]) -> Any:
    """Deterministic pick: object referred to by the lexicographically smallest name."""
    return min(pairs_for_one_object, key=lambda x: x[0])[1]


def _resolve_workflow(
    rollout: str,
    entrypoint: str,
    *,
    workspace_directory: Path | None = None,
) -> tuple[type, Any, str]:
    """Resolve an AgentWorkflow subclass and its config.

    Converts the entrypoint file path to a module, imports it,
    and auto-discovers an AgentWorkflow subclass and optional config.

    Returns (workflow_cls, config, entrypoint_module_name) where config
    may be None.  *entrypoint_module_name* is the ``__name__`` of the
    loaded entrypoint module — callers should use this (not
    ``workflow_cls.__module__``) when discovering a Grader, because the
    workflow class may have been defined in a different file and merely
    imported into the entrypoint.
    """
    from osmosis_ai.rollout.agent_workflow import AgentWorkflow
    from osmosis_ai.rollout.types import AgentWorkflowConfig

    mod = _load_rollout_module(
        rollout,
        entrypoint,
        workspace_directory=workspace_directory,
    )

    workflow_pairs = [
        (n, v)
        for n, v in vars(mod).items()
        if isinstance(v, type)
        and issubclass(v, AgentWorkflow)
        and v is not AgentWorkflow
    ]
    wf_groups = _group_by_object_id(workflow_pairs)
    if len(wf_groups) == 0:
        raise CLIError(f"No AgentWorkflow subclass found in '{entrypoint}'")
    if len(wf_groups) > 1:
        raise CLIError(
            f"Multiple AgentWorkflow subclasses found in '{entrypoint}': "
            f"{_format_ambiguous_binding_names(workflow_pairs)}. "
            "Keep only one AgentWorkflow subclass in the entrypoint module, or move "
            "extra classes to a separate file."
        )
    workflow_cls = _pick_representative(next(iter(wf_groups.values())))

    config_pairs = [
        (n, v) for n, v in vars(mod).items() if isinstance(v, AgentWorkflowConfig)
    ]
    cfg_groups = _group_by_object_id(config_pairs)
    if len(cfg_groups) > 1:
        raise CLIError(
            f"Multiple AgentWorkflowConfig instances found in '{entrypoint}': "
            f"{_format_ambiguous_binding_names(config_pairs)}. "
            "Keep only one AgentWorkflowConfig instance in the entrypoint module, or move "
            "extras to a separate file."
        )
    config = (
        _pick_representative(next(iter(cfg_groups.values())))
        if len(cfg_groups) == 1
        else None
    )

    return workflow_cls, config, mod.__name__


def load_workflow(
    rollout: str,
    entrypoint: str,
    quiet: bool = False,
    console: Console | None = None,
    workspace_directory: Path | None = None,
) -> tuple[type | None, Any, str | None, str | None]:
    """Load an AgentWorkflow class and its config.

    Returns (workflow_cls, workflow_config, entrypoint_module_name, error).
    Raises ModuleNotFoundError when the rollout's dependencies aren't installed
    in this environment (other load failures are returned as the error string).
    """
    if console and not quiet:
        console.print(f"Loading workflow: {entrypoint}")

    try:
        workflow_cls, workflow_config, entrypoint_module = _resolve_workflow(
            rollout=rollout,
            entrypoint=entrypoint,
            workspace_directory=workspace_directory,
        )
    except ModuleNotFoundError:
        # Deps aren't importable here; let the submit preflight skip and defer
        # to the server, which installs from pyproject before validating.
        raise
    except Exception as e:
        detail = str(e)
        if not isinstance(e, (CLIError, ImportError, ValueError, TypeError)):
            detail = f"{type(e).__name__}: {detail}"
        return None, None, None, detail

    if console and not quiet:
        console.print(f"  Workflow: {workflow_cls.__name__}")

    return workflow_cls, workflow_config, entrypoint_module, None


@dataclass(frozen=True)
class _NativeBinding:
    """Statically known native-backend values carried through helper calls."""

    backend_class: type | None = None
    positional: tuple[_NativeBinding, ...] | None = None
    keywords: tuple[tuple[str, _NativeBinding], ...] | None = None

    @classmethod
    def keyword_mapping(cls, values: dict[str, _NativeBinding]) -> _NativeBinding:
        return cls(keywords=tuple(sorted(values.items())))

    def keyword_value(self, name: str) -> _NativeBinding | None:
        if self.keywords is None:
            return None
        return dict(self.keywords).get(name)


_UNKNOWN_NATIVE_BINDING = _NativeBinding()


@dataclass(frozen=True)
class _FunctionSignature:
    positional_only: tuple[str, ...]
    positional_or_keyword: tuple[str, ...]
    keyword_only: tuple[str, ...]
    vararg: str | None
    kwarg: str | None
    defaults: tuple[tuple[str, ast.expr], ...]

    @property
    def positional(self) -> tuple[str, ...]:
        return (*self.positional_only, *self.positional_or_keyword)


class _NativeBackendWiringVisitor(ast.NodeVisitor):
    """Find ``create_rollout_server(backend=<native instance>)`` in one scope."""

    def __init__(
        self,
        backend_classes: dict[str, type],
        server_factories: set[str],
        module_symbols: dict[str, Any],
        native_backend_base: type,
        server_factory: Any,
        function_signatures: dict[str, _FunctionSignature],
        inherited_bindings: dict[str, _NativeBinding] | None = None,
    ) -> None:
        self.backend_classes = backend_classes
        self.server_factories = server_factories
        self.module_symbols = module_symbols
        self.native_backend_base = native_backend_base
        self.server_factory = server_factory
        self.function_signatures = function_signatures
        self.bindings = dict(inherited_bindings or {})
        self.found: type | None = None
        self.called_functions: list[tuple[str, dict[str, _NativeBinding]]] = []

    def _qualified_symbol(self, expression: ast.expr) -> Any:
        if isinstance(expression, ast.Name):
            if expression.id in self.bindings:
                return None
            return self.module_symbols.get(expression.id)
        if isinstance(expression, ast.Attribute):
            parent = self._qualified_symbol(expression.value)
            if isinstance(parent, types.ModuleType):
                return vars(parent).get(expression.attr)
        return None

    def _binding(self, expression: ast.expr) -> _NativeBinding:
        if isinstance(expression, ast.Name):
            return self.bindings.get(expression.id, _UNKNOWN_NATIVE_BINDING)
        if isinstance(expression, ast.Call):
            backend_class = (
                self.backend_classes.get(expression.func.id)
                if isinstance(expression.func, ast.Name)
                else self._qualified_symbol(expression.func)
            )
            if isinstance(backend_class, type) and issubclass(
                backend_class, self.native_backend_base
            ):
                return _NativeBinding(backend_class=backend_class)
        if isinstance(expression, (ast.List, ast.Tuple)):
            return _NativeBinding(
                positional=tuple(self._binding(item) for item in expression.elts)
            )
        if isinstance(expression, ast.Dict):
            values: dict[str, _NativeBinding] = {}
            for key, value in zip(expression.keys, expression.values, strict=True):
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    return _UNKNOWN_NATIVE_BINDING
                values[key.value] = self._binding(value)
            return _NativeBinding.keyword_mapping(values)
        if isinstance(expression, ast.Subscript):
            container = self._binding(expression.value)
            try:
                index = ast.literal_eval(expression.slice)
            except (TypeError, ValueError):
                return _UNKNOWN_NATIVE_BINDING
            if isinstance(index, int) and container.positional is not None:
                try:
                    return container.positional[index]
                except IndexError:
                    return _UNKNOWN_NATIVE_BINDING
            if isinstance(index, str):
                return container.keyword_value(index) or _UNKNOWN_NATIVE_BINDING
        return _UNKNOWN_NATIVE_BINDING

    def _call_bindings(
        self, function_name: str, node: ast.Call | None = None
    ) -> dict[str, _NativeBinding]:
        signature = self.function_signatures[function_name]
        bindings = {
            name: self._binding(default) for name, default in signature.defaults
        }

        positional_values: list[_NativeBinding] = []
        positional_unknown = False
        if node is not None:
            for argument in node.args:
                if isinstance(argument, ast.Starred):
                    expanded = self._binding(argument.value).positional
                    if expanded is None:
                        positional_unknown = True
                    else:
                        positional_values.extend(expanded)
                else:
                    positional_values.append(self._binding(argument))

        positional_names = signature.positional
        if positional_unknown:
            for parameter_name in positional_names:
                bindings[parameter_name] = _UNKNOWN_NATIVE_BINDING
        else:
            for parameter_name, value in zip(
                positional_names, positional_values, strict=False
            ):
                bindings[parameter_name] = value

        if signature.vararg is not None:
            bindings[signature.vararg] = (
                _UNKNOWN_NATIVE_BINDING
                if positional_unknown
                else _NativeBinding(
                    positional=tuple(positional_values[len(positional_names) :])
                )
            )

        keyword_values: dict[str, _NativeBinding] = {}
        keyword_unpack_unknown = False
        if node is not None:
            for keyword in node.keywords:
                if keyword.arg is not None:
                    keyword_values[keyword.arg] = self._binding(keyword.value)
                    continue
                expanded = self._binding(keyword.value)
                if expanded.keywords is None:
                    keyword_unpack_unknown = True
                else:
                    keyword_values.update(expanded.keywords)

        keyword_parameters = {
            *signature.positional_or_keyword,
            *signature.keyword_only,
        }
        extra_keywords: dict[str, _NativeBinding] = {}
        for name, value in keyword_values.items():
            if name in keyword_parameters:
                bindings[name] = value
            elif signature.kwarg is not None:
                extra_keywords[name] = value
        if keyword_unpack_unknown:
            for parameter_name in keyword_parameters:
                bindings[parameter_name] = _UNKNOWN_NATIVE_BINDING
        if signature.kwarg is not None:
            bindings[signature.kwarg] = (
                _UNKNOWN_NATIVE_BINDING
                if keyword_unpack_unknown
                else _NativeBinding.keyword_mapping(extra_keywords)
            )
        return bindings

    def _record_target(self, target: ast.expr, binding: _NativeBinding) -> None:
        if isinstance(target, ast.Name):
            self.bindings[target.id] = binding
            return
        if isinstance(target, ast.Starred):
            self._record_target(target.value, binding)
            return
        if not isinstance(target, (ast.Tuple, ast.List)):
            return

        values = binding.positional
        starred = [
            index
            for index, item in enumerate(target.elts)
            if isinstance(item, ast.Starred)
        ]
        if values is None or len(starred) > 1:
            for item in target.elts:
                self._record_target(item, _UNKNOWN_NATIVE_BINDING)
            return
        if not starred:
            if len(values) != len(target.elts):
                values = tuple(_UNKNOWN_NATIVE_BINDING for _ in target.elts)
            for item, value in zip(target.elts, values, strict=True):
                self._record_target(item, value)
            return

        star_index = starred[0]
        trailing_count = len(target.elts) - star_index - 1
        if len(values) < len(target.elts) - 1:
            for item in target.elts:
                self._record_target(item, _UNKNOWN_NATIVE_BINDING)
            return
        for item, value in zip(
            target.elts[:star_index], values[:star_index], strict=True
        ):
            self._record_target(item, value)
        star_end = len(values) - trailing_count if trailing_count else len(values)
        self._record_target(
            target.elts[star_index],
            _NativeBinding(positional=values[star_index:star_end]),
        )
        if trailing_count:
            for item, value in zip(
                target.elts[-trailing_count:], values[-trailing_count:], strict=True
            ):
                self._record_target(item, value)

    def _record_assignment(self, targets: list[ast.expr], value: ast.expr) -> None:
        binding = self._binding(value)
        for target in targets:
            self._record_target(target, binding)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        self._record_assignment(node.targets, node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is None:
            return
        self.visit(node.value)
        self._record_assignment([node.target], node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._record_assignment([node.target], node.value)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in self.function_signatures:
            self.called_functions.append(
                (node.func.id, self._call_bindings(node.func.id, node))
            )
        is_server_factory = (
            isinstance(node.func, ast.Name) and node.func.id in self.server_factories
        ) or self._qualified_symbol(node.func) is self.server_factory
        if is_server_factory:
            explicit_backend = next(
                (
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg == "backend"
                ),
                None,
            )
            backend_binding = (
                self._binding(explicit_backend)
                if explicit_backend is not None
                else _UNKNOWN_NATIVE_BINDING
            )
            if explicit_backend is None:
                for keyword in node.keywords:
                    if keyword.arg is None:
                        unpacked_backend = self._binding(keyword.value).keyword_value(
                            "backend"
                        )
                        if unpacked_backend is not None:
                            backend_binding = unpacked_backend
            if backend_binding.backend_class is not None:
                self.found = backend_binding.backend_class
        self.generic_visit(node)

    # Nested definitions are different execution scopes. The caller follows only
    # top-level helpers reachable from module execution or ``main``.
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return


def _find_wired_native_backend(
    module: types.ModuleType,
    entrypoint_path: Path,
    native_backend_base: type,
    server_factory: Any,
) -> type | None:
    backend_classes = {
        name: value
        for name, value in vars(module).items()
        if isinstance(value, type) and issubclass(value, native_backend_base)
    }
    server_factories = {
        name for name, value in vars(module).items() if value is server_factory
    }
    module_symbols = dict(vars(module))

    tree = ast.parse(entrypoint_path.read_text(encoding="utf-8"), entrypoint_path)
    function_definitions: dict[
        str, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda
    ] = {
        statement.name: statement
        for statement in tree.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and isinstance(
            statement.value, ast.Lambda
        ):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    function_definitions[target.id] = statement.value
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and isinstance(statement.value, ast.Lambda)
        ):
            function_definitions[statement.target.id] = statement.value
    function_signatures = {
        name: _FunctionSignature(
            positional_only=tuple(
                argument.arg for argument in definition.args.posonlyargs
            ),
            positional_or_keyword=tuple(
                argument.arg for argument in definition.args.args
            ),
            keyword_only=tuple(argument.arg for argument in definition.args.kwonlyargs),
            vararg=(definition.args.vararg.arg if definition.args.vararg else None),
            kwarg=(definition.args.kwarg.arg if definition.args.kwarg else None),
            defaults=tuple(
                zip(
                    (
                        (
                            *(argument.arg for argument in definition.args.posonlyargs),
                            *(argument.arg for argument in definition.args.args),
                        )[-len(definition.args.defaults) :]
                        if definition.args.defaults
                        else ()
                    ),
                    definition.args.defaults,
                    strict=True,
                )
            )
            + tuple(
                (argument.arg, default)
                for argument, default in zip(
                    definition.args.kwonlyargs,
                    definition.args.kw_defaults,
                    strict=True,
                )
                if default is not None
            ),
        )
        for name, definition in function_definitions.items()
    }
    module_visitor = _NativeBackendWiringVisitor(
        backend_classes,
        server_factories,
        module_symbols,
        native_backend_base,
        server_factory,
        function_signatures,
    )
    for statement in tree.body:
        if not isinstance(
            statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            module_visitor.visit(statement)
    if module_visitor.found is not None:
        return module_visitor.found

    pending_functions = [
        (name, bindings)
        for name, bindings in module_visitor.called_functions
        if name in function_definitions
    ]
    if "main" in function_definitions:
        pending_functions.append(("main", module_visitor._call_bindings("main")))
    visited_functions: set[tuple[str, tuple[tuple[str, _NativeBinding], ...]]] = set()
    while pending_functions:
        function_name, call_bindings = pending_functions.pop()
        visit_key = (
            function_name,
            tuple(sorted(call_bindings.items())),
        )
        if visit_key in visited_functions:
            continue
        visited_functions.add(visit_key)
        function_visitor = _NativeBackendWiringVisitor(
            backend_classes,
            server_factories,
            module_symbols,
            native_backend_base,
            server_factory,
            function_signatures,
            inherited_bindings={**module_visitor.bindings, **call_bindings},
        )
        definition = function_definitions[function_name]
        children = (
            [definition.body] if isinstance(definition, ast.Lambda) else definition.body
        )
        for child in children:
            function_visitor.visit(child)
        if function_visitor.found is not None:
            return function_visitor.found
        pending_functions.extend(
            (name, bindings)
            for name, bindings in function_visitor.called_functions
            if name in function_definitions
        )
    return None


def discover_native_backend(
    rollout: str,
    entrypoint: str,
    *,
    workspace_directory: Path | None = None,
) -> type | None:
    """Return the native backend class wired into ``create_rollout_server``.

    Native rollouts carry no Python Grader, so submit preflight uses this to skip
    the Grader requirement. Merely importing ``NativeHarborBackend`` is not enough:
    the entrypoint must construct it and pass that value as the server's ``backend``.
    A load or static-analysis failure returns ``None`` and is surfaced by the normal
    workflow preflight path.
    """
    try:
        from osmosis_ai.rollout.backend.native_harbor.backend import (
            NativeHarborBackend,
        )
        from osmosis_ai.rollout.server import create_rollout_server

        mod = _load_rollout_module(
            rollout, entrypoint, workspace_directory=workspace_directory
        )
        _, entrypoint_path = _resolve_rollout_entrypoint(
            rollout, entrypoint, workspace_directory=workspace_directory
        )
        return _find_wired_native_backend(
            mod,
            entrypoint_path,
            NativeHarborBackend,
            create_rollout_server,
        )
    except Exception:
        return None


def auto_discover_grader(module_name: str) -> tuple[type | None, Any]:
    """Discover a Grader subclass and its config from the entrypoint module.

    The entrypoint file (e.g., ``local_rollout_server_example.py``) typically
    imports the Grader alongside the Workflow, so scanning its namespace is
    sufficient — no need to walk the entire package.

    Returns (grader_cls, grader_config) or (None, None) if not found.
    """
    mod = sys.modules.get(module_name)
    if mod is None:
        return None, None

    return _discover_grader_from_module(mod, module_name)


def _discover_grader_from_module(
    mod: Any,
    entrypoint: str,
) -> tuple[type | None, Any]:
    """Pick Grader subclass and GraderConfig from a loaded module namespace."""
    from osmosis_ai.rollout.grader import Grader
    from osmosis_ai.rollout.types import GraderConfig

    grader_pairs = [
        (n, v)
        for n, v in vars(mod).items()
        if isinstance(v, type) and issubclass(v, Grader) and v is not Grader
    ]
    grader_groups = _group_by_object_id(grader_pairs)
    if len(grader_groups) > 1:
        raise CLIError(
            f"Multiple Grader subclasses found in '{entrypoint}': "
            f"{_format_ambiguous_binding_names(grader_pairs)}. "
            "Keep only one Grader subclass in the entrypoint module, or move extra "
            "classes to a separate file."
        )
    grader_cls: type | None = None
    if len(grader_groups) == 1:
        grader_cls = _pick_representative(next(iter(grader_groups.values())))

    config_pairs = [(n, v) for n, v in vars(mod).items() if isinstance(v, GraderConfig)]
    cfg_groups = _group_by_object_id(config_pairs)
    if len(cfg_groups) > 1:
        raise CLIError(
            f"Multiple GraderConfig instances found in '{entrypoint}': "
            f"{_format_ambiguous_binding_names(config_pairs)}. "
            "Keep only one GraderConfig instance in the entrypoint module, or move "
            "extras to a separate file."
        )
    grader_config = None
    if len(cfg_groups) == 1:
        grader_config = _pick_representative(next(iter(cfg_groups.values())))

    return grader_cls, grader_config


def _resolve_grader(
    module_name: str,
    explicit_grader: str | None = None,
    explicit_config: str | None = None,
) -> tuple[type | None, Any]:
    """Resolve Grader from explicit path or auto-discover from workflow module.

    Only called when [grader] is present in TOML. Returns (None, None) when
    no grader is found.
    """
    from osmosis_ai.rollout.utils.imports import resolve_object

    if explicit_grader:
        from osmosis_ai.rollout.grader import Grader
        from osmosis_ai.rollout.types import GraderConfig

        grader_cls = resolve_object(explicit_grader)
        if (
            not isinstance(grader_cls, type)
            or not issubclass(grader_cls, Grader)
            or grader_cls is Grader
        ):
            raise CLIError(
                f"[grader].module must point to a concrete Grader subclass, "
                f"but '{explicit_grader}' resolved to {grader_cls!r}"
            )

        grader_config = resolve_object(explicit_config) if explicit_config else None
        if grader_config is not None and not isinstance(grader_config, GraderConfig):
            raise CLIError(
                f"[grader].config must point to a GraderConfig instance, "
                f"but '{explicit_config}' resolved to {type(grader_config).__name__}"
            )
        return grader_cls, grader_config

    return auto_discover_grader(module_name)


__all__ = [
    "_resolve_grader",
    "auto_discover_grader",
    "load_workflow",
]
