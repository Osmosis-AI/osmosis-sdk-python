"""Build an installable wheel from an agent/harness project dir.

Stages the user's project (their pyproject.toml, dependencies, and build
backend included), injects a generated shim with literal imports, and exposes
``<package>-agent`` / ``<package>-grade`` console scripts. The wheel installs
anywhere with one ``pip install`` — a rollout container or a user's own box.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PathDistribution
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from osmosis_ai._imports import raise_optional_dependency_error

# Only the Harbor backend packages rollout projects, so these ship with that
# extra rather than the base install.
try:
    import platformdirs
    import toml
except ModuleNotFoundError as _exc:
    raise_optional_dependency_error(
        _exc,
        extra="harbor",
        feature="Rollout bundle packaging",
    )

BUNDLES_DIR = platformdirs.user_cache_path("osmosis") / "bundles"

AGENT_MAIN_TEMPLATE = """\


def agent_main():
    runner.agent_main({workflow_class}, {workflow_config})
"""

GRADER_MAIN_TEMPLATE = """\


def grader_main():
    runner.grader_main({grader_class}, {grader_config})
"""

EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "dist",
    "build",
    "node_modules",
    ".pytest_cache",
    ".ruff_cache",
}

# PyPA's ``src/`` layout, which ``uv init --lib`` also produces.
SRC_LAYOUT_DIR = "src"


@dataclass(frozen=True)
class BundleInfo:
    wheel: Path
    agent_script: str | None
    grader_script: str | None
    requirements: list[str]


def agent_script_name(package: str) -> str:
    return f"{package.replace('_', '-')}-agent"


def grader_script_name(package: str) -> str:
    return f"{package.replace('_', '-')}-grade"


def _uv_executable() -> str:
    """Find uv installed with this interpreter, then fall back to PATH."""
    script_dirs = (Path(sysconfig.get_path("scripts")), Path(sys.executable).parent)
    for script_dir in dict.fromkeys(script_dirs):
        for name in ("uv", "uv.exe"):
            candidate = script_dir / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    executable = shutil.which("uv")
    if executable is None:
        raise RuntimeError(
            "uv is required to build rollout bundles; install osmosis-ai[harbor]"
        )
    return executable


def content_hash(path: Path, *, extra: str = "", exclude: Path | None = None) -> str:
    digest = hashlib.sha256(extra.encode())
    for file in sorted(
        p
        for p in path.rglob("*")
        if p.is_file()
        and not (set(p.relative_to(path).parts) & EXCLUDE_DIRS)
        and not (exclude is not None and p.is_relative_to(exclude))
    ):
        digest.update(str(file.relative_to(path)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()[:32]


def _reject_directory_symlinks(path: Path, *, exclude: Path | None) -> None:
    """Refuse to build through directory (or broken) symlinks.

    ``content_hash`` cannot see through them — ``rglob`` does not recurse into
    symlinked directories — while the ``copytree`` staging below dereferences
    them into the build, so the cache key would not describe what actually
    ships and a mutated link target would keep serving the stale cached wheel.
    A link resolving into the build output tree would even recurse. File
    symlinks stay allowed: hashing and staging both read the target's bytes,
    so the two agree. Walks the same exclusion rules as ``content_hash``.
    """
    for p in path.rglob("*"):
        if set(p.relative_to(path).parts) & EXCLUDE_DIRS:
            continue
        if exclude is not None and p.is_relative_to(exclude):
            continue
        if p.is_symlink() and not p.is_file():
            raise ValueError(
                f"bundle source contains a directory or broken symlink: "
                f"{p} -> {os.readlink(p)}; replace it with a real directory "
                "or file so the bundle cache can hash what it ships"
            )


def _stage_ignore(exclude: Path | None) -> Callable[[str, list[str]], set[str]]:
    """``copytree`` filter: the usual noise, plus a cache dir inside the source.

    ``ignore_patterns`` matches bare names, so it cannot distinguish the one
    ``bundles/`` that is this build's own output directory from any other.
    """
    patterns = shutil.ignore_patterns(*EXCLUDE_DIRS, "*.pyc")

    def ignore(src: str, names: list[str]) -> set[str]:
        ignored = set(patterns(src, names))
        if exclude is not None and exclude.parent == Path(src):
            ignored.add(exclude.name)
        return ignored

    return ignore


def package_roots(code_dir: Path) -> list[Path]:
    """Directories that may hold the import package: flat layout, then ``src/``."""
    roots = [code_dir]
    src = code_dir / SRC_LAYOUT_DIR
    if src.is_dir():
        roots.append(src)
    return roots


def find_package(code_dir: Path) -> str:
    packages = sorted(
        {
            d.name
            for root in package_roots(code_dir)
            for d in root.iterdir()
            if d.is_dir()
            and (d / "__init__.py").exists()
            and d.name not in EXCLUDE_DIRS
        }
    )
    if len(packages) != 1:
        searched = " or ".join(str(root) for root in package_roots(code_dir))
        raise ValueError(
            f"expected exactly one python package in {searched}, "
            f"found {packages or 'none'}; pass package= explicitly"
        )
    return packages[0]


def find_package_dir(code_dir: Path, package: str) -> Path:
    """Resolve *package* to its directory under a flat or ``src/`` layout.

    The shim has to land in the directory the build backend will actually
    package, which is ``src/<package>/`` for a src-layout project — writing it
    to ``<project>/<package>/`` silently produces a wheel with no runner.
    """
    roots = package_roots(code_dir)
    # ``__init__.py`` first, so a regular package always wins the flat-vs-src
    # tie; then any directory, which is all a PEP 420 namespace package has.
    for probe in (lambda d: (d / "__init__.py").is_file(), lambda d: d.is_dir()):
        for root in roots:
            candidate = root / package
            if probe(candidate):
                return candidate
    searched = " or ".join(str(root / package) for root in roots)
    raise ValueError(f"package {package!r} not found; looked in {searched}")


def requirement_name(spec: str) -> str:
    return canonicalize_name(Requirement(spec).name)


def split_ref(ref: str, label: str) -> tuple[str, str]:
    if ":" not in ref:
        raise ValueError(f"{label} must be 'module:attr', got {ref!r}")
    module, attr = ref.rsplit(":", 1)
    return module, attr


def project_dir_for(obj: object) -> Path:
    """Locate the project dir (containing pyproject.toml) of *obj*'s package."""
    import inspect

    package_dir = Path(
        inspect.getfile(type(obj) if not isinstance(obj, type) else obj)
    ).parent
    while (package_dir.parent / "__init__.py").exists():
        package_dir = package_dir.parent
    project_dir = package_dir.parent
    # A src-layout package sits one level deeper than its project root.
    if (
        project_dir.name == SRC_LAYOUT_DIR
        and not (project_dir / "pyproject.toml").is_file()
    ):
        project_dir = project_dir.parent
    if not (project_dir / "pyproject.toml").is_file():
        raise ValueError(
            f"no pyproject.toml next to package {package_dir.name!r} "
            f"(looked in {project_dir}); pass code_dir= explicitly"
        )
    return project_dir


def extra_gated(spec: str) -> bool:
    marker = Requirement(spec).marker
    return marker is not None and "extra ==" in str(marker)


def inspect_bundle(wheel: Path) -> BundleInfo:
    """Read the bundle's console scripts and dependencies from wheel metadata.

    Extra-gated dependencies are dropped; environment markers (python_version,
    sys_platform, …) ship verbatim since pip evaluates them inside the
    container, where they may resolve differently than on this host.
    """
    dist_info = next(
        entry
        for entry in zipfile.Path(wheel).iterdir()
        if entry.name.endswith(".dist-info")
    )
    dist = PathDistribution(dist_info)
    scripts = {
        ep.name: ep.value for ep in dist.entry_points if ep.group == "console_scripts"
    }
    agent = next(
        (n for n, t in scripts.items() if t.endswith(".bundle_main:agent_main")), None
    )
    grader = next(
        (n for n, t in scripts.items() if t.endswith(".bundle_main:grader_main")), None
    )
    if agent is None and grader is None:
        raise ValueError(f"{wheel.name} is not an osmosis bundle (no runner scripts)")
    return BundleInfo(
        wheel=wheel,
        agent_script=agent,
        grader_script=grader,
        requirements=[spec for spec in dist.requires or [] if not extra_gated(spec)],
    )


def build_bundle(
    code_dir: Path,
    *,
    workflow: str | None = None,
    grader: str | None = None,
    workflow_config: str | None = None,
    grader_config: str | None = None,
    deps: list[str] | None = None,
    package: str | None = None,
    bundles_dir: Path = BUNDLES_DIR,
) -> Path:
    """Package the project at *code_dir* into a wheel; rebuild only on change.

    workflow/grader are 'module:Class' refs; the config refs point at
    module-level instances and are passed to the runner as-is. Grader-only
    bundles (workflow=None) carry just the grading script. Entries in *deps*
    replace same-named dependencies from the project's pyproject.toml.
    """
    code_dir = code_dir.resolve()
    pyproject_path = code_dir / "pyproject.toml"
    if not pyproject_path.is_file():
        raise ValueError(f"project dir must contain pyproject.toml: {code_dir}")
    if workflow is None and grader is None:
        raise ValueError("pass workflow and/or grader")
    package = package or find_package(code_dir)

    imports = []
    references = {}
    for label, ref in (
        ("workflow", workflow),
        ("grader", grader),
        ("workflow_config", workflow_config),
        ("grader_config", grader_config),
    ):
        if ref is None:
            references[label] = "None"
            continue
        module, attr = split_ref(ref, label)
        alias = f"_osmosis_{label}"
        imports.append(f"from {module} import {attr} as {alias}")
        references[label] = alias

    shim = "\n".join(imports) + "\nfrom osmosis_ai.rollout.container import runner\n"
    scripts = {}
    if workflow:
        shim += AGENT_MAIN_TEMPLATE.format(
            workflow_class=references["workflow"],
            workflow_config=references["workflow_config"],
        )
        scripts[agent_script_name(package)] = f"{package}.bundle_main:agent_main"
    if grader:
        shim += GRADER_MAIN_TEMPLATE.format(
            grader_class=references["grader"],
            grader_config=references["grader_config"],
        )
        scripts[grader_script_name(package)] = f"{package}.bundle_main:grader_main"

    project = tomllib.loads(pyproject_path.read_text()).get("project", {})
    name = project.get("name") or f"osmosis-harness-{package}"
    wheel_distribution = canonicalize_name(name, validate=True).replace("-", "_")
    build_descriptor = "\0".join(
        (
            pyproject_path.read_text(),
            shim,
            repr(deps or []),
            package,
            sys.implementation.cache_tag or "",
            sysconfig.get_platform(),
        )
    )
    # A bundles_dir inside the project would otherwise feed its own output back
    # into the build: the wheels change the cache key on every call, and the
    # staging dir below becomes a descendant of the tree being copied into it.
    # Compared resolved (code_dir is), but bundles_dir itself is left as the
    # caller wrote it so the returned path keeps its shape.
    resolved_bundles_dir = bundles_dir.resolve()
    if resolved_bundles_dir == code_dir:
        # The exclusion below only knows how to carve out a strict
        # subdirectory. Equal paths would exclude every source file from the
        # cache key and stage the tree into itself until copytree fails.
        raise ValueError(
            f"bundles_dir must not be the project directory itself ({code_dir}); "
            "use a subdirectory such as code_dir / 'bundles'"
        )
    nested_cache = (
        resolved_bundles_dir if resolved_bundles_dir.is_relative_to(code_dir) else None
    )
    _reject_directory_symlinks(code_dir, exclude=nested_cache)
    bundle_key = content_hash(
        code_dir,
        extra=build_descriptor,
        exclude=nested_cache,
    )
    distribution_dir = bundles_dir / wheel_distribution
    cache_dir = distribution_dir / bundle_key
    distribution_dir.mkdir(parents=True, exist_ok=True)

    cached = sorted(path for path in cache_dir.glob("*.whl") if path.is_file())
    if len(cached) == 1:
        return cached[0]
    if cache_dir.exists():
        # Complete entries appear with one atomic directory rename below. Never
        # delete here: another builder may have published between the glob and
        # exists checks, and callers may already hold its returned wheel path.
        cached = sorted(path for path in cache_dir.glob("*.whl") if path.is_file())
        if len(cached) == 1:
            return cached[0]
        raise RuntimeError(
            f"invalid bundle cache entry: {cache_dir}; delete this directory and retry"
        )

    with tempfile.TemporaryDirectory(
        dir=distribution_dir, prefix=".build-"
    ) as build_root:
        root = Path(build_root)
        stage = root / "source"
        output = root / "wheel"
        shutil.copytree(
            code_dir,
            stage,
            dirs_exist_ok=True,
            ignore=_stage_ignore(nested_cache),
        )
        (find_package_dir(stage, package) / "bundle_main.py").write_text(shim)
        staged = tomllib.loads((stage / "pyproject.toml").read_text())
        staged_project = staged.setdefault("project", {})
        overridden = {requirement_name(d) for d in deps or []}
        staged_project["dependencies"] = [
            d
            for d in staged_project.get("dependencies", [])
            if requirement_name(d) not in overridden
        ] + list(deps or [])
        staged_project["scripts"] = {**staged_project.get("scripts", {}), **scripts}
        (stage / "pyproject.toml").write_text(toml.dumps(staged))
        subprocess.run(
            [
                _uv_executable(),
                "build",
                "--wheel",
                "--out-dir",
                str(output),
                str(stage),
            ],
            check=True,
            capture_output=True,
        )
        wheels = sorted(path for path in output.glob("*.whl") if path.is_file())
        if len(wheels) != 1:
            raise RuntimeError(
                f"uv build produced {len(wheels)} wheels in {output}; "
                "expected exactly one"
            )
        wheel_name = wheels[0].name
        try:
            output.replace(cache_dir)
        except OSError:
            # Another process may have published the same content-addressed
            # entry while this build was running. Its complete entry wins.
            cached = sorted(path for path in cache_dir.glob("*.whl") if path.is_file())
            if len(cached) == 1:
                return cached[0]
            raise

    return cache_dir / wheel_name
