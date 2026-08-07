"""Focused tests for lightweight package initializers and public facades."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).parents[2]


def _run_python(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_rollout_root_exports_only_framework_neutral_core() -> None:
    result = _run_python(
        """
        import sys
        import osmosis_ai.rollout as rollout

        expected = {
            "AgentWorkflow",
            "AgentWorkflowConfig",
            "AgentWorkflowContext",
            "AgentWorkflowOutput",
            "BaseConfig",
            "ConcurrencyConfig",
            "ExecutionBackend",
            "ExecutionRequest",
            "ExecutionResult",
            "Grader",
            "GraderCompleteRequest",
            "GraderConfig",
            "GraderContext",
            "GraderInitRequest",
            "GraderInitResponse",
            "GraderStatus",
            "LocalBackend",
            "MessageDict",
            "RolloutCompleteRequest",
            "RolloutContext",
            "RolloutErrorCategory",
            "RolloutInitRequest",
            "RolloutInitResponse",
            "RolloutSample",
            "RolloutStatus",
            "SampleSource",
            "get_rollout_context",
        }
        assert set(rollout.__all__) == expected

        removed = {
            "ControllerAuth",
            "HarborAgentWorkflowContext",
            "OsmosisRolloutModel",
            "OsmosisStrandsAgent",
            "create_rollout_server",
        }
        assert not (removed & set(dir(rollout)))

        optional_roots = {"agents", "fastapi", "harbor", "litellm", "strands"}
        loaded_roots = {name.partition(".")[0] for name in sys.modules}
        assert not (optional_roots & loaded_roots)
        """
    )
    assert result.returncode == 0, result.stderr


def test_top_level_star_import_is_safe_without_rubric_dependencies() -> None:
    result = _run_python(
        """
        import builtins

        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "litellm" or name.startswith("litellm."):
                raise ModuleNotFoundError("No module named 'litellm'", name="litellm")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import
        namespace = {}
        exec("from osmosis_ai import *", namespace)
        assert "evaluate_rubric" not in namespace
        assert "RubricResult" in namespace

        import osmosis_ai
        assert "evaluate_rubric" in dir(osmosis_ai)
        """
    )
    assert result.returncode == 0, result.stderr


def test_integration_namespaces_do_not_select_a_framework() -> None:
    result = _run_python(
        """
        import sys
        import osmosis_ai.rollout.integrations as integrations
        import osmosis_ai.rollout.integrations.agents as agent_integrations

        assert integrations.__all__ == []
        assert agent_integrations.__all__ == []
        optional_roots = {"agents", "litellm", "strands"}
        loaded_roots = {name.partition(".")[0] for name in sys.modules}
        assert not (optional_roots & loaded_roots)
        """
    )
    assert result.returncode == 0, result.stderr


def test_framework_neutral_core_imports_without_optional_dependencies() -> None:
    """A bare install must reach everything a rollout container needs.

    Bundle wheels install `osmosis-ai` inside the task container without the
    harbor extra, so the in-container runner, its file contract, and ATIF
    persistence have to resolve with only the base dependencies.
    """
    result = _run_python(
        """
        import builtins
        import importlib

        blocked = {
            "agents",
            "aiohttp",
            "dockerfile_parse",
            "fastapi",
            "harbor",
            "litellm",
            "mcp",
            "openai",
            "orjson",
            "platformdirs",
            "pyarrow",
            "strands",
            "toml",
            "tqdm",
            "uvicorn",
        }
        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            root = name.partition(".")[0]
            if root in blocked:
                raise ModuleNotFoundError("No module named " + repr(root), name=root)
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import
        for module_name in (
            "osmosis_ai.rollout",
            "osmosis_ai.rollout.container.files",
            "osmosis_ai.rollout.container.runner",
            "osmosis_ai.rollout.container.trajectories",
            "osmosis_ai.rollout.trajectory.atif",
            "osmosis_ai.rollout.trajectory.converter",
            "osmosis_ai.rollout.trajectory.save",
            "osmosis_ai.rollout.types.output",
            "osmosis_ai.rollout.types.protocol",
            "osmosis_ai.rollout.utils.errors",
            "osmosis_ai.rollout.utils.ttl_cache",
        ):
            importlib.import_module(module_name)
        """
    )
    assert result.returncode == 0, result.stderr


def test_facades_resolve_each_symbol_from_its_leaf_module() -> None:
    result = _run_python(
        """
        import sys

        import osmosis_ai.eval.rubric as rubric
        import osmosis_ai.platform.auth as auth
        import osmosis_ai.rollout.server as server
        import osmosis_ai.rollout.trajectory as trajectory
        import osmosis_ai.templates as templates

        assert "osmosis_ai.eval.rubric.engine" not in sys.modules
        assert rubric.RubricResult.__module__ == "osmosis_ai.eval.rubric.types"
        assert "osmosis_ai.eval.rubric.engine" not in sys.modules

        assert "osmosis_ai.rollout.server.app" not in sys.modules
        assert server.ControllerAuth.__module__ == "osmosis_ai.rollout.server.auth"
        assert "osmosis_ai.rollout.server.app" not in sys.modules

        assert "osmosis_ai.rollout.trajectory.converter" not in sys.modules
        assert trajectory.TrajectoryReport.__module__ == (
            "osmosis_ai.rollout.trajectory.report"
        )
        assert "osmosis_ai.rollout.trajectory.converter" not in sys.modules

        assert "osmosis_ai.platform.auth.credentials" not in sys.modules
        assert auth.CONFIG_DIR.name == "osmosis"
        assert "osmosis_ai.platform.auth.credentials" not in sys.modules

        assert "osmosis_ai.templates.registry" not in sys.modules
        assert "list_templates" in dir(templates)
        assert "osmosis_ai.templates.registry" not in sys.modules
        """
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("module_path", "symbol", "missing_module", "extra"),
    [
        (
            "osmosis_ai.rollout.server",
            "create_rollout_server",
            "fastapi",
            "server",
        ),
        (
            "osmosis_ai.rollout.integrations.agents.strands",
            "OsmosisStrandsAgent",
            "strands",
            "strands",
        ),
        (
            "osmosis_ai.rollout.integrations.agents.openai_agents",
            "OsmosisAgent",
            "agents",
            "openai-agents",
        ),
        (
            "osmosis_ai.rollout.integrations.agents.openai_agents",
            "OsmosisAgent",
            "litellm",
            "openai-agents",
        ),
        (
            "osmosis_ai.rollout.backend.harbor",
            "HarborBackend",
            "harbor",
            "harbor",
        ),
        (
            "osmosis_ai.rollout.backend.harbor",
            "TaskMode",
            "harbor",
            "harbor",
        ),
        # The bundle builder reaches osmosis_ai.packaging, whose toml and
        # platformdirs imports ship with the harbor extra.
        (
            "osmosis_ai.rollout.backend.harbor",
            "HarborBackend",
            "toml",
            "harbor",
        ),
        (
            "osmosis_ai.rollout.backend.harbor",
            "HarborBackend",
            "platformdirs",
            "harbor",
        ),
        (
            "osmosis_ai.eval.rubric",
            "evaluate_rubric",
            "litellm",
            "rubric",
        ),
        (
            "osmosis_ai.eval.rubric.cli",
            "RubricCommand",
            "litellm",
            "rubric",
        ),
    ],
)
def test_missing_optional_dependency_names_the_install_extra(
    module_path: str,
    symbol: str,
    missing_module: str,
    extra: str,
) -> None:
    result = _run_python(
        f"""
        import builtins
        import importlib

        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == {missing_module!r} or name.startswith({missing_module!r} + "."):
                raise ModuleNotFoundError(
                    "No module named " + repr({missing_module!r}),
                    name={missing_module!r},
                )
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import
        try:
            module = importlib.import_module({module_path!r})
            getattr(module, {symbol!r})
        except ModuleNotFoundError as exc:
            expected = 'pip install "osmosis-ai[{extra}]"'
            assert expected in str(exc), str(exc)
            assert exc.name == {missing_module!r}
        else:
            raise AssertionError("expected ModuleNotFoundError")
        """
    )
    assert result.returncode == 0, result.stderr


def test_cli_main_from_package_remains_a_module() -> None:
    from osmosis_ai.cli import main

    assert isinstance(main, ModuleType)


def test_agent_integration_classes_use_canonical_module_paths() -> None:
    from osmosis_ai.rollout.integrations.agents.openai_agents import OsmosisAgent
    from osmosis_ai.rollout.integrations.agents.strands import OsmosisStrandsAgent

    assert (
        OsmosisAgent.__module__
        == "osmosis_ai.rollout.integrations.agents.openai_agents"
    )
    assert (
        OsmosisStrandsAgent.__module__
        == "osmosis_ai.rollout.integrations.agents.strands"
    )
