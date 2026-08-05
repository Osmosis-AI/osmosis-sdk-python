from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.cli import workspace_directory_contract
from osmosis_ai.platform.cli.workspace_directory_contract import (
    ensure_context_path,
    ensure_workspace_directory_config_path,
    resolve_workspace_directory_from_cwd,
    validate_workspace_directory_contract,
)


def _make_git_repo(path: Path) -> None:
    subprocess.run(
        ["git", "init", "-b", "main", str(path)], check=True, capture_output=True
    )


def _write_required_scaffold(path: Path) -> None:
    for rel in ("rollouts", "configs/training", "configs/eval", "data"):
        (path / rel).mkdir(parents=True, exist_ok=True)


def _make_workspace_directory(root: Path) -> Path:
    _make_git_repo(root)
    _write_required_scaffold(root)
    (root / "rollouts" / "demo").mkdir(parents=True)
    return root


def test_resolve_workspace_directory_uses_git_top_level_for_subdirectory(
    tmp_path: Path,
) -> None:
    _make_git_repo(tmp_path)
    _write_required_scaffold(tmp_path)
    nested = tmp_path / "rollouts" / "demo"
    nested.mkdir(parents=True)

    assert (
        workspace_directory_contract.resolve_workspace_directory(nested)
        == tmp_path.resolve()
    )


def test_resolve_workspace_directory_uses_git_top_level_for_file_path(
    tmp_path: Path,
) -> None:
    _make_git_repo(tmp_path)
    _write_required_scaffold(tmp_path)
    config = tmp_path / "configs" / "training" / "default.toml"
    config.write_text("[training]\n", encoding="utf-8")

    assert (
        workspace_directory_contract.resolve_workspace_directory(config)
        == tmp_path.resolve()
    )


def test_resolve_workspace_directory_from_cwd_uses_git_top_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _make_workspace_directory(tmp_path / "project")
    monkeypatch.chdir(project / "configs")

    assert resolve_workspace_directory_from_cwd() == project.resolve()


def test_resolve_workspace_directory_from_cwd_reports_missing_workspace_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(CLIError) as exc:
        resolve_workspace_directory_from_cwd()

    assert "Osmosis workspace directory created by Platform" in str(exc.value)
    assert exc.value.code == "WORKSPACE_REQUIRED"


def test_validate_workspace_directory_contract_does_not_require_training_brief(
    tmp_path: Path,
) -> None:
    project = _make_workspace_directory(tmp_path / "project")

    validate_workspace_directory_contract(project)


def test_validate_contract_accepts_scaffold_without_project_toml(
    tmp_path: Path,
) -> None:
    _make_git_repo(tmp_path)
    _write_required_scaffold(tmp_path)

    workspace_directory_contract.validate_workspace_directory_contract(tmp_path)


def test_validate_contract_reports_missing_scaffold_without_requiring_dot_osmosis(
    tmp_path: Path,
) -> None:
    _make_git_repo(tmp_path)
    (tmp_path / "rollouts").mkdir()

    missing = workspace_directory_contract.missing_workspace_directory_paths(tmp_path)

    assert missing == ["configs/training/", "configs/eval/", "data/"]
    assert ".osmosis/project.toml" not in missing

    with pytest.raises(CLIError) as exc:
        workspace_directory_contract.validate_workspace_directory_contract(tmp_path)

    message = str(exc.value)
    assert (
        "This workspace directory is missing required Osmosis scaffold paths."
        in message
    )
    assert "configs/training/" in message
    assert "configs/eval/" in message
    assert "data/" in message
    assert "osmosis doctor --fix" in message
    assert ".osmosis/project.toml" not in message


def test_resolve_workspace_directory_rejects_non_git_directory(tmp_path: Path) -> None:
    with pytest.raises(CLIError) as exc:
        workspace_directory_contract.resolve_workspace_directory(tmp_path)

    assert "Osmosis workspace directory created by Platform" in str(exc.value)
    assert exc.value.code == "WORKSPACE_REQUIRED"


def test_ensure_context_path_accepts_canonical_config(tmp_path: Path) -> None:
    project = _make_workspace_directory(tmp_path / "project")
    config = project / "configs" / "eval" / "default.toml"
    config.write_text("[eval]\n", encoding="utf-8")

    assert (
        ensure_context_path(
            config, project, required_dir="configs/eval", label="eval config"
        )
        == config.resolve()
    )


def test_ensure_context_path_resolves_relative_path_against_workspace_directory(
    tmp_path: Path,
) -> None:
    project = _make_workspace_directory(tmp_path / "project")
    config = project / "configs" / "eval" / "default.toml"
    config.write_text("[eval]\n", encoding="utf-8")

    assert (
        ensure_context_path(
            Path("configs/eval/default.toml"),
            project,
            required_dir="configs/eval",
            label="eval config",
        )
        == config.resolve()
    )


def test_ensure_context_path_rejects_wrong_suffix_under_required_dir(
    tmp_path: Path,
) -> None:
    project = _make_workspace_directory(tmp_path / "project")
    config = project / "configs" / "eval" / "default.yaml"
    config.write_text("eval: {}\n", encoding="utf-8")

    with pytest.raises(
        CLIError, match=r"eval config must be a \.toml file under `configs/eval/`"
    ):
        ensure_context_path(
            config,
            project,
            required_dir="configs/eval",
            label="eval config",
            suffix=".toml",
        )


def test_ensure_workspace_directory_config_path_rejects_wrong_suffix_with_config_error_shape(
    tmp_path: Path,
) -> None:
    project = _make_workspace_directory(tmp_path / "project")
    config = project / "configs" / "training" / "default.yaml"
    config.write_text("training: {}\n", encoding="utf-8")

    with pytest.raises(CLIError) as exc:
        ensure_workspace_directory_config_path(
            config,
            project,
            config_dir="configs/training",
            command_label="train",
        )

    message = str(exc.value)
    assert "train config must be a .toml file under `configs/training/`" in message
    assert "got:" in message
    assert str(config.resolve()) in message


def test_ensure_context_path_rejects_project_external_symlink(tmp_path: Path) -> None:
    project = _make_workspace_directory(tmp_path / "project")
    outside = tmp_path / "outside.toml"
    outside.write_text("[eval]\n", encoding="utf-8")
    link = project / "configs" / "eval" / "outside.toml"
    link.symlink_to(outside)

    with pytest.raises(CLIError, match="must live under `configs/eval/`"):
        ensure_context_path(
            link, project, required_dir="configs/eval", label="eval config"
        )


@pytest.mark.parametrize("required_dir", ["/configs/eval", "configs/../data"])
def test_ensure_context_path_rejects_invalid_required_dir(
    tmp_path: Path, required_dir: str
) -> None:
    project = _make_workspace_directory(tmp_path / "project")
    config = project / "configs" / "eval" / "default.toml"
    config.write_text("[eval]\n", encoding="utf-8")

    with pytest.raises(CLIError, match="required_dir must be relative"):
        ensure_context_path(
            config,
            project,
            required_dir=required_dir,
            label="eval config",
        )


def _make_rollout(
    workspace_directory: Path, name: str, *, dependencies: str, entrypoint: str
) -> None:
    rollout_dir = workspace_directory / "rollouts" / name
    rollout_dir.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\nversion = "0.1.0"\n'
        f"dependencies = [{dependencies}]\n",
        encoding="utf-8",
    )
    (rollout_dir / "main.py").write_text(entrypoint, encoding="utf-8")


# `osmosis-ai` is installed whenever these tests run, so it is a dependency the
# gate can reason about concretely in both directions.
_SATISFIED = '"osmosis-ai"'
_UNSATISFIED = '"osmosis-ai>=999.0.0"'

# A native Harbor entrypoint: the backend runs Harbor's own agent and the
# task verifier supplies the reward, so no AgentWorkflow or Grader exists.
# The load counter proves preflight executes module-level side effects once.
_NATIVE_V2_ENTRYPOINT = """\
from pathlib import Path

from harbor.trial.queue import TrialQueue

from osmosis_ai.rollout.backend.harbor.backend_v2 import HarborBackendV2
from osmosis_ai.rollout.server import create_rollout_server

_count_file = Path(__file__).with_name("load_count")
_count = int(_count_file.read_text()) if _count_file.exists() else 0
_count_file.write_text(str(_count + 1))

backend = HarborBackendV2(
    orchestrator=TrialQueue(n_concurrent=1),
    tasks_dir=Path(__file__).parent / "tasks",
    agent="terminus-2",
    grader=None,
)
app = create_rollout_server(backend=backend)
"""

# A backend with no workflow or grader: CLI preflight must not infer either
# requirement from classes present (or absent) in the module namespace.
_COMPONENT_FREE_BACKEND_ENTRYPOINT = """\
from osmosis_ai.rollout.backend.base import ExecutionBackend
from osmosis_ai.rollout.server import create_rollout_server


class ComponentFreeBackend(ExecutionBackend):
    async def execute(self, request, on_workflow_complete, on_grader_complete=None):
        return None


app = create_rollout_server(backend=ComponentFreeBackend())
"""

# A script-style entrypoint with no module-level backend. Importability is the
# only CLI concern; the script validates its backend when it constructs one.
_WORKFLOW_WITHOUT_GRADER_ENTRYPOINT = """\
from osmosis_ai.rollout.agent_workflow import AgentWorkflow


class DemoWorkflow(AgentWorkflow):
    async def run(self, ctx):
        return None
"""

# A backend whose constructor rejects its configuration: preflight surfaces
# the import-time error instead of running any validation of its own.
_INVALID_BACKEND_ENTRYPOINT = """\
from osmosis_ai.rollout.backend.base import ExecutionBackend
from osmosis_ai.rollout.server import create_rollout_server


class InvalidBackend(ExecutionBackend):
    def __init__(self):
        raise ValueError("backend constructor rejected the configuration")

    async def execute(self, request, on_workflow_complete, on_grader_complete=None):
        return None


app = create_rollout_server(backend=InvalidBackend())
"""

# A rollout naming an enum member that only exists in a newer dependency. The
# import alone cannot tell this from a typo.
_VERSION_SKEW_ENTRYPOINT = (
    "import enum\n\n\nclass E(enum.Enum):\n    A = 'a'\n\n\nE.MISSING_MEMBER\n"
)


def test_unsatisfied_requirements_empty_when_environment_matches(
    tmp_path: Path,
) -> None:
    project = _make_workspace_directory(tmp_path / "project")
    _make_rollout(project, "demo", dependencies=_SATISFIED, entrypoint="")

    assert (
        workspace_directory_contract._unsatisfied_rollout_requirements(
            project / "rollouts" / "demo"
        )
        == []
    )


def test_unsatisfied_requirements_reports_version_skew(tmp_path: Path) -> None:
    project = _make_workspace_directory(tmp_path / "project")
    _make_rollout(project, "demo", dependencies=_UNSATISFIED, entrypoint="")

    unsatisfied = workspace_directory_contract._unsatisfied_rollout_requirements(
        project / "rollouts" / "demo"
    )

    assert len(unsatisfied) == 1
    assert "osmosis-ai" in unsatisfied[0]
    assert ">=999.0.0" in unsatisfied[0]


def test_unsatisfied_requirements_reports_missing_distribution(
    tmp_path: Path,
) -> None:
    project = _make_workspace_directory(tmp_path / "project")
    _make_rollout(
        project, "demo", dependencies='"definitely-not-installed-xyz"', entrypoint=""
    )

    unsatisfied = workspace_directory_contract._unsatisfied_rollout_requirements(
        project / "rollouts" / "demo"
    )

    assert len(unsatisfied) == 1
    assert "not installed" in unsatisfied[0]


@pytest.mark.parametrize(
    "dependency",
    [
        # A direct URL pin carries no version to compare against.
        '"osmosis-ai @ git+https://github.com/Osmosis-AI/osmosis-sdk-python.git@main"',
        # A marker that cannot apply says nothing about this environment.
        "\"definitely-not-installed-xyz; python_version < '3.0'\"",
        # Unparseable requirements are the resolver's problem to report.
        '"!!!not a requirement!!!"',
    ],
)
def test_unsatisfied_requirements_ignores_uncomparable_dependencies(
    tmp_path: Path, dependency: str
) -> None:
    project = _make_workspace_directory(tmp_path / "project")
    _make_rollout(project, "demo", dependencies=dependency, entrypoint="")

    assert (
        workspace_directory_contract._unsatisfied_rollout_requirements(
            project / "rollouts" / "demo"
        )
        == []
    )


def test_unsatisfied_requirements_empty_without_pyproject(tmp_path: Path) -> None:
    project = _make_workspace_directory(tmp_path / "project")
    (project / "rollouts" / "demo").mkdir(parents=True, exist_ok=True)

    assert (
        workspace_directory_contract._unsatisfied_rollout_requirements(
            project / "rollouts" / "demo"
        )
        == []
    )


def test_validate_rollout_backend_accepts_native_backend_without_workflow(
    tmp_path: Path,
) -> None:
    """A backend that runs Harbor's own agent with verifier rewards needs no
    dummy AgentWorkflow or Grader to pass preflight."""
    project = _make_workspace_directory(tmp_path / "project")
    _make_rollout(
        project,
        "demo",
        dependencies=_SATISFIED,
        entrypoint=_NATIVE_V2_ENTRYPOINT,
    )

    warnings = workspace_directory_contract.validate_rollout_backend(
        workspace_directory=project,
        rollout="demo",
        entrypoint="main.py",
        command_label="eval submit",
    )

    assert warnings == []


def test_validate_rollout_backend_does_not_discover_components(
    tmp_path: Path,
) -> None:
    project = _make_workspace_directory(tmp_path / "project")
    _make_rollout(
        project,
        "demo",
        dependencies=_SATISFIED,
        entrypoint=_COMPONENT_FREE_BACKEND_ENTRYPOINT,
    )

    warnings = workspace_directory_contract.validate_rollout_backend(
        workspace_directory=project,
        rollout="demo",
        entrypoint="main.py",
        command_label="eval submit",
    )

    assert warnings == []


def test_validate_rollout_backend_loads_entrypoint_once(tmp_path: Path) -> None:
    """Preflight must not execute user module side effects more than once."""
    project = _make_workspace_directory(tmp_path / "project")
    _make_rollout(
        project,
        "demo",
        dependencies=_SATISFIED,
        entrypoint=_NATIVE_V2_ENTRYPOINT,
    )

    workspace_directory_contract.validate_rollout_backend(
        workspace_directory=project,
        rollout="demo",
        entrypoint="main.py",
        command_label="eval submit",
    )

    assert (project / "rollouts" / "demo" / "load_count").read_text() == "1"


def test_validate_rollout_backend_accepts_script_without_grader(
    tmp_path: Path,
) -> None:
    project = _make_workspace_directory(tmp_path / "project")
    _make_rollout(
        project,
        "demo",
        dependencies=_SATISFIED,
        entrypoint=_WORKFLOW_WITHOUT_GRADER_ENTRYPOINT,
    )

    warnings = workspace_directory_contract.validate_rollout_backend(
        workspace_directory=project,
        rollout="demo",
        entrypoint="main.py",
        command_label="eval submit",
    )

    assert warnings == []


def test_validate_rollout_backend_reports_constructor_error(
    tmp_path: Path,
) -> None:
    project = _make_workspace_directory(tmp_path / "project")
    _make_rollout(
        project,
        "demo",
        dependencies=_SATISFIED,
        entrypoint=_INVALID_BACKEND_ENTRYPOINT,
    )

    with pytest.raises(CLIError, match="backend constructor rejected"):
        workspace_directory_contract.validate_rollout_backend(
            workspace_directory=project,
            rollout="demo",
            entrypoint="main.py",
            command_label="eval submit",
        )


def test_validate_rollout_backend_skips_and_warns_on_version_skew(
    tmp_path: Path,
) -> None:
    """An unrepresentative environment must not fail a rollout it cannot judge."""
    project = _make_workspace_directory(tmp_path / "project")
    _make_rollout(
        project,
        "demo",
        dependencies=_UNSATISFIED,
        entrypoint=_VERSION_SKEW_ENTRYPOINT,
    )

    warnings = workspace_directory_contract.validate_rollout_backend(
        workspace_directory=project,
        rollout="demo",
        entrypoint="main.py",
        command_label="eval submit",
    )

    assert len(warnings) == 1
    assert "rollouts/demo" in warnings[0]
    assert "osmosis-ai" in warnings[0]


def test_validate_rollout_backend_still_fails_when_environment_matches(
    tmp_path: Path,
) -> None:
    """A representative environment means an import failure is the rollout's bug."""
    project = _make_workspace_directory(tmp_path / "project")
    _make_rollout(
        project,
        "demo",
        dependencies=_SATISFIED,
        entrypoint=_VERSION_SKEW_ENTRYPOINT,
    )

    with pytest.raises(CLIError, match="preflight failed"):
        workspace_directory_contract.validate_rollout_backend(
            workspace_directory=project,
            rollout="demo",
            entrypoint="main.py",
            command_label="eval submit",
        )


@pytest.mark.parametrize("rollout", ["demo/..", "../rollouts/demo", "demo/nested"])
def test_validate_rollout_backend_rejects_multi_segment_rollout_names(
    tmp_path: Path, rollout: str
) -> None:
    """`demo/..` resolves back onto `rollouts/` itself, which would still pass the
    containment check and read the wrong pyproject.toml."""
    project = _make_workspace_directory(tmp_path / "project")
    _make_rollout(project, "demo", dependencies=_SATISFIED, entrypoint="")

    with pytest.raises(CLIError, match="single-segment name"):
        workspace_directory_contract.validate_rollout_backend(
            workspace_directory=project,
            rollout=rollout,
            entrypoint="main.py",
            command_label="eval submit",
        )


def test_validate_rollout_backend_rejects_escaping_entrypoints(tmp_path: Path) -> None:
    project = _make_workspace_directory(tmp_path / "project")
    _make_rollout(project, "demo", dependencies=_SATISFIED, entrypoint="")

    with pytest.raises(CLIError, match="entrypoint must resolve under"):
        workspace_directory_contract.validate_rollout_backend(
            workspace_directory=project,
            rollout="demo",
            entrypoint="../../main.py",
            command_label="eval submit",
        )
