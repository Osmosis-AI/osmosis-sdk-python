from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.cli import project_contract
from osmosis_ai.platform.cli.project_contract import (
    ensure_context_path,
    ensure_project_config_path,
    resolve_project_root_from_cwd,
    validate_project_contract,
)


def _make_git_repo(path: Path) -> None:
    subprocess.run(
        ["git", "init", "-b", "main", str(path)], check=True, capture_output=True
    )


def _write_required_scaffold(path: Path) -> None:
    for rel in ("rollouts", "configs/training", "configs/eval", "data"):
        (path / rel).mkdir(parents=True, exist_ok=True)


def _make_project(root: Path) -> Path:
    _make_git_repo(root)
    _write_required_scaffold(root)
    (root / "rollouts" / "demo").mkdir(parents=True)
    return root


def test_resolve_project_root_uses_git_top_level_for_subdirectory(
    tmp_path: Path,
) -> None:
    _make_git_repo(tmp_path)
    _write_required_scaffold(tmp_path)
    nested = tmp_path / "rollouts" / "demo"
    nested.mkdir(parents=True)

    assert project_contract.resolve_project_root(nested) == tmp_path.resolve()


def test_resolve_project_root_from_cwd_uses_git_top_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _make_project(tmp_path / "project")
    monkeypatch.chdir(project / "configs")

    assert resolve_project_root_from_cwd() == project.resolve()


def test_resolve_project_root_from_cwd_reports_missing_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(CLIError) as exc:
        resolve_project_root_from_cwd()

    assert "cloned Osmosis repository created by Platform" in str(exc.value)


def test_validate_project_contract_does_not_require_training_brief(
    tmp_path: Path,
) -> None:
    project = _make_project(tmp_path / "project")

    validate_project_contract(project)


def test_validate_contract_accepts_scaffold_without_project_toml(
    tmp_path: Path,
) -> None:
    _make_git_repo(tmp_path)
    _write_required_scaffold(tmp_path)

    project_contract.validate_project_contract(tmp_path)


def test_validate_contract_reports_missing_scaffold_without_requiring_dot_osmosis(
    tmp_path: Path,
) -> None:
    _make_git_repo(tmp_path)
    (tmp_path / "rollouts").mkdir()

    missing = project_contract.missing_project_paths(tmp_path)

    assert missing == ["configs/training/", "configs/eval/", "data/"]
    assert ".osmosis/project.toml" not in missing

    with pytest.raises(CLIError) as exc:
        project_contract.validate_project_contract(tmp_path)

    message = str(exc.value)
    assert "This checkout is missing required Osmosis scaffold paths." in message
    assert "configs/training/" in message
    assert "configs/eval/" in message
    assert "data/" in message
    assert "osmosis project doctor --fix" in message
    assert ".osmosis/project.toml" not in message


def test_resolve_project_root_rejects_non_git_directory(tmp_path: Path) -> None:
    with pytest.raises(CLIError) as exc:
        project_contract.resolve_project_root(tmp_path)

    assert "cloned Osmosis repository created by Platform" in str(exc.value)


def test_ensure_context_path_accepts_canonical_config(tmp_path: Path) -> None:
    project = _make_project(tmp_path / "project")
    config = project / "configs" / "eval" / "default.toml"
    config.write_text("[eval]\n", encoding="utf-8")

    assert (
        ensure_context_path(
            config, project, required_dir="configs/eval", label="eval config"
        )
        == config.resolve()
    )


def test_ensure_context_path_resolves_relative_path_against_project_root(
    tmp_path: Path,
) -> None:
    project = _make_project(tmp_path / "project")
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
    project = _make_project(tmp_path / "project")
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


def test_ensure_project_config_path_rejects_wrong_suffix_with_config_error_shape(
    tmp_path: Path,
) -> None:
    project = _make_project(tmp_path / "project")
    config = project / "configs" / "training" / "default.yaml"
    config.write_text("training: {}\n", encoding="utf-8")

    with pytest.raises(CLIError) as exc:
        ensure_project_config_path(
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
    project = _make_project(tmp_path / "project")
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
    project = _make_project(tmp_path / "project")
    config = project / "configs" / "eval" / "default.toml"
    config.write_text("[eval]\n", encoding="utf-8")

    with pytest.raises(CLIError, match="required_dir must be relative"):
        ensure_context_path(
            config,
            project,
            required_dir=required_dir,
            label="eval config",
        )
