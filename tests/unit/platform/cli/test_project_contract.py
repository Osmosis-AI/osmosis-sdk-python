from __future__ import annotations

from pathlib import Path

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.cli.project_contract import (
    ensure_context_path,
    ensure_project_config_path,
    resolve_project_root_from_cwd,
    validate_project_contract,
)


def _make_project(root: Path) -> Path:
    (root / ".osmosis").mkdir(parents=True)
    (root / ".osmosis" / "project.toml").write_text("[project]\n", encoding="utf-8")
    (root / ".osmosis" / "program.md").write_text("# Test Program\n", encoding="utf-8")
    (root / "configs" / "eval").mkdir(parents=True)
    (root / "configs" / "training").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "rollouts" / "demo").mkdir(parents=True)
    return root


def test_resolve_project_root_from_cwd_uses_nearest_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = _make_project(tmp_path / "parent")
    child = _make_project(parent / "nested")
    monkeypatch.chdir(child / "configs")

    assert resolve_project_root_from_cwd() == child.resolve()


def test_resolve_project_root_from_cwd_reports_missing_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(CLIError, match="Not in an Osmosis project"):
        resolve_project_root_from_cwd()


def test_validate_project_contract_requires_program_file(tmp_path: Path) -> None:
    project = _make_project(tmp_path / "project")
    (project / ".osmosis" / "program.md").unlink()

    with pytest.raises(CLIError) as exc:
        validate_project_contract(project)

    message = str(exc.value)
    assert ".osmosis/program.md" in message
    assert "osmosis project doctor --fix" in message


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
