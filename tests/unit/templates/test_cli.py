"""Unit tests for the template CLI business logic."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import OutputFormat
from osmosis_ai.cli.output.context import override_output_context
from osmosis_ai.templates.cli import _next_steps, apply_command, list_command


def _make_project(root: Path) -> Path:
    """Create the minimum directory layout that resolve_project_root accepts."""
    subprocess.run(
        ["git", "init", "-b", "main", str(root)],
        check=True,
        capture_output=True,
    )
    return root


# ── list_command ─────────────────────────────────────────────────


def test_list_command_returns_list_result_in_json() -> None:
    with override_output_context(format=OutputFormat.json):
        result = list_command()

    assert result is not None
    assert any(item["name"] == "multiply" for item in result.items)
    assert result.total_count == len(result.items)
    assert result.has_more is False


def test_list_command_returns_none_in_rich() -> None:
    # Rich-mode list_command prints inline through the shared Console
    # (which caches sys.stdout at import time, defeating capsys/capfd);
    # the contract under test here is that the function returns None so
    # the result_callback skips the structured renderer.
    with override_output_context(format=OutputFormat.rich):
        result = list_command()

    assert result is None


# ── apply_command ────────────────────────────────────────────────


def test_next_steps_use_git_scoped_training_flow() -> None:
    next_steps = _next_steps("multiply")
    rendered = "\n".join(next_steps)

    assert "Git Sync" not in rendered
    assert "project link" not in rendered
    assert "project unlink" not in rendered
    assert ".osmosis/project.toml" not in rendered
    assert "linked workspace" not in rendered
    assert "X-Osmosis-Org" not in rendered
    assert "osmosis train submit configs/training/multiply.toml" in next_steps


def test_apply_command_writes_directly_into_project_canonical_layout(
    tmp_path, monkeypatch
) -> None:
    """`apply` lands files at rollouts/<name>/ and configs/{eval,training}/."""
    project_root = _make_project(tmp_path)
    monkeypatch.chdir(project_root)

    with override_output_context(format=OutputFormat.json):
        result = apply_command("multiply")

    rollout_root = project_root / "rollouts" / "multiply"
    rollout_pkg = rollout_root / "multiply_rollout"
    # A rollout in a project is self-describing: entrypoint + deps + README.
    assert (rollout_root / "main.py").is_file()
    assert (rollout_root / "pyproject.toml").is_file()
    assert (rollout_root / "README.md").is_file()
    assert (rollout_pkg / "__init__.py").is_file()
    assert (rollout_pkg / "workflow.py").is_file()
    assert (rollout_pkg / "grader.py").is_file()
    assert (rollout_pkg / "tools.py").is_file()
    assert (rollout_pkg / "utils.py").is_file()
    assert (project_root / "configs" / "training" / "multiply.toml").is_file()
    assert (project_root / "configs" / "eval" / "multiply.toml").is_file()
    assert (project_root / "data" / "multiply.jsonl").is_file()
    # The legacy staging directory must NOT be created.
    assert not (project_root / "templates").exists()

    assert result is not None
    assert result.operation == "template.apply"
    assert result.status == "success"
    assert result.resource is not None
    assert result.resource["name"] == "multiply"
    files = result.resource["files"]
    assert "rollouts/multiply/main.py" in files
    assert "rollouts/multiply/pyproject.toml" in files
    assert "configs/training/multiply.toml" in files
    assert "configs/eval/multiply.toml" in files
    assert "data/multiply.jsonl" in files
    destinations = result.resource["destinations"]
    assert any(dest.endswith("/rollouts/multiply") for dest in destinations)
    assert result.display_next_steps == [
        "pip install -e rollouts/multiply",
        "osmosis eval run configs/eval/multiply.toml --limit 1",
        'git add . && git commit -m "add rollout template"',
        "git push",
        "osmosis train submit configs/training/multiply.toml",
    ]


def test_apply_command_refuses_overwrite_without_force(tmp_path, monkeypatch) -> None:
    project_root = _make_project(tmp_path)
    monkeypatch.chdir(project_root)

    rollout_root = project_root / "rollouts" / "multiply"
    rollout_root.mkdir(parents=True)
    user_edit = rollout_root / "main.py"
    user_edit.write_text("# user-edited\n", encoding="utf-8")

    with (
        override_output_context(format=OutputFormat.json),
        pytest.raises(CLIError) as exc_info,
    ):
        apply_command("multiply")

    assert exc_info.value.code == "CONFLICT"
    assert "rollouts/multiply/" in str(exc_info.value)
    assert "--force" in str(exc_info.value)
    # User's edit must be left untouched.
    assert user_edit.read_text(encoding="utf-8") == "# user-edited\n"


def test_apply_command_force_overwrites_existing_rollout(tmp_path, monkeypatch) -> None:
    project_root = _make_project(tmp_path)
    monkeypatch.chdir(project_root)

    rollout_root = project_root / "rollouts" / "multiply"
    rollout_root.mkdir(parents=True)
    stale = rollout_root / "stale_user_file.txt"
    stale.write_text("user edits", encoding="utf-8")

    with override_output_context(format=OutputFormat.json):
        apply_command("multiply", force=True)

    assert not stale.exists(), "--force must replace the rollout dir wholesale"
    assert (rollout_root / "main.py").is_file()
    assert (rollout_root / "pyproject.toml").is_file()


def test_apply_command_force_overwrites_existing_config(tmp_path, monkeypatch) -> None:
    project_root = _make_project(tmp_path)
    monkeypatch.chdir(project_root)

    config = project_root / "configs" / "eval" / "multiply.toml"
    config.parent.mkdir(parents=True)
    config.write_text("# stale\n", encoding="utf-8")

    with override_output_context(format=OutputFormat.json):
        apply_command("multiply", force=True)

    refreshed = config.read_text(encoding="utf-8")
    assert "# stale" not in refreshed
    assert 'rollout = "multiply"' in refreshed


def test_apply_command_preserves_unrelated_user_files(tmp_path, monkeypatch) -> None:
    project_root = _make_project(tmp_path)
    (project_root / "rollouts").mkdir()
    (project_root / "configs" / "training").mkdir(parents=True)
    user_rollout = project_root / "rollouts" / "user.py"
    user_rollout.write_text("# user code\n", encoding="utf-8")
    user_training_config = project_root / "configs" / "training" / "user.toml"
    user_training_config.write_text("# user training\n", encoding="utf-8")

    monkeypatch.chdir(project_root)
    with override_output_context(format=OutputFormat.json):
        apply_command("multiply")

    # The template only owns rollouts/multiply/ and the matching multiply.toml
    # configs — unrelated rollouts and configs must be left alone.
    assert user_rollout.read_text(encoding="utf-8") == "# user code\n"
    assert user_training_config.read_text(encoding="utf-8") == "# user training\n"
    assert (project_root / "rollouts" / "multiply" / "main.py").is_file()


def test_apply_command_unknown_template_raises_not_found(tmp_path, monkeypatch) -> None:
    project_root = _make_project(tmp_path)
    monkeypatch.chdir(project_root)

    with (
        override_output_context(format=OutputFormat.json),
        pytest.raises(CLIError) as exc_info,
    ):
        apply_command("definitely-not-a-template")

    assert exc_info.value.code == "NOT_FOUND"
    assert "Available templates" in str(exc_info.value)


def test_apply_command_outside_project_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with override_output_context(format=OutputFormat.json), pytest.raises(CLIError):
        apply_command("multiply")
