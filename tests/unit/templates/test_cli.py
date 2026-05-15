"""Unit tests for the template CLI business logic."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import OutputFormat
from osmosis_ai.cli.output.context import override_output_context
from osmosis_ai.templates import cli as template_cli
from osmosis_ai.templates.cli import _next_steps, apply_command, list_command


def _write_workspace_template(root: Path) -> Path:
    """Create a minimal workspace-template checkout for template CLI tests."""
    rollout_root = root / "rollouts" / "multiply-local-strands"
    rollout_root.mkdir(parents=True)
    (rollout_root / "main.py").write_text("# rollout\n", encoding="utf-8")
    (rollout_root / "pyproject.toml").write_text(
        "[project]\nname = 'multiply-local-strands'\n", encoding="utf-8"
    )
    (rollout_root / "README.md").write_text("# Multiply\n", encoding="utf-8")
    openai_root = root / "rollouts" / "multiply-local-openai"
    openai_root.mkdir(parents=True)
    (openai_root / "main.py").write_text("# openai rollout\n", encoding="utf-8")
    (openai_root / "pyproject.toml").write_text(
        "[project]\nname = 'multiply-local-openai'\n", encoding="utf-8"
    )
    (openai_root / "README.md").write_text("# Multiply OpenAI\n", encoding="utf-8")
    harbor_root = root / "rollouts" / "multiply-harbor-strands"
    harbor_root.mkdir(parents=True)
    (harbor_root / "main.py").write_text("# harbor rollout\n", encoding="utf-8")
    (harbor_root / "pyproject.toml").write_text(
        "[project]\nname = 'multiply-harbor-strands'\n", encoding="utf-8"
    )
    (harbor_root / "README.md").write_text("# Multiply Harbor\n", encoding="utf-8")
    (root / "configs" / "eval").mkdir(parents=True)
    (root / "configs" / "eval" / "multiply-local-strands.toml").write_text(
        'rollout = "multiply-local-strands"\n', encoding="utf-8"
    )
    (root / "configs" / "eval" / "multiply-local-openai.toml").write_text(
        'rollout = "multiply-local-openai"\n', encoding="utf-8"
    )
    (root / "configs" / "eval" / "multiply-harbor-strands.toml").write_text(
        'rollout = "multiply-harbor-strands"\n', encoding="utf-8"
    )
    (root / "configs" / "training").mkdir(parents=True)
    (root / "configs" / "training" / "multiply-local-strands.toml").write_text(
        'rollout = "multiply-local-strands"\n', encoding="utf-8"
    )
    (root / "configs" / "training" / "multiply-local-openai.toml").write_text(
        'rollout = "multiply-local-openai"\n', encoding="utf-8"
    )
    (root / "configs" / "training" / "multiply-harbor-strands.toml").write_text(
        'rollout = "multiply-harbor-strands"\n', encoding="utf-8"
    )
    (root / "data").mkdir()
    (root / "data" / "multiply.jsonl").write_text("{}\n", encoding="utf-8")
    return root


@pytest.fixture
def workspace_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = _write_workspace_template(tmp_path / "workspace-template")
    monkeypatch.setenv("OSMOSIS_WORKSPACE_TEMPLATE_PATH", str(root))
    return root


def _make_workspace_directory(root: Path) -> Path:
    """Create the minimum directory layout that resolve_workspace_directory accepts."""
    subprocess.run(
        ["git", "init", "-b", "main", str(root)],
        check=True,
        capture_output=True,
    )
    return root


# ── list_command ─────────────────────────────────────────────────


def test_list_command_returns_list_result_in_json(workspace_template: Path) -> None:
    with override_output_context(format=OutputFormat.json):
        result = list_command()

    assert result is not None
    assert any(item["name"] == "multiply-local-strands" for item in result.items)
    assert result.total_count == len(result.items)
    assert result.has_more is False


def test_list_command_returns_none_in_rich(workspace_template: Path) -> None:
    # Rich-mode list_command prints inline through the shared Console
    # (which caches sys.stdout at import time, defeating capsys/capfd);
    # the contract under test here is that the function returns None so
    # the result_callback skips the structured renderer.
    with override_output_context(format=OutputFormat.rich):
        result = list_command()

    assert result is None


def test_list_command_empty_message_uses_user_facing_template_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(template_cli, "list_templates", lambda: [])

    error = template_cli._format_unknown_template("missing")
    message = str(error).lower()

    assert "no templates are currently available" in message
    assert "recipe" not in message
    assert "workspace template" not in message


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
    assert "osmosis dataset upload data/multiply.jsonl" in next_steps
    assert (
        "Edit configs/training/multiply.toml with the uploaded dataset ID and target model"
        in next_steps
    )
    assert "osmosis train submit configs/training/multiply.toml" in next_steps


def test_apply_command_writes_directly_into_project_canonical_layout(
    tmp_path, monkeypatch, workspace_template
) -> None:
    """`apply` lands files at rollouts/<name>/ and configs/{eval,training}/."""
    workspace_directory = _make_workspace_directory(tmp_path)
    monkeypatch.chdir(workspace_directory)

    with override_output_context(format=OutputFormat.json):
        result = apply_command("multiply-local-strands")

    rollout_root = workspace_directory / "rollouts" / "multiply-local-strands"
    # A rollout in a project is self-describing: entrypoint + deps + README.
    assert (rollout_root / "main.py").is_file()
    assert (rollout_root / "pyproject.toml").is_file()
    assert (rollout_root / "README.md").is_file()
    assert (
        workspace_directory / "configs" / "training" / "multiply-local-strands.toml"
    ).is_file()
    assert (
        workspace_directory / "configs" / "eval" / "multiply-local-strands.toml"
    ).is_file()
    assert (workspace_directory / "data" / "multiply.jsonl").is_file()
    # The legacy staging directory must NOT be created.
    assert not (workspace_directory / "templates").exists()

    assert result is not None
    assert result.operation == "template.apply"
    assert result.status == "success"
    assert result.resource is not None
    assert result.resource["name"] == "multiply-local-strands"
    files = result.resource["files"]
    assert "rollouts/multiply-local-strands/main.py" in files
    assert "rollouts/multiply-local-strands/pyproject.toml" in files
    assert "configs/training/multiply-local-strands.toml" in files
    assert "configs/eval/multiply-local-strands.toml" in files
    assert "data/multiply.jsonl" in files
    destinations = result.resource["destinations"]
    assert any(
        dest.endswith("/rollouts/multiply-local-strands") for dest in destinations
    )
    assert result.display_next_steps == [
        "pip install -e rollouts/multiply-local-strands",
        "osmosis eval run configs/eval/multiply-local-strands.toml --limit 1",
        "git push",
        "Confirm Git Sync is connected in the Osmosis Platform",
        "osmosis train submit configs/training/multiply-local-strands.toml",
    ]


def test_apply_command_refuses_overwrite_without_force(
    tmp_path, monkeypatch, workspace_template
) -> None:
    workspace_directory = _make_workspace_directory(tmp_path)
    monkeypatch.chdir(workspace_directory)

    rollout_root = workspace_directory / "rollouts" / "multiply-local-strands"
    rollout_root.mkdir(parents=True)
    user_edit = rollout_root / "main.py"
    user_edit.write_text("# user-edited\n", encoding="utf-8")

    with (
        override_output_context(format=OutputFormat.json),
        pytest.raises(CLIError) as exc_info,
    ):
        apply_command("multiply-local-strands")

    assert exc_info.value.code == "CONFLICT"
    assert "rollouts/multiply-local-strands/" in str(exc_info.value)
    assert "--force" in str(exc_info.value)
    # User's edit must be left untouched.
    assert user_edit.read_text(encoding="utf-8") == "# user-edited\n"


def test_apply_command_force_overwrites_existing_rollout(
    tmp_path, monkeypatch, workspace_template
) -> None:
    workspace_directory = _make_workspace_directory(tmp_path)
    monkeypatch.chdir(workspace_directory)

    rollout_root = workspace_directory / "rollouts" / "multiply-local-strands"
    rollout_root.mkdir(parents=True)
    stale = rollout_root / "stale_user_file.txt"
    stale.write_text("user edits", encoding="utf-8")

    with override_output_context(format=OutputFormat.json):
        apply_command("multiply-local-strands", force=True)

    assert not stale.exists(), "--force must replace the rollout dir wholesale"
    assert (rollout_root / "main.py").is_file()
    assert (rollout_root / "pyproject.toml").is_file()


def test_apply_command_force_rejects_file_at_owned_directory_path(
    tmp_path, monkeypatch, workspace_template
) -> None:
    workspace_directory = _make_workspace_directory(tmp_path)
    monkeypatch.chdir(workspace_directory)

    rollout_root = workspace_directory / "rollouts" / "multiply-local-strands"
    rollout_root.parent.mkdir(parents=True)
    rollout_root.write_text("not a directory\n", encoding="utf-8")

    with (
        override_output_context(format=OutputFormat.json),
        pytest.raises(CLIError) as exc_info,
    ):
        apply_command("multiply-local-strands", force=True)

    assert exc_info.value.code == "CONFLICT"
    assert "rollouts/multiply-local-strands" in str(exc_info.value)
    assert "`--force` only replaces existing directories" in str(exc_info.value)
    assert rollout_root.read_text(encoding="utf-8") == "not a directory\n"


def test_apply_command_force_overwrites_existing_config(
    tmp_path, monkeypatch, workspace_template
) -> None:
    workspace_directory = _make_workspace_directory(tmp_path)
    monkeypatch.chdir(workspace_directory)

    config = workspace_directory / "configs" / "eval" / "multiply-local-strands.toml"
    config.parent.mkdir(parents=True)
    config.write_text("# stale\n", encoding="utf-8")

    with override_output_context(format=OutputFormat.json):
        apply_command("multiply-local-strands", force=True)

    refreshed = config.read_text(encoding="utf-8")
    assert "# stale" not in refreshed
    assert 'rollout = "multiply-local-strands"' in refreshed


def test_apply_command_refreshes_workspace_template_before_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osmosis_ai.templates import source

    monkeypatch.delenv("OSMOSIS_WORKSPACE_TEMPLATE_PATH", raising=False)
    monkeypatch.setattr(source, "CACHE_DIR", tmp_path / "cache")
    downloads: list[Path] = []

    def fake_download_workspace_template(
        repo: str, ref: str, destination: Path
    ) -> None:
        del repo, ref
        downloads.append(destination)
        if destination.exists():
            shutil.rmtree(destination)
        _write_workspace_template(destination)
        rollout = destination / "rollouts" / "multiply-local-strands" / "main.py"
        rollout.write_text(f"# rollout {len(downloads)}\n", encoding="utf-8")

    monkeypatch.setattr(
        source, "_download_workspace_template", fake_download_workspace_template
    )
    workspace_directory = _make_workspace_directory(tmp_path / "project")
    monkeypatch.chdir(workspace_directory)

    with override_output_context(format=OutputFormat.json):
        apply_command("multiply-local-strands")
    first = workspace_directory / "rollouts" / "multiply-local-strands" / "main.py"
    assert first.read_text(encoding="utf-8") == "# rollout 1\n"

    with override_output_context(format=OutputFormat.json):
        apply_command("multiply-local-strands", force=True)

    assert first.read_text(encoding="utf-8") == "# rollout 2\n"
    assert len(downloads) == 2
    assert downloads[0] == downloads[1]


def test_apply_command_refuses_existing_data_without_force(
    tmp_path, monkeypatch, workspace_template
) -> None:
    workspace_directory = _make_workspace_directory(tmp_path)
    monkeypatch.chdir(workspace_directory)

    data = workspace_directory / "data" / "multiply.jsonl"
    data.parent.mkdir(parents=True)
    data.write_text("user data\n", encoding="utf-8")

    with (
        override_output_context(format=OutputFormat.json),
        pytest.raises(CLIError) as exc_info,
    ):
        apply_command("multiply-local-strands")

    assert exc_info.value.code == "CONFLICT"
    assert "data/multiply.jsonl" in str(exc_info.value)
    assert data.read_text(encoding="utf-8") == "user data\n"


def test_apply_command_reuses_shared_dataset_from_another_template(
    tmp_path, monkeypatch, workspace_template
) -> None:
    workspace_directory = _make_workspace_directory(tmp_path)
    monkeypatch.chdir(workspace_directory)

    with override_output_context(format=OutputFormat.json):
        apply_command("multiply-local-strands")

    data = workspace_directory / "data" / "multiply.jsonl"
    assert data.read_text(encoding="utf-8") == "{}\n"

    for name in ("multiply-local-openai", "multiply-harbor-strands"):
        with override_output_context(format=OutputFormat.json):
            result = apply_command(name)

        assert result is not None
        assert result.status == "success"
        assert (workspace_directory / "rollouts" / name / "main.py").is_file()
        assert data.read_text(encoding="utf-8") == "{}\n"


def test_apply_command_preserves_unrelated_user_files(
    tmp_path, monkeypatch, workspace_template
) -> None:
    workspace_directory = _make_workspace_directory(tmp_path)
    (workspace_directory / "rollouts").mkdir()
    (workspace_directory / "configs" / "training").mkdir(parents=True)
    user_rollout = workspace_directory / "rollouts" / "user.py"
    user_rollout.write_text("# user code\n", encoding="utf-8")
    user_training_config = workspace_directory / "configs" / "training" / "user.toml"
    user_training_config.write_text("# user training\n", encoding="utf-8")

    monkeypatch.chdir(workspace_directory)
    with override_output_context(format=OutputFormat.json):
        apply_command("multiply-local-strands")

    # The template only owns rollouts/multiply/ and the matching multiply.toml
    # configs — unrelated rollouts and configs must be left alone.
    assert user_rollout.read_text(encoding="utf-8") == "# user code\n"
    assert user_training_config.read_text(encoding="utf-8") == "# user training\n"
    assert (
        workspace_directory / "rollouts" / "multiply-local-strands" / "main.py"
    ).is_file()


def test_apply_command_unknown_template_raises_not_found(
    tmp_path, monkeypatch, workspace_template
) -> None:
    workspace_directory = _make_workspace_directory(tmp_path)
    monkeypatch.chdir(workspace_directory)

    with (
        override_output_context(format=OutputFormat.json),
        pytest.raises(CLIError) as exc_info,
    ):
        apply_command("definitely-not-a-template")

    assert exc_info.value.code == "NOT_FOUND"
    assert "Available templates" in str(exc_info.value)
    message = str(exc_info.value).lower()
    assert "recipe" not in message
    assert "workspace template" not in message


def test_apply_command_outside_project_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with override_output_context(format=OutputFormat.json), pytest.raises(CLIError):
        apply_command("multiply")
