"""End-to-end CLI contracts for `osmosis template` (rich + JSON + plain)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from typer.main import get_command

from osmosis_ai.cli import main as cli


def _write_workspace_template(root: Path) -> Path:
    """Create a minimal workspace-template checkout for CLI contract tests."""
    rollout_root = root / "rollouts" / "multiply-local-strands"
    rollout_root.mkdir(parents=True)
    (rollout_root / "main.py").write_text("# rollout\n", encoding="utf-8")
    (rollout_root / "pyproject.toml").write_text(
        "[project]\nname = 'multiply-local-strands'\n", encoding="utf-8"
    )
    (root / "configs" / "eval").mkdir(parents=True)
    (root / "configs" / "eval" / "multiply-local-strands.toml").write_text(
        "[eval]\n", encoding="utf-8"
    )
    (root / "configs" / "training").mkdir(parents=True)
    (root / "configs" / "training" / "multiply-local-strands.toml").write_text(
        "[experiment]\n", encoding="utf-8"
    )
    (root / "data").mkdir()
    (root / "data" / "multiply.jsonl").write_text("{}\n", encoding="utf-8")
    return root


@pytest.fixture
def workspace_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = _write_workspace_template(tmp_path / "workspace-template")
    monkeypatch.setenv("OSMOSIS_WORKSPACE_TEMPLATE_PATH", str(root))
    return root


def _make_project(root: Path) -> Path:
    (root / ".osmosis").mkdir(parents=True, exist_ok=True)
    (root / ".osmosis" / "project.toml").write_text(
        "[project]\nsetup_source = 'test'\n",
        encoding="utf-8",
    )
    return root


# ── help ─────────────────────────────────────────────────────────


def test_template_help_lists_subcommands(capfd) -> None:
    rc = cli.main(["template", "--help"])
    out = capfd.readouterr().out

    assert rc == 0
    assert "list" in out
    assert "apply" in out


def test_template_help_uses_user_facing_template_terms(capfd) -> None:
    rc = cli.main(["template", "--help"])
    out = capfd.readouterr().out.lower()

    assert rc == 0
    assert "recipe" not in out
    assert "workspace template" not in out


@pytest.mark.parametrize(
    "args",
    [["template", "list", "--help"], ["template", "apply", "--help"]],
)
def test_template_subcommand_help_uses_user_facing_template_terms(
    args: list[str], capfd
) -> None:
    rc = cli.main(args)
    out = capfd.readouterr().out.lower()

    assert rc == 0
    assert "recipe" not in out
    assert "workspace template" not in out


# ── list ─────────────────────────────────────────────────────────


def test_template_list_json_returns_items_envelope(capsys, workspace_template) -> None:
    rc = cli.main(["--json", "template", "list"])
    captured = capsys.readouterr()

    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["schema_version"] == 1
    names = [item["name"] for item in payload["items"]]
    assert "multiply-local-strands" in names
    assert payload["total_count"] == len(names)
    assert payload["has_more"] is False


def test_template_list_plain_emits_one_name_per_line(
    capsys, workspace_template
) -> None:
    rc = cli.main(["--plain", "template", "list"])
    captured = capsys.readouterr()

    assert rc == 0
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert "multiply-local-strands" in lines


def test_template_list_rich_exits_zero(workspace_template) -> None:
    # Rich-mode output goes through the shared Console (which caches
    # sys.stdout at module import) so capsys/capfd can't reliably observe
    # it; assert the contract by exit code and rely on the JSON/plain
    # tests above to lock the actual content.
    rc = cli.main(["template", "list"])
    assert rc == 0


def test_zsh_completion_for_template_commands_does_not_require_comp_words(
    workspace_template,
) -> None:
    cli._register_commands()
    result = CliRunner().invoke(
        get_command(cli.app),
        [],
        prog_name="osmosis",
        env={
            "_OSMOSIS_COMPLETE": "complete_zsh",
            "_TYPER_COMPLETE_ARGS": "osmosis template ",
        },
    )

    assert result.exit_code == 0
    assert result.exception is None
    assert "COMP_WORDS" not in result.output
    assert "apply" in result.output
    assert "list" in result.output


def test_completion_registration_uses_public_typer_api() -> None:
    main_source = Path(cli.__file__).read_text(encoding="utf-8")

    assert "typer._completion_classes" not in main_source
    assert (
        "from typer.completion import get_completion_inspect_parameters" in main_source
    )


# ── apply ────────────────────────────────────────────────────────


def test_template_apply_json_writes_into_project_canonical_layout(
    monkeypatch, tmp_path, capsys, workspace_template
) -> None:
    project_root = _make_project(tmp_path)
    monkeypatch.chdir(project_root)

    rc = cli.main(["--json", "template", "apply", "multiply-local-strands"])
    captured = capsys.readouterr()

    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["schema_version"] == 1
    assert payload["status"] == "success"
    assert payload["operation"] == "template.apply"
    assert payload["resource"]["name"] == "multiply-local-strands"
    files = payload["resource"]["files"]
    assert "rollouts/multiply-local-strands/main.py" in files
    assert "rollouts/multiply-local-strands/pyproject.toml" in files
    assert "configs/eval/multiply-local-strands.toml" in files
    assert "configs/training/multiply-local-strands.toml" in files
    assert "data/multiply.jsonl" in files

    assert (project_root / "rollouts" / "multiply-local-strands" / "main.py").is_file()
    assert (project_root / "configs" / "eval" / "multiply-local-strands.toml").is_file()
    assert (project_root / "data" / "multiply.jsonl").is_file()
    # The legacy staging directory must NOT be created.
    assert not (project_root / "templates").exists()


def test_template_apply_refuses_overwrite_without_force(
    monkeypatch, tmp_path, capsys, workspace_template
) -> None:
    project_root = _make_project(tmp_path)
    monkeypatch.chdir(project_root)

    rollout_root = project_root / "rollouts" / "multiply-local-strands"
    rollout_root.mkdir(parents=True)
    (rollout_root / "main.py").write_text("# user\n", encoding="utf-8")

    rc = cli.main(["--json", "template", "apply", "multiply-local-strands"])
    captured = capsys.readouterr()

    assert rc != 0
    err = json.loads(captured.err)
    assert err["error"]["code"] == "CONFLICT"
    assert "rollouts/multiply-local-strands/" in err["error"]["message"]
    assert "--force" in err["error"]["message"]
    # User edits left intact.
    assert (rollout_root / "main.py").read_text(encoding="utf-8") == "# user\n"


def test_template_apply_force_overwrites_existing_rollout(
    monkeypatch, tmp_path, workspace_template
) -> None:
    project_root = _make_project(tmp_path)
    monkeypatch.chdir(project_root)

    rollout_root = project_root / "rollouts" / "multiply-local-strands"
    rollout_root.mkdir(parents=True)
    stale = rollout_root / "STALE.txt"
    stale.write_text("stale", encoding="utf-8")

    rc = cli.main(["--json", "template", "apply", "multiply-local-strands", "--force"])

    assert rc == 0
    assert not stale.exists()
    assert (rollout_root / "main.py").is_file()


def test_template_apply_unknown_template_returns_not_found_in_json(
    monkeypatch, tmp_path, capsys, workspace_template
) -> None:
    project_root = _make_project(tmp_path)
    monkeypatch.chdir(project_root)

    rc = cli.main(["--json", "template", "apply", "nope"])
    captured = capsys.readouterr()

    assert rc != 0
    err = json.loads(captured.err)
    assert err["error"]["code"] == "NOT_FOUND"
    assert "Available templates" in err["error"]["message"]
    message = err["error"]["message"].lower()
    assert "recipe" not in message
    assert "workspace template" not in message


def test_zsh_completion_for_template_apply_names(workspace_template) -> None:
    cli._register_commands()
    result = CliRunner().invoke(
        get_command(cli.app),
        [],
        prog_name="osmosis",
        env={
            "_OSMOSIS_COMPLETE": "complete_zsh",
            "_TYPER_COMPLETE_ARGS": "osmosis template apply multiply-local",
        },
    )

    assert result.exit_code == 0
    assert result.exception is None
    assert "multiply-local-strands" in result.output
    assert "multiply-local-openai" in result.output


def test_template_apply_outside_project_errors(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    rc = cli.main(["template", "apply", "multiply"])
    captured = capsys.readouterr()

    assert rc != 0
    assert "Not in an Osmosis project" in captured.err


# ── bare `osmosis template` ──────────────────────────────────────


def test_bare_template_shows_help(capfd) -> None:
    rc = cli.main(["template"])
    out = capfd.readouterr().out

    assert rc == 0
    assert "list" in out
    assert "apply" in out
