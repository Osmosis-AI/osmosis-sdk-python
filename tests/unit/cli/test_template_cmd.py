"""End-to-end CLI contracts for `osmosis template` (rich + JSON + plain)."""

from __future__ import annotations

import json
from pathlib import Path

from osmosis_ai.cli import main as cli


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


# ── list ─────────────────────────────────────────────────────────


def test_template_list_json_returns_items_envelope(capsys) -> None:
    rc = cli.main(["--json", "template", "list"])
    captured = capsys.readouterr()

    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["schema_version"] == 1
    names = [item["name"] for item in payload["items"]]
    assert "multiply" in names
    assert payload["total_count"] == len(names)
    assert payload["has_more"] is False


def test_template_list_plain_emits_one_name_per_line(capsys) -> None:
    rc = cli.main(["--plain", "template", "list"])
    captured = capsys.readouterr()

    assert rc == 0
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert "multiply" in lines


def test_template_list_rich_exits_zero() -> None:
    # Rich-mode output goes through the shared Console (which caches
    # sys.stdout at module import) so capsys/capfd can't reliably observe
    # it; assert the contract by exit code and rely on the JSON/plain
    # tests above to lock the actual content.
    rc = cli.main(["template", "list"])
    assert rc == 0


# ── apply ────────────────────────────────────────────────────────


def test_template_apply_json_writes_into_project_canonical_layout(
    monkeypatch, tmp_path, capsys
) -> None:
    project_root = _make_project(tmp_path)
    monkeypatch.chdir(project_root)

    rc = cli.main(["--json", "template", "apply", "multiply"])
    captured = capsys.readouterr()

    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["schema_version"] == 1
    assert payload["status"] == "success"
    assert payload["operation"] == "template.apply"
    assert payload["resource"]["name"] == "multiply"
    files = payload["resource"]["files"]
    assert "rollouts/multiply/main.py" in files
    assert "rollouts/multiply/pyproject.toml" in files
    assert "configs/eval/multiply.toml" in files

    assert (project_root / "rollouts" / "multiply" / "main.py").is_file()
    assert (project_root / "configs" / "eval" / "multiply.toml").is_file()
    # The legacy staging directory must NOT be created.
    assert not (project_root / "templates").exists()


def test_template_apply_refuses_overwrite_without_force(
    monkeypatch, tmp_path, capsys
) -> None:
    project_root = _make_project(tmp_path)
    monkeypatch.chdir(project_root)

    rollout_root = project_root / "rollouts" / "multiply"
    rollout_root.mkdir(parents=True)
    (rollout_root / "main.py").write_text("# user\n", encoding="utf-8")

    rc = cli.main(["--json", "template", "apply", "multiply"])
    captured = capsys.readouterr()

    assert rc != 0
    err = json.loads(captured.err)
    assert err["error"]["code"] == "CONFLICT"
    assert "rollouts/multiply/" in err["error"]["message"]
    assert "--force" in err["error"]["message"]
    # User edits left intact.
    assert (rollout_root / "main.py").read_text(encoding="utf-8") == "# user\n"


def test_template_apply_force_overwrites_existing_rollout(
    monkeypatch, tmp_path
) -> None:
    project_root = _make_project(tmp_path)
    monkeypatch.chdir(project_root)

    rollout_root = project_root / "rollouts" / "multiply"
    rollout_root.mkdir(parents=True)
    stale = rollout_root / "STALE.txt"
    stale.write_text("stale", encoding="utf-8")

    rc = cli.main(["--json", "template", "apply", "multiply", "--force"])

    assert rc == 0
    assert not stale.exists()
    assert (rollout_root / "main.py").is_file()


def test_template_apply_unknown_template_returns_not_found_in_json(
    monkeypatch, tmp_path, capsys
) -> None:
    project_root = _make_project(tmp_path)
    monkeypatch.chdir(project_root)

    rc = cli.main(["--json", "template", "apply", "nope"])
    captured = capsys.readouterr()

    assert rc != 0
    err = json.loads(captured.err)
    assert err["error"]["code"] == "NOT_FOUND"
    assert "Available templates" in err["error"]["message"]


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
