"""Init command JSON/plain contracts."""

from __future__ import annotations

import json

from osmosis_ai.cli import main as cli


def _stub_init_dependencies(monkeypatch) -> None:
    """Stub side-effecting helpers so init() runs without touching disk/git."""
    import osmosis_ai.platform.cli.init as init_module

    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: "git")
    monkeypatch.setattr(init_module, "_write_scaffold", lambda *args, **kwargs: None)
    monkeypatch.setattr(init_module, "_git_init", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        init_module, "_git_initial_commit", lambda *args, **kwargs: None
    )


def test_init_json_returns_created_paths_and_next_steps(
    monkeypatch, tmp_path, capsys
) -> None:

    monkeypatch.chdir(tmp_path)
    _stub_init_dependencies(monkeypatch)

    exit_code = cli.main(["--json", "init", "demo"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["status"] == "success"
    assert payload["operation"] == "init"
    assert payload["resource"]["workspace"] is None
    assert payload["resource"]["linked"] is False
    assert payload["resource"]["mode"] == "create"
    assert payload["resource"]["created_paths"] == [str((tmp_path / "demo").resolve())]
    assert payload["next_steps_structured"]


def test_init_plain_uses_renderer_without_rich_panels(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_init_dependencies(monkeypatch)

    exit_code = cli.main(["--plain", "init", "demo"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.startswith("Initialized project in ")
    assert "get started" not in captured.out


def test_init_json_rejects_removed_workspace_flag(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_init_dependencies(monkeypatch)

    exit_code = cli.main(["--json", "init", "demo", "--workspace", "ws_1"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "No such option" in json.loads(captured.err)["error"]["message"]


def test_project_init_json_matches_top_level_init(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_init_dependencies(monkeypatch)

    exit_code = cli.main(["--json", "project", "init", "demo"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["operation"] == "init"
    assert payload["resource"]["workspace"] is None
    assert payload["resource"]["linked"] is False
    assert payload["resource"]["mode"] == "create"
