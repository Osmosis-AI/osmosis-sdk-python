"""Init command JSON/plain contracts."""

from __future__ import annotations

import json

from osmosis_ai.cli import main as cli


def test_init_json_returns_created_paths_and_next_steps(
    monkeypatch, tmp_path, capsys
) -> None:
    import osmosis_ai.platform.cli.init as init_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: "git")
    monkeypatch.setattr(init_module, "_write_scaffold", lambda *args, **kwargs: None)
    monkeypatch.setattr(init_module, "_git_init", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        init_module, "_git_initial_commit", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        init_module,
        "_selected_workspace_git_context",
        lambda: {
            "workspace_id": "ws_1",
            "workspace_name": "default",
            "git_sync_url": "https://platform.osmosis.ai/default/integrations/git",
            "has_github_app_installation": False,
            "connected_repo_url": None,
        },
    )

    exit_code = cli.main(["--json", "init", "demo"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["status"] == "success"
    assert payload["operation"] == "init"
    assert payload["resource"]["workspace"] == {"id": "ws_1", "name": "default"}
    assert payload["resource"]["created_paths"] == [str((tmp_path / "demo").resolve())]
    assert payload["resource"]["git_sync_url"].endswith("/default/integrations/git")
    assert payload["next_steps_structured"]


def test_init_plain_uses_renderer_without_rich_panels(
    monkeypatch, tmp_path, capsys
) -> None:
    import osmosis_ai.platform.cli.init as init_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: "git")
    monkeypatch.setattr(init_module, "_write_scaffold", lambda *args, **kwargs: None)
    monkeypatch.setattr(init_module, "_git_init", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        init_module, "_git_initial_commit", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        init_module,
        "_selected_workspace_git_context",
        lambda: {
            "workspace_id": None,
            "workspace_name": None,
            "git_sync_url": None,
            "has_github_app_installation": False,
            "connected_repo_url": None,
        },
    )

    exit_code = cli.main(["--plain", "init", "demo"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.startswith("Initialized workspace in ")
    assert "get started" not in captured.out
