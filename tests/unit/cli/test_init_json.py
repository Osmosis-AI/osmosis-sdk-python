"""Init command JSON/plain contracts."""

from __future__ import annotations

import json
from typing import Any

from osmosis_ai.cli import main as cli


class _FakeCreds:
    def is_expired(self) -> bool:
        return False


def _stub_init_dependencies(monkeypatch) -> None:
    """Stub side-effecting helpers so init() runs without touching disk/git."""
    import osmosis_ai.platform.cli.init as init_module

    monkeypatch.setattr(init_module.shutil, "which", lambda cmd: "git")
    monkeypatch.setattr(init_module, "_write_scaffold", lambda *args, **kwargs: None)
    monkeypatch.setattr(init_module, "_git_init", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        init_module, "_git_initial_commit", lambda *args, **kwargs: None
    )


def _patch_workspace_context(
    monkeypatch, *, workspace_id: str | None, git_context: dict[str, Any]
) -> None:
    """Pin auth + workspace metadata so init's auth gate is satisfied."""
    import osmosis_ai.platform.cli.init as init_module

    workspace_name = git_context.get("workspace_name")
    monkeypatch.setattr(init_module, "_require_link_credentials", lambda: _FakeCreds())
    monkeypatch.setattr(
        init_module,
        "_resolve_workspace_for_link",
        lambda ref, credentials: {
            "id": workspace_id or ref,
            "name": workspace_name or "default",
            "has_github_app_installation": git_context.get(
                "has_github_app_installation", False
            ),
            "connected_repo": (
                {"repo_url": git_context["connected_repo_url"]}
                if git_context.get("connected_repo_url")
                else None
            ),
        },
    )


def test_init_json_returns_created_paths_and_next_steps(
    monkeypatch, tmp_path, capsys
) -> None:
    import osmosis_ai.platform.cli.init as init_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(init_module, "CONFIG_FILE", tmp_path / "config.json")
    _stub_init_dependencies(monkeypatch)
    _patch_workspace_context(
        monkeypatch,
        workspace_id="ws_1",
        git_context={
            "workspace_id": "ws_1",
            "workspace_name": "default",
            "git_sync_url": "https://platform.osmosis.ai/default/integrations/git",
            "has_github_app_installation": False,
            "connected_repo_url": None,
        },
    )

    exit_code = cli.main(["--json", "init", "demo", "--workspace", "ws_1"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["status"] == "success"
    assert payload["operation"] == "init"
    assert payload["resource"]["workspace"] == {"id": "ws_1", "name": "default"}
    assert payload["resource"]["linked"] is True
    assert payload["resource"]["mode"] == "create"
    assert payload["resource"]["created_paths"] == [str((tmp_path / "demo").resolve())]
    assert payload["next_steps_structured"]


def test_init_plain_uses_renderer_without_rich_panels(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_init_dependencies(monkeypatch)
    _patch_workspace_context(
        monkeypatch,
        workspace_id="ws_solo",
        git_context={
            "workspace_id": "ws_solo",
            "workspace_name": "solo-team",
            "git_sync_url": ("https://platform.osmosis.ai/solo-team/integrations/git"),
            "has_github_app_installation": False,
            "connected_repo_url": None,
        },
    )

    exit_code = cli.main(["--plain", "init", "demo", "--no-link"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.startswith("Initialized project in ")
    assert "get started" not in captured.out


def test_init_json_noninteractive_requires_link_choice(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_init_dependencies(monkeypatch)

    exit_code = cli.main(["--json", "init", "demo"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["code"] == "INTERACTIVE_REQUIRED"


def test_project_init_json_matches_top_level_init(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_init_dependencies(monkeypatch)

    exit_code = cli.main(["--json", "project", "init", "demo", "--no-link"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["operation"] == "init"
    assert payload["resource"]["workspace"] is None
    assert payload["resource"]["linked"] is False
    assert payload["resource"]["mode"] == "create"
