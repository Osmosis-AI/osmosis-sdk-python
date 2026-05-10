"""Workspace command JSON/plain contracts (platform workspace)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from osmosis_ai.cli import main as cli


def test_bare_workspace_in_json_fails_interactive_required(capsys) -> None:
    exit_code = cli.main(["--json", "workspace"])

    captured = capsys.readouterr()
    assert exit_code != 0
    assert captured.out == ""
    envelope = json.loads(captured.err)
    assert envelope["error"]["code"] == "INTERACTIVE_REQUIRED"
    assert "workspace list" in envelope["error"]["message"]


def test_workspace_list_json_returns_list_result(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace.require_credentials", lambda: object()
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace.platform_request",
        lambda *args, **kwargs: {
            "workspaces": [
                {"id": "ws_1", "name": "default", "has_subscription": True},
                {"id": "ws_2", "name": "research", "has_subscription": False},
            ]
        },
    )

    exit_code = cli.main(["--json", "workspace", "list"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["schema_version"] == 1
    assert payload["total_count"] == 2
    assert payload["items"][0]["name"] == "default"
    assert "is_active" not in payload["items"][0]


def test_workspace_switch_is_removed(monkeypatch, capsys) -> None:
    from osmosis_ai.cli.errors import CLIError
    from osmosis_ai.cli.main import main

    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace.require_credentials",
        lambda: (_ for _ in ()).throw(CLIError("switch handler should not run")),
    )

    rc = main(["workspace", "switch", "team-alpha"])

    assert rc != 0
    assert "No such command" in capsys.readouterr().err


def test_workspace_create_json_returns_operation_result(monkeypatch, capsys) -> None:
    creds = object()
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace.require_credentials",
        lambda: creds,
    )

    class FakeClient:
        def create_workspace(self, name, timezone, *, credentials=None):
            assert credentials is creds
            return {"id": "ws_new", "name": name, "timezone": timezone}

    monkeypatch.setattr("osmosis_ai.platform.api.client.OsmosisClient", FakeClient)

    exit_code = cli.main(
        ["--json", "workspace", "create", "new-team", "--timezone", "UTC"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["operation"] == "workspace.create"
    assert payload["resource"]["id"] == "ws_new"
    assert payload["resource"]["name"] == "new-team"
    assert "workspace.switch" not in json.dumps(payload)


def test_workspace_delete_json_without_yes_fails_interactive_required(capsys) -> None:
    exit_code = cli.main(["--json", "workspace", "delete", "default"])

    captured = capsys.readouterr()
    assert exit_code != 0
    assert captured.out == ""
    envelope = json.loads(captured.err)
    assert envelope["error"]["code"] == "INTERACTIVE_REQUIRED"


def test_workspace_delete_json_with_yes_returns_operation_result(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace.require_credentials",
        lambda: object(),
    )

    class FakeClient:
        def list_workspaces(self, *, credentials=None):
            return {"workspaces": [{"id": "ws_1", "name": "Default"}]}

        def get_workspace_deletion_status(self, workspace_id, *, credentials=None):
            return SimpleNamespace(
                is_owner=True,
                is_last_workspace=False,
                has_running_processes=False,
                feature_pipelines=SimpleNamespace(valid=True, count=0),
                training_runs=SimpleNamespace(valid=True, count=0),
                models=SimpleNamespace(valid=True, count=0),
            )

        def delete_workspace(self, workspace_id, *, credentials=None):
            return True

    monkeypatch.setattr("osmosis_ai.platform.api.client.OsmosisClient", FakeClient)

    exit_code = cli.main(["--json", "workspace", "delete", "default", "--yes"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["operation"] == "workspace.delete"
    assert payload["resource"] == {"id": "ws_1", "name": "Default"}


def test_workspace_delete_json_preserves_platform_error_code(
    monkeypatch, capsys
) -> None:
    from osmosis_ai.platform.auth.platform_client import PlatformAPIError

    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace.require_credentials",
        lambda: object(),
    )

    class FakeClient:
        def list_workspaces(self, *, credentials=None):
            return {"workspaces": [{"id": "ws_1", "name": "default"}]}

        def get_workspace_deletion_status(self, workspace_id, *, credentials=None):
            raise PlatformAPIError("Authentication failed.", status_code=401)

    monkeypatch.setattr("osmosis_ai.platform.api.client.OsmosisClient", FakeClient)

    exit_code = cli.main(["--json", "workspace", "delete", "default", "--yes"])

    captured = capsys.readouterr()
    assert exit_code == 1
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "AUTH_REQUIRED"
