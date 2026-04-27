"""Tests for `osmosis auth whoami` across rich/json/plain."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from osmosis_ai.cli import main as cli
from osmosis_ai.platform.auth.credentials import Credentials, UserInfo
from osmosis_ai.platform.auth.flow import VerifyResult


@pytest.fixture
def fake_credentials() -> Credentials:
    return Credentials(
        access_token="token",
        token_type="Bearer",
        expires_at=datetime.now(UTC) + timedelta(days=30),
        created_at=datetime.now(UTC),
        user=UserInfo(id="u1", email="brian@example.com", name="Brian"),
        token_id="tok_1",
    )


def _patch_auth(monkeypatch, creds: Credentials | None, workspace=None) -> None:
    monkeypatch.setattr("osmosis_ai.platform.auth.load_credentials", lambda: creds)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.ensure_active_workspace",
        lambda *args, **kwargs: workspace,
    )


@pytest.fixture
def fake_verify_result() -> VerifyResult:
    return VerifyResult(
        user=UserInfo(id="env_u1", email="env@example.com", name="Env User"),
        expires_at=datetime.now(UTC) + timedelta(days=30),
        token_id="env_tok_1",
    )


def test_whoami_json_envelope(monkeypatch, capsys, fake_credentials) -> None:
    _patch_auth(
        monkeypatch,
        fake_credentials,
        {"id": "ws_1", "name": "default"},
    )
    exit_code = cli.main(["--json", "auth", "whoami"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["schema_version"] == 1
    data = payload["data"]
    assert data["email"] == "brian@example.com"
    assert data["name"] == "Brian"
    assert data["workspace"] == {"id": "ws_1", "name": "default"}
    assert data["source"] == "credentials"
    assert "expires_at" in data


def test_whoami_json_with_env_token_uses_verified_identity(
    monkeypatch, capsys, fake_verify_result
) -> None:
    monkeypatch.setenv("OSMOSIS_TOKEN", "env-token")
    verify_calls = []
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token: verify_calls.append(token) or fake_verify_result,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda: pytest.fail("env token whoami should not use stored credentials"),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.get_active_workspace",
        lambda: {"id": "stored_ws", "name": "stored"},
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_request",
        lambda *args, **kwargs: {"workspaces": [{"id": "env_ws", "name": "env"}]},
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.ensure_active_workspace",
        lambda *args, **kwargs: pytest.fail(
            "env token whoami should not use cached workspace resolution"
        ),
    )

    exit_code = cli.main(["--json", "auth", "whoami"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    data = payload["data"]
    assert data["email"] == "env@example.com"
    assert data["name"] == "Env User"
    assert data["workspace"] == {"id": "env_ws", "name": "env"}
    assert data["source"] == "environment"
    assert verify_calls == ["env-token"]


def test_whoami_json_with_env_token_keeps_matching_local_workspace(
    monkeypatch, capsys, fake_verify_result
) -> None:
    monkeypatch.setenv("OSMOSIS_TOKEN", "env-token")
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token", lambda token: fake_verify_result
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.get_active_workspace",
        lambda: {"id": "env_ws_2", "name": "old-env-2"},
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_request",
        lambda *args, **kwargs: {
            "workspaces": [
                {"id": "env_ws_1", "name": "env-1"},
                {"id": "env_ws_2", "name": "env-2"},
            ]
        },
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda: Credentials(
            access_token="stored-token",
            token_type="Bearer",
            expires_at=datetime.now(UTC) + timedelta(days=30),
            created_at=datetime.now(UTC),
            user=UserInfo(id="stored_u1", email="stored@example.com", name="Stored"),
            token_id="stored_tok_1",
        ),
    )

    exit_code = cli.main(["--json", "auth", "whoami"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    data = payload["data"]
    assert data["email"] == "env@example.com"
    assert data["workspace"] == {"id": "env_ws_2", "name": "env-2"}
    assert data["source"] == "environment"


def test_whoami_json_with_env_token_ignores_mismatched_local_workspace(
    monkeypatch, capsys, fake_verify_result
) -> None:
    monkeypatch.setenv("OSMOSIS_TOKEN", "env-token")
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token", lambda token: fake_verify_result
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.get_active_workspace",
        lambda: {"id": "stored_ws", "name": "stored"},
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_request",
        lambda *args, **kwargs: {
            "workspaces": [
                {"id": "env_ws_1", "name": "env-1"},
                {"id": "env_ws_2", "name": "env-2"},
            ]
        },
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda: Credentials(
            access_token="stored-token",
            token_type="Bearer",
            expires_at=datetime.now(UTC) + timedelta(days=30),
            created_at=datetime.now(UTC),
            user=UserInfo(id="stored_u1", email="stored@example.com", name="Stored"),
            token_id="stored_tok_1",
        ),
    )

    exit_code = cli.main(["--json", "auth", "whoami"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    data = payload["data"]
    assert data["email"] == "env@example.com"
    assert data["workspace"] is None
    assert data["source"] == "environment"


def test_whoami_json_with_env_token_verify_failure_emits_auth_required(
    monkeypatch, capsys
) -> None:
    from osmosis_ai.platform.auth import LoginError

    monkeypatch.setenv("OSMOSIS_TOKEN", "env-token")
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token: (_ for _ in ()).throw(
            LoginError("Authentication failed.", status_code=401)
        ),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda: pytest.fail(
            "env token whoami should not fall back to stored credentials"
        ),
    )

    exit_code = cli.main(["--json", "auth", "whoami"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    envelope = json.loads(captured.err)
    assert envelope["error"]["code"] == "AUTH_REQUIRED"


def test_whoami_json_with_env_token_malformed_response_emits_platform_error(
    monkeypatch, capsys
) -> None:
    from osmosis_ai.platform.auth import LoginError

    monkeypatch.setenv("OSMOSIS_TOKEN", "env-token")
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token: (_ for _ in ()).throw(
            LoginError("Invalid response from platform")
        ),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda: pytest.fail(
            "env token whoami should not fall back to stored credentials"
        ),
    )

    exit_code = cli.main(["--json", "auth", "whoami"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    envelope = json.loads(captured.err)
    assert envelope["error"]["code"] == "PLATFORM_ERROR"


def test_whoami_json_when_logged_out_emits_auth_required(monkeypatch, capsys) -> None:
    _patch_auth(monkeypatch, None)
    exit_code = cli.main(["--json", "auth", "whoami"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    envelope = json.loads(captured.err)
    assert envelope["error"]["code"] == "AUTH_REQUIRED"


def test_whoami_plain_renders_label_value_lines(
    monkeypatch, capsys, fake_credentials
) -> None:
    _patch_auth(
        monkeypatch,
        fake_credentials,
        {"id": "ws_1", "name": "default"},
    )
    exit_code = cli.main(["--plain", "auth", "whoami"])
    captured = capsys.readouterr()
    assert exit_code == 0
    lines = captured.out.splitlines()
    assert "Email: brian@example.com" in lines
    assert "Workspace: default" in lines


def test_whoami_rich_still_uses_table(monkeypatch, capsys, fake_credentials) -> None:
    _patch_auth(
        monkeypatch,
        fake_credentials,
        {"id": "ws_1", "name": "default"},
    )
    exit_code = cli.main(["auth", "whoami"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Email" in captured.out
    assert "brian@example.com" in captured.out
