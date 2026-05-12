"""auth login + logout JSON/plain contracts."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from osmosis_ai.cli import main as cli
from osmosis_ai.platform.auth.credentials import Credentials, UserInfo
from osmosis_ai.platform.auth.flow import LoginError, VerifyResult


@pytest.fixture
def fake_verify_result() -> VerifyResult:
    return VerifyResult(
        user=UserInfo(id="u1", email="brian@example.com", name="Brian"),
        expires_at=datetime.now(UTC) + timedelta(days=30),
        token_id="tok_1",
    )


@pytest.fixture(autouse=True)
def _fail_workspace_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_request",
        lambda *args, **kwargs: pytest.fail(
            "auth login must not look up platform workspaces"
        ),
    )


def _credentials(
    *,
    access_token: str = "t",
    user_id: str = "u",
    email: str = "x@example.com",
    token_id: str = "tok",
    include_env: bool = True,
) -> Credentials:
    return Credentials(
        access_token=access_token,
        token_type="Bearer",
        expires_at=datetime.now(UTC) + timedelta(days=1),
        created_at=datetime.now(UTC),
        user=UserInfo(id=user_id, email=email, name=None),
        token_id=token_id,
    )


def test_login_json_with_token_returns_operation_result(
    monkeypatch, capsys, fake_verify_result
) -> None:
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token", lambda token: fake_verify_result
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials", lambda creds: "keyring"
    )

    exit_code = cli.main(["--json", "auth", "login", "--token", "secret"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["status"] == "success"
    assert payload["operation"] == "auth.login"
    assert payload["resource"]["email"] == "brian@example.com"
    assert payload["resource"]["workspace"] is None
    assert "workspace_count" not in payload["resource"]
    assert "workspace_lookup_error" not in payload["resource"]
    assert payload["resource"]["verified"] is True
    assert payload["resource"]["saved"] is True
    assert [step["action"] for step in payload["next_steps_structured"]] == [
        "platform.clone_repository",
        "project.doctor",
    ]
    serialized = json.dumps(payload)
    assert "project.link" not in serialized
    assert "project link" not in serialized
    assert "workspace.switch" not in serialized


def test_login_json_with_token_points_to_clone_and_doctor(
    monkeypatch, capsys, fake_verify_result
) -> None:
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token", lambda token: fake_verify_result
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials", lambda creds: "keyring"
    )

    exit_code = cli.main(["--json", "auth", "login", "--token", "secret"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    serialized = json.dumps(payload)
    assert "workspace_count" not in payload["resource"]
    assert "workspace_lookup_error" not in payload["resource"]
    assert "Create or open a project in the Osmosis Platform" in serialized
    assert "osmosis project doctor" in serialized
    assert "osmosis workspace" not in serialized
    assert "workspace create" not in serialized
    assert "workspace list" not in serialized
    assert "osmosis project link --workspace <workspace-id-or-name>" not in serialized
    assert "project.validate" not in serialized
    assert "project.link" not in serialized


def test_login_plain_with_token_prints_clone_and_doctor_next_steps(
    monkeypatch, capsys, fake_verify_result
) -> None:
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token", lambda token: fake_verify_result
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials", lambda creds: "keyring"
    )

    exit_code = cli.main(["--plain", "auth", "login", "--token", "secret"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Logged in as brian@example.com." in captured.out
    assert "Create or open a project in the Osmosis Platform" in captured.out
    assert "osmosis project doctor" in captured.out
    assert "osmosis project link --workspace <workspace-id-or-name>" not in captured.out
    assert "project.validate" not in captured.out
    assert "project.link" not in captured.out
    assert "workspace switch" not in captured.out


def test_login_plain_with_token_omits_workspace_lookup_fields(
    monkeypatch, capsys, fake_verify_result
) -> None:
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token", lambda token: fake_verify_result
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials", lambda creds: "keyring"
    )

    exit_code = cli.main(["--plain", "auth", "login", "--token", "secret"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Logged in as brian@example.com." in captured.out
    assert "Create or open a project in the Osmosis Platform" in captured.out
    assert "osmosis project doctor" in captured.out
    assert "workspace_count" not in captured.out
    assert "workspace_lookup_error" not in captured.out
    assert "osmosis workspace" not in captured.out
    assert "workspace create" not in captured.out
    assert "workspace list" not in captured.out
    assert "osmosis project link --workspace <workspace-id-or-name>" not in captured.out


def test_login_json_force_with_invalid_token_preserves_existing_session(
    monkeypatch, capsys
) -> None:
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    side_effects = []
    monkeypatch.setattr("osmosis_ai.platform.auth.load_credentials", _credentials)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token: (_ for _ in ()).throw(
            LoginError("Authentication failed.", status_code=401)
        ),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.delete_credentials",
        lambda: side_effects.append("delete_credentials"),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials",
        lambda creds: side_effects.append("save_credentials"),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.local_config.clear_all_local_data",
        lambda: side_effects.append("clear_all_local_data"),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.revoke_cli_token",
        lambda creds: side_effects.append("revoke_cli_token"),
    )

    exit_code = cli.main(["--json", "auth", "login", "--force", "--token", "bad"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    envelope = json.loads(captured.err)
    assert envelope["error"]["code"] == "AUTH_REQUIRED"
    assert side_effects == []


def test_login_json_with_platform_verify_error_is_platform_error(
    monkeypatch, capsys
) -> None:
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token: (_ for _ in ()).throw(
            LoginError(
                "Osmosis platform encountered an internal error. Please try again later.",
                status_code=500,
            )
        ),
    )

    exit_code = cli.main(["--json", "auth", "login", "--token", "secret"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    envelope = json.loads(captured.err)
    assert envelope["error"]["code"] == "PLATFORM_ERROR"
    assert envelope["error"]["details"]["status_code"] == 500
    assert "internal error" in envelope["error"]["message"]


def test_login_json_with_malformed_verify_response_is_platform_error(
    monkeypatch, capsys
) -> None:
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token: (_ for _ in ()).throw(
            LoginError("Invalid response from platform")
        ),
    )

    exit_code = cli.main(["--json", "auth", "login", "--token", "secret"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    envelope = json.loads(captured.err)
    assert envelope["error"]["code"] == "PLATFORM_ERROR"
    assert envelope["error"]["details"] == {}
    assert envelope["error"]["message"] == "Invalid response from platform"


def test_login_json_force_revokes_and_cleans_before_saving_new_token(
    monkeypatch, capsys, fake_verify_result
) -> None:
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    calls = []
    monkeypatch.setattr("osmosis_ai.platform.auth.load_credentials", _credentials)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token", lambda token: fake_verify_result
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials",
        lambda creds: calls.append("save_credentials") or "keyring",
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.revoke_cli_token",
        lambda creds: calls.append("revoke_cli_token") or True,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.local_config.clear_all_local_data",
        lambda: calls.append("clear_all_local_data"),
    )

    exit_code = cli.main(["--json", "auth", "login", "--force", "--token", "secret"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out)["status"] == "success"
    assert calls == ["revoke_cli_token", "clear_all_local_data", "save_credentials"]


def test_login_json_force_with_token_leaves_new_credentials_saved(
    monkeypatch, capsys, fake_verify_result
) -> None:
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    saved_tokens: list[str] = ["old-token"]
    calls: list[str] = []
    monkeypatch.setattr("osmosis_ai.platform.auth.load_credentials", _credentials)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token", lambda token: fake_verify_result
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials",
        lambda creds: saved_tokens.append(creds.access_token) or "keyring",
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.local_config.clear_all_local_data",
        lambda: (calls.append("clear_all_local_data"), saved_tokens.clear()),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.revoke_cli_token",
        lambda creds: calls.append("revoke_cli_token") or True,
    )

    exit_code = cli.main(["--json", "auth", "login", "--force", "--token", "secret"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out)["status"] == "success"
    assert saved_tokens == ["secret"]
    assert calls == ["revoke_cli_token", "clear_all_local_data"]


def test_login_json_with_token_loads_stored_credentials_when_env_is_set(
    monkeypatch, capsys, fake_verify_result
) -> None:
    monkeypatch.setenv("OSMOSIS_TOKEN", "env-token")
    calls = []
    old_credentials = _credentials(
        access_token="stored-token",
        user_id=fake_verify_result.user.id,
        token_id="tok_old",
    )

    def load_credentials(*, include_env: bool = True):
        calls.append(("load_credentials", include_env))
        return old_credentials

    monkeypatch.setattr("osmosis_ai.platform.auth.load_credentials", load_credentials)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token", lambda token: fake_verify_result
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials",
        lambda creds: (
            calls.append(("save_credentials", creds.access_token)) or "keyring"
        ),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.revoke_cli_token",
        lambda creds: (
            calls.append(("revoke_cli_token", creds.access_token, creds.token_id))
            or True
        ),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.local_config.clear_all_local_data",
        lambda: calls.append(("clear_all_local_data",)),
    )

    exit_code = cli.main(["--json", "auth", "login", "--token", "new-token"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out)["status"] == "success"
    assert ("load_credentials", False) in calls
    assert ("revoke_cli_token", "stored-token", "tok_old") in calls
    assert ("clear_all_local_data",) not in calls


def test_login_json_with_env_token_is_verify_only(
    monkeypatch, capsys, fake_verify_result
) -> None:
    monkeypatch.setenv("OSMOSIS_TOKEN", "env-token")
    save_calls = []
    delete_calls = []
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token", lambda token: fake_verify_result
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials",
        lambda creds: save_calls.append(creds),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.delete_credentials",
        lambda: delete_calls.append(True),
    )

    exit_code = cli.main(["--json", "auth", "login"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["resource"]["source"] == "environment"
    assert payload["resource"]["verified"] is True
    assert payload["resource"]["saved"] is False
    assert save_calls == []
    assert delete_calls == []


def test_login_rich_with_env_token_is_verify_only(
    monkeypatch, capsys, fake_verify_result
) -> None:
    monkeypatch.setenv("OSMOSIS_TOKEN", "env-token")
    verify_calls = []
    save_calls = []
    delete_calls = []
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token: verify_calls.append(token) or fake_verify_result,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials",
        lambda creds: save_calls.append(creds),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.delete_credentials",
        lambda: delete_calls.append(True),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.device_login",
        lambda: pytest.fail("device login should not run when OSMOSIS_TOKEN is set"),
    )

    exit_code = cli.main(["auth", "login"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Verified OSMOSIS_TOKEN for brian@example.com." in captured.out
    assert "OSMOSIS_TOKEN was not saved to local credentials." in captured.out
    assert verify_calls == ["env-token"]
    assert save_calls == []
    assert delete_calls == []


@pytest.mark.parametrize(
    "error_code",
    ["AUTH_HEADER_MISSING", "TOKEN_MISSING", "TOKEN_INVALID", "UNKNOWN_AUTH_ERROR"],
)
def test_login_rich_with_invalid_env_token_mentions_unset(
    monkeypatch, capsys, error_code: str
) -> None:
    monkeypatch.setenv("OSMOSIS_TOKEN", "bad-env-token")
    errors = []
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token: (_ for _ in ()).throw(
            LoginError("Token is invalid.", code=error_code)
        ),
    )
    monkeypatch.setattr(
        "osmosis_ai.cli.commands.auth.console.print_error",
        lambda message: errors.append(message),
    )

    exit_code = cli.main(["auth", "login"])

    capsys.readouterr()
    message = errors[0]
    assert exit_code == 1
    assert "OSMOSIS_TOKEN environment variable is invalid or expired" in message
    assert "unset OSMOSIS_TOKEN" in message


@pytest.mark.parametrize(
    "error_code",
    ["AUTH_HEADER_MISSING", "TOKEN_MISSING", "TOKEN_INVALID", "UNKNOWN_AUTH_ERROR"],
)
def test_login_json_with_invalid_env_token_mentions_unset(
    monkeypatch, capsys, error_code: str
) -> None:
    monkeypatch.setenv("OSMOSIS_TOKEN", "bad-env-token")
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token: (_ for _ in ()).throw(
            LoginError("Token is invalid.", code=error_code)
        ),
    )

    exit_code = cli.main(["--json", "auth", "login"])

    captured = capsys.readouterr()
    assert exit_code == 1
    envelope = json.loads(captured.err)
    assert envelope["error"]["code"] == "AUTH_REQUIRED"
    assert "OSMOSIS_TOKEN environment variable" in envelope["error"]["message"]
    assert "unset OSMOSIS_TOKEN" in envelope["error"]["message"]


def test_login_json_with_generic_401_env_token_mentions_unset(
    monkeypatch, capsys
) -> None:
    monkeypatch.setenv("OSMOSIS_TOKEN", "bad-env-token")
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token: (_ for _ in ()).throw(
            LoginError("Authentication failed.", status_code=401)
        ),
    )

    exit_code = cli.main(["--json", "auth", "login"])

    captured = capsys.readouterr()
    assert exit_code == 1
    envelope = json.loads(captured.err)
    assert envelope["error"]["code"] == "AUTH_REQUIRED"
    assert "OSMOSIS_TOKEN environment variable" in envelope["error"]["message"]
    assert "unset OSMOSIS_TOKEN" in envelope["error"]["message"]


def test_login_json_without_token_or_env_fails_interactive_required(
    monkeypatch, capsys
) -> None:
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)

    exit_code = cli.main(["--json", "auth", "login"])

    captured = capsys.readouterr()
    assert exit_code != 0
    assert captured.out == ""
    envelope = json.loads(captured.err)
    assert envelope["error"]["code"] == "INTERACTIVE_REQUIRED"


def test_logout_json_without_yes_fails_fast(monkeypatch, capsys) -> None:
    monkeypatch.setattr("osmosis_ai.platform.auth.load_credentials", _credentials)

    exit_code = cli.main(["--json", "auth", "logout"])

    captured = capsys.readouterr()
    assert exit_code != 0
    assert captured.out == ""
    envelope = json.loads(captured.err)
    assert envelope["error"]["code"] == "INTERACTIVE_REQUIRED"


def test_logout_json_with_yes_returns_operation_result(monkeypatch, capsys) -> None:
    monkeypatch.setattr("osmosis_ai.platform.auth.load_credentials", _credentials)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.local_config.reset_session", lambda: None
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.revoke_cli_token", lambda c: True
    )

    exit_code = cli.main(["--json", "auth", "logout", "--yes"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["status"] == "success"
    assert payload["operation"] == "auth.logout"
    assert payload["resource"]["revoked"] is True


def test_logout_json_with_env_token_only_does_not_reset_session(
    monkeypatch, capsys
) -> None:
    monkeypatch.setenv("OSMOSIS_TOKEN", "env-token")
    calls = []

    def load_credentials(*, include_env: bool = True):
        calls.append(("load_credentials", include_env))
        return _credentials(access_token="env-token") if include_env else None

    monkeypatch.setattr("osmosis_ai.platform.auth.load_credentials", load_credentials)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.local_config.reset_session",
        lambda: calls.append(("reset_session",)),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.revoke_cli_token",
        lambda creds: calls.append(("revoke_cli_token",)),
    )

    exit_code = cli.main(["--json", "auth", "logout", "--yes"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["operation"] == "auth.logout"
    assert payload["status"] == "noop"
    assert payload["resource"]["env_token_set"] is True
    assert payload["next_steps_structured"] == [
        {"action": "unset_env", "name": "OSMOSIS_TOKEN"}
    ]
    assert "unset OSMOSIS_TOKEN" in json.dumps(payload)
    assert calls == [("load_credentials", False)]
