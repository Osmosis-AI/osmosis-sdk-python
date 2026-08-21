"""Auth login and logout CLI contracts."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from io import StringIO

import pytest

from osmosis_ai.cli import main as cli
from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.auth.credentials import Credentials, UserInfo
from osmosis_ai.platform.auth.flow import LoginError, VerifyResult
from osmosis_ai.platform.cli import auth as auth_module


@pytest.fixture
def fake_verify_result() -> VerifyResult:
    return VerifyResult(
        user=UserInfo(id="u1", email="brian@example.com", name="Brian"),
        expires_at=datetime.now(UTC) + timedelta(days=30),
        token_id="tok_1",
    )


@pytest.fixture(autouse=True)
def _auth_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_request",
        lambda *args, **kwargs: pytest.fail(
            "auth login must not look up platform workspaces"
        ),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.get_credential_store",
        lambda *, include_env=True: None,
    )


def _credentials(
    *,
    access_token: str = "t",
    user_id: str = "u",
    email: str = "x@example.com",
    token_id: str = "tok",
) -> Credentials:
    return Credentials(
        access_token=access_token,
        token_type="Bearer",
        expires_at=datetime.now(UTC) + timedelta(days=1),
        created_at=datetime.now(UTC),
        user=UserInfo(id=user_id, email=email, name=None),
        token_id=token_id,
    )


def _force_console_width(monkeypatch: pytest.MonkeyPatch, width: int) -> StringIO:
    rich_output = StringIO()
    monkeypatch.setattr(console, "_file", rich_output)
    monkeypatch.setattr(console, "_rich_size", {"width": width, "height": 25})
    monkeypatch.setattr(console, "_rich_stdout", None)
    return rich_output


def _patch_persistent_login(
    monkeypatch: pytest.MonkeyPatch, credentials: Credentials | None
) -> None:
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.get_credential_store",
        lambda *, include_env=False: "keyring",
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda *, include_env=False: credentials,
    )


def test_login_json_with_token_returns_saved_operation(
    monkeypatch, capsys, fake_verify_result
) -> None:
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token, git_identity=None: fake_verify_result,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials",
        lambda credentials, **kwargs: "keyring",
    )

    exit_code = cli.main(["--json", "auth", "login", "--token", "secret"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["operation"] == "auth.login"
    assert payload["status"] == "success"
    assert payload["resource"]["email"] == "brian@example.com"
    assert payload["resource"]["source"] == "token"
    assert payload["resource"]["token_store"] == "keyring"
    assert payload["resource"]["saved"] is True
    assert payload["resource"]["workspace"] is None
    assert [step["action"] for step in payload["next_steps_structured"]] == [
        "platform.clone_repository",
        "doctor",
    ]


def test_login_plain_with_token_prints_clone_and_doctor(
    monkeypatch, capsys, fake_verify_result
) -> None:
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token, git_identity=None: fake_verify_result,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials",
        lambda credentials, **kwargs: "keyring",
    )

    exit_code = cli.main(["--plain", "auth", "login", "--token", "secret"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Logged in as brian@example.com." in captured.out
    assert "Create or open a workspace in the Osmosis Platform" in captured.out
    assert "osmosis doctor" in captured.out
    assert "workspace link" not in captured.out


def test_login_rich_with_token_prints_success_panel(
    monkeypatch, capsys, fake_verify_result
) -> None:
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token, git_identity=None: fake_verify_result,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials",
        lambda credentials, **kwargs: "keyring",
    )
    rich_output = _force_console_width(monkeypatch, width=112)

    exit_code = cli.main(["auth", "login", "--token", "secret"])

    capsys.readouterr()
    rendered = rich_output.getvalue()
    assert exit_code == 0
    assert "Osmosis AI" in rendered
    assert "Login Successful" in rendered
    assert "brian@example.com" in rendered


def test_invalid_token_does_not_mutate_existing_session(monkeypatch, capsys) -> None:
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials", lambda **kwargs: _credentials()
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token, git_identity=None: (_ for _ in ()).throw(
            LoginError("Authentication failed.", status_code=401)
        ),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials",
        lambda credentials, **kwargs: pytest.fail("invalid token must not be saved"),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.revoke_cli_token",
        lambda credentials: pytest.fail("existing token must not be revoked"),
    )

    exit_code = cli.main(["--json", "auth", "login", "--token", "bad"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert json.loads(captured.err)["error"]["code"] == "AUTH_REQUIRED"


def test_login_json_with_platform_verify_error_is_platform_error(
    monkeypatch, capsys
) -> None:
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token, git_identity=None: (_ for _ in ()).throw(
            LoginError("Platform error", status_code=500)
        ),
    )

    exit_code = cli.main(["--json", "auth", "login", "--token", "secret"])

    envelope = json.loads(capsys.readouterr().err)
    assert exit_code == 1
    assert envelope["error"]["code"] == "PLATFORM_ERROR"
    assert envelope["error"]["details"]["status_code"] == 500


def test_login_json_with_426_preserves_upgrade_details(monkeypatch, capsys) -> None:
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token, git_identity=None: (_ for _ in ()).throw(
            LoginError(
                "Upgrade required",
                code="UPGRADE_REQUIRED",
                status_code=426,
                details={"status": "unsupported"},
            )
        ),
    )

    exit_code = cli.main(["--json", "auth", "login", "--token", "secret"])

    envelope = json.loads(capsys.readouterr().err)
    assert exit_code == 1
    assert envelope["error"]["code"] == "UPGRADE_REQUIRED"
    assert envelope["error"]["details"] == {
        "status_code": 426,
        "status": "unsupported",
    }


def test_login_json_with_malformed_response_is_platform_error(
    monkeypatch, capsys
) -> None:
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token, git_identity=None: (_ for _ in ()).throw(
            LoginError("Invalid response from platform")
        ),
    )

    exit_code = cli.main(["--json", "auth", "login", "--token", "secret"])

    envelope = json.loads(capsys.readouterr().err)
    assert exit_code == 1
    assert envelope["error"]["code"] == "PLATFORM_ERROR"


def test_login_revokes_before_cleaning_up_old_keyring_token(
    monkeypatch, capsys, fake_verify_result
) -> None:
    calls: list[str] = []
    old_credentials = _credentials(access_token="old", token_id="tok_old")
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda *, include_env=False: old_credentials,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token, git_identity=None: fake_verify_result,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials",
        lambda credentials, **kwargs: calls.append("save") or "keyring",
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.revoke_cli_token",
        lambda credentials: calls.append("revoke") or True,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials._cleanup_replaced_credentials",
        lambda old, current: calls.append("cleanup") or True,
    )

    exit_code = cli.main(["--json", "auth", "login", "--token", "secret"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "success"
    assert calls == ["save", "revoke", "cleanup"]


def test_login_save_failure_does_not_revoke_old_token(
    monkeypatch, fake_verify_result
) -> None:
    calls: list[str] = []
    old_credentials = _credentials(access_token="old", token_id="tok_old")
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda *, include_env=False: old_credentials,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token, git_identity=None: fake_verify_result,
    )

    def fail_save(credentials: Credentials, **kwargs) -> str:
        calls.append("save")
        raise OSError("disk full")

    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials", fail_save
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.revoke_cli_token",
        lambda credentials: calls.append("revoke") or True,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials._cleanup_replaced_credentials",
        lambda old, current: calls.append("cleanup") or True,
    )

    with pytest.raises(OSError, match="disk full"):
        auth_module._login_with_token(token="new-token")

    assert calls == ["save"]


def test_explicit_token_overrides_active_environment_token(
    monkeypatch, capsys, fake_verify_result
) -> None:
    monkeypatch.setenv("OSMOSIS_TOKEN", "env-token")
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda token, git_identity=None: fake_verify_result,
    )
    saved: list[str] = []
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.save_credentials",
        lambda credentials, **kwargs: (
            saved.append(credentials.access_token) or "keyring"
        ),
    )

    exit_code = cli.main(["--json", "auth", "login", "--token", "new-token"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "success"
    assert saved == ["new-token"]


def test_login_with_env_token_reports_conflict_without_verifying(
    monkeypatch, capsys
) -> None:
    monkeypatch.setenv("OSMOSIS_TOKEN", "env-token")
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.verify_token",
        lambda *args, **kwargs: pytest.fail("auth login must not verify env token"),
    )

    exit_code = cli.main(["--json", "auth", "login"])

    captured = capsys.readouterr()
    envelope = json.loads(captured.err)
    assert exit_code == 1
    assert envelope["error"]["code"] == "CONFLICT"
    assert "unset" in envelope["error"]["message"]
    assert "CI/CD" in envelope["error"]["message"]


def test_login_json_without_token_or_env_requires_interactive(
    monkeypatch, capsys
) -> None:
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)

    exit_code = cli.main(["--json", "auth", "login"])

    envelope = json.loads(capsys.readouterr().err)
    assert exit_code == 1
    assert envelope["error"]["code"] == "INTERACTIVE_REQUIRED"


def test_logout_json_without_yes_fails_fast(monkeypatch, capsys) -> None:
    _patch_persistent_login(monkeypatch, _credentials())

    exit_code = cli.main(["--json", "auth", "logout"])

    envelope = json.loads(capsys.readouterr().err)
    assert exit_code == 1
    assert envelope["error"]["code"] == "INTERACTIVE_REQUIRED"


def test_logout_json_with_yes_removes_persistent_login(monkeypatch, capsys) -> None:
    _patch_persistent_login(monkeypatch, _credentials())
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.local_config.reset_session", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.revoke_cli_token",
        lambda credentials: True,
    )

    exit_code = cli.main(["--json", "auth", "logout", "--yes"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "success"
    assert payload["resource"]["revoked"] is True
    assert payload["resource"]["logged_in"] is False
    assert payload["resource"]["persistent_login"] is False
    assert payload["resource"]["effective_source"] is None


def test_logout_reports_failure_when_keyring_is_unavailable(
    monkeypatch, capsys
) -> None:
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.get_credential_store",
        lambda *, include_env=False: "keyring",
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.load_credentials",
        lambda *, include_env=False: (_ for _ in ()).throw(
            CLIError("keyring locked", code="KEYRING_UNAVAILABLE")
        ),
    )
    reset_kwargs: list[dict[str, bool]] = []

    def fail_reset(**kwargs) -> None:
        reset_kwargs.append(kwargs)
        raise CLIError("keyring locked", code="KEYRING_UNAVAILABLE")

    monkeypatch.setattr(
        "osmosis_ai.platform.auth.local_config.reset_session",
        fail_reset,
    )

    exit_code = cli.main(["--json", "auth", "logout", "--yes"])

    envelope = json.loads(capsys.readouterr().err)
    assert exit_code == 1
    assert envelope["error"]["code"] == "KEYRING_UNAVAILABLE"
    assert reset_kwargs == [{"recover_invalid_credentials": True}]


def test_logout_with_env_token_only_does_not_reset_session(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OSMOSIS_TOKEN", "env-token")
    calls: list[str] = []
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.local_config.reset_session",
        lambda **kwargs: calls.append("reset"),
    )

    exit_code = cli.main(["--json", "auth", "logout", "--yes"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "noop"
    assert payload["resource"]["logged_in"] is True
    assert payload["resource"]["persistent_login"] is False
    assert payload["resource"]["effective_source"] == "environment"
    assert calls == []


def test_logout_removes_saved_login_but_env_token_remains_active(
    monkeypatch, capsys
) -> None:
    monkeypatch.setenv("OSMOSIS_TOKEN", "env-token")
    _patch_persistent_login(monkeypatch, _credentials(access_token="saved"))
    calls: list[str] = []
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.local_config.reset_session",
        lambda **kwargs: calls.append("reset"),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.revoke_cli_token",
        lambda credentials: calls.append("revoke") or True,
    )

    exit_code = cli.main(["--json", "auth", "logout", "--yes"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["resource"]["logged_in"] is True
    assert payload["resource"]["persistent_login"] is False
    assert payload["resource"]["effective_source"] == "environment"
    assert "still active" in payload["message"]
    assert calls == ["revoke", "reset"]


def test_logout_cleans_stale_metadata_without_a_token(monkeypatch, capsys) -> None:
    _patch_persistent_login(monkeypatch, None)
    calls: list[str] = []
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.local_config.reset_session",
        lambda **kwargs: calls.append("reset"),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.revoke_cli_token",
        lambda credentials: pytest.fail("missing token cannot be revoked"),
    )

    exit_code = cli.main(["--json", "auth", "logout", "--yes"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "success"
    assert payload["resource"]["revoked"] is False
    assert calls == ["reset"]


def test_logout_rich_with_env_token_only_prints_unset_once(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OSMOSIS_TOKEN", "env-token")

    exit_code = cli.main(["auth", "logout"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.count("Run 'unset OSMOSIS_TOKEN' to logout.") == 1


def test_logout_escape_behaves_like_decline(monkeypatch, capsys) -> None:
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    _patch_persistent_login(monkeypatch, _credentials())
    monkeypatch.setattr("osmosis_ai.cli.prompts.confirm", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.local_config.reset_session",
        lambda: pytest.fail("cancelled logout must not reset the session"),
    )

    exit_code = cli.main(["auth", "logout"])

    assert exit_code == 0
    assert "Cancelled." in capsys.readouterr().out
