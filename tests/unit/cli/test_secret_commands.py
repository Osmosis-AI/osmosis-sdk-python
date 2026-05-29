"""Tests for `osmosis secret` commands (list + add).

These focus on the security-critical invariants:
  * `list` returns names + metadata only — never a value.
  * `add` accepts the value from --env or a hidden prompt, never as a
    plaintext argv flag, and never echoes it back in any output mode.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import osmosis_ai.cli.main as cli
import osmosis_ai.platform.cli.secret as secret_module
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import OutputFormat, override_output_context
from osmosis_ai.platform.api.models import (
    EnvironmentSecretInfo,
    PaginatedEnvironmentSecrets,
)

GIT_IDENTITY = "acme/rollouts"
REPO_URL = "https://github.com/acme/rollouts.git"
PROJECT_ROOT = "/repo"
SECRETS_URL = "https://platform.osmosis.ai/acme/secrets"

# A unique sentinel used as the secret value so we can assert it never leaks
# into any output stream.
SENTINEL_VALUE = "sk-NEVER-PRINT-THIS-1a2b3c4d"


def assert_git_context(data: dict[str, object]) -> None:
    assert data["workspace_directory"] == PROJECT_ROOT
    assert data["git"] == {"identity": GIT_IDENTITY, "remote_url": REPO_URL}


def _stub_git_context(monkeypatch: pytest.MonkeyPatch) -> object:
    fake_credentials = object()
    context = SimpleNamespace(
        workspace_directory=PROJECT_ROOT,
        git_identity=GIT_IDENTITY,
        repo_url=REPO_URL,
        credentials=fake_credentials,
    )
    monkeypatch.setattr(
        secret_module,
        "require_git_workspace_directory_context",
        lambda: context,
    )
    return fake_credentials


def _secret(
    *,
    id: str = "sec_1",
    name: str = "OPENAI_API_KEY",
    creator_name: str | None = "Ada",
    scope: str | None = "workspace",
    platform_url: str | None = None,
) -> EnvironmentSecretInfo:
    return EnvironmentSecretInfo(
        id=id,
        name=name,
        created_at="2026-05-01T00:00:00Z",
        updated_at="2026-05-01T00:00:01Z",
        creator_name=creator_name,
        scope=scope,
        platform_url=platform_url,
    )


# ── list ──────────────────────────────────────────────────────────


def test_secret_list_json_envelope_has_no_value(monkeypatch, capsys) -> None:
    fake_credentials = _stub_git_context(monkeypatch)

    class FakeClient:
        def list_environment_secrets(
            self, *, limit, offset, scope, git_identity, credentials=None
        ):
            assert credentials is fake_credentials
            assert git_identity == GIT_IDENTITY
            assert limit == 10
            assert offset == 0
            return PaginatedEnvironmentSecrets(
                environment_secrets=[_secret()],
                total_count=20,
                has_more=True,
                next_offset=10,
                platform_url=SECRETS_URL,
            )

    monkeypatch.setattr(secret_module, "OsmosisClient", FakeClient)

    exit_code = cli.main(["--json", "secret", "list", "--limit", "10"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["schema_version"] == 1
    item = payload["items"][0]
    assert item["id"] == "sec_1"
    assert item["name"] == "OPENAI_API_KEY"
    # No value field anywhere in the serialized item.
    assert "value" not in item
    assert "secret_value" not in item
    assert payload["total_count"] == 20
    assert payload["has_more"] is True
    assert payload["next_offset"] == 10
    assert payload["platform_url"] == SECRETS_URL
    assert_git_context(payload)


def test_secret_list_plain_emits_names(monkeypatch, capsys) -> None:
    _stub_git_context(monkeypatch)

    class FakeClient:
        def list_environment_secrets(
            self, *, limit, offset, scope, git_identity, credentials=None
        ):
            return PaginatedEnvironmentSecrets(
                environment_secrets=[
                    _secret(id="sec_1", name="ALPHA"),
                    _secret(id="sec_2", name="BETA"),
                ],
                total_count=2,
                has_more=False,
                platform_url=SECRETS_URL,
            )

    monkeypatch.setattr(secret_module, "OsmosisClient", FakeClient)

    exit_code = cli.main(["--plain", "secret", "list"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "ALPHA" in captured.out
    assert "BETA" in captured.out


def test_secret_list_requires_linked_workspace(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    rc = cli.main(["--json", "secret", "list"])

    assert rc == 1
    assert "workspace directory" in capsys.readouterr().err.lower()


def test_secret_list_forwards_scope_to_client(monkeypatch, capsys) -> None:
    _stub_git_context(monkeypatch)
    seen: dict[str, object] = {}

    class FakeClient:
        def list_environment_secrets(
            self, *, limit, offset, scope, git_identity, credentials=None
        ):
            seen["scope"] = scope
            return PaginatedEnvironmentSecrets(
                environment_secrets=[_secret(scope="user")],
                total_count=1,
                has_more=False,
                platform_url=SECRETS_URL,
            )

    monkeypatch.setattr(secret_module, "OsmosisClient", FakeClient)
    exit_code = cli.main(["--json", "secret", "list", "--scope", "user"])
    assert exit_code == 0
    assert seen["scope"] == "user"


def test_secret_list_default_scope_is_all(monkeypatch, capsys) -> None:
    _stub_git_context(monkeypatch)
    seen: dict[str, object] = {}

    class FakeClient:
        def list_environment_secrets(
            self, *, limit, offset, scope, git_identity, credentials=None
        ):
            seen["scope"] = scope
            return PaginatedEnvironmentSecrets(
                environment_secrets=[],
                total_count=0,
                has_more=False,
            )

    monkeypatch.setattr(secret_module, "OsmosisClient", FakeClient)
    assert cli.main(["--json", "secret", "list"]) == 0
    assert seen["scope"] == "all"


def test_secret_list_invalid_scope_is_validation_error(monkeypatch, capsys) -> None:
    _stub_git_context(monkeypatch)
    exit_code = cli.main(["--json", "secret", "list", "--scope", "team"])
    captured = capsys.readouterr()
    assert exit_code == 1
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "VALIDATION"


def test_secret_list_plain_shows_scope(monkeypatch, capsys) -> None:
    _stub_git_context(monkeypatch)

    class FakeClient:
        def list_environment_secrets(
            self, *, limit, offset, scope, git_identity, credentials=None
        ):
            return PaginatedEnvironmentSecrets(
                environment_secrets=[
                    _secret(id="a", name="ALPHA", scope="workspace"),
                    _secret(id="b", name="BETA", scope="user"),
                ],
                total_count=2,
                has_more=False,
            )

    monkeypatch.setattr(secret_module, "OsmosisClient", FakeClient)
    assert cli.main(["--plain", "secret", "list"]) == 0
    out = capsys.readouterr().out
    assert "workspace" in out
    assert "user" in out


# ── set (upsert) ──────────────────────────────────────────────────


def test_secret_set_from_env_success_omits_value(monkeypatch, capsys) -> None:
    fake_credentials = _stub_git_context(monkeypatch)
    monkeypatch.setenv("SOURCE_VAR", SENTINEL_VALUE)
    seen: dict[str, object] = {}

    class FakeClient:
        def set_environment_secret(
            self, name, value, *, scope, git_identity, credentials=None
        ):
            assert credentials is fake_credentials
            assert git_identity == GIT_IDENTITY
            seen["name"] = name
            seen["value"] = value
            seen["scope"] = scope
            return _secret(name=name, scope=scope, platform_url=SECRETS_URL)

    monkeypatch.setattr(secret_module, "OsmosisClient", FakeClient)

    exit_code = cli.main(
        ["--json", "secret", "set", "OPENAI_API_KEY", "--scope", "workspace",
         "--env", "SOURCE_VAR"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    # The value was read from the env var and forwarded to the client...
    assert seen == {
        "name": "OPENAI_API_KEY",
        "value": SENTINEL_VALUE,
        "scope": "workspace",
    }
    payload = json.loads(captured.out)
    assert payload["operation"] == "secret.set"
    assert payload["status"] == "success"
    assert payload["resource"]["name"] == "OPENAI_API_KEY"
    assert payload["platform_url"] == SECRETS_URL
    assert_git_context(payload["resource"])
    # ...but it must never appear in stdout or stderr.
    assert SENTINEL_VALUE not in captured.out
    assert SENTINEL_VALUE not in captured.err
    assert "value" not in payload["resource"]


def test_secret_set_from_env_forwards_name_value_scope(monkeypatch, capsys) -> None:
    fake_credentials = _stub_git_context(monkeypatch)
    monkeypatch.setenv("SOURCE_VAR", SENTINEL_VALUE)
    seen: dict[str, object] = {}

    class FakeClient:
        def set_environment_secret(
            self, name, value, *, scope, git_identity, credentials=None
        ):
            assert credentials is fake_credentials
            assert git_identity == GIT_IDENTITY
            seen.update(name=name, value=value, scope=scope)
            return _secret(name=name, scope=scope, platform_url=SECRETS_URL)

    monkeypatch.setattr(secret_module, "OsmosisClient", FakeClient)

    exit_code = cli.main(
        ["--json", "secret", "set", "OPENAI_API_KEY",
         "--scope", "user", "--env", "SOURCE_VAR"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert seen == {"name": "OPENAI_API_KEY", "value": SENTINEL_VALUE, "scope": "user"}
    payload = json.loads(captured.out)
    assert payload["operation"] == "secret.set"
    assert payload["status"] == "success"
    assert payload["resource"]["name"] == "OPENAI_API_KEY"
    assert payload["resource"]["scope"] == "user"
    assert SENTINEL_VALUE not in captured.out
    assert SENTINEL_VALUE not in captured.err
    assert "value" not in payload["resource"]


def test_secret_set_defaults_scope_to_workspace(monkeypatch, capsys) -> None:
    _stub_git_context(monkeypatch)
    monkeypatch.setenv("SOURCE_VAR", SENTINEL_VALUE)
    seen: dict[str, object] = {}

    class FakeClient:
        def set_environment_secret(
            self, name, value, *, scope, git_identity, credentials=None
        ):
            seen["scope"] = scope
            return _secret(name=name, scope=scope)

    monkeypatch.setattr(secret_module, "OsmosisClient", FakeClient)

    exit_code = cli.main(
        ["--json", "secret", "set", "OPENAI_API_KEY", "--env", "SOURCE_VAR"]
    )
    assert exit_code == 0
    assert seen["scope"] == "workspace"


def test_secret_set_invalid_scope_is_validation_error(monkeypatch, capsys) -> None:
    _stub_git_context(monkeypatch)
    monkeypatch.setenv("SOURCE_VAR", SENTINEL_VALUE)

    exit_code = cli.main(
        ["--json", "secret", "set", "OPENAI_API_KEY",
         "--scope", "team", "--env", "SOURCE_VAR"]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "VALIDATION"
    assert SENTINEL_VALUE not in captured.err


def test_secret_set_env_unset_is_validation_error(monkeypatch, capsys) -> None:
    _stub_git_context(monkeypatch)
    monkeypatch.delenv("MISSING_VAR", raising=False)

    exit_code = cli.main(
        ["--json", "secret", "set", "MY_SECRET", "--scope", "workspace",
         "--env", "MISSING_VAR"]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "VALIDATION"
    assert "MISSING_VAR" in payload["error"]["message"]


def test_secret_set_without_value_source_requires_interactive(
    monkeypatch, capsys
) -> None:
    _stub_git_context(monkeypatch)

    exit_code = cli.main(["--json", "secret", "set", "MY_SECRET", "--scope", "workspace"])
    captured = capsys.readouterr()

    assert exit_code == 1
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "INTERACTIVE_REQUIRED"
    assert "--env" in payload["error"]["message"]


def test_secret_set_without_value_source_checks_before_workspace(
    monkeypatch, capsys
) -> None:
    def fail_if_workspace_resolved():
        raise AssertionError("workspace context should not be resolved")

    monkeypatch.setattr(
        secret_module,
        "require_git_workspace_directory_context",
        fail_if_workspace_resolved,
    )

    exit_code = cli.main(["--json", "secret", "set", "MY_SECRET", "--scope", "workspace"])
    captured = capsys.readouterr()

    assert exit_code == 1
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "INTERACTIVE_REQUIRED"
    assert "--env" in payload["error"]["message"]


def test_secret_set_invalid_name_is_validation_error(monkeypatch, capsys) -> None:
    _stub_git_context(monkeypatch)
    monkeypatch.setenv("SOURCE_VAR", SENTINEL_VALUE)

    exit_code = cli.main(
        ["--json", "secret", "set", "bad name!", "--scope", "workspace",
         "--env", "SOURCE_VAR"]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "VALIDATION"
    assert SENTINEL_VALUE not in captured.err


def test_secret_name_with_trailing_newline_is_invalid() -> None:
    with pytest.raises(CLIError) as exc:
        secret_module._validate_secret_name("MY_SECRET\n")

    assert exc.value.code == "VALIDATION"


@pytest.mark.parametrize(
    "name",
    ["OPENAI_API_KEY", "A", "DATABASE_URL", "X1", "A_B_C"],
)
def test_validate_secret_name_accepts_screaming_snake_case(name: str) -> None:
    secret_module._validate_secret_name(name)  # no raise


@pytest.mark.parametrize(
    "name",
    ["lowercase", "1LEADING_DIGIT", "_LEADING_UNDERSCORE", "HAS-DASH", "HAS SPACE", ""],
)
def test_validate_secret_name_rejects_non_env_style(name: str) -> None:
    with pytest.raises(CLIError) as exc:
        secret_module._validate_secret_name(name)
    assert exc.value.code == "VALIDATION"


def test_secret_set_conflict_maps_to_conflict_code(monkeypatch, capsys) -> None:
    _stub_git_context(monkeypatch)
    monkeypatch.setenv("SOURCE_VAR", SENTINEL_VALUE)

    from osmosis_ai.platform.auth import PlatformAPIError

    class FakeClient:
        def set_environment_secret(
            self, name, value, *, scope, git_identity, credentials=None
        ):
            raise PlatformAPIError(
                "A secret with this name already exists",
                status_code=409,
            )

    monkeypatch.setattr(secret_module, "OsmosisClient", FakeClient)

    exit_code = cli.main(
        ["--json", "secret", "set", "OPENAI_API_KEY", "--scope", "workspace",
         "--env", "SOURCE_VAR"]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "CONFLICT"
    assert SENTINEL_VALUE not in captured.err


def test_secret_set_redacts_value_from_platform_error(monkeypatch, capsys) -> None:
    _stub_git_context(monkeypatch)
    monkeypatch.setenv("SOURCE_VAR", SENTINEL_VALUE)

    from osmosis_ai.platform.auth import PlatformAPIError

    class FakeClient:
        def set_environment_secret(
            self, name, value, *, scope, git_identity, credentials=None
        ):
            assert value == SENTINEL_VALUE
            raise PlatformAPIError(
                f"Rejected value {SENTINEL_VALUE}",
                status_code=400,
                error_code=f"bad_{SENTINEL_VALUE}",
                field=f"field_{SENTINEL_VALUE}",
                details={
                    "value": SENTINEL_VALUE,
                    "nested": {"message": f"contains {SENTINEL_VALUE}"},
                    "items": [SENTINEL_VALUE],
                },
            )

    monkeypatch.setattr(secret_module, "OsmosisClient", FakeClient)

    exit_code = cli.main(
        ["--json", "secret", "set", "OPENAI_API_KEY", "--scope", "user",
         "--env", "SOURCE_VAR"]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert SENTINEL_VALUE not in captured.out
    assert SENTINEL_VALUE not in captured.err
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "VALIDATION"
    assert "[REDACTED]" in payload["error"]["message"]
    assert payload["error"]["details"]["platform_code"] == "bad_[REDACTED]"
    assert payload["error"]["details"]["field"] == "field_[REDACTED]"
    assert payload["error"]["details"]["value"] == "[REDACTED]"
    assert payload["error"]["details"]["nested"]["message"] == "contains [REDACTED]"
    assert payload["error"]["details"]["items"] == ["[REDACTED]"]


def test_secret_set_redacted_platform_error_suppresses_original_cause(
    monkeypatch,
) -> None:
    _stub_git_context(monkeypatch)
    monkeypatch.setenv("SOURCE_VAR", SENTINEL_VALUE)

    from osmosis_ai.platform.auth import PlatformAPIError

    class FakeClient:
        def set_environment_secret(
            self, name, value, *, scope, git_identity, credentials=None
        ):
            raise PlatformAPIError(
                f"Rejected value {SENTINEL_VALUE}",
                status_code=400,
            )

    monkeypatch.setattr(secret_module, "OsmosisClient", FakeClient)

    with override_output_context(format=OutputFormat.json, interactive=False):
        with pytest.raises(PlatformAPIError) as exc:
            secret_module.set_secret(
                name="OPENAI_API_KEY", scope="workspace", env="SOURCE_VAR"
            )

    assert exc.value.__cause__ is None
    assert SENTINEL_VALUE not in str(exc.value)


# ── value resolution unit tests (incl. hidden prompt path) ────────


def _rich_interactive_output() -> SimpleNamespace:
    return SimpleNamespace(format=OutputFormat.rich, interactive=True)


def test_resolve_value_from_hidden_prompt(monkeypatch) -> None:
    import osmosis_ai.cli.prompts as prompts

    monkeypatch.setattr(prompts, "password", lambda *a, **k: SENTINEL_VALUE)

    value = secret_module._resolve_secret_value(
        env=None, output=_rich_interactive_output()
    )

    assert value == SENTINEL_VALUE


def test_resolve_value_prompt_cancel_returns_none(monkeypatch) -> None:
    import osmosis_ai.cli.prompts as prompts

    monkeypatch.setattr(prompts, "password", lambda *a, **k: None)

    value = secret_module._resolve_secret_value(
        env=None, output=_rich_interactive_output()
    )

    assert value is None


def test_resolve_value_empty_prompt_is_validation_error(monkeypatch) -> None:
    import osmosis_ai.cli.prompts as prompts

    monkeypatch.setattr(prompts, "password", lambda *a, **k: "")

    with pytest.raises(CLIError) as exc:
        secret_module._resolve_secret_value(env=None, output=_rich_interactive_output())

    assert exc.value.code == "VALIDATION"


def test_resolve_value_from_env_preserves_value_verbatim(monkeypatch) -> None:
    spaced = "  value with spaces and\nnewline  "
    monkeypatch.setenv("SRC", spaced)

    value = secret_module._resolve_secret_value(
        env="SRC", output=_rich_interactive_output()
    )

    assert value == spaced


# ── delete ────────────────────────────────────────────────────────


def test_secret_delete_forwards_name_scope(monkeypatch, capsys) -> None:
    fake_credentials = _stub_git_context(monkeypatch)
    seen: dict[str, object] = {}

    class FakeClient:
        def delete_environment_secret(
            self, name, *, scope, git_identity, credentials=None
        ):
            assert credentials is fake_credentials
            assert git_identity == GIT_IDENTITY
            seen.update(name=name, scope=scope)

    monkeypatch.setattr(secret_module, "OsmosisClient", FakeClient)
    exit_code = cli.main(
        ["--json", "secret", "delete", "OPENAI_API_KEY",
         "--scope", "user", "--yes"]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert seen == {"name": "OPENAI_API_KEY", "scope": "user"}
    payload = json.loads(captured.out)
    assert payload["operation"] == "secret.delete"
    assert payload["status"] == "success"
    assert payload["resource"]["name"] == "OPENAI_API_KEY"
    assert payload["resource"]["scope"] == "user"


def test_secret_delete_requires_scope_confirmation_without_yes(
    monkeypatch, capsys
) -> None:
    _stub_git_context(monkeypatch)
    called = {"deleted": False}

    class FakeClient:
        def delete_environment_secret(
            self, name, *, scope, git_identity, credentials=None
        ):
            called["deleted"] = True

    monkeypatch.setattr(secret_module, "OsmosisClient", FakeClient)
    # --json + non-interactive => require_confirmation emits INTERACTIVE_REQUIRED
    exit_code = cli.main(
        ["--json", "secret", "delete", "OPENAI_API_KEY", "--scope", "workspace"]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "INTERACTIVE_REQUIRED"
    assert called["deleted"] is False


def test_secret_delete_invalid_scope_is_validation_error(monkeypatch, capsys) -> None:
    _stub_git_context(monkeypatch)
    exit_code = cli.main(
        ["--json", "secret", "delete", "OPENAI_API_KEY", "--scope", "all", "--yes"]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "VALIDATION"


def test_secret_delete_not_found_maps_to_not_found(monkeypatch, capsys) -> None:
    _stub_git_context(monkeypatch)
    from osmosis_ai.platform.auth import PlatformAPIError

    class FakeClient:
        def delete_environment_secret(
            self, name, *, scope, git_identity, credentials=None
        ):
            raise PlatformAPIError("Secret not found", status_code=404)

    monkeypatch.setattr(secret_module, "OsmosisClient", FakeClient)
    exit_code = cli.main(
        ["--json", "secret", "delete", "MISSING", "--scope", "user", "--yes"]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "NOT_FOUND"


# ── typer wiring ──────────────────────────────────────────────────


def test_secret_add_subcommand_is_removed(monkeypatch, capsys) -> None:
    _stub_git_context(monkeypatch)
    monkeypatch.setenv("SOURCE_VAR", SENTINEL_VALUE)
    # The old `add` verb must no longer be a recognized subcommand.
    exit_code = cli.main(
        ["--json", "secret", "add", "OPENAI_API_KEY", "--env", "SOURCE_VAR"]
    )
    assert exit_code != 0


def test_secret_set_wires_scope_default_workspace(monkeypatch, capsys) -> None:
    _stub_git_context(monkeypatch)
    monkeypatch.setenv("SOURCE_VAR", SENTINEL_VALUE)
    seen: dict[str, object] = {}

    class FakeClient:
        def set_environment_secret(
            self, name, value, *, scope, git_identity, credentials=None
        ):
            seen["scope"] = scope
            return _secret(name=name, scope=scope)

    monkeypatch.setattr(secret_module, "OsmosisClient", FakeClient)
    assert cli.main(["--json", "secret", "set", "X", "--env", "SOURCE_VAR"]) == 0
    assert seen["scope"] == "workspace"


def test_secret_delete_subcommand_wires_through(monkeypatch, capsys) -> None:
    _stub_git_context(monkeypatch)
    seen: dict[str, object] = {}

    class FakeClient:
        def delete_environment_secret(
            self, name, *, scope, git_identity, credentials=None
        ):
            seen.update(name=name, scope=scope)

    monkeypatch.setattr(secret_module, "OsmosisClient", FakeClient)
    assert cli.main(["--json", "secret", "delete", "X", "--scope", "user", "--yes"]) == 0
    assert seen == {"name": "X", "scope": "user"}
