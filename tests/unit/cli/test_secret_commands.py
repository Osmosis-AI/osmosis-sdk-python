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
from osmosis_ai.cli.output import OutputFormat
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
    platform_url: str | None = None,
) -> EnvironmentSecretInfo:
    return EnvironmentSecretInfo(
        id=id,
        name=name,
        created_at="2026-05-01T00:00:00Z",
        updated_at="2026-05-01T00:00:01Z",
        creator_name=creator_name,
        platform_url=platform_url,
    )


# ── list ──────────────────────────────────────────────────────────


def test_secret_list_json_envelope_has_no_value(monkeypatch, capsys) -> None:
    fake_credentials = _stub_git_context(monkeypatch)

    class FakeClient:
        def list_environment_secrets(
            self, *, limit, offset, git_identity, credentials=None
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
            self, *, limit, offset, git_identity, credentials=None
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


# ── add (--env) ───────────────────────────────────────────────────


def test_secret_add_from_env_success_omits_value(monkeypatch, capsys) -> None:
    fake_credentials = _stub_git_context(monkeypatch)
    monkeypatch.setenv("SOURCE_VAR", SENTINEL_VALUE)
    seen: dict[str, object] = {}

    class FakeClient:
        def create_environment_secret(
            self, name, value, *, git_identity, credentials=None
        ):
            assert credentials is fake_credentials
            assert git_identity == GIT_IDENTITY
            seen["name"] = name
            seen["value"] = value
            return _secret(name=name, platform_url=SECRETS_URL)

    monkeypatch.setattr(secret_module, "OsmosisClient", FakeClient)

    exit_code = cli.main(
        ["--json", "secret", "add", "OPENAI_API_KEY", "--env", "SOURCE_VAR"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    # The value was read from the env var and forwarded to the client...
    assert seen == {"name": "OPENAI_API_KEY", "value": SENTINEL_VALUE}
    payload = json.loads(captured.out)
    assert payload["operation"] == "secret.add"
    assert payload["status"] == "success"
    assert payload["resource"]["name"] == "OPENAI_API_KEY"
    assert payload["platform_url"] == SECRETS_URL
    assert_git_context(payload["resource"])
    # ...but it must never appear in stdout or stderr.
    assert SENTINEL_VALUE not in captured.out
    assert SENTINEL_VALUE not in captured.err
    assert "value" not in payload["resource"]


def test_secret_add_env_unset_is_validation_error(monkeypatch, capsys) -> None:
    _stub_git_context(monkeypatch)
    monkeypatch.delenv("MISSING_VAR", raising=False)

    exit_code = cli.main(
        ["--json", "secret", "add", "MY_SECRET", "--env", "MISSING_VAR"]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "VALIDATION"
    assert "MISSING_VAR" in payload["error"]["message"]


def test_secret_add_without_value_source_requires_interactive(
    monkeypatch, capsys
) -> None:
    _stub_git_context(monkeypatch)

    exit_code = cli.main(["--json", "secret", "add", "MY_SECRET"])
    captured = capsys.readouterr()

    assert exit_code == 1
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "INTERACTIVE_REQUIRED"
    assert "--env" in payload["error"]["message"]


def test_secret_add_invalid_name_is_validation_error(monkeypatch, capsys) -> None:
    _stub_git_context(monkeypatch)
    monkeypatch.setenv("SOURCE_VAR", SENTINEL_VALUE)

    exit_code = cli.main(
        ["--json", "secret", "add", "bad name!", "--env", "SOURCE_VAR"]
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


def test_secret_add_conflict_maps_to_conflict_code(monkeypatch, capsys) -> None:
    _stub_git_context(monkeypatch)
    monkeypatch.setenv("SOURCE_VAR", SENTINEL_VALUE)

    from osmosis_ai.platform.auth import PlatformAPIError

    class FakeClient:
        def create_environment_secret(
            self, name, value, *, git_identity, credentials=None
        ):
            raise PlatformAPIError(
                "A secret with this name already exists",
                status_code=409,
            )

    monkeypatch.setattr(secret_module, "OsmosisClient", FakeClient)

    exit_code = cli.main(
        ["--json", "secret", "add", "OPENAI_API_KEY", "--env", "SOURCE_VAR"]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "CONFLICT"
    assert SENTINEL_VALUE not in captured.err


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
