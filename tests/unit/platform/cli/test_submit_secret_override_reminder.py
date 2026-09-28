"""Tests for separate secrets table rendering and missing-secret error enrichment."""

from __future__ import annotations

import osmosis_ai.platform.cli.shared_submit as shared_submit
from osmosis_ai.platform.api.models import (
    EnvironmentSecretInfo,
    PaginatedEnvironmentSecrets,
)


def test_fetch_secret_scopes_collects_all_pages_and_partitions(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeClient:
        def list_environment_secrets(
            self, *, limit, offset, scope, credentials, git_identity
        ):
            calls.append({"scope": scope, "offset": offset})
            if offset == 0:
                return PaginatedEnvironmentSecrets(
                    environment_secrets=[
                        EnvironmentSecretInfo(id="A", name="A", scope="workspace")
                    ],
                    total_count=2,
                    has_more=True,
                    next_offset=1,
                )
            return PaginatedEnvironmentSecrets(
                environment_secrets=[
                    EnvironmentSecretInfo(id="B", name="B", scope="user")
                ],
                total_count=2,
                has_more=False,
            )

    result = shared_submit._fetch_secret_scopes(
        FakeClient(), credentials=object(), git_identity="acme/x"
    )
    assert result == ({"A"}, {"B"})
    assert all(c["scope"] == "all" for c in calls)


def test_fetch_secret_scopes_returns_none_on_error(monkeypatch) -> None:
    class FakeClient:
        def list_environment_secrets(self, **kwargs):
            raise RuntimeError("network down")

    result = shared_submit._fetch_secret_scopes(
        FakeClient(), credentials=object(), git_identity="acme/x"
    )
    assert result is None


def test_missing_secret_message_lists_set_commands() -> None:
    msg = shared_submit._missing_secret_message(["OPENAI_API_KEY", "WANDB_API_KEY"])
    assert "Could not find secret(s): OPENAI_API_KEY, WANDB_API_KEY" in msg
    assert "osmosis secret set OPENAI_API_KEY" in msg
    assert "osmosis secret set WANDB_API_KEY" in msg
    assert "Secrets default to personal scope" in msg
    assert "--scope workspace" in msg


def test_enrich_missing_secret_error_adds_hint() -> None:
    from osmosis_ai.platform.auth.platform_client import PlatformAPIError

    exc = PlatformAPIError(
        "Secret(s) not found: OPENAI_API_KEY, WANDB_API_KEY",
        404,
        details={
            "error": "Secret(s) not found: OPENAI_API_KEY, WANDB_API_KEY",
            "platform_url": "https://platform.example.test/my-workspace/secrets",
        },
    )
    enriched = shared_submit._enrich_missing_secret_error(exc)
    assert enriched is not None
    msg = str(enriched)
    assert "osmosis secret set OPENAI_API_KEY" in msg
    assert "osmosis secret set WANDB_API_KEY" in msg
    assert "Secrets default to personal scope" in msg
    assert "--scope workspace" in msg
    assert "https://platform.example.test/my-workspace/secrets" in msg


def test_enrich_missing_secret_error_reads_legacy_error_field() -> None:
    from osmosis_ai.platform.auth.platform_client import PlatformAPIError

    exc = PlatformAPIError(
        "One or more secrets could not be resolved.",
        404,
        details={
            "error": "Secret(s) not found: OPENAI_API_KEY",
            "message": "One or more secrets could not be resolved.",
        },
    )
    enriched = shared_submit._enrich_missing_secret_error(exc)
    assert enriched is not None
    assert "osmosis secret set OPENAI_API_KEY" in str(enriched)


def test_enrich_missing_secret_error_returns_none_for_other_errors() -> None:
    from osmosis_ai.platform.auth.platform_client import PlatformAPIError

    exc = PlatformAPIError("Some other error", 500)
    assert shared_submit._enrich_missing_secret_error(exc) is None
