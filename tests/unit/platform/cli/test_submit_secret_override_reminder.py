"""Tests for separate secrets table rendering and missing-secret error enrichment."""

from __future__ import annotations

import osmosis_ai.platform.cli.shared_submit as shared_submit
from osmosis_ai.platform.api.models import (
    EnvironmentSecretInfo,
    PaginatedEnvironmentSecrets,
)
from osmosis_ai.platform.cli.shared_config import build_secret_table_rows


def test_build_secret_table_rows_annotates_scope() -> None:
    rows = build_secret_table_rows(
        ["OPENAI_API_KEY", "GITHUB_TOKEN", "MY_PERSONAL"],
        user_secret_names={"OPENAI_API_KEY", "MY_PERSONAL"},
        workspace_secret_names={"OPENAI_API_KEY", "GITHUB_TOKEN"},
    )
    assert ("GITHUB_TOKEN", "Workspace") in rows
    assert ("OPENAI_API_KEY", "Personal (overrides workspace)") in rows
    # personal-only: no workspace secret of that name → not an override
    assert ("MY_PERSONAL", "Personal") in rows


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
    assert "not found: OPENAI_API_KEY, WANDB_API_KEY" in msg
    assert "osmosis secret set OPENAI_API_KEY" in msg
    assert "osmosis secret set WANDB_API_KEY" in msg


def test_enrich_missing_secret_error_adds_hint() -> None:
    from osmosis_ai.platform.auth.platform_client import PlatformAPIError

    exc = PlatformAPIError(
        "Secret(s) not found: OPENAI_API_KEY, WANDB_API_KEY",
        404,
        details={
            "error": "Secret(s) not found: OPENAI_API_KEY, WANDB_API_KEY",
            "platform_url": "https://platform.osmosis.ai/my-workspace/secrets",
        },
    )
    enriched = shared_submit._enrich_missing_secret_error(exc)
    assert enriched is not None
    msg = str(enriched)
    assert "osmosis secret set OPENAI_API_KEY" in msg
    assert "osmosis secret set WANDB_API_KEY" in msg
    assert "https://platform.osmosis.ai/my-workspace/secrets" in msg


def test_enrich_missing_secret_error_returns_none_for_other_errors() -> None:
    from osmosis_ai.platform.auth.platform_client import PlatformAPIError

    exc = PlatformAPIError("Some other error", 500)
    assert shared_submit._enrich_missing_secret_error(exc) is None
