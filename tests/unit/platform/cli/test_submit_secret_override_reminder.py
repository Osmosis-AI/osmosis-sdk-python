"""The submit confirmation reminds the user which referenced secrets resolve
to their personal (user-scope) value vs the shared workspace value."""

from __future__ import annotations

import osmosis_ai.platform.cli.shared_submit as shared_submit
from osmosis_ai.platform.api.models import (
    EnvironmentSecretInfo,
    PaginatedEnvironmentSecrets,
)


def _user_secrets_page(names: list[str]) -> PaginatedEnvironmentSecrets:
    return PaginatedEnvironmentSecrets(
        environment_secrets=[
            EnvironmentSecretInfo(id=n, name=n, scope="user") for n in names
        ],
        total_count=len(names),
        has_more=False,
    )


def test_annotates_rows_with_personal_vs_workspace() -> None:
    rows = [
        ("Rollout secrets (2)", "OPENAI_API_KEY, DATABASE_URL"),
    ]
    annotated = shared_submit._annotate_secret_override_row(
        rows,
        referenced=["OPENAI_API_KEY", "DATABASE_URL"],
        user_secret_names={"OPENAI_API_KEY"},
    )
    # the secrets row value now spells out per-secret resolution
    label, value = annotated[0]
    assert label == "Rollout secrets (2)"
    assert "OPENAI_API_KEY (personal)" in value
    assert "DATABASE_URL (workspace)" in value


def test_no_secrets_row_is_unchanged() -> None:
    rows = [("Rollout", "calculator")]
    annotated = shared_submit._annotate_secret_override_row(
        rows, referenced=[], user_secret_names=set()
    )
    assert annotated == rows


def test_fetch_user_secret_names_collects_all_pages(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeClient:
        def list_environment_secrets(
            self, *, limit, offset, scope, credentials, git_identity
        ):
            calls.append({"scope": scope, "offset": offset})
            if offset == 0:
                return PaginatedEnvironmentSecrets(
                    environment_secrets=[
                        EnvironmentSecretInfo(id="A", name="A", scope="user")
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

    names = shared_submit._fetch_user_secret_names(
        FakeClient(), credentials=object(), git_identity="acme/x"
    )
    assert names == {"A", "B"}
    assert all(c["scope"] == "user" for c in calls)


def test_fetch_user_secret_names_swallows_errors(monkeypatch) -> None:
    class FakeClient:
        def list_environment_secrets(self, **kwargs):
            raise RuntimeError("network down")

    # A failure to fetch the reminder must not block submit.
    names = shared_submit._fetch_user_secret_names(
        FakeClient(), credentials=object(), git_identity="acme/x"
    )
    assert names == set()
