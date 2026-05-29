"""serialize_environment_secret must expose scope but never a value."""

from __future__ import annotations

from osmosis_ai.cli.output import serialize_environment_secret
from osmosis_ai.platform.api.models import EnvironmentSecretInfo


def test_serialize_includes_scope_and_no_value() -> None:
    out = serialize_environment_secret(
        EnvironmentSecretInfo(
            id="sec_1",
            name="OPENAI_API_KEY",
            created_at="2026-05-01T00:00:00Z",
            updated_at="2026-05-01T00:00:01Z",
            creator_name="Ada",
            scope="user",
        )
    )
    assert out["scope"] == "user"
    assert out["name"] == "OPENAI_API_KEY"
    assert "value" not in out
    assert "secret_value" not in out
