"""Redaction of provided secrets echoed in platform API errors."""

from __future__ import annotations

from pathlib import Path

import pytest

from osmosis_ai.cli.output.context import OutputFormat, override_output_context
from osmosis_ai.cli.output.error import classify_error
from osmosis_ai.platform.auth.platform_client import (
    PlatformAPIError,
    SubscriptionRequiredError,
)
from osmosis_ai.platform.cli.secret_redact import redact_provided_secrets
from osmosis_ai.platform.cli.shared_submit import confirm_remote_fetch_and_post

SENTINEL = "super-secret-value-xyz"


def test_redact_provided_secrets_scrubs_message_and_details() -> None:
    exc = PlatformAPIError(
        f"Rejected {SENTINEL}",
        status_code=400,
        error_code=f"bad_{SENTINEL}",
        field=f"field_{SENTINEL}",
        details={"nested": {"message": f"contains {SENTINEL}"}, "items": [SENTINEL]},
    )

    redacted = redact_provided_secrets(exc, [SENTINEL, "other"])

    assert SENTINEL not in str(redacted)
    assert "[REDACTED]" in str(redacted)
    assert SENTINEL not in (redacted.error_code or "")
    assert SENTINEL not in (redacted.field or "")
    assert SENTINEL not in str(redacted.details)


def test_redaction_preserves_subscription_error_classification() -> None:
    exc = SubscriptionRequiredError(
        f"Subscription required for {SENTINEL}",
        error_code="SUBSCRIPTION_REQUIRED",
        field=f"secret_{SENTINEL}",
        details={"echo": SENTINEL},
    )

    redacted = redact_provided_secrets(exc, [SENTINEL])

    assert isinstance(redacted, SubscriptionRequiredError)
    assert classify_error(redacted).code == "SUBSCRIPTION_REQUIRED"
    assert SENTINEL not in str(redacted)
    assert SENTINEL not in (redacted.field or "")
    assert SENTINEL not in str(redacted.details)


def test_confirm_remote_fetch_and_post_redacts_provided_secrets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.shared_submit.print_remote_fetch_notice",
        lambda *args, **kwargs: ([], []),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.shared_submit.require_confirmation",
        lambda *args, **kwargs: None,
    )

    def post() -> None:
        raise PlatformAPIError(
            f"echoed {SENTINEL}",
            status_code=400,
            details={"msg": SENTINEL},
        )

    with (
        override_output_context(format=OutputFormat.json),
        pytest.raises(PlatformAPIError) as exc_info,
    ):
        confirm_remote_fetch_and_post(
            yes=True,
            confirm_prompt="Submit?",
            full_summary=[],
            workspace_directory=tmp_path,
            status_message="Submitting...",
            post=post,
            provided_secrets={"OPENAI_API_KEY": SENTINEL},
        )

    assert SENTINEL not in str(exc_info.value)
    assert SENTINEL not in str(exc_info.value.details)
    assert "[REDACTED]" in str(exc_info.value)
    assert exc_info.value.__cause__ is None
