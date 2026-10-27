"""Redaction of provided secrets echoed in platform API errors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osmosis_ai._redaction import invocation_redaction, register_invocation_secrets
from osmosis_ai.cli.output.context import (
    OutputContext,
    OutputFormat,
    override_output_context,
)
from osmosis_ai.cli.output.error import classify_error
from osmosis_ai.cli.output.renderer import render
from osmosis_ai.cli.output.result import ListResult, OperationResult
from osmosis_ai.platform.auth.platform_client import (
    PlatformAPIError,
    SubscriptionRequiredError,
)
from osmosis_ai.platform.cli.secret_redact import (
    redact_provided_secrets,
    redact_secret_text,
    redact_secret_value,
)
from osmosis_ai.platform.cli.shared_submit import confirm_remote_fetch_and_post

SENTINEL = "super-secret-value-xyz"


def test_success_output_redacts_every_presentation_field_without_changing_schema(
    capsys: pytest.CaptureFixture[str],
) -> None:
    original = OperationResult(
        operation="eval.retry",
        status="success",
        resource={"id": "id", "retryable_samples": 1, "nested": {SENTINEL: SENTINEL}},
        message=SENTINEL,
        display_next_steps=[SENTINEL],
        next_steps_structured=[{"action": "open_url", "url": SENTINEL}],
        extra={"url": SENTINEL},
    )
    with invocation_redaction():
        register_invocation_secrets([SENTINEL, "id", "1"])
        render(original, OutputContext(format=OutputFormat.json, interactive=False))
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "schema_version": 1,
        "status": "success",
        "operation": "eval.retry",
        "resource": {
            "id": "[REDACTED]",
            "retryable_samples": "[REDACTED]",
            "nested": {"[REDACTED]": "[REDACTED]"},
        },
        "message": "[REDACTED]",
        "next_steps_structured": [{"action": "open_url", "url": "[REDACTED]"}],
        "url": "[REDACTED]",
    }
    # The command's own result is untouched; only its presentation is scrubbed.
    assert original.message == SENTINEL
    # JSON omits display_next_steps; plain output prints it.
    with invocation_redaction():
        register_invocation_secrets([SENTINEL])
        render(original, OutputContext(format=OutputFormat.plain, interactive=False))
    plain = capsys.readouterr().out
    assert SENTINEL not in plain
    assert plain.count("[REDACTED]") == 2


def test_list_extras_named_like_protocol_fields_are_redacted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = ListResult(
        title="Runs",
        items=[],
        total_count=0,
        has_more=False,
        next_offset=None,
        extra={"status": SENTINEL, "operation": SENTINEL},
    )
    with invocation_redaction():
        register_invocation_secrets([SENTINEL])
        render(result, OutputContext(format=OutputFormat.json, interactive=False))
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == payload["operation"] == "[REDACTED]"


@pytest.mark.parametrize(
    "value", [9876543210123456789, 987654321.125, True, False, None]
)
def test_redaction_covers_json_scalar_leaves(value) -> None:
    secret = json.dumps(value)
    details = {"received": [value, {"nested": value}], "safe": "context"}
    original = PlatformAPIError("Rejected input", details=details)

    redacted = redact_provided_secrets(original, [secret])

    assert secret not in json.dumps(redacted.details)
    assert redacted.details["safe"] == "context"
    assert original.details == details
    assert original.details["received"][0] is value


def test_redact_empty_secret_leaves_input_unchanged() -> None:
    assert redact_secret_value("ab", "") == "ab"
    assert redact_secret_text("ab", "") == "ab"


def test_redact_never_rebuilds_the_secret_from_the_marker() -> None:
    # Plain replacement turns "]]]" into "[REDACTED]]", which ends in the secret.
    assert redact_secret_text("]]]", "]]") == "[REDACTED]"
    assert redact_secret_value({"]]]": ["]]]"]}, "]]") == {"[REDACTED]": ["[REDACTED]"]}
    # A secret inside the marker itself survives any marker-based replacement.
    assert redact_secret_text("ARE", "RE") == ""
    assert redact_secret_value({"value": "x"}, "RE") == {"value": ""}
    # Redacting "y" one secret at a time turned "yx" into "[REDACTED]x".
    redacted = redact_provided_secrets(PlatformAPIError("yx"), ["]x", "y"])
    assert str(redacted) == "[REDACTED]"


def test_redact_secret_value_redacts_keys_containing_secret() -> None:
    data = {
        f"echo-{SENTINEL}": "ok",
        "value": "not-the-secret",
        "secret": "also-not-the-secret",
        "secret_value": "still-not-the-secret",
        "nested": {f"{SENTINEL}-k": SENTINEL},
    }

    redacted = redact_secret_value(data, SENTINEL)

    assert SENTINEL not in str(redacted)
    assert redacted["echo-[REDACTED]"] == "ok"
    assert redacted["value"] == "[REDACTED]"
    assert redacted["secret"] == "[REDACTED]"
    assert redacted["secret_value"] == "[REDACTED]"
    assert redacted["nested"] == {"[REDACTED]-k": "[REDACTED]"}


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
