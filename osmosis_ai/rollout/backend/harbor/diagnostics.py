"""Trial observability: phase timings, token totals, failure diagnostics,
and secret redaction. Everything reads Harbor's TrialResult duck-typed, so
tests can pass any object with the same fields.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from osmosis_ai.rollout.types import RolloutErrorCategory

logger: logging.Logger = logging.getLogger(__name__)

REDACTED = "[REDACTED]"
SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "credentials",
        "password",
        "secret",
        "token",
    }
)


def sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in SENSITIVE_KEYS or normalized.endswith(
        tuple(f"_{name}" for name in SENSITIVE_KEYS)
    )


def redact_secrets(value: Any, api_key: str | None = None) -> Any:
    """Replace credential-bearing dict leaves and api-key substrings."""
    if isinstance(value, dict):
        return {
            key: REDACTED if sensitive_key(str(key)) else redact_secrets(child, api_key)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_secrets(child, api_key) for child in value]
    if api_key and isinstance(value, str) and api_key in value:
        return REDACTED
    return value


def span_seconds(info: Any) -> float | None:
    if info is None or info.started_at is None or info.finished_at is None:
        return None
    return max(0.0, (info.finished_at - info.started_at).total_seconds())


def trial_timings(result: Any) -> dict[str, float]:
    """Per-phase durations from Harbor's own TimingInfo records."""
    if result is None:
        return {}
    spans = {
        "environment_setup": result.environment_setup,
        "agent_setup": result.agent_setup,
        "agent": result.agent_execution,
        "verifier": result.verifier,
        "total": result,
    }
    return {
        name: round(seconds, 2)
        for name, info in spans.items()
        if (seconds := span_seconds(info)) is not None
    }


def _as_utc(dt: datetime) -> datetime:
    """Normalize to UTC. Harbor <0.21 recorded ``ExceptionInfo.occurred_at``
    as a naive host-local datetime; 0.21+ records it timezone-aware."""
    return dt.astimezone(UTC)


def agent_phase_failure(result: Any) -> Any | None:
    """The recorded exception, when it struck before verification started.

    Harbor records agent failures and still runs the verifier, so both can
    coexist; the UTC-normalized timestamps separate them, since harbor's
    exception class names are open-ended.
    """
    err = getattr(result, "exception_info", None)
    if err is None:
        return None
    verifier_started = getattr(getattr(result, "verifier", None), "started_at", None)
    if verifier_started is None:
        return err
    return err if _as_utc(err.occurred_at) < _as_utc(verifier_started) else None


def failure_phase(result: Any) -> str:
    """The phase a failure belongs to.

    Normally the furthest phase the trial reached, but a pre-verification
    exception belongs to the agent-side phase that produced it.
    """
    if result is None:
        return "setup"
    spans = [
        ("verifier", result.verifier),
        ("agent", result.agent_execution),
        ("agent_setup", result.agent_setup),
        ("environment_setup", result.environment_setup),
    ]
    if agent_phase_failure(result) is not None:
        spans = spans[1:]
    for name, info in spans:
        if info is not None:
            return name
    return "setup"


def trial_metrics(result: Any) -> dict[str, Any]:
    """Token and cost totals Harbor accumulated across the trial."""
    if result is None:
        return {}
    try:
        input_tokens, cached_tokens, output_tokens, cost_usd = (
            result.compute_token_cost_totals()
        )
    except Exception:
        logger.warning("Failed to read token totals from trial result", exc_info=True)
        return {}
    values = {
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
    }
    return {key: value for key, value in values.items() if value is not None}


def diagnostic_payload(
    *,
    phase: str,
    category: RolloutErrorCategory | None,
    exception_type: str | None,
    timings: dict[str, float],
) -> dict[str, Any]:
    return {
        "backend": "harbor",
        "phase": phase,
        "harbor_exception_type": exception_type,
        "category": category.value if category else None,
        "timings_sec": timings,
    }
