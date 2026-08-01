"""Trial observability: phase timings, token totals, failure diagnostics,
and secret redaction. Everything reads Harbor's TrialResult duck-typed, so
tests can pass any object with the same fields.
"""

from __future__ import annotations

import logging
from typing import Any

from osmosis_ai.rollout.types import RolloutErrorCategory

logger: logging.Logger = logging.getLogger(__name__)

REDACTED = "[REDACTED]"
SENSITIVE_KEYS = frozenset(
    {"api_key", "apikey", "authorization", "credential", "credentials", "password", "secret", "token"}
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


def failure_phase(result: Any) -> str:
    """The furthest phase the trial reached; a failure happened there."""
    if result is None:
        return "setup"
    for name, info in (
        ("verifier", result.verifier),
        ("agent", result.agent_execution),
        ("agent_setup", result.agent_setup),
        ("environment_setup", result.environment_setup),
    ):
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
        "backend": "harbor-v2",
        "phase": phase,
        "harbor_exception_type": exception_type,
        "category": category.value if category else None,
        "timings_sec": timings,
    }
