"""Shared exception-to-wire-category mapping used by execution backends."""

from __future__ import annotations

from osmosis_ai.rollout.types import RolloutErrorCategory


def categorize_exception(exc: BaseException) -> RolloutErrorCategory:
    """Map backend exceptions onto the wire error vocabulary."""
    if isinstance(exc, TimeoutError):
        return RolloutErrorCategory.TIMEOUT
    if isinstance(exc, (ValueError, TypeError, AssertionError)):
        return RolloutErrorCategory.VALIDATION_ERROR
    return RolloutErrorCategory.AGENT_ERROR
