"""Shared exception-to-wire-category mapping used by execution backends."""

from __future__ import annotations

from osmosis_ai.rollout.types import RolloutErrorCategory

# These OpenSandbox types identify provider failures without an HTTP status.
# Generic API, internal, readiness, and pool-lifecycle errors can instead be
# task-specific. Harbor drops structured status codes from ExceptionInfo, so
# those ambiguous names must not trigger the controller's run-wide breaker.
_SANDBOX_PROVIDER_ERROR_TYPES = frozenset(
    {
        "SandboxConnectionException",
        "SandboxRateLimitException",
        "PoolStateStoreUnavailableException",
    }
)
_SANDBOX_API_ERROR_TYPES = frozenset({"SandboxApiException", "SandboxAPIError"})
_PROVIDER_HTTP_STATUSES = frozenset({401, 402, 403, 429, 500, 502, 503, 504})


def categorize_error_type(
    exception_type: str | None, *, status_code: int | None = None
) -> RolloutErrorCategory:
    """Classify provider failures only from explicit types or structured status.

    A name-only API error may be a task's missing file (404) or bad request
    (400). Never recover status codes by scraping its message or traceback.
    """
    if exception_type in _SANDBOX_PROVIDER_ERROR_TYPES or (
        exception_type in _SANDBOX_API_ERROR_TYPES
        and status_code in _PROVIDER_HTTP_STATUSES
    ):
        return RolloutErrorCategory.HTTP_ERROR
    return RolloutErrorCategory.AGENT_ERROR


def categorize_exception(exc: BaseException) -> RolloutErrorCategory:
    """Map backend exceptions onto the wire error vocabulary."""
    if isinstance(exc, TimeoutError):
        return RolloutErrorCategory.TIMEOUT
    if isinstance(exc, (ValueError, TypeError, AssertionError)):
        return RolloutErrorCategory.VALIDATION_ERROR
    status_code = getattr(exc, "status_code", None)
    if not isinstance(status_code, int):
        status_code = None
    # Inspect the hierarchy without importing optional provider SDKs. A live
    # subclass retains its provider identity, unlike Harbor's name-only record.
    for cls in type(exc).__mro__:
        category = categorize_error_type(cls.__name__, status_code=status_code)
        if category is RolloutErrorCategory.HTTP_ERROR:
            return category
    return RolloutErrorCategory.AGENT_ERROR
