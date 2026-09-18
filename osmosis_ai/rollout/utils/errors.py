"""Shared exception-to-wire-category mapping used by execution backends."""

from __future__ import annotations

from osmosis_ai.rollout.types import RolloutErrorCategory

# Upstream sandbox/environment provider errors, matched by class name:
# harbor's ExceptionInfo preserves only the class name (the numeric HTTP
# status is discarded), and the provider SDKs are optional imports here.
# Mirrors the OpenSandbox SDK's exceptions/sandbox.py (InvalidArgumentException
# and other input errors are intentionally excluded) plus the SkyPilot
# Sandbox SDK's SandboxAPIError.
SANDBOX_ERROR_TYPES = frozenset(
    {
        # OpenSandbox SDK
        "SandboxException",
        "SandboxApiException",
        "SandboxInternalException",
        "SandboxConnectionException",
        "SandboxRateLimitException",
        "SandboxTimeoutException",
        "SandboxReadyTimeoutException",
        "SandboxUnhealthyException",
        "PoolAcquireFailedException",
        "PoolEmptyException",
        "PoolNotRunningException",
        "PoolDestroyedException",
        "PoolDestroyIncompleteException",
        "PoolStateStoreUnavailableException",
        "PoolStateStoreContentionException",
        # SkyPilot Sandbox SDK (early access) / retired Daytona SDK
        "SandboxAPIError",
    }
)


def categorize_error_type(exception_type: str | None) -> RolloutErrorCategory:
    """Classify a recorded exception by its class name alone."""
    if exception_type in SANDBOX_ERROR_TYPES:
        return RolloutErrorCategory.HTTP_ERROR
    return RolloutErrorCategory.AGENT_ERROR


def categorize_exception(exc: BaseException) -> RolloutErrorCategory:
    """Map backend exceptions onto the wire error vocabulary."""
    if isinstance(exc, TimeoutError):
        return RolloutErrorCategory.TIMEOUT
    if isinstance(exc, (ValueError, TypeError, AssertionError)):
        return RolloutErrorCategory.VALIDATION_ERROR
    return categorize_error_type(type(exc).__name__)
