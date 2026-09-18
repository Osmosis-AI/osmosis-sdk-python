"""Provider outages must stay distinct from task-level sandbox API failures."""

import pytest

from osmosis_ai.rollout.types import RolloutErrorCategory
from osmosis_ai.rollout.utils.errors import categorize_error_type, categorize_exception


@pytest.mark.parametrize(
    "name",
    [
        "SandboxConnectionException",
        "SandboxRateLimitException",
        "PoolStateStoreUnavailableException",
    ],
)
def test_explicit_provider_failures_survive_name_only_records(name):
    assert categorize_error_type(name) is RolloutErrorCategory.HTTP_ERROR
    exception_type = type(name, (Exception,), {})
    assert categorize_exception(exception_type()) is RolloutErrorCategory.HTTP_ERROR


@pytest.mark.parametrize(
    "name",
    [
        None,
        "SandboxApiException",
        "SandboxAPIError",
        "SandboxException",
        "SandboxInternalException",
        "SandboxTimeoutException",
        "SandboxReadyTimeoutException",
        "SandboxUnhealthyException",
        "PoolAcquireFailedException",
        "PoolEmptyException",
        "PoolNotRunningException",
        "PoolDestroyedException",
        "PoolDestroyIncompleteException",
        "PoolStateStoreContentionException",
        "InvalidArgumentException",
        "NonZeroAgentExitCodeError",
    ],
)
def test_ambiguous_recorded_failures_do_not_indicate_provider_outages(name):
    assert categorize_error_type(name) is RolloutErrorCategory.AGENT_ERROR


@pytest.mark.parametrize("name", ["SandboxApiException", "SandboxAPIError"])
@pytest.mark.parametrize(
    "status, expected",
    [
        (400, RolloutErrorCategory.AGENT_ERROR),
        (404, RolloutErrorCategory.AGENT_ERROR),
        (409, RolloutErrorCategory.AGENT_ERROR),
        (422, RolloutErrorCategory.AGENT_ERROR),
        (501, RolloutErrorCategory.AGENT_ERROR),
        (None, RolloutErrorCategory.AGENT_ERROR),
        ("503", RolloutErrorCategory.AGENT_ERROR),
        ([503], RolloutErrorCategory.AGENT_ERROR),
        (401, RolloutErrorCategory.HTTP_ERROR),
        (402, RolloutErrorCategory.HTTP_ERROR),
        (403, RolloutErrorCategory.HTTP_ERROR),
        (429, RolloutErrorCategory.HTTP_ERROR),
        (500, RolloutErrorCategory.HTTP_ERROR),
        (502, RolloutErrorCategory.HTTP_ERROR),
        (503, RolloutErrorCategory.HTTP_ERROR),
        (504, RolloutErrorCategory.HTTP_ERROR),
    ],
)
def test_live_api_error_uses_structured_status_not_message(name, status, expected):
    exception_type = type(name, (Exception,), {"status_code": status})
    # Status-like text is deliberately misleading; only the attribute counts.
    assert categorize_exception(exception_type("HTTP 503")) is expected


def test_unrelated_http_exception_is_not_a_sandbox_provider_failure():
    exception_type = type("AgentToolError", (Exception,), {"status_code": 503})
    assert categorize_exception(exception_type()) is RolloutErrorCategory.AGENT_ERROR


@pytest.mark.parametrize("name", ["SandboxApiException", "SandboxConnectionException"])
def test_live_provider_subclass_keeps_provider_identity(name):
    base = type(name, (Exception,), {"status_code": 503})
    subclass = type("CustomProviderError", (base,), {})
    assert categorize_exception(subclass()) is RolloutErrorCategory.HTTP_ERROR


@pytest.mark.parametrize(
    "base, expected",
    [
        (TimeoutError, RolloutErrorCategory.TIMEOUT),
        (ValueError, RolloutErrorCategory.VALIDATION_ERROR),
        (TypeError, RolloutErrorCategory.VALIDATION_ERROR),
        (AssertionError, RolloutErrorCategory.VALIDATION_ERROR),
    ],
)
def test_existing_timeout_and_validation_classification_takes_precedence(
    base, expected
):
    exception_type = type("SandboxApiException", (base,), {"status_code": 503})
    assert categorize_exception(exception_type()) is expected
