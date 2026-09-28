"""Contracts shared by the cloud and local inference gateways."""

from copy import deepcopy
from typing import Any

import pytest

from osmosis_ai.eval.request_policy import (
    BridgeRequestError,
    build_chat_kwargs,
    inference_parameters,
)


def test_extra_body_merges_like_the_openai_client_without_mutating_inputs() -> None:
    body = {
        "model": "client-alias",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning": {"effort": "high", "summary": "auto"},
        "extra_body": {"reasoning": {"effort": "low"}, "provider": {"sort": "latency"}},
    }
    original = deepcopy(body)
    parameters = inference_parameters(body)
    assert parameters == {
        "reasoning": {"effort": "low", "summary": "auto"},
        "provider": {"sort": "latency"},
    }
    parameters["reasoning"]["effort"] = "medium"
    assert body == original


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "field",
    [
        "api_key",
        "base_url",
        "callbacks",
        "ssl_verify",
        "mock_response",
        "aws_bedrock_runtime_endpoint",
        "vertex_credentials",
        "azure_ad_token",
        "allowed_openai_params",
        "drop_params",
        "_hidden_params",
        "token",
        "sagemaker_base_url",
        "wx_credentials",
        "oci_key_file",
        "zen_api_key",
        "model_id",
        "account_id",
        "predibase_tenant_id",
        "project_id",
        "space_id",
        "region",
        "workspace_id",
        "anthropic-workspace-id",
        "anthropic_workspace_id",
        "s3_endpoint_url",
        "endpoint_url",
        "default_headers",
        "api_type",
        "use_psc_endpoint_format",
        "method",
        "user_config",
        "user_agent",
        "databricks_user_agent",
        "braintrust_api_key",
        "braintrust_host",
        "braintrust_project",
        "lunary_public_key",
        "posthog_host",
        "slack_webhook_url",
    ],
)
def test_runtime_controls_are_rejected_instead_of_becoming_provider_kwargs(
    field: str,
    nested: bool,
) -> None:
    fields = {field: "private-value-must-not-appear-in-errors"}
    with pytest.raises(BridgeRequestError, match=field) as error:
        inference_parameters({"extra_body": fields} if nested else fields)
    assert "private-value" not in str(error.value)


@pytest.mark.parametrize("extra", [[], "invalid", 3])
def test_extra_body_requires_an_object(extra: Any) -> None:
    with pytest.raises(BridgeRequestError, match="JSON object"):
        inference_parameters({"extra_body": extra})


def test_more_than_one_completion_is_explicitly_rejected() -> None:
    assert inference_parameters({"n": 1}) == {}
    with pytest.raises(BridgeRequestError, match="n=1"):
        inference_parameters({"n": 3})


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-sonnet-4-5",
        "azure_ai/claude-sonnet-4-5",
        "vertex_ai/claude-sonnet-4-5",
    ],
)
def test_native_provider_keeps_litellm_translation_and_custom_kwargs(
    model: str,
) -> None:
    kwargs = build_chat_kwargs(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "max_completion_tokens": 64,
            "reasoning_effort": "low",
            "context_management": {"edits": []},
            "custom_parameter": {"enabled": True},
            "metadata": {"user_id": "test-user"},
        },
        model=model,
    )
    assert kwargs["max_completion_tokens"] == 64
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["context_management"] == {"edits": []}
    assert kwargs["custom_parameter"] == {"enabled": True}
    assert kwargs["metadata"] == {"user_id": "test-user"}
    assert kwargs["drop_params"] is False
    assert "extra_body" not in kwargs


@pytest.mark.parametrize(
    "model", ["openrouter/openai/gpt-5-mini", "openrouter/custom/unlisted"]
)
def test_model_catalog_does_not_change_openrouter_forwarding(model: str) -> None:
    parameters = {
        "reasoning": {"effort": "low"},
        "temperature": 0.7,
        "provider": {"sort": "latency"},
    }
    kwargs = build_chat_kwargs(parameters, model=model)
    assert kwargs["extra_body"] == parameters
    assert kwargs["drop_params"] is False


def test_official_openai_preserves_translation_and_sends_metadata_as_body_data() -> (
    None
):
    kwargs = build_chat_kwargs(
        {"max_tokens": 64, "metadata": {"experiment": "test"}},
        model="openai/gpt-5-mini",
        api_base="https://api.openai.com/v1",
    )
    assert kwargs["max_tokens"] == 64
    assert kwargs["extra_body"] == {"metadata": {"experiment": "test"}}
    assert "metadata" not in kwargs
    assert kwargs["drop_params"] is False


def test_native_metadata_cannot_be_silently_reduced_by_litellm() -> None:
    with pytest.raises(BridgeRequestError, match="metadata"):
        build_chat_kwargs(
            {"metadata": {"user_id": "a-user", "custom_tag": "keep-me"}},
            model="anthropic/claude-sonnet-4-5",
        )


def test_bridge_envelope_remains_configured_and_request_messages_are_detached() -> None:
    body = {
        "model": "osmosis-rollout",
        "stream": True,
        "messages": [{"role": "user", "content": "original"}],
        "extra_body": {"model": "other", "messages": [], "stream": True, "top_k": 3},
    }
    kwargs = build_chat_kwargs(
        body, model="openai/configured", api_base="https://example.com/v1"
    )
    assert kwargs["model"] == "openai/configured"
    assert kwargs["messages"] == body["messages"]
    assert kwargs["extra_body"] == {"top_k": 3}
    kwargs["messages"][0]["content"] = "changed"
    assert body["messages"][0]["content"] == "original"
