"""Request parameter handling shared by local and cloud evaluation bridges."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from functools import cache
from typing import Any

_ENVELOPE_FIELDS = frozenset({"model", "messages", "stream", "stream_options", "n"})
# Native provider and callback controls not covered by LiteLLM's shared schemas.
# Recheck these consumers when updating LiteLLM.
_CONTROL_FIELDS = frozenset(
    {
        "base_url",
        "extra_headers",
        "deployment_id",
        "drop_params",
        "additional_drop_params",
        "callbacks",
        "success_callback",
        "failure_callback",
        "ssl_verify",
        "mock_delay",
        "mock_tool_calls",
        "initial_prompt_value",
        "original_function",
        "engine",
        "config",
        "user_api_key_end_user_id",
        "wx_credentials",
        "apikey",
        "url",
        "custom_endpoint",
        "deployment_url",
        "zen_api_key",
        "endpoint_url",
        "default_headers",
        "api_type",
        "model_id",
        "account_id",
        "predibase_tenant_id",
        "project_id",
        "space_id",
        "region",
        "workspace_id",
        "anthropic-workspace-id",
        "anthropic_workspace_id",
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
        # These replace the configured model, accounting, or translated prompt.
        "models",
        "preset",
        "usage",
        "contents",
        "systemInstruction",
        "system_instruction",
        "system",
        "prompt",
        "input",
        "inputText",
    }
)
_CONTROL_PREFIXES = (
    "_",
    "litellm_",
    "aws_",
    "azure_",
    "vertex_",
    "watsonx_",
    "sagemaker_",
    "oci_",
    "wx_",
    "s3_",
)


class BridgeRequestError(ValueError):
    """A request attempts to change bridge configuration or has an invalid shape."""


@cache
def _control_fields() -> frozenset[str]:
    import litellm
    from litellm.types.router import GenericLiteLLMParams
    from litellm.types.utils import all_litellm_params

    # Provider metadata is request data even though LiteLLM also uses this name.
    return (
        _CONTROL_FIELDS
        | frozenset(all_litellm_params)
        | frozenset(GenericLiteLLMParams.model_fields)
        | frozenset(litellm.common_cloud_provider_auth_params["params"])
    ) - {"metadata"}


def _merge_body(base: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in extra.items():
        previous = base.get(key)
        if isinstance(previous, dict) and isinstance(value, dict):
            base[key] = _merge_body(previous, value)
        else:
            base[key] = deepcopy(value)
    return base


def inference_parameters(body: Mapping[str, Any]) -> dict[str, Any]:
    """Return detached inference fields, applying OpenAI extra-body precedence.

    Client model aliases and response streaming belong to the bridge envelope.
    Other runtime controls are rejected rather than executed or silently dropped.
    """
    extra = body.get("extra_body")
    if extra is not None and not isinstance(extra, dict):
        raise BridgeRequestError("extra_body must be a JSON object")
    fields = _merge_body(
        deepcopy({key: value for key, value in body.items() if key != "extra_body"}),
        extra or {},
    )
    if fields.get("n") not in (None, 1):
        raise BridgeRequestError("The eval bridge supports only n=1")
    for key in _ENVELOPE_FIELDS:
        fields.pop(key, None)
    blocked = sorted(
        key
        for key in fields
        if key in _control_fields() or key.startswith(_CONTROL_PREFIXES)
    )
    if blocked:
        raise BridgeRequestError(
            "Request fields cannot override eval bridge configuration: "
            + ", ".join(blocked)
        )
    if "extra_body" in fields:
        raise BridgeRequestError("extra_body cannot contain another extra_body")
    return fields


def is_official_openai_endpoint(api_base: str | None = None) -> bool:
    """Resolve explicit, global, and environment bases using LiteLLM's precedence."""
    import litellm

    effective_base = litellm.OpenAIGPTConfig.get_api_base(api_base)
    return (effective_base or "").rstrip("/") == "https://api.openai.com/v1"


def build_chat_kwargs(
    body: Mapping[str, Any],
    *,
    model: str,
    api_key: str | None = None,
    api_base: str | None = None,
) -> dict[str, Any]:
    """Build a LiteLLM request without opting into silent parameter removal.

    OpenAI-compatible endpoints receive inference fields as JSON, regardless of
    LiteLLM's model catalog. Native providers retain LiteLLM's protocol conversion.
    """
    import litellm
    from litellm.constants import openai_compatible_providers

    fields = inference_parameters(body)
    resolved_model, provider, _, _ = litellm.get_llm_provider(
        model=model, api_base=api_base
    )
    native_claude = provider in {"azure_ai", "vertex_ai"} and "claude" in resolved_model
    openai_wire = not native_claude and (
        provider == "openrouter"
        or provider in openai_compatible_providers
        or (provider == "openai" and not is_official_openai_endpoint(api_base))
    )
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": deepcopy(body.get("messages", [])),
        "drop_params": False,
    }
    if openai_wire:
        if fields:
            kwargs["extra_body"] = fields
    else:
        metadata = fields.pop("metadata", None)
        if metadata is not None:
            if provider in {"openai", "azure"}:
                kwargs["extra_body"] = {"metadata": metadata}
            elif provider == "anthropic" or native_claude:
                if not isinstance(metadata, dict) or set(metadata) - {"user_id"}:
                    raise BridgeRequestError("Anthropic metadata supports only user_id")
                fields["metadata"] = metadata
            else:
                raise BridgeRequestError(
                    f"LiteLLM cannot forward metadata to {provider}; use its provider-specific parameter"
                )
        kwargs.update(fields)
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["base_url"] = api_base
    return kwargs
