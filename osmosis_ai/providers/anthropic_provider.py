from __future__ import annotations

from typing import Any, Dict

try:  # pragma: no cover - optional dependency
    import anthropic  # type: ignore
    from anthropic import APIError  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    anthropic = None  # type: ignore[assignment]
    APIError = None  # type: ignore[assignment]

from ..rubric_types import ModelNotFoundError, ProviderRequestError, RewardRubricRunResult
from .base import DEFAULT_REQUEST_TIMEOUT_SECONDS, ProviderRequest, RubricProvider
from .shared import dump_model, extract_structured_score, reward_schema_definition


class AnthropicProvider(RubricProvider):
    name = "anthropic"

    def default_timeout(self, model: str) -> float:
        return DEFAULT_REQUEST_TIMEOUT_SECONDS

    def run(self, request: ProviderRequest) -> RewardRubricRunResult:
        if anthropic is None or APIError is None:
            raise ProviderRequestError(
                self.name,
                request.model,
                "Anthropic SDK is required. Install it via `pip install anthropic`.",
            )

        client = anthropic.Anthropic(api_key=request.api_key)
        tool_name = "emit_reward_rubric_response"
        schema_definition = reward_schema_definition()
        tool = {
            "name": tool_name,
            "description": "Return the reward rubric score and explanation as structured JSON.",
            "input_schema": schema_definition,
        }

        try:
            response = client.messages.create(
                model=request.model,
                system=request.system_content,
                messages=[{"role": "user", "content": [{"type": "text", "text": request.user_content}]}],
                tools=[tool],
                tool_choice={"type": "tool", "name": tool_name},
                max_tokens=512,
                temperature=0,
                timeout=request.timeout,
            )
        except APIError as err:
            detail = getattr(err, "message", None)
            if not isinstance(detail, str) or not detail.strip():
                detail = str(err)
            status_code = getattr(err, "status_code", None)
            if status_code == 404:
                not_found_detail = (
                    f"Model '{request.model}' was not found. Confirm your Anthropic account has access "
                    "to the requested snapshot or update the model identifier."
                )
                raise ModelNotFoundError(self.name, request.model, not_found_detail) from err
            raise ProviderRequestError(self.name, request.model, detail) from err
        except Exception as err:
            detail = str(err).strip() or "Unexpected error during Anthropic request."
            raise ProviderRequestError(self.name, request.model, detail) from err

        raw = dump_model(response)

        payload: Dict[str, Any] | None = None
        content_blocks = raw.get("content") if isinstance(raw, dict) else None
        if isinstance(content_blocks, list):
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == tool_name:
                    maybe_input = block.get("input")
                    if isinstance(maybe_input, dict):
                        payload = maybe_input
                    break
        if payload is None:
            raise ProviderRequestError(self.name, request.model, "Model response missing expected tool output.")
        score, explanation = extract_structured_score(payload)
        bounded = max(request.score_min, min(request.score_max, score))
        return {"score": bounded, "explanation": explanation, "raw": raw}


__all__ = ["AnthropicProvider"]
