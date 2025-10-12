from __future__ import annotations

from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    from openai import BadRequestError, OpenAI, OpenAIError  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]
    BadRequestError = None  # type: ignore[assignment]
    OpenAIError = None  # type: ignore[assignment]

from ..rubric_types import RewardRubricRunResult
from .base import DEFAULT_REQUEST_TIMEOUT_SECONDS, ProviderRequest, RubricProvider
from .shared import (
    debug_payload,
    dump_model,
    reward_json_schema,
    reward_schema_definition,
    sanitize_json,
)


def _should_use_openai_responses(model_id: str) -> bool:
    normalized = model_id.strip().lower()
    return any(normalized.startswith(prefix) for prefix in ("gpt-4.1", "gpt-4o", "gpt-5", "o3", "o4"))


def _is_openai_gpt5_family(model_id: str) -> bool:
    return model_id.strip().lower().startswith("gpt-5")


def _extract_openai_responses_text(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None

    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    if isinstance(output_text, list):
        for item in output_text:
            if isinstance(item, str) and item.strip():
                return item

    response_obj = payload.get("response")
    if isinstance(response_obj, dict):
        text = response_obj.get("output_text")
        if isinstance(text, str) and text.strip():
            return text

    output_list = payload.get("output")
    if isinstance(output_list, list):
        for entry in output_list:
            if not isinstance(entry, dict):
                continue
            for part in entry.get("content", []):
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        return text

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content
            content_list = first_choice.get("content")
            if isinstance(content_list, list):
                for part in content_list:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text")
                        if isinstance(text, str) and text.strip():
                            return text

    return None


def _openai_error_message(err: Exception) -> str:
    message = getattr(err, "message", None)
    if isinstance(message, str) and message.strip():
        return message.strip()
    body = getattr(err, "body", None)
    if isinstance(body, dict):
        error_field = body.get("error")
        if isinstance(error_field, dict):
            detail = error_field.get("message") or error_field.get("code")
            if isinstance(detail, str) and detail.strip():
                return detail.strip()
        elif isinstance(error_field, str) and error_field.strip():
            return error_field.strip()
    return str(err)


def _call_openai_family(
    provider: str,
    model: str,
    api_key: str,
    system_content: str,
    user_content: str,
    score_min: float,
    score_max: float,
    timeout: float,
    req_id: str,
    *,
    base_url: Optional[str] = None,
    force_responses_api: bool = False,
) -> RewardRubricRunResult:
    if OpenAI is None or BadRequestError is None or OpenAIError is None:
        raise RuntimeError(
            "OpenAI SDK is required for provider "
            f"'{provider}'. Install it via `pip install openai>=1.0.0`."
        )
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    schema_definition = reward_schema_definition()
    schema_payload = reward_json_schema()
    schema_format = {"type": "json_schema", "json_schema": schema_payload}

    temperature_kwargs: Dict[str, Any] = {}
    if not _is_openai_gpt5_family(model):
        temperature_kwargs["temperature"] = 0

    input_messages = [
        {"role": "system", "content": [{"type": "input_text", "text": system_content}]},
        {"role": "user", "content": [{"type": "input_text", "text": user_content}]},
    ]
    chat_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    request_preview = {
        "model": model,
        "mode": "responses" if force_responses_api or _should_use_openai_responses(model) else "chat.completions",
        "system_chars": len(system_content),
        "user_chars": len(user_content),
        "base_url": base_url or "https://api.openai.com/v1",
    }
    debug_payload(req_id, provider, "request", request_preview, [api_key])

    def _finalise(raw_response: Any, text: Optional[str]) -> RewardRubricRunResult:
        if not text:
            text = _extract_openai_responses_text(raw_response)
        if not text:
            raise RuntimeError("Model response did not include any content.")
        score, explanation = sanitize_json(text)
        bounded = max(score_min, min(score_max, score))
        debug_payload(req_id, provider, "response", raw_response, [api_key])
        return {"score": bounded, "explanation": explanation, "raw": raw_response}

    use_responses_api = force_responses_api or _should_use_openai_responses(model)
    response_format = schema_format
    responses_supports_schema = True
    chat_supports_schema = True

    if use_responses_api:
        try:
            response = client.responses.create(
                model=model,
                input=input_messages,
                max_output_tokens=512,
                timeout=timeout,
                **temperature_kwargs,
                **({"response_format": response_format} if responses_supports_schema else {}),
            )
            raw = dump_model(response)
            text = getattr(response, "output_text", None)
            return _finalise(raw, text)
        except TypeError as err:
            message = str(err)
            if "response_format" in message:
                responses_supports_schema = False
                try:
                    import openai as _openai_module  # type: ignore
                    sdk_version = getattr(_openai_module, "__version__", None)
                except Exception:  # pragma: no cover - best effort metadata
                    sdk_version = None
                capability_note = {
                    "message": "OpenAI SDK does not support response_format; falling back to instruction-only JSON.",
                    "sdk_version": sdk_version,
                }
                debug_payload(req_id, provider, "capability", capability_note, [api_key])
                response = client.responses.create(
                    model=model,
                    input=input_messages,
                    max_output_tokens=512,
                    timeout=timeout,
                    **temperature_kwargs,
                )
                raw = dump_model(response)
                text = getattr(response, "output_text", None)
                return _finalise(raw, text)
            raise
        except BadRequestError as err:
            message = _openai_error_message(err)
            if "response_format" in message.lower() and "json_schema" in message.lower():
                if responses_supports_schema:
                    fallback_format = {"type": "json_object"}
                    try:
                        response = client.responses.create(
                            model=model,
                            input=input_messages,
                            response_format=fallback_format,
                            max_output_tokens=512,
                            timeout=timeout,
                            **temperature_kwargs,
                        )
                        raw = dump_model(response)
                        text = getattr(response, "output_text", None)
                        return _finalise(raw, text)
                    except OpenAIError as fallback_err:
                        fallback_message = _openai_error_message(fallback_err)
                        raise RuntimeError(f"Model request failed. {fallback_message}") from fallback_err
            raise RuntimeError(f"Model request failed. {message}") from err
        except OpenAIError as err:
            message = _openai_error_message(err)
            raise RuntimeError(f"Model request failed. {message}") from err

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=chat_messages,
            timeout=timeout,
            **temperature_kwargs,
            **({"response_format": response_format} if chat_supports_schema else {}),
        )
        raw = dump_model(completion)
        text = _extract_openai_responses_text(raw)
        return _finalise(raw, text)
    except TypeError as err:
        message = str(err)
        if "response_format" in message:
            chat_supports_schema = False
            try:
                import openai as _openai_module  # type: ignore
                sdk_version = getattr(_openai_module, "__version__", None)
            except Exception:  # pragma: no cover - best effort metadata
                sdk_version = None
            capability_note = {
                "message": "OpenAI SDK chat.completions does not support response_format; falling back to instruction-only JSON.",
                "sdk_version": sdk_version,
            }
            debug_payload(req_id, provider, "capability", capability_note, [api_key])
            completion = client.chat.completions.create(
                model=model,
                messages=chat_messages,
                timeout=timeout,
                **temperature_kwargs,
            )
            raw = dump_model(completion)
            text = _extract_openai_responses_text(raw)
            return _finalise(raw, text)
        raise
    except BadRequestError as err:
        message = _openai_error_message(err)
        if "response_format" in message.lower() and "json_schema" in message.lower():
            if chat_supports_schema:
                try:
                    fallback_completion = client.chat.completions.create(
                        model=model,
                        messages=chat_messages,
                        response_format={"type": "json_object"},
                        timeout=timeout,
                        **temperature_kwargs,
                    )
                    raw = dump_model(fallback_completion)
                    text = _extract_openai_responses_text(raw)
                    return _finalise(raw, text)
                except OpenAIError as fallback_err:
                    fallback_message = _openai_error_message(fallback_err)
                    raise RuntimeError(f"Model request failed. {fallback_message}") from fallback_err
        raise RuntimeError(f"Model request failed. {message}") from err
    except OpenAIError as err:
        message = _openai_error_message(err)
        raise RuntimeError(f"Model request failed. {message}") from err


class OpenAIProvider(RubricProvider):
    name = "openai"

    def default_timeout(self, model: str) -> float:
        return DEFAULT_REQUEST_TIMEOUT_SECONDS

    def run(self, request: ProviderRequest) -> RewardRubricRunResult:
        return _call_openai_family(
            provider=self.name,
            model=request.model,
            api_key=request.api_key,
            system_content=request.system_content,
            user_content=request.user_content,
            score_min=request.score_min,
            score_max=request.score_max,
            timeout=request.timeout,
            req_id=request.req_id,
        )


class XAIProvider(OpenAIProvider):
    name = "xai"

    def default_timeout(self, model: str) -> float:
        normalized = model.strip().lower()
        if normalized.startswith("grok-4"):
            return 60.0
        return 45.0

    def run(self, request: ProviderRequest) -> RewardRubricRunResult:
        return _call_openai_family(
            provider=self.name,
            model=request.model,
            api_key=request.api_key,
            system_content=request.system_content,
            user_content=request.user_content,
            score_min=request.score_min,
            score_max=request.score_max,
            timeout=request.timeout,
            req_id=request.req_id,
            base_url="https://api.x.ai/v1",
            force_responses_api=True,
        )


__all__ = [
    "OpenAIProvider",
    "XAIProvider",
]
