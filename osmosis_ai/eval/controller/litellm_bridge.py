from __future__ import annotations

import json
import logging
import time
from typing import Any

from osmosis_ai.cli.errors import CLIError

logger = logging.getLogger(__name__)

SLIME_CONSUMED_PROVIDER_FIELDS = frozenset(
    {"messages", "temperature", "top_p", "max_tokens", "tools"}
)
SYSTEMIC_EXCEPTIONS = {
    "AuthenticationError",
    "BudgetExceededError",
    "NotFoundError",
    "UnsupportedParamsError",
    "APIConnectionError",
}


def _get_litellm() -> Any:
    try:
        import litellm
    except ImportError as exc:
        raise CLIError(
            "LiteLLM is required. Install with: pip install litellm"
        ) from exc
    litellm.suppress_debug_info = True
    if hasattr(litellm, "_async_client_cleanup_registered"):
        litellm._async_client_cleanup_registered = True
    return litellm


def _getattr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _message_content(choice: Any) -> tuple[str, Any]:
    message = _getattr_or_key(choice, "message")
    if message is None:
        delta = _getattr_or_key(choice, "delta", {})
        return str(_getattr_or_key(delta, "content", "") or ""), _getattr_or_key(
            delta, "tool_calls"
        )
    return str(_getattr_or_key(message, "content", "") or ""), _getattr_or_key(
        message, "tool_calls"
    )


def _usage_payload(response: Any) -> dict[str, int] | None:
    usage = _getattr_or_key(response, "usage")
    if usage is None:
        return None
    return {
        "prompt_tokens": int(_getattr_or_key(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(_getattr_or_key(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(_getattr_or_key(usage, "total_tokens", 0) or 0),
    }


def _plain_data(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {key: _plain_data(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain_data(item) for item in value]
    if hasattr(value, "model_dump"):
        return _plain_data(value.model_dump(exclude_none=True))
    if hasattr(value, "__dict__"):
        return {
            key: _plain_data(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value


def _stream_tool_calls_payload(tool_calls: Any) -> list[Any]:
    plain = _plain_data(tool_calls)
    if not isinstance(plain, list):
        plain = [plain]

    normalized: list[Any] = []
    for index, item in enumerate(plain):
        if isinstance(item, dict) and "index" not in item:
            normalized.append({"index": index, **item})
        else:
            normalized.append(item)
    return normalized


def model_response_to_payload(
    response: Any, *, request_model: str, stream: bool
) -> dict[str, Any]:
    choices: list[dict[str, Any]] = []
    for index, choice in enumerate(_getattr_or_key(response, "choices", []) or []):
        content, tool_calls = _message_content(choice)
        finish_reason = _getattr_or_key(choice, "finish_reason", "stop")
        choice_index = int(_getattr_or_key(choice, "index", index) or index)
        if stream:
            delta: dict[str, Any] = {"content": content}
            if tool_calls:
                delta["tool_calls"] = _stream_tool_calls_payload(tool_calls)
            choices.append(
                {
                    "index": choice_index,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            )
        else:
            message: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                message["tool_calls"] = _plain_data(tool_calls)
            choices.append(
                {
                    "index": choice_index,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            )

    payload: dict[str, Any] = {
        "id": _getattr_or_key(response, "id", "chatcmpl-eval"),
        "object": "chat.completion.chunk" if stream else "chat.completion",
        "created": int(
            _getattr_or_key(response, "created", time.time()) or time.time()
        ),
        "model": request_model,
        "choices": choices,
    }
    if stream:
        payload["stream"] = True
    usage = _usage_payload(response)
    if usage:
        payload["usage"] = usage
    return payload


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def build_sse_error_event(message: str) -> str:
    return f"event: error\ndata: {compact_json({'error': message})}\n\n"


class LiteLLMBridge:
    def __init__(
        self, *, model: str, api_key: str | None = None, base_url: str | None = None
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._tokens: dict[str, int] = {}
        self._systemic_errors: dict[str, str] = {}

    def build_kwargs(self, body: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": body.get("messages", []),
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url
        for key in SLIME_CONSUMED_PROVIDER_FIELDS - {"messages"}:
            if key in body:
                kwargs[key] = body[key]
        return kwargs

    async def preflight_check(self) -> None:
        litellm = _get_litellm()
        try:
            litellm.get_llm_provider(model=self.model, api_base=self.base_url)
        except Exception as exc:
            msg = getattr(exc, "message", str(exc))
            raise CLIError(
                f"Invalid LiteLLM model format. Use 'provider/model' "
                f"(e.g. openai/gpt-5-mini). Received: '{self.model}'. Details: {msg}"
            ) from exc
        try:
            await litellm.acompletion(
                **self.build_kwargs(
                    {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
                )
            )
        except Exception as exc:
            ename = type(exc).__name__
            if ename == "RateLimitError":
                return
            msg = getattr(exc, "message", str(exc))
            if ename in SYSTEMIC_EXCEPTIONS:
                raise CLIError(f"LLM preflight check failed: {msg}") from exc
            logger.debug("Preflight non-fatal error: %s", exc)

    async def complete(
        self, rollout_id: str, sample_id: str, body: dict[str, Any]
    ) -> Any:
        litellm = _get_litellm()
        try:
            response = await litellm.acompletion(**self.build_kwargs(body))
        except Exception as exc:
            if type(exc).__name__ in SYSTEMIC_EXCEPTIONS:
                self._systemic_errors[rollout_id] = getattr(exc, "message", str(exc))
            raise
        usage = _usage_payload(response)
        if usage:
            self._tokens[rollout_id] = (
                self._tokens.get(rollout_id, 0) + usage["total_tokens"]
            )
        return response

    def collect_tokens(self, rollout_id: str) -> int:
        return self._tokens.pop(rollout_id, 0)

    def collect_systemic_error(self, rollout_id: str) -> str | None:
        return self._systemic_errors.pop(rollout_id, None)


__all__ = [
    "LiteLLMBridge",
    "build_sse_error_event",
    "compact_json",
    "model_response_to_payload",
]
