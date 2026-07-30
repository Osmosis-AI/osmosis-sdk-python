"""Per-rollout protocol translation for installed native Harbor agents.

The gateway keeps routing state independent of FastAPI so importing the native
backend does not require the optional server dependencies.  The rollout server
installs the HTTP routes from :mod:`osmosis_ai.rollout.server.native_harbor_gateway`.
"""

from __future__ import annotations

import logging
import secrets
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

logger: logging.Logger = logging.getLogger(__name__)

_NO_CONTROLLER_KEY = "osmosis-no-controller-key"
_ROUTING_MODE = "header_token"
_ROUTING_ONLY_FIELDS = frozenset(
    {
        "api_base",
        "api_key",
        "base_url",
        "client",
        "custom_llm_provider",
        "extra_headers",
        "headers",
        "use_chat_completions_api",
    }
)


class NativeHarborGatewayError(Exception):
    """An HTTP-safe gateway error before or during protocol translation."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class NativeHarborGatewayRoute:
    """Controller endpoint and credential owned by one active rollout."""

    chat_completions_url: str
    controller_api_key: str | None


def _normalize_gateway_base_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("gateway_base_url must be a non-empty absolute HTTP(S) URL")
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("gateway_base_url must be a non-empty absolute HTTP(S) URL")
    if parsed.query or parsed.fragment:
        raise ValueError("gateway_base_url must not include a query or fragment")
    if parsed.path not in {"", "/"}:
        raise ValueError(
            "gateway_base_url must be an origin without a path; the gateway "
            "serves /v1/messages and /v1/responses"
        )
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _bridge_model(model: Any) -> str:
    if not isinstance(model, str) or not model.strip():
        raise NativeHarborGatewayError(
            "request field 'model' must be a non-empty string",
            status_code=400,
        )
    model = model.strip()
    if model.startswith("hosted_vllm/"):
        return model
    # LiteLLM's hosted_vllm provider is its OpenAI-compatible Chat Completions
    # path.  Unlike the openai provider, it cannot silently route an Anthropic
    # thinking request back to the Responses API.
    return f"hosted_vllm/{model}"


def _request_body(body: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(body)
    for field in _ROUTING_ONLY_FIELDS:
        data.pop(field, None)
    return data


class NativeHarborTranslationGateway:
    """Fixed-base gateway with opaque per-rollout header-token routing."""

    def __init__(self, base_url: str) -> None:
        self.base_url = _normalize_gateway_base_url(base_url)
        self._routes: dict[str, NativeHarborGatewayRoute] = {}
        self._lock = threading.RLock()

    @property
    def active_routes(self) -> int:
        with self._lock:
            return len(self._routes)

    @property
    def routing_mode(self) -> str:
        """Stable health-check name for opaque header-token routing."""
        return _ROUTING_MODE

    def register(
        self,
        *,
        chat_completions_url: str,
        controller_api_key: str | None,
    ) -> str:
        if not chat_completions_url:
            raise ValueError("chat_completions_url is required for gateway routing")
        route = NativeHarborGatewayRoute(
            chat_completions_url=chat_completions_url,
            controller_api_key=controller_api_key,
        )
        with self._lock:
            while True:
                token = secrets.token_urlsafe(32)
                if token not in self._routes:
                    self._routes[token] = route
                    return token

    def unregister(self, token: str) -> None:
        with self._lock:
            self._routes.pop(token, None)

    def resolve_headers(self, headers: Mapping[str, str]) -> NativeHarborGatewayRoute:
        authorization = (headers.get("authorization") or "").strip()
        bearer_token: str | None = None
        scheme, separator, credential = authorization.partition(" ")
        if separator and scheme.lower() == "bearer" and credential.strip():
            bearer_token = credential.strip()

        api_key_token = (headers.get("x-api-key") or "").strip() or None
        if (
            bearer_token is not None
            and api_key_token is not None
            and not secrets.compare_digest(bearer_token, api_key_token)
        ):
            raise NativeHarborGatewayError(
                "conflicting gateway credentials",
                status_code=401,
            )
        token = bearer_token or api_key_token
        if token is None:
            raise NativeHarborGatewayError(
                "missing gateway credential",
                status_code=401,
            )
        with self._lock:
            route = self._routes.get(token)
        if route is None:
            raise NativeHarborGatewayError(
                "invalid or expired gateway credential",
                status_code=401,
            )
        return route

    async def anthropic_messages(
        self,
        body: Mapping[str, Any],
        route: NativeHarborGatewayRoute,
    ) -> Any:
        """Translate an Anthropic Messages request through Chat Completions."""
        import litellm

        data = _request_body(body)
        model = _bridge_model(data.pop("model", None))
        if "messages" not in data:
            raise NativeHarborGatewayError(
                "request field 'messages' is required",
                status_code=400,
            )
        if "max_tokens" not in data:
            raise NativeHarborGatewayError(
                "request field 'max_tokens' is required",
                status_code=400,
            )

        # LiteLLM 1.91.1 otherwise leaks both Anthropic names unchanged into
        # the OpenAI Chat body.  `stop` is the direct Chat equivalent; top_k
        # has none and is deliberately dropped, matching LiteLLM drop-params
        # semantics rather than sending an invalid upstream request.
        stop_sequences = data.pop("stop_sequences", None)
        if stop_sequences is not None:
            data["stop"] = stop_sequences
        if data.pop("top_k", None) is not None:
            logger.warning(
                "Dropping Anthropic top_k because Chat Completions has no equivalent"
            )

        return await litellm.anthropic_messages(
            model=model,
            custom_llm_provider="hosted_vllm",
            api_base=route.chat_completions_url,
            api_key=route.controller_api_key or _NO_CONTROLLER_KEY,
            **data,
        )

    async def openai_responses(
        self,
        body: Mapping[str, Any],
        route: NativeHarborGatewayRoute,
    ) -> Any:
        """Translate an OpenAI Responses request through Chat Completions."""
        import litellm

        data = _request_body(body)
        model = _bridge_model(data.pop("model", None))
        if "input" not in data:
            raise NativeHarborGatewayError(
                "request field 'input' is required",
                status_code=400,
            )
        return await litellm.aresponses(
            model=model,
            custom_llm_provider="hosted_vllm",
            api_base=route.chat_completions_url,
            api_key=route.controller_api_key or _NO_CONTROLLER_KEY,
            use_chat_completions_api=True,
            **data,
        )


__all__ = [
    "NativeHarborGatewayError",
    "NativeHarborGatewayRoute",
    "NativeHarborTranslationGateway",
]
