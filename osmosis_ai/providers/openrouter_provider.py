from __future__ import annotations

try:  # pragma: no cover - optional dependency
    from openai import OpenAI, OpenAIError  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]
    OpenAIError = None  # type: ignore[assignment]

from ..rubric_types import ProviderRequestError, RewardRubricRunResult
from .base import ProviderRequest, RubricProvider
from .openai_family import _call_openai_family


class OpenRouterProvider(RubricProvider):
    """
    OpenRouter provider using OpenAI SDK with custom base URL.

    OpenRouter provides access to hundreds of AI models through a unified API,
    using OpenAI-compatible endpoints.

    Documentation: https://openrouter.ai/docs
    """
    name = "openrouter"

    def default_timeout(self, model: str) -> float:
        # OpenRouter routes to different providers, so use a reasonable timeout
        return 60.0

    def run(self, request: ProviderRequest) -> RewardRubricRunResult:
        # Guard: SDK available
        if OpenAI is None or OpenAIError is None:
            raise ProviderRequestError(
                self.name,
                request.model,
                "OpenAI SDK is required for OpenRouter. Install it via `pip install 'openai>=2.0.0'`.",
            )

        # OpenRouter uses the OpenAI-compatible API with a custom base URL
        return _call_openai_family(
            provider=self.name,
            model=request.model,
            api_key=request.api_key,
            system_content=request.system_content,
            user_content=request.user_content,
            score_min=request.score_min,
            score_max=request.score_max,
            timeout=request.timeout,
            base_url="https://openrouter.ai/api/v1",
        )


__all__ = [
    "OpenRouterProvider",
]
