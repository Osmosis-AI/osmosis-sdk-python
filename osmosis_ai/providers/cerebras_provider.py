from __future__ import annotations

from typing import Any, Dict

try:  # pragma: no cover - optional dependency
    from cerebras.cloud.sdk import (  # type: ignore
        APIStatusError,
        BadRequestError,
        Cerebras,
    )
except ImportError:  # pragma: no cover - optional dependency
    Cerebras = None  # type: ignore[assignment]
    BadRequestError = None  # type: ignore[assignment]
    APIStatusError = None  # type: ignore[assignment]

from ..rubric_types import ProviderRequestError, RewardRubricRunResult
from .base import ProviderRequest, RubricProvider
from .shared import dump_model, reward_json_schema, sanitize_json


class CerebrasProvider(RubricProvider):
    """
    Cerebras Cloud SDK provider for fast inference.

    Cerebras provides high-performance AI inference using their Wafer-Scale Engine.

    Documentation: https://inference-docs.cerebras.ai/
    SDK: https://github.com/Cerebras/cerebras-cloud-sdk-python
    """
    name = "cerebras"

    def default_timeout(self, model: str) -> float:
        # Cerebras is optimized for fast inference, use their default timeout
        return 60.0

    def run(self, request: ProviderRequest) -> RewardRubricRunResult:
        # Guard: SDK available
        if Cerebras is None or BadRequestError is None:
            raise ProviderRequestError(
                self.name,
                request.model,
                "Cerebras Cloud SDK is required. Install it via `pip install cerebras_cloud_sdk`.",
            )

        try:
            # Initialize Cerebras client
            client = Cerebras(api_key=request.api_key)

            # Get JSON schema for structured output
            schema_payload = reward_json_schema()

            # Build messages
            messages = [
                {"role": "system", "content": request.system_content},
                {"role": "user", "content": request.user_content},
            ]

            # Build request parameters
            params: Dict[str, Any] = {
                "model": request.model,
                "messages": messages,
                "temperature": 0,  # For deterministic scoring
            }

            # Try with JSON schema first
            try:
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": schema_payload,
                }

                completion = client.chat.completions.create(
                    **params,
                    timeout=request.timeout,
                )

            except (BadRequestError, TypeError) as schema_err:
                # BadRequestError: Server-side rejection of unsupported parameters
                # TypeError: Client-side SDK rejection of unsupported parameters
                # Both indicate json_schema mode is not supported, fallback to json_object
                error_msg = str(schema_err).lower()
                if "response_format" in error_msg or "json_schema" in error_msg:
                    params["response_format"] = {"type": "json_object"}

                    completion = client.chat.completions.create(
                        **params,
                        timeout=request.timeout,
                    )
                else:
                    raise

            # Extract response
            raw = dump_model(completion)

            # Get content from the completion
            if not isinstance(raw, dict):
                raise ProviderRequestError(
                    self.name,
                    request.model,
                    "Unexpected response format from Cerebras API",
                )

            choices = raw.get("choices", [])
            if not choices or not isinstance(choices, list):
                raise ProviderRequestError(
                    self.name,
                    request.model,
                    "No choices returned in Cerebras response",
                )

            first_choice = choices[0]
            if not isinstance(first_choice, dict):
                raise ProviderRequestError(
                    self.name,
                    request.model,
                    "Invalid choice format in Cerebras response",
                )

            message = first_choice.get("message", {})
            content = message.get("content", "")

            if not content or not isinstance(content, str):
                raise ProviderRequestError(
                    self.name,
                    request.model,
                    "No content returned from Cerebras model",
                )

            # Parse JSON response
            try:
                score, explanation = sanitize_json(content)
            except ValueError as err:
                raise ProviderRequestError(
                    self.name,
                    request.model,
                    f"Failed to parse Cerebras response: {err}",
                ) from err

            # Bound the score to the requested range
            bounded_score = max(request.score_min, min(request.score_max, score))

            return {
                "score": bounded_score,
                "explanation": explanation,
                "raw": raw,
            }

        except ProviderRequestError:
            # Re-raise our own errors
            raise

        except Exception as err:
            # Catch any other errors from the Cerebras SDK
            error_msg = str(err)

            # Try to extract more details from SDK exceptions
            if hasattr(err, "message"):
                error_msg = getattr(err, "message")
            elif hasattr(err, "body"):
                body = getattr(err, "body")
                if isinstance(body, dict):
                    error_field = body.get("error")
                    if isinstance(error_field, dict):
                        error_msg = error_field.get("message") or error_msg
                    elif isinstance(error_field, str):
                        error_msg = error_field

            raise ProviderRequestError(
                self.name,
                request.model,
                f"Cerebras API request failed: {error_msg}",
            ) from err


__all__ = [
    "CerebrasProvider",
]
