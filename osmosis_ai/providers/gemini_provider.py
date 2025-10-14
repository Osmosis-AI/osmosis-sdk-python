from __future__ import annotations

from contextlib import contextmanager
import time
import warnings
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from google import genai as genai_module  # type: ignore
    from google.genai import types as genai_types_module  # type: ignore

from ..rubric_types import ProviderRequestError, RewardRubricRunResult
from .base import DEFAULT_REQUEST_TIMEOUT_SECONDS, ProviderRequest, RubricProvider
from .shared import dump_model, reward_schema_definition, sanitize_json


_GENAI_MODULE: Any | None = None
_GENAI_TYPES_MODULE: Any | None = None
_PYDANTIC_ANY_WARNING_MESSAGE = r".*<built-in function any> is not a Python type.*"

GEMINI_DEFAULT_TIMEOUT_SECONDS = 60.0
GEMINI_MIN_TIMEOUT_SECONDS = 5.0
GEMINI_MAX_TIMEOUT_SECONDS = 180.0
GEMINI_RETRY_ATTEMPTS = 3
GEMINI_TIMEOUT_BACKOFF = 1.5
GEMINI_RETRY_SLEEP_SECONDS = (0.5, 1.0, 2.0)


@contextmanager
def _suppress_pydantic_any_warning() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=_PYDANTIC_ANY_WARNING_MESSAGE,
            category=UserWarning,
            module=r"pydantic\._internal\._generate_schema",
        )
        yield


def _load_google_genai() -> Tuple[Any, Any]:
    """
    Lazily import the Google Generative AI SDK so that environments without the optional
    dependency avoid import-time side effects (like pydantic warnings) unless the Gemini
    provider is actually used.
    """
    global _GENAI_MODULE, _GENAI_TYPES_MODULE
    if _GENAI_MODULE is not None and _GENAI_TYPES_MODULE is not None:
        return _GENAI_MODULE, _GENAI_TYPES_MODULE

    try:  # pragma: no cover - optional dependency
        with _suppress_pydantic_any_warning():
            from google import genai as genai_mod  # type: ignore
            from google.genai import types as genai_types_mod  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Google Generative AI SDK is required for provider 'gemini'. "
            "Install it via `pip install google-genai`."
        ) from exc

    _GENAI_MODULE = genai_mod
    _GENAI_TYPES_MODULE = genai_types_mod
    return _GENAI_MODULE, _GENAI_TYPES_MODULE


def _normalize_gemini_model(model_id: str) -> str:
    import re

    return re.sub(r"^models/", "", model_id, flags=re.IGNORECASE)


def _json_schema_to_genai(
    schema: Dict[str, Any],
    genai_types: Any,
) -> "genai_types_module.Schema":  # type: ignore[name-defined]

    type_map = {
        "object": genai_types.Type.OBJECT,
        "string": genai_types.Type.STRING,
        "number": genai_types.Type.NUMBER,
        "integer": genai_types.Type.INTEGER,
        "boolean": genai_types.Type.BOOLEAN,
        "array": genai_types.Type.ARRAY,
    }

    kwargs: Dict[str, Any] = {}
    type_value = schema.get("type")
    if isinstance(type_value, str):
        mapped = type_map.get(type_value.lower())
        if mapped is not None:
            kwargs["type"] = mapped

    required = schema.get("required")
    if isinstance(required, list):
        filtered_required = [name for name in required if isinstance(name, str)]
        if filtered_required:
            kwargs["required"] = filtered_required

    properties = schema.get("properties")
    if isinstance(properties, dict):
        converted_properties = {}
        for key, value in properties.items():
            if isinstance(key, str) and isinstance(value, dict):
                converted_properties[key] = _json_schema_to_genai(value, genai_types)
        if converted_properties:
            kwargs["properties"] = converted_properties

    items = schema.get("items")
    if isinstance(items, dict):
        kwargs["items"] = _json_schema_to_genai(items, genai_types)

    enum_values = schema.get("enum")
    if isinstance(enum_values, list):
        filtered_enum = [str(option) for option in enum_values]
        if filtered_enum:
            kwargs["enum"] = filtered_enum

    description = schema.get("description")
    if isinstance(description, str):
        kwargs["description"] = description

    minimum = schema.get("minimum")
    if isinstance(minimum, (int, float)):
        kwargs["minimum"] = float(minimum)

    maximum = schema.get("maximum")
    if isinstance(maximum, (int, float)):
        kwargs["maximum"] = float(maximum)

    min_items = schema.get("min_items")
    if isinstance(min_items, int):
        kwargs["min_items"] = min_items

    max_items = schema.get("max_items")
    if isinstance(max_items, int):
        kwargs["max_items"] = max_items

    min_length = schema.get("min_length")
    if isinstance(min_length, int):
        kwargs["min_length"] = min_length

    max_length = schema.get("max_length")
    if isinstance(max_length, int):
        kwargs["max_length"] = max_length

    nullable = schema.get("nullable")
    if isinstance(nullable, bool):
        kwargs["nullable"] = nullable

    with _suppress_pydantic_any_warning():
        return genai_types.Schema(**kwargs)


def _build_retry_timeouts(requested_timeout: float) -> List[float]:
    # Keep the first attempt generous, then increase for retries while capping growth.
    base = max(requested_timeout, GEMINI_MIN_TIMEOUT_SECONDS, GEMINI_DEFAULT_TIMEOUT_SECONDS)
    timeouts: List[float] = []
    current = base
    for _ in range(GEMINI_RETRY_ATTEMPTS):
        timeouts.append(min(current, GEMINI_MAX_TIMEOUT_SECONDS))
        current = min(current * GEMINI_TIMEOUT_BACKOFF, GEMINI_MAX_TIMEOUT_SECONDS)
    return timeouts


def _seconds_to_millis(seconds: float) -> int:
    # Gemini client expects timeout in milliseconds. Clamp to at least 1ms.
    return max(int(round(seconds * 1000)), 1)


class GeminiProvider(RubricProvider):
    name = "gemini"

    def default_timeout(self, model: str) -> float:
        return max(DEFAULT_REQUEST_TIMEOUT_SECONDS, GEMINI_DEFAULT_TIMEOUT_SECONDS)

    def run(self, request: ProviderRequest) -> RewardRubricRunResult:
        try:
            genai, genai_types = _load_google_genai()
        except RuntimeError as exc:
            detail = str(exc).strip() or "Google Generative AI SDK is required."
            raise ProviderRequestError(self.name, request.model, detail) from exc

        try:
            requested_timeout = float(request.timeout)
        except (TypeError, ValueError):
            requested_timeout = float(DEFAULT_REQUEST_TIMEOUT_SECONDS)

        retry_timeouts = _build_retry_timeouts(requested_timeout)
        max_timeout = max(retry_timeouts)

        with _suppress_pydantic_any_warning():
            client = genai.Client(
                api_key=request.api_key,
                http_options={"timeout": _seconds_to_millis(max_timeout)},
            )
        schema_definition = reward_schema_definition()
        gemini_schema = _json_schema_to_genai(schema_definition, genai_types)
        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=gemini_schema,
            temperature=0,
        )

        combined_prompt = f"{request.system_content}\n\n{request.user_content}"

        response: Any | None = None
        last_error: Exception | None = None

        for attempt_index, attempt_timeout in enumerate(retry_timeouts, start=1):
            try:
                with _suppress_pydantic_any_warning():
                    try:
                        response = client.models.generate_content(
                            model=_normalize_gemini_model(request.model),
                            contents=combined_prompt,
                            config=config,
                            request_options={"timeout": _seconds_to_millis(attempt_timeout)},
                        )
                    except TypeError as err:
                        # Older SDKs may not accept request_options; retry without it.
                        if "request_options" not in str(err):
                            raise
                        response = client.models.generate_content(
                            model=_normalize_gemini_model(request.model),
                            contents=combined_prompt,
                            config=config,
                        )
                break
            except Exception as err:  # pragma: no cover - network failures depend on runtime
                last_error = err
                if attempt_index >= len(retry_timeouts):
                    detail = str(err).strip() or "Gemini request failed."
                    raise ProviderRequestError(self.name, request.model, detail) from err
                sleep_idx = min(attempt_index - 1, len(GEMINI_RETRY_SLEEP_SECONDS) - 1)
                time.sleep(GEMINI_RETRY_SLEEP_SECONDS[sleep_idx])

        if response is None and last_error is not None:
            detail = str(last_error).strip() or "Gemini request failed."
            raise ProviderRequestError(self.name, request.model, detail) from last_error

        raw = dump_model(response)

        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            candidates = raw.get("candidates") if isinstance(raw, dict) else None
            if isinstance(candidates, list) and candidates:
                first = candidates[0]
                if isinstance(first, dict):
                    content = first.get("content")
                    if isinstance(content, dict):
                        parts = content.get("parts")
                        if isinstance(parts, list):
                            for part in parts:
                                if isinstance(part, dict):
                                    candidate_text = part.get("text")
                                    if isinstance(candidate_text, str) and candidate_text.strip():
                                        text = candidate_text
                                        break
        if not isinstance(text, str) or not text.strip():
            raise ProviderRequestError(self.name, request.model, "Model response did not include any text content.")
        try:
            score, explanation = sanitize_json(text)
        except ValueError as err:
            raise ProviderRequestError(self.name, request.model, str(err)) from err
        bounded = max(request.score_min, min(request.score_max, score))
        return {"score": bounded, "explanation": explanation, "raw": raw}


__all__ = ["GeminiProvider"]
