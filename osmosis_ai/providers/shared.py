from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, Iterable, Mapping, Tuple

DEBUG = os.getenv("REWARD_RUBRIC_DEBUG", "").lower() in {"1", "true"}
LOG_BODIES = os.getenv("REWARD_RUBRIC_LOG_BODIES", "").lower() in {"1", "true"}


def mask_token(text: str) -> str:
    return re.sub(r"(Bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1***", text, flags=re.IGNORECASE)


def redact_secrets(text: str, extra_secrets: Iterable[str]) -> str:
    masked = mask_token(text)
    for secret in extra_secrets:
        if not secret:
            continue
        masked = re.sub(re.escape(secret), "***", masked)
    return masked


def safe_stringify(value: Any) -> str:
    try:
        return json.dumps(value)
    except TypeError:
        seen: set[int] = set()

        def _convert(obj: Any) -> Any:
            obj_id = id(obj)
            if obj_id in seen:
                return "[Circular]"
            if isinstance(obj, dict):
                seen.add(obj_id)
                out = {k: _convert(v) for k, v in obj.items()}
                seen.remove(obj_id)
                return out
            if isinstance(obj, list):
                seen.add(obj_id)
                out_list = [_convert(v) for v in obj]
                seen.remove(obj_id)
                return out_list
            return obj

        return json.dumps(_convert(value), indent=2)


def preview(text: str, limit: int = 50000 if LOG_BODIES else 2000) -> str:
    return text if len(text) <= limit else f"{text[:limit]}...[+{len(text) - limit} more]"


def debug_log(*args: Any) -> None:
    if DEBUG:
        print(*args, file=sys.stderr)


def debug_payload(req_id: str, provider: str, stage: str, payload: Any, extra_mask: Iterable[str]) -> None:
    if not DEBUG:
        return
    serialised = safe_stringify(payload)
    redacted = redact_secrets(serialised, extra_mask)
    debug_log(f"[{req_id}] {provider} {stage}: {preview(redacted)}")


def dump_model(obj: Any) -> Any:
    for attr in ("model_dump", "dict", "to_dict"):
        method = getattr(obj, attr, None)
        if callable(method):
            return method()
    json_attr = getattr(obj, "model_dump_json", None)
    if callable(json_attr):
        try:
            return json.loads(json_attr())
        except (TypeError, ValueError):
            pass
    return obj


def reward_schema_definition() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "score": {"type": "number"},
            "explanation": {"type": "string"},
        },
        "required": ["score", "explanation"],
        "additionalProperties": False,
    }


def reward_json_schema() -> Dict[str, Any]:
    return {
        "name": "reward_rubric_response",
        "strict": True,
        "schema": reward_schema_definition(),
    }


def extract_structured_score(payload: Mapping[str, Any]) -> Tuple[float, str]:
    score_raw = payload.get("score")
    explanation_raw = payload.get("explanation")
    if not isinstance(score_raw, (int, float)):
        raise ValueError("Model response did not include a numeric score.")
    score = float(score_raw)
    if not float("-inf") < score < float("inf"):
        raise ValueError("Model response did not include a numeric score.")
    if not isinstance(explanation_raw, str) or not explanation_raw.strip():
        raise ValueError("Model response did not include an explanation string.")
    return score, explanation_raw.strip()


def sanitize_json(raw: str) -> Tuple[float, str]:
    trimmed = raw.strip()
    without_fence = re.sub(r"^```(?:json)?\s*", "", trimmed, flags=re.IGNORECASE)
    without_fence = re.sub(r"```$", "", without_fence, flags=re.IGNORECASE).strip()

    try:
        parsed = json.loads(without_fence)
    except json.JSONDecodeError as err:
        raise ValueError(
            "Model response was not valid JSON. Please refine the rubric instructions and try again."
        ) from err

    if not isinstance(parsed, dict):
        raise ValueError("Model response did not contain the expected JSON object.")

    score_raw = parsed.get("score")
    explanation_raw = parsed.get("explanation")

    if not isinstance(score_raw, (int, float)):
        raise ValueError("Model response must include a numeric 'score'.")

    score = float(score_raw)
    if not float("-inf") < score < float("inf"):
        raise ValueError("Model response must include a finite numeric 'score'.")

    if not isinstance(explanation_raw, str) or not explanation_raw.strip():
        raise ValueError("Model response must include a non-empty 'explanation' string.")

    return score, explanation_raw.strip()


__all__ = [
    "DEBUG",
    "LOG_BODIES",
    "debug_log",
    "debug_payload",
    "dump_model",
    "extract_structured_score",
    "mask_token",
    "preview",
    "redact_secrets",
    "reward_json_schema",
    "reward_schema_definition",
    "safe_stringify",
    "sanitize_json",
]
