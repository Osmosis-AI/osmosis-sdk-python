from __future__ import annotations

import json
import re
from typing import Any, Dict, Mapping, Tuple


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
    "dump_model",
    "extract_structured_score",
    "reward_json_schema",
    "reward_schema_definition",
    "sanitize_json",
]
