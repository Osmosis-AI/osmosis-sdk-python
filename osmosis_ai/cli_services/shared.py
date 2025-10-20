from __future__ import annotations

from statistics import mean, pvariance, pstdev
from typing import Any, Optional

from .errors import CLIError


def coerce_optional_float(value: Any, field_name: str, source_label: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise CLIError(
        f"Expected '{field_name}' in {source_label} to be numeric, got {type(value).__name__}."
    )


def collapse_preview_text(value: Any, *, max_length: int = 140) -> Optional[str]:
    if not isinstance(value, str):
        return None
    collapsed = " ".join(value.strip().split())
    if not collapsed:
        return None
    if len(collapsed) > max_length:
        collapsed = collapsed[: max_length - 3].rstrip() + "..."
    return collapsed


def calculate_statistics(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {
            "average": 0.0,
            "variance": 0.0,
            "stdev": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    average = mean(scores)
    variance = pvariance(scores)
    std_dev = pstdev(scores)
    return {
        "average": average,
        "variance": variance,
        "stdev": std_dev,
        "min": min(scores),
        "max": max(scores),
    }


def calculate_stat_deltas(baseline: dict[str, float], current: dict[str, float]) -> dict[str, float]:
    delta: dict[str, float] = {}
    for key, current_value in current.items():
        if key not in baseline:
            continue
        try:
            baseline_value = float(baseline[key])
            current_numeric = float(current_value)
        except (TypeError, ValueError):
            continue
        delta[key] = current_numeric - baseline_value
    return delta
