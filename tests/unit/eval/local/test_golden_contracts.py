"""Golden contract fixtures, mirrored with the platform (design §16.3).

These files are the SDK half of a mirrored pair: the same JSON is meant to be
consumed by the monolith so ``index.jsonl`` validity and the metrics formula can
never drift silently between producer and consumer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osmosis_ai.eval.local.results import aggregate_metrics, lint_index_row

GOLDEN_DIR = Path(__file__).resolve().parents[3] / "golden" / "eval_local"
METRICS_DIR = GOLDEN_DIR / "metrics"
INDEX_CASES = GOLDEN_DIR / "index" / "contract_cases.json"


def _metrics_fixtures() -> list[Path]:
    fixtures = sorted(METRICS_DIR.glob("*.json"))
    assert fixtures, f"no metrics fixtures found under {METRICS_DIR}"
    return fixtures


def _assert_matches(actual: Any, expected: Any, *, where: str) -> None:
    """Compare nested summaries; ``pytest.approx`` refuses nested mappings."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), where
        assert sorted(actual) == sorted(expected), where
        for key in expected:
            _assert_matches(actual[key], expected[key], where=f"{where}.{key}")
        return
    if isinstance(expected, list):
        assert isinstance(actual, list), where
        assert len(actual) == len(expected), where
        for index, item in enumerate(expected):
            _assert_matches(actual[index], item, where=f"{where}[{index}]")
        return
    if isinstance(expected, float):
        assert actual == pytest.approx(expected), where
        return
    assert actual == expected, where


@pytest.mark.parametrize("fixture", _metrics_fixtures(), ids=lambda path: path.stem)
def test_metrics_match_the_golden_aggregate(fixture: Path) -> None:
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    summary = aggregate_metrics(
        payload["index"], pass_threshold=payload["pass_threshold"]
    )
    _assert_matches(summary, payload["expected"], where=fixture.stem)


def _index_cases() -> list[dict[str, Any]]:
    payload = json.loads(INDEX_CASES.read_text(encoding="utf-8"))
    cases = payload["cases"]
    assert cases, f"no index contract cases found in {INDEX_CASES}"
    return cases


@pytest.mark.parametrize("case", _index_cases(), ids=lambda case: case["name"])
def test_index_contract_lint_matches_the_golden_case(case: dict[str, Any]) -> None:
    problems = lint_index_row(case["row"])
    expected = case["problems"]
    assert len(problems) == len(expected), (
        f"{case['name']}: expected {expected}, got {problems}"
    )
    for fragment, problem in zip(expected, problems, strict=True):
        assert fragment in problem


def test_every_valid_golden_row_serializes_without_nulls() -> None:
    for case in _index_cases():
        if case["problems"]:
            continue
        line = json.dumps(case["row"], allow_nan=False)
        assert "null" not in line, case["name"]
