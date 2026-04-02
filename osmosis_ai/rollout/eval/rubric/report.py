from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev, pvariance
from typing import Any


def calculate_statistics(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {"average": 0.0, "variance": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
    return {
        "average": mean(scores),
        "variance": pvariance(scores),
        "stdev": pstdev(scores),
        "min": min(scores),
        "max": max(scores),
    }


@dataclass
class RecordResult:
    record_index: int
    label: str
    scores: list[float]
    explanations: list[str]
    errors: list[str]
    statistics: dict[str, float]


@dataclass
class RubricReport:
    model: str
    rubric_text: str
    data_path: Path
    number: int
    results: list[RecordResult]
    overall_statistics: dict[str, float]


class ConsoleReportRenderer:
    """Renders rubric evaluation report to stdout."""

    def __init__(self, printer: Callable[[str], None] = print):
        self._printer = printer

    def render(self, report: RubricReport) -> None:
        self._printer(f"Model: {report.model}")
        self._printer(
            f"Evaluated {len(report.results)} record(s) from {report.data_path}"
        )
        self._printer(f"Runs per record: {report.number}")
        self._printer("")

        for result in report.results:
            self._printer(f"[{result.label}]")
            for i, score in enumerate(result.scores):
                self._printer(f"  Run {i + 1:02d}: score={score:.4f}")
                if i < len(result.explanations):
                    self._printer(f"    explanation: {result.explanations[i]}")
            for error in result.errors:
                self._printer(f"  ERROR: {error}")
            stats = result.statistics
            if len(result.scores) > 1:
                self._printer(
                    f"  Summary: avg={stats.get('average', 0):.4f} "
                    f"stdev={stats.get('stdev', 0):.4f} "
                    f"min={stats.get('min', 0):.4f} max={stats.get('max', 0):.4f}"
                )
            self._printer("")

        overall = report.overall_statistics
        self._printer("Overall Statistics:")
        self._printer(f"  average:  {overall.get('average', 0):.4f}")
        self._printer(f"  stdev:    {overall.get('stdev', 0):.4f}")
        self._printer(
            f"  min/max:  {overall.get('min', 0):.4f} / {overall.get('max', 0):.4f}"
        )


class JsonReportWriter:
    """Writes rubric evaluation report to JSON."""

    def write(self, report: RubricReport, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": report.model,
            "rubric": report.rubric_text,
            "data_path": str(report.data_path),
            "number": report.number,
            "overall_statistics": report.overall_statistics,
            "records": [
                {
                    "index": r.record_index,
                    "label": r.label,
                    "scores": r.scores,
                    "explanations": r.explanations,
                    "errors": r.errors,
                    "statistics": r.statistics,
                }
                for r in report.results
            ],
        }
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        return output_path
