from __future__ import annotations

import copy
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from tqdm import tqdm

from ..rubric_eval import evaluate_rubric
from ..rubric_types import MissingAPIKeyError, ModelNotFoundError, ProviderRequestError
from .config import RubricConfig
from .dataset import DatasetRecord
from .errors import CLIError
from .shared import calculate_statistics, coerce_optional_float, collapse_preview_text


class RubricEvaluator:
    """Thin wrapper over evaluate_rubric to enable injection during tests."""

    def __init__(self, evaluate_fn: Any = evaluate_rubric):
        self._evaluate_fn = evaluate_fn

    def run(self, config: RubricConfig, record: DatasetRecord) -> dict[str, Any]:
        messages = record.message_payloads()
        if not messages:
            label = record.conversation_id or record.rubric_id or "<record>"
            raise CLIError(f"Record '{label}' must include a non-empty 'messages' list.")

        score_min = coerce_optional_float(
            record.score_min if record.score_min is not None else config.score_min,
            "score_min",
            f"record '{record.conversation_id or '<record>'}'",
        )
        score_max = coerce_optional_float(
            record.score_max if record.score_max is not None else config.score_max,
            "score_max",
            f"record '{record.conversation_id or '<record>'}'",
        )

        try:
            return self._evaluate_fn(
                rubric=config.rubric_text,
                messages=messages,
                model_info=copy.deepcopy(config.model_info),
                ground_truth=record.ground_truth if record.ground_truth is not None else config.ground_truth,
                system_message=record.system_message if record.system_message is not None else config.system_message,
                original_input=record.original_input if record.original_input is not None else config.original_input,
                extra_info=record.merged_extra_info(config.extra_info),
                score_min=score_min,
                score_max=score_max,
                return_details=True,
            )
        except (MissingAPIKeyError, ProviderRequestError, ModelNotFoundError) as exc:
            raise CLIError(str(exc)) from exc


@dataclass
class EvaluationRun:
    run_index: int
    status: str
    score: Optional[float]
    explanation: Optional[str]
    preview: Optional[str]
    duration_seconds: float
    started_at: datetime
    completed_at: datetime
    error: Optional[str]
    raw: Any


@dataclass
class EvaluationRecordResult:
    record_index: int
    record: DatasetRecord
    conversation_label: str
    runs: list[EvaluationRun]
    statistics: dict[str, float]


@dataclass
class EvaluationReport:
    rubric_config: RubricConfig
    config_path: Path
    data_path: Path
    number: int
    record_results: list[EvaluationRecordResult]
    overall_statistics: dict[str, float]


class RubricEvaluationEngine:
    """Executes rubric evaluations across a dataset and aggregates statistics."""

    def __init__(self, evaluator: Optional[RubricEvaluator] = None):
        self._evaluator = evaluator or RubricEvaluator()

    def execute(
        self,
        *,
        rubric_config: RubricConfig,
        config_path: Path,
        data_path: Path,
        records: Sequence[DatasetRecord],
        number: int,
    ) -> EvaluationReport:
        record_results: list[EvaluationRecordResult] = []
        aggregate_scores: list[float] = []
        total_runs = 0
        total_successes = 0

        progress_total = len(records) * number
        show_progress = progress_total > 1 and getattr(sys.stderr, "isatty", lambda: False)()
        progress = (
            tqdm(
                total=progress_total,
                file=sys.stderr,
                dynamic_ncols=True,
                leave=False,
            )
            if show_progress
            else None
        )

        try:
            for record_index, record in enumerate(records, start=1):
                conversation_label = record.conversation_label(record_index)
                fallback_preview = record.assistant_preview()

                runs: list[EvaluationRun] = []
                scores: list[float] = []

                for attempt in range(1, number + 1):
                    started_at = datetime.now(timezone.utc)
                    timer_start = time.perf_counter()
                    status = "success"
                    error_message: Optional[str] = None
                    score_value: Optional[float] = None
                    explanation_value: Optional[str] = None
                    preview_value: Optional[str] = None
                    raw_payload: Any = None

                    try:
                        result = self._evaluator.run(rubric_config, record)
                    except CLIError as exc:
                        status = "error"
                        error_message = str(exc)
                        result = None
                    except Exception as exc:  # pragma: no cover - unexpected path
                        status = "error"
                        error_message = f"{type(exc).__name__}: {exc}"
                        result = None

                    duration_seconds = time.perf_counter() - timer_start
                    completed_at = datetime.now(timezone.utc)

                    if status == "success" and isinstance(result, dict):
                        raw_payload = result.get("raw")
                        score_value = _extract_float(result.get("score"))
                        explanation_value = _normalize_optional_text(result.get("explanation"))
                        preview_value = self._resolve_preview_text(result, fallback_preview)
                        if score_value is not None:
                            scores.append(score_value)
                            aggregate_scores.append(score_value)
                            total_successes += 1
                    else:
                        preview_value = fallback_preview

                    total_runs += 1

                    runs.append(
                        EvaluationRun(
                            run_index=attempt,
                            status=status,
                            score=score_value,
                            explanation=explanation_value,
                            preview=preview_value,
                            duration_seconds=duration_seconds,
                            started_at=started_at,
                            completed_at=completed_at,
                            error=error_message,
                            raw=raw_payload,
                        )
                    )

                    if progress:
                        progress.update()

                statistics = calculate_statistics(scores)
                statistics["total_runs"] = len(runs)
                statistics["success_count"] = len(scores)
                statistics["failure_count"] = len(runs) - len(scores)
                record_results.append(
                    EvaluationRecordResult(
                        record_index=record_index,
                        record=record,
                        conversation_label=conversation_label,
                        runs=runs,
                        statistics=statistics,
                    )
                )
        finally:
            if progress:
                progress.close()

        overall_statistics = calculate_statistics(aggregate_scores)
        overall_statistics["total_runs"] = total_runs
        overall_statistics["success_count"] = total_successes
        overall_statistics["failure_count"] = total_runs - total_successes
        return EvaluationReport(
            rubric_config=rubric_config,
            config_path=config_path,
            data_path=data_path,
            number=number,
            record_results=record_results,
            overall_statistics=overall_statistics,
        )

    @staticmethod
    def _resolve_preview_text(result: Optional[dict[str, Any]], fallback: Optional[str]) -> Optional[str]:
        if not isinstance(result, dict):
            return fallback
        preview = collapse_preview_text(result.get("preview"))
        if preview:
            return preview

        raw_payload = result.get("raw")
        if isinstance(raw_payload, dict):
            for key in ("preview", "summary", "text"):
                preview = collapse_preview_text(raw_payload.get(key))
                if preview:
                    return preview
        return fallback


def _extract_float(value: Any) -> Optional[float]:
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None
    except (TypeError, ValueError):
        return None


def _normalize_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
