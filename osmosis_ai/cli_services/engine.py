from __future__ import annotations

import copy
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import inspect
from typing import Any, Optional, Sequence

from tqdm import tqdm

from ..rubric_eval import DEFAULT_API_KEY_ENV, evaluate_rubric
from ..rubric_types import MissingAPIKeyError, ModelNotFoundError, ProviderRequestError
from .config import RubricConfig
from .dataset import DatasetRecord
from .errors import CLIError
from .shared import calculate_statistics, coerce_optional_float, collapse_preview_text


def _normalize_config_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _compose_extra_info_context(
    base: Optional[dict[str, Any]],
    *,
    rubric_text: str,
    provider: Optional[str],
    model: Optional[str],
    system_prompt: Optional[str],
    original_input: Optional[str],
    api_key: Optional[str],
    api_key_env: Optional[str],
    score_min: Optional[float],
    score_max: Optional[float],
    model_info: dict[str, Any],
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    """
    Build the runtime context passed to rubric functions along with a sanitised
    copy safe for prompt injection.
    """
    base_payload = copy.deepcopy(base) if isinstance(base, dict) else {}
    decorated_payload = copy.deepcopy(base_payload)

    if provider:
        decorated_payload["provider"] = provider
    if model:
        decorated_payload["model"] = model
    if api_key:
        decorated_payload["api_key"] = api_key
        decorated_payload.pop("api_key_env", None)
    elif api_key_env:
        decorated_payload["api_key_env"] = api_key_env
        decorated_payload.pop("api_key", None)
    if rubric_text:
        decorated_payload["rubric"] = rubric_text
    if score_min is not None:
        decorated_payload["score_min"] = float(score_min)
    if score_max is not None:
        decorated_payload["score_max"] = float(score_max)
    if system_prompt:
        decorated_payload["system_prompt"] = system_prompt
    if original_input and isinstance(original_input, str):
        decorated_payload["original_input"] = original_input

    model_info_copy = copy.deepcopy(model_info)
    if isinstance(model_info_copy, dict):
        if api_key and "api_key" not in model_info_copy:
            model_info_copy["api_key"] = api_key
        if api_key_env and "api_key_env" not in model_info_copy:
            model_info_copy["api_key_env"] = api_key_env
    decorated_payload["model_info"] = model_info_copy

    prompt_payload: Optional[dict[str, Any]] = None
    base_metadata = decorated_payload.get("metadata")
    if isinstance(base_metadata, dict):
        prompt_payload = copy.deepcopy(base_metadata)
    elif base_metadata is not None:
        prompt_payload = dict(base_metadata)

    dataset_metadata = decorated_payload.get("dataset_metadata")
    if isinstance(dataset_metadata, dict):
        if prompt_payload is None:
            prompt_payload = {}
        prompt_payload.setdefault("dataset_metadata", copy.deepcopy(dataset_metadata))

    if prompt_payload is not None:
        decorated_payload["metadata"] = copy.deepcopy(prompt_payload)
    else:
        decorated_payload.pop("metadata", None)

    return decorated_payload, prompt_payload


def _merge_system_prompts(
    prepend_prompt: Optional[str],
    base_prompt: Optional[str],
) -> Optional[str]:
    prompts: list[str] = []
    if prepend_prompt:
        prompts.append(prepend_prompt)
    if base_prompt:
        prompts.append(base_prompt)
    if not prompts:
        return None
    return "\n\n".join(prompts)


class RubricEvaluator:
    """Thin wrapper over evaluate_rubric to enable injection during tests."""

    def __init__(self, evaluate_fn: Any = evaluate_rubric):
        self._evaluate_fn = evaluate_fn

    def run(self, config: RubricConfig, record: DatasetRecord) -> dict[str, Any]:
        solution = record.solution_str
        if not isinstance(solution, str) or not solution.strip():
            label = record.conversation_id or record.rubric_id or "<record>"
            raise CLIError(f"Record '{label}' must include a non-empty 'solution_str' string.")

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

        ground_truth = record.ground_truth if record.ground_truth is not None else config.ground_truth
        original_input = record.original_input if record.original_input is not None else config.original_input

        provider_value = _normalize_config_str(config.model_info.get("provider"))
        model_value = _normalize_config_str(config.model_info.get("model"))
        system_prompt_value = _normalize_config_str(config.system_prompt)
        api_key_value = _normalize_config_str(config.model_info.get("api_key"))
        api_key_env_value = _normalize_config_str(config.model_info.get("api_key_env"))
        if api_key_env_value is None and provider_value:
            default_env = DEFAULT_API_KEY_ENV.get(provider_value.lower())
            if default_env:
                api_key_env_value = default_env

        try:
            model_info_payload = copy.deepcopy(config.model_info)
            base_system_prompt = _normalize_config_str(model_info_payload.get("system_prompt"))
            combined_system_prompt = _merge_system_prompts(system_prompt_value, base_system_prompt)
            if combined_system_prompt is not None:
                model_info_payload["system_prompt"] = combined_system_prompt
            else:
                model_info_payload.pop("system_prompt", None)

            decorated_extra, prompt_metadata = _compose_extra_info_context(
                record.merged_extra_info(),
                rubric_text=config.rubric_text,
                provider=provider_value,
                model=model_value,
                system_prompt=system_prompt_value,
                original_input=original_input,
                api_key=api_key_value,
                api_key_env=api_key_env_value,
                score_min=score_min,
                score_max=score_max,
                model_info=model_info_payload,
            )

            signature = inspect.signature(self._evaluate_fn)
            parameters = signature.parameters
            accepts_var_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
            )
            is_evaluate_rubric_style = accepts_var_kwargs or "rubric" in parameters or "model_info" in parameters

            if is_evaluate_rubric_style:
                return self._evaluate_fn(
                    rubric=config.rubric_text,
                    solution_str=solution,
                    model_info=model_info_payload,
                    ground_truth=ground_truth,
                    original_input=original_input,
                    metadata=prompt_metadata,
                    score_min=score_min,
                    score_max=score_max,
                    return_details=True,
                )

            call_args: list[Any] = []
            call_kwargs: dict[str, Any] = {}
            for param in parameters.values():
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue

                if param.name == "solution_str":
                    value = solution
                elif param.name == "ground_truth":
                    value = ground_truth
                elif param.name == "extra_info":
                    value = decorated_extra
                else:
                    continue

                if param.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ):
                    call_args.append(value)
                else:
                    call_kwargs[param.name] = value

            return self._evaluate_fn(*call_args, **call_kwargs)
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

                    if status == "success":
                        if isinstance(result, dict):
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
                            numeric_score = _extract_float(result)
                            if numeric_score is not None:
                                score_value = numeric_score
                                raw_payload = result
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
