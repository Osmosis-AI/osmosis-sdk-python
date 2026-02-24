from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.rubric.eval import ensure_api_key_available
from osmosis_ai.rubric.types import MissingAPIKeyError

from .config import (
    RubricConfig,
    RubricSuite,
    discover_rubric_config_path,
    load_rubric_suite,
)
from .dataset import DatasetLoader, DatasetRecord
from .engine import EvaluationReport, RubricEvaluationEngine
from .reporting import BaselineComparator, BaselineStatistics, JsonReportWriter

logger: logging.Logger = logging.getLogger(__name__)

_CACHE_ROOT = Path("~/.cache/osmosis/rubric").expanduser()
_OLD_RUBRIC_CACHE_ROOT = Path("~/.cache/osmosis/eval_result").expanduser()


_migration_done = False


def _migrate_rubric_cache() -> None:
    """Migrate rubric cache from old path to new path.

    Old: ~/.cache/osmosis/eval_result
    New: ~/.cache/osmosis/rubric
    """
    global _migration_done
    if _migration_done:
        return
    _migration_done = True

    if not _OLD_RUBRIC_CACHE_ROOT.exists():
        return

    if _CACHE_ROOT.exists():
        # Both exist — warn but don't overwrite
        logger.warning(
            "Both old and new rubric cache directories exist.\n"
            "  Old: %s\n"
            "  New: %s\n"
            "  Please manually merge or remove the old directory.",
            _OLD_RUBRIC_CACHE_ROOT,
            _CACHE_ROOT,
        )
        return

    try:
        _CACHE_ROOT.parent.mkdir(parents=True, exist_ok=True)
        os.rename(_OLD_RUBRIC_CACHE_ROOT, _CACHE_ROOT)
    except OSError as e:
        # TOCTOU race or permission issue — fall back silently
        logger.warning("Could not migrate rubric cache: %s", e)


def _sanitise_rubric_folder(rubric_id: str) -> str:
    """Produce a filesystem-safe folder name for the rubric id."""
    clean = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in rubric_id.strip()
    )
    clean = clean.strip("_") or "rubric"
    return clean.lower()


@dataclass(frozen=True)
class EvaluationSessionRequest:
    rubric_id: str
    data_path: Path
    number: int = 1
    config_path: Path | None = None
    output_path: Path | None = None
    output_identifier: str | None = None
    baseline_path: Path | None = None


@dataclass
class EvaluationSessionResult:
    request: EvaluationSessionRequest
    config_path: Path
    data_path: Path
    rubric_config: RubricConfig
    records: Sequence[DatasetRecord]
    report: EvaluationReport
    baseline: BaselineStatistics | None
    written_path: Path | None
    output_identifier: str | None


class EvaluationSession:
    """Coordinates rubric evaluation end-to-end for reusable orchestration."""

    def __init__(
        self,
        *,
        config_locator: Callable[
            [str | None, Path], Path
        ] = discover_rubric_config_path,
        suite_loader: Callable[[Path], RubricSuite] = load_rubric_suite,
        dataset_loader: DatasetLoader | None = None,
        engine: RubricEvaluationEngine | None = None,
        baseline_comparator: BaselineComparator | None = None,
        report_writer: JsonReportWriter | None = None,
        identifier_factory: Callable[[], str] | None = None,
    ):
        self._config_locator = config_locator
        self._suite_loader = suite_loader
        self._dataset_loader = dataset_loader or DatasetLoader()
        self._engine = engine or RubricEvaluationEngine()
        self._baseline_comparator = baseline_comparator or BaselineComparator()
        self._report_writer = report_writer or JsonReportWriter()
        self._identifier_factory = identifier_factory or self._default_identifier

    def execute(self, request: EvaluationSessionRequest) -> EvaluationSessionResult:
        _migrate_rubric_cache()
        rubric_id = request.rubric_id.strip()
        if not rubric_id:
            raise CLIError("Rubric identifier cannot be empty.")

        number_value = request.number if request.number is not None else 1
        number = int(number_value)
        if number < 1:
            raise CLIError("Number of runs must be a positive integer.")

        data_path = request.data_path.expanduser()
        if not data_path.exists():
            raise CLIError(f"Data path '{data_path}' does not exist.")
        if data_path.is_dir():
            raise CLIError(
                f"Expected a JSONL file but received directory '{data_path}'."
            )

        config_override = (
            str(request.config_path.expanduser()) if request.config_path else None
        )
        config_path = self._config_locator(config_override, data_path)
        suite = self._suite_loader(config_path)
        rubric_config = suite.get(rubric_id)

        try:
            ensure_api_key_available(rubric_config.model_info)
        except (MissingAPIKeyError, TypeError) as exc:
            raise CLIError(str(exc)) from exc

        all_records = self._dataset_loader.load(data_path)
        matching_records = [
            record
            for record in all_records
            if record.rubric_id.lower() == rubric_id.lower()
        ]
        if not matching_records:
            raise CLIError(
                f"No records in '{data_path}' reference rubric '{rubric_id}'."
            )

        baseline_stats = self._load_baseline(request.baseline_path)

        resolved_output_path, resolved_identifier = self._resolve_output_path(
            request.output_path,
            request.output_identifier,
            rubric_id=rubric_id,
        )

        report = self._engine.execute(
            rubric_config=rubric_config,
            config_path=config_path,
            data_path=data_path,
            records=matching_records,
            number=number,
        )

        written_path = None
        if resolved_output_path is not None:
            written_path = self._report_writer.write(
                report,
                output_path=resolved_output_path,
                output_identifier=resolved_identifier,
                baseline=baseline_stats,
            )

        return EvaluationSessionResult(
            request=request,
            config_path=config_path,
            data_path=data_path,
            rubric_config=rubric_config,
            records=matching_records,
            report=report,
            baseline=baseline_stats,
            written_path=written_path,
            output_identifier=resolved_identifier,
        )

    def _load_baseline(self, baseline_path: Path | None) -> BaselineStatistics | None:
        if baseline_path is None:
            return None
        resolved = baseline_path.expanduser()
        return self._baseline_comparator.load(resolved)

    def _resolve_output_path(
        self,
        output_candidate: Path | None,
        output_identifier: str | None,
        *,
        rubric_id: str,
    ) -> tuple[Path | None, str | None]:
        if output_candidate is None:
            identifier = output_identifier or self._identifier_factory()
            target_dir = _CACHE_ROOT / _sanitise_rubric_folder(rubric_id)
            target_dir.mkdir(parents=True, exist_ok=True)
            return target_dir / f"rubric_eval_result_{identifier}.json", identifier

        candidate = output_candidate.expanduser()
        if candidate.suffix:
            if candidate.exists() and candidate.is_dir():
                raise CLIError(f"Output path '{candidate}' is a directory.")
            return candidate, output_identifier

        candidate.mkdir(parents=True, exist_ok=True)
        identifier = output_identifier or self._identifier_factory()
        return candidate / f"rubric_eval_result_{identifier}.json", identifier

    @staticmethod
    def _default_identifier() -> str:
        return str(int(time.time()))
