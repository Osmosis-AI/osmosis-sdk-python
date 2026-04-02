from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from osmosis_ai.cli.errors import CLIError

from .dataset import RubricRecord, load_rubric_dataset
from .engine import evaluate_rubric
from .report import (
    ConsoleReportRenderer,
    JsonReportWriter,
    RecordResult,
    RubricReport,
    calculate_statistics,
)
from .types import MissingAPIKeyError, ProviderRequestError


class RubricCommand:
    """Business logic for `osmosis eval rubric`."""

    def run(
        self,
        *,
        data: str,
        rubric: str,
        model: str,
        number: int = 1,
        output_path: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        score_min: float = 0.0,
        score_max: float = 1.0,
    ) -> int:
        if number < 1:
            raise CLIError("--number must be at least 1.")

        data_path = Path(data).expanduser()
        if not data_path.exists():
            raise CLIError(f"Data path '{data_path}' does not exist.")
        if data_path.is_dir():
            raise CLIError(f"Expected a file but received directory '{data_path}'.")

        rubric_text = self._resolve_rubric_text(rubric)
        records = load_rubric_dataset(data_path)

        results = asyncio.run(
            self._evaluate_all(
                records=records,
                rubric_text=rubric_text,
                model=model,
                number=number,
                api_key=api_key,
                timeout=timeout,
                score_min=score_min,
                score_max=score_max,
            )
        )

        all_scores = [s for r in results for s in r.scores]
        report = RubricReport(
            model=model,
            rubric_text=rubric_text,
            data_path=data_path,
            number=number,
            results=results,
            overall_statistics=calculate_statistics(all_scores),
        )

        ConsoleReportRenderer().render(report)

        if output_path:
            out = Path(output_path).expanduser()
            if out.is_dir():
                out = out / "rubric_eval_result.json"
            written = JsonReportWriter().write(report, out)
            print(f"Wrote results to {written}")

        has_errors = any(r.errors for r in results)
        return 1 if has_errors else 0

    @staticmethod
    def _resolve_rubric_text(rubric: str) -> str:
        if rubric.startswith("@"):
            path = Path(rubric[1:]).expanduser()
            if not path.exists():
                raise CLIError(f"Rubric file '{path}' does not exist.")
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                raise CLIError(f"Rubric file '{path}' is empty.")
            return text
        text = rubric.strip()
        if not text:
            raise CLIError("Rubric text must not be empty.")
        return text

    @staticmethod
    async def _evaluate_all(
        *,
        records: list[RubricRecord],
        rubric_text: str,
        model: str,
        number: int,
        api_key: str | None,
        timeout: float | None,
        score_min: float,
        score_max: float,
    ) -> list[RecordResult]:
        total = len(records) * number
        show_progress = total > 1 and getattr(sys.stderr, "isatty", lambda: False)()
        progress = None
        if show_progress:
            from tqdm import tqdm

            progress = tqdm(
                total=total, file=sys.stderr, dynamic_ncols=True, leave=False
            )

        # Cap concurrency to avoid overwhelming the LLM provider.
        sem = asyncio.Semaphore(8)

        async def _run_single(
            record: RubricRecord,
        ) -> tuple[float | None, str | None, str | None]:
            async with sem:
                try:
                    result = await evaluate_rubric(
                        solution_str=record.solution_str,
                        rubric=rubric_text,
                        model=model,
                        ground_truth=record.ground_truth,
                        original_input=record.original_input,
                        metadata=record.metadata,
                        score_min=score_min,
                        score_max=score_max,
                        api_key=api_key,
                        timeout=timeout,
                    )
                    return result.score, result.explanation, None
                except (
                    ProviderRequestError,
                    MissingAPIKeyError,
                    CLIError,
                    TypeError,
                    ValueError,
                ) as exc:
                    return None, None, str(exc)
                finally:
                    if progress:
                        progress.update()

        # Flatten all (record, run_index) pairs into concurrent tasks so that
        # number > 1 runs are truly parallel, not serialized per record.
        tasks = [_run_single(record) for record in records for _ in range(number)]

        try:
            flat_results = await asyncio.gather(*tasks)
        finally:
            if progress:
                progress.close()

        results: list[RecordResult] = []
        for idx, record in enumerate(records):
            chunk = flat_results[idx * number : (idx + 1) * number]
            scores = [s for s, _, _ in chunk if s is not None]
            explanations = [e for _, e, _ in chunk if e is not None]
            errors = [err for _, _, err in chunk if err is not None]
            results.append(
                RecordResult(
                    record_index=idx + 1,
                    label=record.label(idx + 1),
                    scores=scores,
                    explanations=explanations,
                    errors=errors,
                    statistics=calculate_statistics(scores),
                )
            )
        return results
