from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .engine import EvaluationRecordResult, EvaluationReport, EvaluationRun
from .errors import CLIError
from .shared import calculate_stat_deltas


class TextReportFormatter:
    """Builds human-readable text lines for an evaluation report."""

    def build(
        self,
        report: EvaluationReport,
        baseline: Optional["BaselineStatistics"] = None,
    ) -> list[str]:
        lines: list[str] = []
        provider = str(report.rubric_config.model_info.get("provider", "")).strip() or "<unknown>"
        model_name = str(report.rubric_config.model_info.get("model", "")).strip() or "<unspecified>"

        lines.append(
            f"Rubric '{report.rubric_config.rubric_id}' "
            f"({report.rubric_config.source_label}) -> provider '{provider}' model '{model_name}'"
        )
        lines.append(f"Loaded {len(report.record_results)} matching record(s) from {report.data_path}")
        lines.append(f"Running {report.number} evaluation(s) per record")
        lines.append("")

        for record_result in report.record_results:
            lines.extend(self._format_record(record_result))
        if baseline is not None:
            lines.extend(self._format_baseline(report, baseline))
        return lines

    def _format_record(self, record_result: EvaluationRecordResult) -> list[str]:
        lines: list[str] = [f"[{record_result.conversation_label}]"]
        total_runs = len(record_result.runs)
        for index, run in enumerate(record_result.runs):
            lines.extend(self._format_run(run))
            if index < total_runs - 1:
                lines.append("")

        summary_lines = self._format_summary(record_result.statistics, len(record_result.runs))
        if summary_lines:
            lines.extend(summary_lines)
        lines.append("")
        return lines

    def _format_run(self, run: EvaluationRun) -> list[str]:
        lines: list[str] = [f"  Run {run.run_index:02d} [{run.status.upper()}]"]
        if run.status == "success":
            score_text = "n/a" if run.score is None else f"{run.score:.4f}"
            lines.append(self._format_detail_line("score", score_text))
            if run.preview:
                lines.append(self._format_detail_line("preview", run.preview))
            explanation = run.explanation or "(no explanation provided)"
            lines.append(self._format_detail_line("explanation", explanation))
        else:
            error_text = run.error or "(no error message provided)"
            lines.append(self._format_detail_line("error", error_text))
            if run.preview:
                lines.append(self._format_detail_line("preview", run.preview))
            if run.explanation:
                lines.append(self._format_detail_line("explanation", run.explanation))
        lines.append(self._format_detail_line("duration", f"{run.duration_seconds:.2f}s"))
        return lines

    def _format_summary(self, statistics: dict[str, float], total_runs: int) -> list[str]:
        if not statistics:
            return []
        success_count = int(round(statistics.get("success_count", total_runs)))
        failure_count = int(round(statistics.get("failure_count", total_runs - success_count)))
        if total_runs <= 1 and failure_count == 0:
            return []

        lines = ["  Summary:"]
        lines.append(f"    total:     {int(round(statistics.get('total_runs', total_runs)))}")
        lines.append(f"    successes: {success_count}")
        lines.append(f"    failures:  {failure_count}")
        if success_count > 0:
            lines.append(f"    average:   {statistics.get('average', 0.0):.4f}")
            lines.append(f"    variance:  {statistics.get('variance', 0.0):.6f}")
            lines.append(f"    stdev:     {statistics.get('stdev', 0.0):.4f}")
            lines.append(f"    min/max:   {statistics.get('min', 0.0):.4f} / {statistics.get('max', 0.0):.4f}")
        else:
            lines.append("    average:   n/a")
            lines.append("    variance:  n/a")
            lines.append("    stdev:     n/a")
            lines.append("    min/max:   n/a")
        return lines

    def _format_baseline(
        self,
        report: EvaluationReport,
        baseline: "BaselineStatistics",
    ) -> list[str]:
        lines = [f"Baseline comparison (source: {baseline.source_path}):"]
        deltas = baseline.delta(report.overall_statistics)
        keys = ["average", "variance", "stdev", "min", "max", "success_count", "failure_count", "total_runs"]

        for key in keys:
            if key not in baseline.statistics or key not in report.overall_statistics:
                continue
            baseline_value = float(baseline.statistics[key])
            current_value = float(report.overall_statistics[key])
            delta_value = float(deltas.get(key, current_value - baseline_value))

            if key in {"success_count", "failure_count", "total_runs"}:
                baseline_str = f"{int(round(baseline_value))}"
                current_str = f"{int(round(current_value))}"
                delta_str = f"{delta_value:+.0f}"
            else:
                precision = 6 if key == "variance" else 4
                baseline_str = format(baseline_value, f".{precision}f")
                current_str = format(current_value, f".{precision}f")
                delta_str = format(delta_value, f"+.{precision}f")

            lines.append(
                f"  {key:12s} baseline={baseline_str} current={current_str} delta={delta_str}"
            )
        return lines

    @staticmethod
    def _format_detail_line(label: str, value: str, *, indent: int = 4) -> str:
        indent_str = " " * indent
        value_str = str(value)
        continuation_indent = indent_str + "  "
        value_str = value_str.replace("\n", f"\n{continuation_indent}")
        return f"{indent_str}{label}: {value_str}"


class ConsoleReportRenderer:
    """Pretty prints evaluation reports to stdout (or any printer function)."""

    def __init__(
        self,
        printer: Callable[[str], None] = print,
        formatter: Optional[TextReportFormatter] = None,
    ):
        self._printer = printer
        self._formatter = formatter or TextReportFormatter()

    def render(
        self,
        report: EvaluationReport,
        baseline: Optional["BaselineStatistics"] = None,
    ) -> None:
        for line in self._formatter.build(report, baseline):
            self._printer(line)


class BaselineStatistics:
    def __init__(self, source_path: Path, statistics: dict[str, float]):
        self.source_path = source_path
        self.statistics = statistics

    def delta(self, current: dict[str, float]) -> dict[str, float]:
        return calculate_stat_deltas(self.statistics, current)


class BaselineComparator:
    """Loads baseline JSON payloads and extracts statistics."""

    def load(self, path: Path) -> BaselineStatistics:
        if not path.exists():
            raise CLIError(f"Baseline path '{path}' does not exist.")
        if path.is_dir():
            raise CLIError(f"Baseline path '{path}' is a directory, expected JSON file.")

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CLIError(f"Failed to parse baseline JSON: {exc}") from exc

        if not isinstance(payload, dict):
            raise CLIError("Baseline JSON must contain an object.")

        source = None
        if isinstance(payload.get("overall_statistics"), dict):
            source = payload["overall_statistics"]
        elif all(key in payload for key in ("average", "variance", "stdev")):
            source = payload
        if source is None:
            raise CLIError(
                "Baseline JSON must include an 'overall_statistics' object or top-level statistics."
            )

        statistics: dict[str, float] = {}
        for key, value in source.items():
            try:
                statistics[key] = float(value)
            except (TypeError, ValueError):
                continue
        if not statistics:
            raise CLIError("Baseline statistics could not be parsed into numeric values.")

        return BaselineStatistics(source_path=path, statistics=statistics)


class JsonReportFormatter:
    """Builds JSON-serialisable payloads for evaluation reports."""

    def build(
        self,
        report: EvaluationReport,
        *,
        output_identifier: Optional[str],
        baseline: Optional[BaselineStatistics],
    ) -> dict[str, Any]:
        provider = str(report.rubric_config.model_info.get("provider", "")).strip() or "<unknown>"
        model_name = str(report.rubric_config.model_info.get("model", "")).strip() or "<unspecified>"

        generated: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rubric_id": report.rubric_config.rubric_id,
            "rubric_source": report.rubric_config.source_label,
            "provider": provider,
            "model": model_name,
            "number": report.number,
            "config_path": str(report.config_path),
            "data_path": str(report.data_path),
            "overall_statistics": _normalise_statistics(report.overall_statistics),
            "records": [],
        }
        if output_identifier is not None:
            generated["output_identifier"] = output_identifier

        for record_result in report.record_results:
            conversation_label = record_result.conversation_label
            record_identifier = record_result.record.record_identifier(conversation_label)

            record_payload: dict[str, Any] = {
                "id": record_identifier,
                "record_index": record_result.record_index,
                "conversation_id": conversation_label,
                "input_record": record_result.record.payload,
                "statistics": _normalise_statistics(record_result.statistics),
                "runs": [],
            }

            for run in record_result.runs:
                record_payload["runs"].append(
                    {
                        "run_index": run.run_index,
                        "status": run.status,
                        "started_at": run.started_at.isoformat(),
                        "completed_at": run.completed_at.isoformat(),
                        "duration_seconds": run.duration_seconds,
                        "score": run.score,
                        "explanation": run.explanation,
                        "preview": run.preview,
                        "error": run.error,
                        "raw": run.raw,
                    }
                )

            generated["records"].append(record_payload)

        if baseline is not None:
            generated["baseline_comparison"] = {
                "source_path": str(baseline.source_path),
                "baseline_statistics": _normalise_statistics(baseline.statistics),
                "delta_statistics": _normalise_statistics(baseline.delta(report.overall_statistics)),
            }

        return generated


class JsonReportWriter:
    """Serialises an evaluation report to disk."""

    def __init__(self, formatter: Optional[JsonReportFormatter] = None):
        self._formatter = formatter or JsonReportFormatter()

    def write(
        self,
        report: EvaluationReport,
        *,
        output_path: Path,
        output_identifier: Optional[str],
        baseline: Optional[BaselineStatistics],
    ) -> Path:
        parent_dir = output_path.parent
        if parent_dir and not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)

        payload = self._formatter.build(
            report,
            output_identifier=output_identifier,
            baseline=baseline,
        )
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        return output_path

def _normalise_statistics(stats: dict[str, float]) -> dict[str, Any]:
    normalised: dict[str, Any] = {}
    for key, value in stats.items():
        if key in {"success_count", "failure_count", "total_runs"}:
            normalised[key] = int(round(value))
        else:
            normalised[key] = float(value)
    return normalised
