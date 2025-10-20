from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .cli_services import (
    CLIError,
    ParsedItem,
    BaselineComparator,
    ConsoleReportRenderer,
    DatasetLoader,
    EvaluationSession,
    EvaluationSessionRequest,
    JsonReportWriter,
    RubricEvaluationEngine,
    RubricSuite,
    discover_rubric_config_path,
    load_jsonl_records,
    load_rubric_configs,
    load_rubric_suite,
    render_json_records,
    render_yaml_items,
)


class PreviewCommand:
    """Handler for `osmosis preview`."""

    def __init__(
        self,
        *,
        yaml_loader: Callable[[Path], list[ParsedItem]] = load_rubric_configs,
        json_loader: Callable[[Path], list[dict[str, Any]]] = load_jsonl_records,
    ):
        self._yaml_loader = yaml_loader
        self._json_loader = json_loader

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.set_defaults(handler=self.run)
        parser.add_argument(
            "-p",
            "--path",
            dest="path",
            required=True,
            help="Path to the YAML or JSONL file to inspect.",
        )

    def run(self, args: argparse.Namespace) -> int:
        path = Path(args.path).expanduser()
        if not path.exists():
            raise CLIError(f"Path '{path}' does not exist.")
        if path.is_dir():
            raise CLIError(f"Expected a file path but got directory '{path}'.")

        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            items = self._yaml_loader(path)
            print(f"Loaded {len(items)} rubric config(s) from {path}")
            print(render_yaml_items(items, label="Rubric config"))
        elif suffix == ".jsonl":
            records = self._json_loader(path)
            print(f"Loaded {len(records)} JSONL record(s) from {path}")
            print(render_json_records(records))
        else:
            raise CLIError(f"Unsupported file extension '{suffix}'. Expected .yaml, .yml, or .jsonl.")

        return 0


class EvalCommand:
    """Handler for `osmosis eval`."""

    def __init__(
        self,
        *,
        session: Optional[EvaluationSession] = None,
        config_locator: Callable[[Optional[str], Path], Path] = discover_rubric_config_path,
        suite_loader: Callable[[Path], RubricSuite] = load_rubric_suite,
        dataset_loader: Optional[DatasetLoader] = None,
        engine: Optional[RubricEvaluationEngine] = None,
        renderer: Optional[ConsoleReportRenderer] = None,
        report_writer: Optional[JsonReportWriter] = None,
        baseline_comparator: Optional[BaselineComparator] = None,
    ):
        self._renderer = renderer or ConsoleReportRenderer()
        if session is not None:
            self._session = session
        else:
            self._session = EvaluationSession(
                config_locator=config_locator,
                suite_loader=suite_loader,
                dataset_loader=dataset_loader,
                engine=engine,
                baseline_comparator=baseline_comparator,
                report_writer=report_writer,
                identifier_factory=self._generate_output_identifier,
            )

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.set_defaults(handler=self.run)
        parser.add_argument(
            "-r",
            "--rubric",
            dest="rubric_id",
            required=True,
            help="Rubric identifier declared in the rubric config file.",
        )
        parser.add_argument(
            "-d",
            "--data",
            dest="data_path",
            required=True,
            help="Path to the JSONL file containing evaluation records.",
        )
        parser.add_argument(
            "-n",
            "--number",
            dest="number",
            type=int,
            default=1,
            help="Run the evaluation multiple times to sample provider variance (default: 1).",
        )
        parser.add_argument(
            "-c",
            "--config",
            dest="config_path",
            help="Path to the rubric config YAML (defaults to searching near the data file).",
        )
        parser.add_argument(
            "-o",
            "--output",
            dest="output_path",
            help="Optional path to write evaluation results as JSON.",
        )
        parser.add_argument(
            "-b",
            "--baseline",
            dest="baseline_path",
            help="Optional path to a prior evaluation JSON to compare against.",
        )

    def run(self, args: argparse.Namespace) -> int:
        rubric_id_raw = getattr(args, "rubric_id", "")
        rubric_id = str(rubric_id_raw).strip()
        if not rubric_id:
            raise CLIError("Rubric identifier cannot be empty.")

        data_path = Path(args.data_path).expanduser()
        config_path_value = getattr(args, "config_path", None)
        output_path_value = getattr(args, "output_path", None)
        baseline_path_value = getattr(args, "baseline_path", None)

        number_value = getattr(args, "number", None)
        number = int(number_value) if number_value is not None else 1

        request = EvaluationSessionRequest(
            rubric_id=rubric_id,
            data_path=data_path,
            number=number,
            config_path=Path(config_path_value).expanduser() if config_path_value else None,
            output_path=Path(output_path_value).expanduser() if output_path_value else None,
            baseline_path=Path(baseline_path_value).expanduser() if baseline_path_value else None,
        )

        try:
            result = self._session.execute(request)
        except KeyboardInterrupt:
            print("Evaluation cancelled by user.")
            return 1
        self._renderer.render(result.report, result.baseline)

        if result.written_path is not None:
            print(f"Wrote evaluation results to {result.written_path}")

        return 0

    @staticmethod
    def _generate_output_identifier() -> str:
        return str(int(time.time()))
