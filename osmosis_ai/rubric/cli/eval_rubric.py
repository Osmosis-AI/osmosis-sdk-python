from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import typer

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.rubric.services import (
    BaselineComparator,
    ConsoleReportRenderer,
    DatasetLoader,
    EvaluationSession,
    EvaluationSessionRequest,
    JsonReportWriter,
    RubricEvaluationEngine,
    RubricSuite,
    discover_rubric_config_path,
    load_rubric_suite,
)

app: typer.Typer = typer.Typer(help="Evaluate JSONL conversations against a rubric.")


class EvalRubricCommand:
    """Handler for `osmosis eval-rubric`."""

    def __init__(
        self,
        *,
        session: EvaluationSession | None = None,
        config_locator: Callable[
            [str | None, Path], Path
        ] = discover_rubric_config_path,
        suite_loader: Callable[[Path], RubricSuite] = load_rubric_suite,
        dataset_loader: DatasetLoader | None = None,
        engine: RubricEvaluationEngine | None = None,
        renderer: ConsoleReportRenderer | None = None,
        report_writer: JsonReportWriter | None = None,
        baseline_comparator: BaselineComparator | None = None,
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

    def run(
        self,
        *,
        rubric_id: str,
        data_path: str,
        number: int = 1,
        config_path: str | None = None,
        output_path: str | None = None,
        baseline_path: str | None = None,
    ) -> int:
        rubric_id = str(rubric_id).strip()
        if not rubric_id:
            raise CLIError("Rubric identifier cannot be empty.")

        data_path_obj = Path(data_path).expanduser()

        request = EvaluationSessionRequest(
            rubric_id=rubric_id,
            data_path=data_path_obj,
            number=number,
            config_path=Path(config_path).expanduser() if config_path else None,
            output_path=Path(output_path).expanduser() if output_path else None,
            baseline_path=Path(baseline_path).expanduser() if baseline_path else None,
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


@app.callback(invoke_without_command=True)
def eval_rubric(
    rubric_id: str = typer.Option(
        ...,
        "-r",
        "--rubric",
        help="Rubric identifier declared in the rubric config file.",
    ),
    data_path: str = typer.Option(
        ...,
        "-d",
        "--data",
        help="Path to the JSONL file containing evaluation records.",
    ),
    number: int = typer.Option(
        1, "-n", "--number", help="Run the evaluation multiple times (default: 1)."
    ),
    config_path: str | None = typer.Option(
        None, "-c", "--config", help="Path to the rubric config YAML."
    ),
    output_path: str | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Optional path to write evaluation results as JSON.",
    ),
    baseline_path: str | None = typer.Option(
        None,
        "-b",
        "--baseline",
        help="Optional path to a prior evaluation JSON to compare against.",
    ),
) -> None:
    """Evaluate JSONL conversations against a rubric."""
    cmd = EvalRubricCommand()
    rc = cmd.run(
        rubric_id=rubric_id,
        data_path=data_path,
        number=number,
        config_path=config_path,
        output_path=output_path,
        baseline_path=baseline_path,
    )
    if rc:
        raise typer.Exit(rc)
