import json
import sys
from unittest.mock import AsyncMock

from osmosis_ai.cli import main as cli
from osmosis_ai.cli.console import Console
from osmosis_ai.eval.rubric.types import RubricResult


def test_eval_rubric_rich_output_contract(tmp_path, monkeypatch, capsys):
    data_path = tmp_path / "records.jsonl"
    data_path.write_text(
        json.dumps({"solution_str": "answer"}) + "\n", encoding="utf-8"
    )
    output_path = tmp_path / "results.json"
    monkeypatch.setattr(
        "osmosis_ai.eval.rubric.cli.evaluate_rubric",
        AsyncMock(return_value=RubricResult(score=0.85, explanation="Good response")),
    )
    capsys.readouterr()  # Ignore optional-dependency import diagnostics.
    monkeypatch.setattr(
        "osmosis_ai.cli.console.console",
        Console(file=sys.stdout, force_terminal=False, width=1000),
    )

    exit_code = cli.main(
        [
            "eval",
            "rubric",
            "-d",
            str(data_path),
            "-r",
            "Score quality.",
            "--model",
            "openai/gpt-5.4",
            "-n",
            "1",
            "-o",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert captured.out == (
        "Model: openai/gpt-5.4\n"
        f"Evaluated 1 record(s) from {data_path}\n"
        "Runs per record: 1\n"
        "\n"
        "[record[1]]\n"
        "  Run 01: score=0.8500\n"
        "    explanation: Good response\n"
        "\n"
        "Overall Statistics:\n"
        "  average:  0.8500\n"
        "  stdev:    0.0000\n"
        "  min/max:  0.8500 / 0.8500\n"
        f"Wrote results to {output_path}\n"
    )
    assert output_path.is_file()


# =============================================================================
# eval rubric — input validation
# =============================================================================


def test_eval_rubric_missing_data_path(tmp_path, capsys):
    """eval rubric fails when data path does not exist."""
    missing_path = tmp_path / "missing.jsonl"

    exit_code = cli.main(
        [
            "eval",
            "rubric",
            "-d",
            str(missing_path),
            "--rubric",
            "Score quality.",
            "--model",
            "openai/gpt-5.4",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert f"Data path '{missing_path}' does not exist." in captured.err


# =============================================================================
# Non-rubric CLI tests
# =============================================================================


def test_main_without_subcommand_shows_help(capsys):
    exit_code = cli.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "osmosis" in captured.out.lower()


def test_fuzzy_suggestion(capsys):
    exit_code = cli.main(["auht"])  # typo for "auth"
    captured = capsys.readouterr()
    assert exit_code != 0
    assert "Did you mean 'auth'?" in captured.err
