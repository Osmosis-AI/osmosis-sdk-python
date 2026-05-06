import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from osmosis_ai.cli import main as cli
from osmosis_ai.eval.rubric.types import RubricResult

# =============================================================================
# eval rubric — happy-path
# =============================================================================


def test_eval_rubric_basic(tmp_path, monkeypatch, capsys):
    """eval rubric runs successfully with mocked evaluate_rubric."""
    data_path = tmp_path / "records.jsonl"
    record = {
        "messages": [
            {"role": "user", "content": "Help me"},
            {"role": "assistant", "content": "Sure, I can help."},
        ]
    }
    data_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    mock_eval = AsyncMock(
        return_value=RubricResult(score=0.85, explanation="Good response", raw={})
    )
    monkeypatch.setattr("osmosis_ai.eval.rubric.cli.evaluate_rubric", mock_eval)

    exit_code = cli.main(
        [
            "eval",
            "rubric",
            "-d",
            str(data_path),
            "--rubric",
            "Score quality of the assistant response.",
            "--model",
            "openai/gpt-5.4",
        ]
    )

    capsys.readouterr()
    assert exit_code == 0
    mock_eval.assert_called_once()


def test_eval_rubric_with_output(tmp_path, monkeypatch, capsys):
    """eval rubric writes JSON output when --output is specified."""
    data_path = tmp_path / "records.jsonl"
    record = {
        "messages": [
            {"role": "user", "content": "Help me"},
            {"role": "assistant", "content": "Sure, I can help."},
        ]
    }
    data_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    mock_eval = AsyncMock(
        return_value=RubricResult(score=0.85, explanation="Good response", raw={})
    )
    monkeypatch.setattr("osmosis_ai.eval.rubric.cli.evaluate_rubric", mock_eval)

    output_path = tmp_path / "results.json"
    exit_code = cli.main(
        [
            "eval",
            "rubric",
            "-d",
            str(data_path),
            "--rubric",
            "Score quality of the assistant response.",
            "--model",
            "openai/gpt-5.4",
            "-o",
            str(output_path),
        ]
    )

    capsys.readouterr()
    assert exit_code == 0
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "overall_statistics" in payload
    assert "records" in payload
    assert payload["overall_statistics"]["average"] == pytest.approx(0.85, rel=1e-6)


def test_eval_rubric_multiple_runs(tmp_path, monkeypatch, capsys):
    """eval rubric correctly handles --number for multiple runs per record."""
    data_path = tmp_path / "records.jsonl"
    record = {
        "messages": [
            {"role": "user", "content": "Help me"},
            {"role": "assistant", "content": "Sure, I can help."},
        ]
    }
    data_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    call_count = 0

    async def mock_eval_fn(**kwargs):
        nonlocal call_count
        call_count += 1
        score = 0.4 + 0.1 * call_count
        return RubricResult(
            score=score, explanation=f"run-{call_count}", raw={"call": call_count}
        )

    monkeypatch.setattr("osmosis_ai.eval.rubric.cli.evaluate_rubric", mock_eval_fn)

    exit_code = cli.main(
        [
            "eval",
            "rubric",
            "-d",
            str(data_path),
            "--rubric",
            "Score quality.",
            "--model",
            "openai/gpt-5.4",
            "-n",
            "3",
        ]
    )

    assert exit_code == 0
    assert call_count == 3


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


def test_eval_run_requires_project_context_before_config_lookup(capsys):
    """eval run first requires a current Osmosis project."""
    exit_code = cli.main(
        [
            "eval",
            "run",
            "nonexistent.toml",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Not in an Osmosis project" in captured.err


def test_eval_run_rejects_missing_config_inside_project(tmp_path, monkeypatch, capsys):
    """eval run fails when a project-local config file does not exist."""
    for rel_path in (
        ".osmosis",
        ".osmosis/research",
        "rollouts",
        "configs",
        "configs/eval",
        "configs/training",
        "data",
    ):
        (tmp_path / rel_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / ".osmosis" / "project.toml").write_text(
        "[project]\nname='test'\n",
        encoding="utf-8",
    )
    (tmp_path / ".osmosis" / "research" / "program.md").write_text(
        "# Test\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(["eval", "run", "configs/eval/nonexistent.toml"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Config file not found" in captured.err


def test_eval_run_rejects_fresh_and_retry_failed(tmp_path, capsys):
    """eval run rejects --fresh and --retry-failed together."""
    config_path = tmp_path / "eval.toml"
    config_path.write_text(
        '[eval]\nmodule = "mod:Agent"\ndataset = "data.jsonl"\n\n[llm]\nmodel = "openai/gpt-5.4"\n',
        encoding="utf-8",
    )
    exit_code = cli.main(
        [
            "eval",
            "run",
            str(config_path),
            "--fresh",
            "--retry-failed",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "--fresh and --retry-failed are mutually exclusive" in captured.err


def test_fuzzy_suggestion(capsys):
    exit_code = cli.main(["auht"])  # typo for "auth"
    captured = capsys.readouterr()
    assert exit_code != 0
    assert "Did you mean 'auth'?" in captured.err
