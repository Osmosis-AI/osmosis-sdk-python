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
            "openai/gpt-4o",
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
            "openai/gpt-4o",
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
            "openai/gpt-4o",
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
            "openai/gpt-4o",
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


def test_rollout_eval_rejects_n_runs_zero(capsys):
    exit_code = cli.main(
        [
            "eval",
            "run",
            "-m",
            "my_agent:MyAgentLoop",
            "-d",
            "data.jsonl",
            "--model",
            "openai/gpt-4o",
            "--eval-fn",
            "rewards:score",
            "--n",
            "0",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error: --n must be >= 1." in captured.err


def test_rollout_eval_rejects_batch_size_zero(capsys):
    exit_code = cli.main(
        [
            "eval",
            "run",
            "-m",
            "my_agent:MyAgentLoop",
            "-d",
            "data.jsonl",
            "--model",
            "openai/gpt-4o",
            "--eval-fn",
            "rewards:score",
            "--batch-size",
            "0",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error: --batch-size must be >= 1." in captured.err


def test_rollout_eval_accepts_any_model_with_base_url(capsys):
    """With --base-url, any model name should be accepted and displayed as-is."""
    cli.main(
        [
            "eval",
            "run",
            "-m",
            "my_agent:MyAgentLoop",
            "-d",
            "data.jsonl",
            "--model",
            "Qwen/Qwen3-0.6B",
            "--base-url",
            "http://localhost:1234/v1",
            "--eval-fn",
            "rewards:score",
        ]
    )
    captured = capsys.readouterr()

    # Should pass model validation (may fail later at connectivity/auth)
    assert "Invalid model/provider format" not in captured.err
    # Model should be displayed without openai/ prefix
    assert "Model: Qwen/Qwen3-0.6B" in captured.out


def test_fuzzy_suggestion(capsys):
    exit_code = cli.main(["auht"])  # typo for "auth"
    captured = capsys.readouterr()
    assert exit_code != 0
    assert "Did you mean 'auth'?" in captured.err


def test_rollout_test_accepts_any_model_with_base_url(capsys):
    """With --base-url, any model name should be accepted and displayed as-is."""
    cli.main(
        [
            "rollout",
            "test",
            "-m",
            "my_agent:MyAgentLoop",
            "-d",
            "data.jsonl",
            "--model",
            "Qwen/Qwen3-0.6B",
            "--base-url",
            "http://localhost:1234/v1",
        ]
    )
    captured = capsys.readouterr()

    # Should pass model validation (may fail later at connectivity/auth)
    assert "Invalid model/provider format" not in captured.err
    # Model should be displayed without openai/ prefix
    assert "Model: Qwen/Qwen3-0.6B" in captured.out
