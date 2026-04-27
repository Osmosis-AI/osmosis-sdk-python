from __future__ import annotations

import json
import sys
from unittest.mock import AsyncMock

import pytest

from osmosis_ai.cli import main as cli
from osmosis_ai.eval.rubric.types import RubricResult


def test_eval_rubric_json_returns_operation_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys,
) -> None:
    data_path = tmp_path / "records.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Help me"},
                    {"role": "assistant", "content": "Sure."},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    mock_eval = AsyncMock(
        return_value=RubricResult(score=0.8, explanation="good", raw={})
    )
    monkeypatch.setattr("osmosis_ai.eval.rubric.cli.evaluate_rubric", mock_eval)

    exit_code = cli.main(
        [
            "--json",
            "eval",
            "rubric",
            "-d",
            str(data_path),
            "--rubric",
            "Score quality.",
            "--model",
            "openai/gpt-5.4",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["schema_version"] == 1
    assert payload["status"] == "success"
    assert payload["operation"] == "eval.rubric"
    assert payload["resource"]["statistics"]["average"] == pytest.approx(0.8)
    assert payload["resource"]["records"][0]["scores"] == [0.8]


def test_eval_rubric_json_output_file_omits_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys,
) -> None:
    data_path = tmp_path / "records.jsonl"
    data_path.write_text(
        json.dumps({"solution_str": "answer", "id": "row-1"}) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "result.json"
    monkeypatch.setattr(
        "osmosis_ai.eval.rubric.cli.evaluate_rubric",
        AsyncMock(return_value=RubricResult(score=1.0, explanation="ok", raw={})),
    )

    exit_code = cli.main(
        [
            "--json",
            "eval",
            "rubric",
            "-d",
            str(data_path),
            "--rubric",
            "Score quality.",
            "--model",
            "openai/gpt-5.4",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["resource"]["output_path"] == str(output_path)
    assert "records" not in payload["resource"]
    assert output_path.exists()


def test_eval_rubric_json_suppresses_tty_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys,
) -> None:
    data_path = tmp_path / "records.jsonl"
    data_path.write_text(
        json.dumps({"solution_str": "answer", "id": "row-1"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    monkeypatch.setattr(
        "osmosis_ai.eval.rubric.cli.evaluate_rubric",
        AsyncMock(return_value=RubricResult(score=1.0, explanation="ok", raw={})),
    )

    exit_code = cli.main(
        [
            "--json",
            "eval",
            "rubric",
            "-d",
            str(data_path),
            "--rubric",
            "Score quality.",
            "--model",
            "openai/gpt-5.4",
            "--number",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["status"] == "success"


def test_eval_rubric_plain_allows_tty_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys,
) -> None:
    data_path = tmp_path / "records.jsonl"
    data_path.write_text(
        json.dumps({"solution_str": "answer", "id": "row-1"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    monkeypatch.setattr(
        "osmosis_ai.eval.rubric.cli.evaluate_rubric",
        AsyncMock(return_value=RubricResult(score=1.0, explanation="ok", raw={})),
    )

    exit_code = cli.main(
        [
            "--plain",
            "eval",
            "rubric",
            "-d",
            str(data_path),
            "--rubric",
            "Score quality.",
            "--model",
            "openai/gpt-5.4",
            "--number",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err != ""
