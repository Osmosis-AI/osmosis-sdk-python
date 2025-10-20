import json
from pathlib import Path

import pytest

from osmosis_ai.cli_services import CLIError
from osmosis_ai.cli_services.dataset import DatasetLoader
from osmosis_ai.cli_services.reporting import BaselineComparator
from osmosis_ai.cli_services.session import EvaluationSession, EvaluationSessionRequest


def test_baseline_comparator_missing_path(tmp_path: Path) -> None:
    comparator = BaselineComparator()
    missing = tmp_path / "baseline.json"

    with pytest.raises(CLIError, match=f"Baseline path '{missing}' does not exist."):
        comparator.load(missing)


def test_baseline_comparator_directory_path(tmp_path: Path) -> None:
    comparator = BaselineComparator()
    directory = tmp_path / "baseline_dir"
    directory.mkdir()

    with pytest.raises(CLIError, match=f"Baseline path '{directory}' is a directory"):
        comparator.load(directory)


def test_baseline_comparator_invalid_json(tmp_path: Path) -> None:
    comparator = BaselineComparator()
    target = tmp_path / "baseline.json"
    target.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(CLIError, match="Failed to parse baseline JSON"):
        comparator.load(target)


def test_baseline_comparator_missing_statistics(tmp_path: Path) -> None:
    comparator = BaselineComparator()
    target = tmp_path / "baseline.json"
    target.write_text(json.dumps({"metadata": {"average": 0.5}}), encoding="utf-8")

    with pytest.raises(
        CLIError, match="Baseline JSON must include an 'overall_statistics' object or top-level statistics."
    ):
        comparator.load(target)


def test_baseline_comparator_non_numeric_statistics(tmp_path: Path) -> None:
    comparator = BaselineComparator()
    target = tmp_path / "baseline.json"
    target.write_text(json.dumps({"overall_statistics": {"average": "bad"}}), encoding="utf-8")

    with pytest.raises(CLIError, match="Baseline statistics could not be parsed into numeric values."):
        comparator.load(target)


def test_evaluation_session_errors_when_no_matching_records(tmp_path: Path) -> None:
    config_content = """rubrics:
  - id: support_followup
    rubric: Score responses.
    model_info:
      provider: openai
      model: gpt-5-mini
      api_key: dummy
"""
    config_path = tmp_path / "rubric_configs.yaml"
    config_path.write_text(config_content, encoding="utf-8")

    record = {
        "rubric_id": "other_rubric",
        "messages": [
            {"role": "user", "content": "Hi there"},
            {"role": "assistant", "content": "Hello"},
        ],
    }
    dataset_path = tmp_path / "records.jsonl"
    dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    session = EvaluationSession()
    request = EvaluationSessionRequest(
        rubric_id="support_followup",
        data_path=dataset_path,
        config_path=config_path,
    )

    message = f"No records in '{dataset_path}' reference rubric 'support_followup'."
    with pytest.raises(CLIError, match=message):
        session.execute(request)


def test_resolve_output_path_defaults_to_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("osmosis_ai.cli_services.session._CACHE_ROOT", tmp_path)

    session = EvaluationSession(identifier_factory=lambda: "12345")
    path, identifier = session._resolve_output_path(None, None, rubric_id="My Rubric/ID")

    expected_dir = tmp_path / "my_rubric_id"
    expected_path = expected_dir / "rubric_eval_result_12345.json"

    assert identifier == "12345"
    assert path == expected_path
    assert expected_dir.is_dir()


def _write_records(path: Path, lines: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")


def test_dataset_loader_invalid_json(tmp_path: Path) -> None:
    loader = DatasetLoader()
    data_path = tmp_path / "records.jsonl"
    data_path.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(CLIError, match="Invalid JSON on line 1"):
        loader.load(data_path)


def test_dataset_loader_non_object_record(tmp_path: Path) -> None:
    loader = DatasetLoader()
    data_path = tmp_path / "records.jsonl"
    data_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(CLIError, match="Expected JSON object on line 1"):
        loader.load(data_path)


def test_dataset_loader_missing_messages(tmp_path: Path) -> None:
    loader = DatasetLoader()
    data_path = tmp_path / "records.jsonl"
    record = {"rubric_id": "support_followup"}
    _write_records(data_path, [record])

    with pytest.raises(CLIError, match="Record 'support_followup' must include a non-empty 'messages' list."):
        loader.load(data_path)


def test_dataset_loader_message_not_mapping(tmp_path: Path) -> None:
    loader = DatasetLoader()
    data_path = tmp_path / "records.jsonl"
    record = {
        "rubric_id": "support_followup",
        "messages": ["not a dict"],
    }
    _write_records(data_path, [record])

    with pytest.raises(CLIError, match="Message 0 in support_followup must be an object, got str."):
        loader.load(data_path)


def test_dataset_loader_empty_file(tmp_path: Path) -> None:
    loader = DatasetLoader()
    data_path = tmp_path / "records.jsonl"
    data_path.write_text("\n", encoding="utf-8")

    with pytest.raises(CLIError, match=f"No JSON records found in '{data_path}'"):
        loader.load(data_path)


def test_dataset_record_assistant_preview_truncates(tmp_path: Path) -> None:
    loader = DatasetLoader()
    data_path = tmp_path / "records.jsonl"
    long_text = "A" * 160
    record = {
        "rubric_id": "support_followup",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": long_text},
        ],
    }
    _write_records(data_path, [record])

    loaded = loader.load(data_path)[0]
    preview = loaded.assistant_preview()

    assert preview == ("A" * 137) + "..."


def test_dataset_record_assistant_preview_returns_none_without_assistant(tmp_path: Path) -> None:
    loader = DatasetLoader()
    data_path = tmp_path / "records.jsonl"
    record = {
        "rubric_id": "support_followup",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Instructions"},
        ],
    }
    _write_records(data_path, [record])

    loaded = loader.load(data_path)[0]

    assert loaded.assistant_preview() is None
