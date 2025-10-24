import copy
import json
from pathlib import Path
from typing import Any

import pytest

from osmosis_ai import osmosis_rubric
from osmosis_ai.cli_services import CLIError
from osmosis_ai.cli_services.config import RubricConfig, load_rubric_suite
from osmosis_ai.cli_services.dataset import DatasetLoader, DatasetRecord
from osmosis_ai.cli_services.engine import RubricEvaluator
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
        "solution_str": "Hello there",
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


def test_dataset_loader_missing_solution_str(tmp_path: Path) -> None:
    loader = DatasetLoader()
    data_path = tmp_path / "records.jsonl"
    record = {"rubric_id": "support_followup"}
    _write_records(data_path, [record])

    with pytest.raises(CLIError, match="Record 'support_followup' must include a non-empty 'solution_str' string."):
        loader.load(data_path)


def test_dataset_loader_blank_solution_str(tmp_path: Path) -> None:
    loader = DatasetLoader()
    data_path = tmp_path / "records.jsonl"
    record = {
        "rubric_id": "support_followup",
        "solution_str": "   ",
    }
    _write_records(data_path, [record])

    with pytest.raises(CLIError, match="Record 'support_followup' must include a non-empty 'solution_str' string."):
        loader.load(data_path)


def test_dataset_loader_empty_file(tmp_path: Path) -> None:
    loader = DatasetLoader()
    data_path = tmp_path / "records.jsonl"
    data_path.write_text("\n", encoding="utf-8")

    with pytest.raises(CLIError, match=f"No JSON records found in '{data_path}'"):
        loader.load(data_path)


def test_dataset_loader_supports_original_input_in_extra_info(tmp_path: Path) -> None:
    loader = DatasetLoader()
    data_path = tmp_path / "records.jsonl"
    record = {
        "rubric_id": "support_followup",
        "solution_str": "Assistant reply",
        "extra_info": {"original_input": "Please help me troubleshoot my purifier."},
    }
    _write_records(data_path, [record])

    loaded = loader.load(data_path)[0]

    assert loaded.original_input == "Please help me troubleshoot my purifier."
    assert loaded.extra_info == {"original_input": "Please help me troubleshoot my purifier."}


def test_rubric_config_rejects_extra_info(tmp_path: Path) -> None:
    config_path = tmp_path / "rubric_configs.yaml"
    config_content = """version: 1
rubrics:
  - id: support_followup
    rubric: Score responses.
    model_info:
      provider: openai
      model: gpt-5-mini
    extra_info:
      unexpected: true
"""
    config_path.write_text(config_content, encoding="utf-8")

    with pytest.raises(CLIError, match="must not include 'extra_info'"):
        load_rubric_suite(config_path)


def test_rubric_config_rejects_default_extra_info(tmp_path: Path) -> None:
    config_path = tmp_path / "rubric_configs.yaml"
    config_content = """version: 1
default_extra_info:
  capture_details: true
rubrics:
  - id: support_followup
    rubric: Score responses.
    model_info:
      provider: openai
      model: gpt-5-mini
"""
    config_path.write_text(config_content, encoding="utf-8")

    with pytest.raises(CLIError, match="default_extra_info"):
        load_rubric_suite(config_path)


def test_dataset_record_assistant_preview_truncates(tmp_path: Path) -> None:
    loader = DatasetLoader()
    data_path = tmp_path / "records.jsonl"
    long_text = "A" * 160
    record = {
        "rubric_id": "support_followup",
        "solution_str": long_text,
    }
    _write_records(data_path, [record])

    loaded = loader.load(data_path)[0]
    preview = loaded.assistant_preview()

    assert preview == ("A" * 137) + "..."


def test_dataset_record_assistant_preview_returns_none_without_assistant(tmp_path: Path) -> None:
    record = DatasetRecord(
        payload={},
        rubric_id="support_followup",
        conversation_id=None,
        record_id=None,
        solution_str="   ",
        ground_truth=None,
        original_input=None,
        metadata=None,
        extra_info=None,
        score_min=None,
        score_max=None,
    )

    assert record.assistant_preview() is None


def test_rubric_evaluator_enriches_extra_info() -> None:
    captured: dict[str, Any] = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"score": 0.75, "explanation": "ok", "raw": {"call": 1}}

    evaluator = RubricEvaluator(evaluate_fn=fake_evaluate)

    record = DatasetRecord(
        payload={},
        rubric_id="support_followup",
        conversation_id="conv-001",
        record_id=None,
        solution_str="Hello",
        ground_truth=None,
        original_input="Please help me troubleshoot my purifier.",
        metadata={"channel": "chat"},
        extra_info={"capture_details": True, "metadata": {"product": "AirPure"}},
        score_min=None,
        score_max=None,
    )

    config = RubricConfig(
        rubric_id="support_followup",
        rubric_text="Score how well the assistant resolves the issue.",
        model_info={"provider": "openai", "model": "gpt-5-mini", "api_key": "dummy", "system_prompt": "Judge fairly."},
        score_min=0.0,
        score_max=1.0,
        system_prompt="System override prompt.",
        original_input=None,
        ground_truth="Reference answer.",
        source_label="tests",
    )

    evaluator.run(config, record)

    system_prompt_value = captured["model_info"].get("system_prompt")
    assert system_prompt_value == "System override prompt.\n\nJudge fairly."
    assert captured["ground_truth"] == "Reference answer."
    assert captured["original_input"] == "Please help me troubleshoot my purifier."
    assert "system_prompt" not in captured

    metadata = captured["metadata"]
    assert metadata == {"product": "AirPure", "dataset_metadata": {"channel": "chat"}}


def test_rubric_evaluator_overrides_conflicting_extra_info() -> None:
    captured: dict[str, Any] = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"score": 0.5, "explanation": "ok", "raw": {"call": "noop"}}

    evaluator = RubricEvaluator(evaluate_fn=fake_evaluate)

    record = DatasetRecord(
        payload={},
        rubric_id="support_followup",
        conversation_id=None,
        record_id=None,
        solution_str="Assistant reply",
        ground_truth=None,
        original_input=None,
        metadata=None,
        extra_info={
            "provider": "anthropic",
            "model": "claude-0",
            "rubric": "stale rubric",
            "score_min": "0.75",
            "score_max": "1.25",
            "capsule": "retain-me",
        },
        score_min=None,
        score_max=None,
    )

    config = RubricConfig(
        rubric_id="support_followup",
        rubric_text="Judge response quality.",
        model_info={"provider": "openai", "model": "gpt-5-mini", "api_key": "dummy"},
        score_min=0.0,
        score_max=1.0,
        system_prompt=None,
        original_input=None,
        ground_truth=None,
        source_label="tests",
    )

    evaluator.run(config, record)

    assert captured["metadata"] is None


def test_rubric_evaluator_passes_required_context_to_decorated_functions() -> None:
    captured: dict[str, Any] = {}

    @osmosis_rubric
    def fake_rubric(solution_str: str, ground_truth: str, extra_info: dict) -> float:
        captured["solution_str"] = solution_str
        captured["ground_truth"] = ground_truth
        captured["extra_info"] = copy.deepcopy(extra_info)
        return 0.5

    evaluator = RubricEvaluator(evaluate_fn=fake_rubric)

    record = DatasetRecord(
        payload={},
        rubric_id="support_followup",
        conversation_id="conv-002",
        record_id=None,
        solution_str="Assistant reply",
        ground_truth=None,
        original_input="Original request.",
        metadata={"channel": "chat"},
        extra_info={"metadata": {"product": "AirPure"}, "capture_details": True},
        score_min=None,
        score_max=None,
    )

    config = RubricConfig(
        rubric_id="support_followup",
        rubric_text="Judge response quality.",
        model_info={"provider": "openai", "model": "gpt-5-mini"},
        score_min=0.0,
        score_max=1.0,
        system_prompt="Judge fairly.",
        original_input=None,
        ground_truth="Reference answer.",
        source_label="tests",
    )

    result = evaluator.run(config, record)

    assert result == 0.5
    assert captured["solution_str"] == "Assistant reply"
    assert captured["ground_truth"] == "Reference answer."

    extra_info = captured["extra_info"]
    assert extra_info["provider"] == "openai"
    assert extra_info["model"] == "gpt-5-mini"
    assert extra_info["api_key_env"] == "OPENAI_API_KEY"
    assert extra_info["rubric"] == "Judge response quality."
    assert extra_info["score_min"] == pytest.approx(0.0, abs=1e-12)
    assert extra_info["score_max"] == pytest.approx(1.0, abs=1e-12)
    assert extra_info["system_prompt"] == "Judge fairly."
    assert extra_info["original_input"] == "Original request."
    assert extra_info["model_info"]["provider"] == "openai"
    assert extra_info["model_info"]["model"] == "gpt-5-mini"
    assert extra_info["model_info"]["api_key_env"] == "OPENAI_API_KEY"
    assert extra_info["metadata"]["product"] == "AirPure"
    assert extra_info["capture_details"] is True
    assert extra_info["dataset_metadata"] == {"channel": "chat"}
