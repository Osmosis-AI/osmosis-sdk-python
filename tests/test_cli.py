import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from osmosis_ai import cli


def test_preview_yaml_single_config(tmp_path, capsys):
    yaml_content = """rubric: >
  Score how well the assistant resolves a smart appliance troubleshooting request.
score_min: 0.0
score_max: 1.0
ground_truth: >
  The assistant should verify the warranty and gather error diagnostics.
default_model_info:
  provider: openai
  model: gpt-5-mini
"""
    path = tmp_path / "config.yaml"
    path.write_text(yaml_content, encoding="utf-8")

    exit_code = cli.main(["preview", "--path", str(path)])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Loaded 1 rubric config(s)" in out
    assert "Rubric config #1" in out
    assert "default_model_info" in out


def test_preview_yaml_multiple_configs(tmp_path, capsys):
    yaml_content = """- rubric: First rubric description.
  score_min: 0
  score_max: 1
- rubric: Second rubric description.
  score_min: 1
  score_max: 5
"""
    path = tmp_path / "multi.yaml"
    path.write_text(yaml_content, encoding="utf-8")

    exit_code = cli.main(["preview", "--path", str(path)])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Loaded 2 rubric config(s)" in out
    assert "Rubric config #2" in out
    assert "Second rubric description." in out


def test_preview_jsonl(tmp_path, capsys):
    record_one = {
        "conversation_id": "ticket-001",
        "rubric_id": "support_followup",
        "original_input": "My AirPure X2 purifier keeps flashing a red light and refuses to start.",
        "solution_str": "I'm sorry the purifier is down. Could you share the order number so I can confirm the warranty?",
        "ground_truth": "Assistant verifies warranty information, gathers diagnostics, and suggests safe troubleshooting steps.",
        "metadata": {"customer_tier": "gold", "language": "en"},
    }
    record_two = {
        "conversation_id": "ticket-047",
        "rubric_id": "policy_grounding",
        "original_input": "Can I switch from Essential to Premium mid-cycle and still keep my existing discounts?",
        "solution_str": "Absolutely—you can upgrade anytime and the discounts roll over automatically.",
        "ground_truth": "Assistant cites the subscription policy, clarifies prorated billing, and avoids promising unauthorized discounts.",
        "metadata": {"policy_version": "2024-05-01"},
    }
    jsonl_content = "\n".join([json.dumps(record_one, ensure_ascii=False), json.dumps(record_two, ensure_ascii=False)])
    path = tmp_path / "data.jsonl"
    path.write_text(jsonl_content, encoding="utf-8")

    exit_code = cli.main(["preview", "--path", str(path)])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Loaded 2 JSONL record(s)" in out
    assert "JSONL record #1" in out
    assert '"rubric_id": "support_followup"' in out
    assert '"ground_truth": "Assistant verifies warranty information, gathers diagnostics, and suggests safe troubleshooting steps."' in out


def test_eval_command_output_json(tmp_path, monkeypatch, capsys):
    config_content = """rubrics:
  - id: support_followup
    rubric: Score the assistant response quality.
    model_info:
      provider: openai
      model: gpt-5-mini
"""
    config_path = tmp_path / "rubric_configs.yaml"
    config_path.write_text(config_content, encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    record_payload = {
        "rubric_id": "support_followup",
        "conversation_id": "conv-001",
        "original_input": "Help me with my device.",
        "solution_str": "Sure, let's troubleshoot.",
        "ground_truth": "Assistant verifies warranty information, gathers diagnostics, and suggests safe troubleshooting steps.",
        "metadata": {"language": "en"},
    }
    jsonl_content = json.dumps(record_payload, ensure_ascii=False) + "\n"
    data_path = tmp_path / "records.jsonl"
    data_path.write_text(jsonl_content, encoding="utf-8")

    class FakeEvaluator:
        def __init__(self):
            self.calls = 0

        def run(self, config, record):
            self.calls += 1
            score = 0.4 + 0.1 * self.calls
            return {
                "score": score,
                "explanation": f"explanation-{self.calls}",
                "raw": {
                    "call": self.calls,
                    "conversation": record.conversation_id or f"record-{self.calls}",
                },
            }

    from osmosis_ai.cli_commands import EvalCommand

    fake_evaluator = FakeEvaluator()
    monkeypatch.setattr("osmosis_ai.cli_services.engine.RubricEvaluator", lambda: fake_evaluator)
    monkeypatch.setattr(EvalCommand, "_generate_output_identifier", staticmethod(lambda: "1700000000"))

    output_stem = tmp_path / "results"
    exit_code = cli.main(
        [
            "eval",
            "--rubric",
            "support_followup",
            "--data",
            str(data_path),
            "--config",
            str(config_path),
            "--number",
            "2",
            "--output",
            str(output_stem),
        ]
    )

    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Wrote evaluation results" in out
    target_file = output_stem / "rubric_eval_result_1700000000.json"
    assert str(target_file) in out

    generated_path = target_file
    assert generated_path.exists()

    payload = json.loads(generated_path.read_text(encoding="utf-8"))
    keys = list(payload.keys())
    assert keys.index("overall_statistics") < keys.index("records")
    assert payload["output_identifier"] == "1700000000"
    assert payload["rubric_id"] == "support_followup"
    assert payload["number"] == 2
    assert payload["overall_statistics"]["average"] == pytest.approx(0.55, rel=1e-6)
    assert payload["overall_statistics"]["variance"] == pytest.approx(0.0025, rel=1e-6)

    assert len(payload["records"]) == 1
    record_payload = payload["records"][0]
    assert record_payload["id"] == "conv-001"
    assert record_payload["conversation_id"] == "conv-001"
    assert record_payload["input_record"]["ground_truth"] == (
        "Assistant verifies warranty information, gathers diagnostics, and suggests safe troubleshooting steps."
    )
    assert record_payload["input_record"]["metadata"] == {"language": "en"}
    assert len(record_payload["runs"]) == 2
    assert record_payload["runs"][0]["raw"]["call"] == 1
    assert "started_at" in record_payload["runs"][0]
    assert "duration_seconds" in record_payload["runs"][0]
    assert record_payload["statistics"]["max"] == pytest.approx(0.6, rel=1e-6)


def test_eval_command_missing_api_key_fails_fast(tmp_path, monkeypatch, capsys):
    env_name = "OSMOSIS_EVAL_MISSING_API_KEY_TEST"
    monkeypatch.delenv(env_name, raising=False)

    config_content = f"""rubrics:
  - id: support_followup
    rubric: Score the assistant response quality.
    model_info:
      provider: openai
      model: gpt-5-mini
      api_key_env: {env_name}
"""
    config_path = tmp_path / "rubric_configs.yaml"
    config_path.write_text(config_content, encoding="utf-8")

    jsonl_content = (
        '{"rubric_id": "support_followup", "conversation_id": "conv-001", '
        '"original_input": "Help me with my device.", '
        '"solution_str": "Sure, let\'s troubleshoot."}\n'
    )
    data_path = tmp_path / "records.jsonl"
    data_path.write_text(jsonl_content, encoding="utf-8")

    exit_code = cli.main(
        [
            "eval",
            "--rubric",
            "support_followup",
            "--data",
            str(data_path),
            "--config",
            str(config_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert f"Environment variable '{env_name}' is not set." in captured.err
    assert "Wrote evaluation results" not in captured.out


def test_eval_command_with_baseline(tmp_path, monkeypatch, capsys):
    config_content = """rubrics:
  - id: support_followup
    rubric: Score the assistant response quality.
    model_info:
      provider: openai
      model: gpt-5-mini
"""
    config_path = tmp_path / "rubric_configs.yaml"
    config_path.write_text(config_content, encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    jsonl_content = (
        '{"rubric_id": "support_followup", "conversation_id": "conv-001", '
        '"original_input": "Help me with my device.", '
        '"solution_str": "Sure, let\'s troubleshoot."}\n'
    )
    data_path = tmp_path / "records.jsonl"
    data_path.write_text(jsonl_content, encoding="utf-8")

    baseline_path = tmp_path / "baseline.json"
    baseline_payload = {
        "overall_statistics": {
            "average": 0.5,
            "variance": 0.0025,
            "stdev": 0.05,
            "min": 0.4,
            "max": 0.6,
        }
    }
    baseline_path.write_text(json.dumps(baseline_payload), encoding="utf-8")

    class FakeEvaluator:
        def __init__(self):
            self.calls = 0

        def run(self, config, record):
            self.calls += 1
            score = 0.5 + 0.1 * self.calls
            return {
                "score": score,
                "explanation": f"baseline-{self.calls}",
                "raw": {"call": self.calls},
            }

    from osmosis_ai.cli_commands import EvalCommand

    fake_evaluator = FakeEvaluator()
    monkeypatch.setattr("osmosis_ai.cli_services.engine.RubricEvaluator", lambda: fake_evaluator)
    monkeypatch.setattr(EvalCommand, "_generate_output_identifier", staticmethod(lambda: "1700000001"))

    output_dir = tmp_path / "baseline_results"
    exit_code = cli.main(
        [
            "eval",
            "--rubric",
            "support_followup",
            "--data",
            str(data_path),
            "--config",
            str(config_path),
            "--number",
            "2",
            "--output",
            str(output_dir),
            "--baseline",
            str(baseline_path),
        ]
    )

    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Baseline comparison" in out
    assert "delta=+0.1500" in out

    target_file = output_dir / "rubric_eval_result_1700000001.json"
    assert target_file.exists()

    payload = json.loads(target_file.read_text(encoding="utf-8"))
    comparison = payload["baseline_comparison"]
    assert comparison["source_path"] == str(baseline_path)
    assert comparison["delta_statistics"]["average"] == pytest.approx(0.15, rel=1e-6)
    assert comparison["delta_statistics"]["variance"] == pytest.approx(0.0, abs=1e-12)
    assert payload["overall_statistics"]["average"] == pytest.approx(0.65, rel=1e-6)


def test_eval_command_output_json_custom_file_path(tmp_path, monkeypatch, capsys):
    config_content = """rubrics:
  - id: support_followup
    rubric: Score the assistant response quality.
    model_info:
      provider: openai
      model: gpt-5-mini
"""
    config_path = tmp_path / "rubric_configs.yaml"
    config_path.write_text(config_content, encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    jsonl_content = (
        '{"rubric_id": "support_followup", "conversation_id": "conv-001", '
        '"original_input": "Help me with my device.", '
        '"solution_str": "Sure, let\'s troubleshoot."}\n'
    )
    data_path = tmp_path / "records.jsonl"
    data_path.write_text(jsonl_content, encoding="utf-8")

    class FakeEvaluator:
        def __init__(self):
            self.calls = 0

        def run(self, config, record):
            self.calls += 1
            score = 0.5 + 0.1 * self.calls
            return {
                "score": score,
                "explanation": f"custom-{self.calls}",
                "raw": {"call": self.calls},
            }

    monkeypatch.setattr("osmosis_ai.cli_services.engine.RubricEvaluator", lambda: FakeEvaluator())

    output_path = tmp_path / "reports" / "custom_output.txt"
    exit_code = cli.main(
        [
            "eval",
            "--rubric",
            "support_followup",
            "--data",
            str(data_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ]
    )

    out = capsys.readouterr().out

    assert exit_code == 0
    assert output_path.exists()
    assert str(output_path) in out

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    keys = list(payload.keys())
    assert keys.index("overall_statistics") < keys.index("records")
    assert "output_identifier" not in payload
    assert payload["records"][0]["id"] == "conv-001"


def test_preview_missing_file(tmp_path, capsys):
    missing_path = tmp_path / "missing.yaml"
    exit_code = cli.main(["preview", "--path", str(missing_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert f"Path '{missing_path}' does not exist." in captured.err


def test_preview_path_is_directory(tmp_path, capsys):
    exit_code = cli.main(["preview", "--path", str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert f"Expected a file path but got directory '{tmp_path}'." in captured.err


def test_preview_unsupported_extension(tmp_path, capsys):
    bad_path = tmp_path / "config.txt"
    bad_path.write_text("rubric: test", encoding="utf-8")

    exit_code = cli.main(["preview", "--path", str(bad_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Unsupported file extension '.txt'." in captured.err


def test_main_without_subcommand_shows_help(capsys):
    exit_code = cli.main([])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "usage: osmosis" in captured.out


def test_eval_command_empty_rubric_id_rejected(tmp_path, capsys):
    data_path = tmp_path / "records.jsonl"

    exit_code = cli.main(
        [
            "eval",
            "--rubric",
            "   ",
            "--data",
            str(data_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Rubric identifier cannot be empty." in captured.err


def test_eval_command_number_must_be_positive(tmp_path, capsys):
    data_path = tmp_path / "records.jsonl"

    exit_code = cli.main(
        [
            "eval",
            "--rubric",
            "support_followup",
            "--data",
            str(data_path),
            "--number",
            "-1",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Number of runs must be a positive integer." in captured.err


def test_eval_command_number_zero_rejected(tmp_path, capsys):
    data_path = tmp_path / "records.jsonl"

    exit_code = cli.main(
        [
            "eval",
            "--rubric",
            "support_followup",
            "--data",
            str(data_path),
            "--number",
            "0",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Number of runs must be a positive integer." in captured.err


def test_eval_command_missing_data_path(tmp_path, capsys):
    missing_path = tmp_path / "missing.jsonl"

    exit_code = cli.main(
        [
            "eval",
            "--rubric",
            "support_followup",
            "--data",
            str(missing_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert f"Data path '{missing_path}' does not exist." in captured.err


def test_eval_command_output_path_directory_with_suffix(tmp_path, capsys):
    config_content = """rubrics:
  - id: support_followup
    rubric: Validate assistant response quality.
    model_info:
      provider: openai
      model: gpt-5-mini
      api_key: dummy
"""
    config_path = tmp_path / "rubric_configs.yaml"
    config_path.write_text(config_content, encoding="utf-8")

    record = {
        "rubric_id": "support_followup",
        "conversation_id": "conv-001",
        "original_input": "Hi",
        "solution_str": "Hello",
    }
    data_path = tmp_path / "records.jsonl"
    data_path.write_text(json.dumps(record), encoding="utf-8")

    output_path = tmp_path / "report.json"
    output_path.mkdir()

    exit_code = cli.main(
        [
            "eval",
            "--rubric",
            "support_followup",
            "--data",
            str(data_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert f"Output path '{output_path}' is a directory." in captured.err
