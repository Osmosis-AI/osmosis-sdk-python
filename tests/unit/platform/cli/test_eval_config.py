from __future__ import annotations

from pathlib import Path

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.cli.eval_config import (
    load_eval_submit_config,
    validate_eval_submit_context_paths,
)


def _write_config(path: Path, body: str) -> Path:
    path.write_text(body.strip() + "\n", encoding="utf-8")
    return path


def test_load_eval_submit_config_accepts_new_cloud_schema(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path / "eval.toml",
        """
[experiment]
rollout = "calculator"
entrypoint = "main.py"
dataset = "multiply"
commit_sha = "deadbeef"

[llm]
model_path = "openai/gpt-5-mini"
base_url = "https://api.openai.com/v1"

[evaluation]
limit = 200
n = 2
batch_size = 3
pass_threshold = 0.75
agent_workflow_timeout_s = 450
grader_timeout_s = 150

[env]
LOG_LEVEL = "INFO"

[secrets]
OPENAI_API_KEY = "openai-api-key"
""",
    )

    config = load_eval_submit_config(path)

    assert config.experiment_rollout == "calculator"
    assert config.experiment_entrypoint == "main.py"
    assert config.experiment_dataset == "multiply"
    assert config.experiment_commit_sha == "deadbeef"
    assert config.llm_model_path == "openai/gpt-5-mini"
    assert config.llm_base_url == "https://api.openai.com/v1"
    assert config.evaluation_config == {
        "limit": 200,
        "n": 2,
        "batch_size": 3,
        "pass_threshold": 0.75,
        "agent_workflow_timeout_s": 450.0,
        "grader_timeout_s": 150.0,
    }
    assert config.rollout_env == {"LOG_LEVEL": "INFO"}
    assert config.rollout_secret_refs == {"OPENAI_API_KEY": "openai-api-key"}
    assert config.eval_config == {
        "experiment": {
            "rollout": "calculator",
            "entrypoint": "main.py",
            "dataset": "multiply",
            "commit_sha": "deadbeef",
        },
        "llm": {
            "model_path": "openai/gpt-5-mini",
            "base_url": "https://api.openai.com/v1",
        },
        "evaluation": {
            "limit": 200,
            "n": 2,
            "batch_size": 3,
            "pass_threshold": 0.75,
            "agent_workflow_timeout_s": 450.0,
            "grader_timeout_s": 150.0,
        },
    }


def test_load_eval_submit_config_defaults_optional_evaluation_fields(
    tmp_path: Path,
) -> None:
    path = _write_config(
        tmp_path / "eval.toml",
        """
[experiment]
rollout = "calculator"
entrypoint = "main.py"
dataset = "multiply"

[llm]
model_path = "openai/gpt-5-mini"
""",
    )

    config = load_eval_submit_config(path)

    assert config.experiment_commit_sha is None
    assert config.evaluation_config == {}
    assert "evaluation" not in config.eval_config
    assert config.rollout_env == {}
    assert config.rollout_secret_refs == {}


def test_load_eval_submit_config_rejects_old_local_eval_schema(
    tmp_path: Path,
) -> None:
    path = _write_config(
        tmp_path / "eval.toml",
        """
[eval]
rollout = "calculator"
entrypoint = "main.py"
dataset = "data/multiply.jsonl"

[llm]
model = "openai/gpt-5-mini"
""",
    )

    with pytest.raises(CLIError, match=r"Missing \[experiment\] section"):
        load_eval_submit_config(path)


def test_load_eval_submit_config_rejects_invalid_env_keys(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path / "eval.toml",
        """
[experiment]
rollout = "calculator"
entrypoint = "main.py"
dataset = "multiply"

[llm]
model_path = "openai/gpt-5-mini"

[env]
_OSMOSIS_INTERNAL = "bad"
""",
    )

    with pytest.raises(CLIError, match="reserved by the platform"):
        load_eval_submit_config(path)


def test_load_eval_submit_config_defers_evaluation_value_validation_to_backend(
    tmp_path: Path,
) -> None:
    path = _write_config(
        tmp_path / "eval.toml",
        """
[experiment]
rollout = "calculator"
entrypoint = "main.py"
dataset = "multiply"

[llm]
model_path = "openai/gpt-5-mini"

[evaluation]
limit = "many"
pass_threshold = "strict"
""",
    )

    config = load_eval_submit_config(path)

    assert config.evaluation_config == {
        "limit": "many",
        "pass_threshold": "strict",
    }


def test_validate_eval_submit_context_paths_rejects_entrypoint_escape(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path / "eval.toml",
        """
[experiment]
rollout = "calculator"
entrypoint = "../main.py"
dataset = "multiply"

[llm]
model_path = "openai/gpt-5-mini"
""",
    )
    config = load_eval_submit_config(config_path)

    with pytest.raises(CLIError, match="entrypoint must resolve under"):
        validate_eval_submit_context_paths(config, tmp_path)
