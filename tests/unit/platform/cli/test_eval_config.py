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
    assert config.env == {"LOG_LEVEL": "INFO"}
    assert config.secrets == {"OPENAI_API_KEY": "openai-api-key"}
    assert config.experiment_config == {
        "rollout": "calculator",
        "entrypoint": "main.py",
        "dataset": "multiply",
        "commit_sha": "deadbeef",
    }
    assert config.llm_config == {
        "model_path": "openai/gpt-5-mini",
        "base_url": "https://api.openai.com/v1",
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
    assert config.llm_base_url is None
    assert config.evaluation_config == {}
    assert config.llm_config == {"model_path": "openai/gpt-5-mini"}
    assert config.env == {}
    assert config.secrets == {}


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


def _write_eval_config_with_rollout(path: Path, rollout: str) -> Path:
    return _write_config(
        path,
        f"""
[experiment]
rollout = "{rollout}"
entrypoint = "main.py"
dataset = "multiply"

[llm]
model_path = "openai/gpt-5-mini"
""",
    )


def test_validate_eval_submit_context_paths_rejects_rollout_with_separator(
    tmp_path: Path,
) -> None:
    config_path = _write_eval_config_with_rollout(tmp_path / "eval.toml", "../outside")
    config = load_eval_submit_config(config_path)

    with pytest.raises(CLIError, match="single-segment name"):
        validate_eval_submit_context_paths(config, tmp_path)


def test_validate_eval_submit_context_paths_rejects_rollout_with_forward_slash(
    tmp_path: Path,
) -> None:
    config_path = _write_eval_config_with_rollout(tmp_path / "eval.toml", "foo/bar")
    config = load_eval_submit_config(config_path)

    with pytest.raises(CLIError, match="single-segment name"):
        validate_eval_submit_context_paths(config, tmp_path)


@pytest.mark.parametrize("rollout", ["", ".", ".."])
def test_validate_eval_submit_context_paths_rejects_non_logical_rollout_name(
    tmp_path: Path,
    rollout: str,
) -> None:
    config_path = _write_eval_config_with_rollout(tmp_path / "eval.toml", rollout)
    config = load_eval_submit_config(config_path)

    with pytest.raises(CLIError, match="not a valid rollout name"):
        validate_eval_submit_context_paths(config, tmp_path)


def test_validate_eval_submit_context_paths_rejects_absolute_rollout(
    tmp_path: Path,
) -> None:
    absolute_rollout = str(tmp_path / "outside")
    config_path = _write_eval_config_with_rollout(
        tmp_path / "eval.toml",
        absolute_rollout,
    )
    config = load_eval_submit_config(config_path)

    with pytest.raises(CLIError, match="logical rollout name"):
        validate_eval_submit_context_paths(config, tmp_path)
