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
model_path = "openai/gpt-5-mini"
dataset = "multiply"
commit_sha = "deadbeef"

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
required = ["OPENAI_API_KEY", "GITHUB_TOKEN"]
""",
    )

    config = load_eval_submit_config(path)

    assert config.experiment_rollout == "calculator"
    assert config.experiment_entrypoint == "main.py"
    assert config.experiment_dataset == "multiply"
    assert config.experiment_commit_sha == "deadbeef"
    assert config.experiment_model_path == "openai/gpt-5-mini"
    assert config.evaluation_config == {
        "limit": 200,
        "n": 2,
        "batch_size": 3,
        "pass_threshold": 0.75,
        "agent_workflow_timeout_s": 450.0,
        "grader_timeout_s": 150.0,
    }
    assert config.env == {"LOG_LEVEL": "INFO"}
    assert config.secrets == ["OPENAI_API_KEY", "GITHUB_TOKEN"]
    assert config.experiment_config == {
        "rollout": "calculator",
        "entrypoint": "main.py",
        "model_path": "openai/gpt-5-mini",
        "dataset": "multiply",
        "commit_sha": "deadbeef",
    }


def test_load_eval_submit_config_accepts_full_length_commit_sha(
    tmp_path: Path,
) -> None:
    full_sha = "a" * 40
    path = _write_config(
        tmp_path / "eval.toml",
        f"""
[experiment]
rollout = "calculator"
entrypoint = "main.py"
model_path = "openai/gpt-5-mini"
dataset = "multiply"
commit_sha = "{full_sha}"

[secrets]
required = []
""",
    )

    config = load_eval_submit_config(path)

    assert config.experiment_commit_sha == full_sha


@pytest.mark.parametrize(
    "bad_sha",
    [
        "main",  # branch name, not a SHA
        "abc123",  # too short (6 chars)
        "z" * 12,  # non-hex characters
        "a" * 41,  # too long
        "dead beef",  # contains whitespace
    ],
)
def test_load_eval_submit_config_rejects_malformed_commit_sha(
    tmp_path: Path,
    bad_sha: str,
) -> None:
    path = _write_config(
        tmp_path / "eval.toml",
        f"""
[experiment]
rollout = "calculator"
entrypoint = "main.py"
model_path = "openai/gpt-5-mini"
dataset = "multiply"
commit_sha = "{bad_sha}"

[secrets]
required = []
""",
    )

    with pytest.raises(CLIError) as exc_info:
        load_eval_submit_config(path)
    message = str(exc_info.value)
    assert "experiment.commit_sha" in message
    assert "hexadecimal Git commit SHA" in message


def test_load_eval_submit_config_defaults_optional_evaluation_fields(
    tmp_path: Path,
) -> None:
    path = _write_config(
        tmp_path / "eval.toml",
        """
[experiment]
rollout = "calculator"
entrypoint = "main.py"
model_path = "openai/gpt-5-mini"
dataset = "multiply"

[secrets]
required = []
""",
    )

    config = load_eval_submit_config(path)

    assert config.experiment_commit_sha is None
    assert config.experiment_model_path == "openai/gpt-5-mini"
    assert config.evaluation_config == {}
    assert config.advanced_config == {}
    assert config.env == {}
    assert config.secrets == []


def test_secrets_required_field_is_required_for_eval(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path / "eval.toml",
        """
[experiment]
rollout = "calculator"
entrypoint = "main.py"
model_path = "openai/gpt-5-mini"
dataset = "multiply"

[secrets]
""",
    )

    with pytest.raises(CLIError) as exc_info:
        load_eval_submit_config(path)
    assert "Missing 'required' in [secrets]" in str(exc_info.value)


def test_secrets_section_is_required_for_eval(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path / "eval.toml",
        """
[experiment]
rollout = "calculator"
entrypoint = "main.py"
model_path = "openai/gpt-5-mini"
dataset = "multiply"
""",
    )

    with pytest.raises(CLIError) as exc_info:
        load_eval_submit_config(path)
    assert "Missing [secrets] section" in str(exc_info.value)


def test_secrets_rejects_unknown_keys(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path / "eval.toml",
        """
[secrets]
OPENAI_API_KEY = "OPENAI_API_KEY"

[experiment]
rollout = "calculator"
entrypoint = "main.py"
model_path = "openai/gpt-5-mini"
dataset = "multiply"
""",
    )

    with pytest.raises(CLIError) as exc_info:
        load_eval_submit_config(path)
    message = str(exc_info.value)
    assert "only supports the 'required' field" in message
    assert "OPENAI_API_KEY" in message


def test_load_eval_submit_config_advanced_section_preserves_backend_params(
    tmp_path: Path,
) -> None:
    path = _write_config(
        tmp_path / "eval.toml",
        """
[experiment]
rollout = "calculator"
entrypoint = "main.py"
model_path = "openai/gpt-5-mini"
dataset = "multiply"

[advanced]
custom_eval_flag = true
future_knob = "experimental"

[secrets]
required = []
""",
    )

    config = load_eval_submit_config(path)

    assert config.advanced_config == {
        "custom_eval_flag": True,
        "future_knob": "experimental",
    }


def test_load_eval_submit_config_rejects_extra_param_fields_outside_advanced(
    tmp_path: Path,
) -> None:
    path = _write_config(
        tmp_path / "eval.toml",
        """
[experiment]
rollout = "calculator"
entrypoint = "main.py"
model_path = "openai/gpt-5-mini"
dataset = "multiply"

[evaluation]
unknown_eval_knob = 42

[secrets]
required = []
""",
    )

    with pytest.raises(
        CLIError, match=r"evaluation\.unknown_eval_knob: Unrecognized key"
    ):
        load_eval_submit_config(path)


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


def test_load_eval_submit_config_rejects_legacy_llm_section(
    tmp_path: Path,
) -> None:
    path = _write_config(
        tmp_path / "eval.toml",
        """
[experiment]
rollout = "calculator"
entrypoint = "main.py"
model_path = "openai/gpt-5-mini"
dataset = "multiply"

[llm]
model_path = "openai/gpt-5-mini"

[secrets]
required = []
""",
    )

    with pytest.raises(CLIError, match=r"llm: Unrecognized section"):
        load_eval_submit_config(path)


def test_load_eval_submit_config_rejects_invalid_env_keys(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path / "eval.toml",
        """
[experiment]
rollout = "calculator"
entrypoint = "main.py"
model_path = "openai/gpt-5-mini"
dataset = "multiply"

[env]
_OSMOSIS_INTERNAL = "bad"

[secrets]
required = []
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
model_path = "openai/gpt-5-mini"
dataset = "multiply"

[evaluation]
limit = "many"
pass_threshold = "strict"

[secrets]
required = []
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
model_path = "openai/gpt-5-mini"
dataset = "multiply"

[secrets]
required = []
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
model_path = "openai/gpt-5-mini"
dataset = "multiply"

[secrets]
required = []
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
