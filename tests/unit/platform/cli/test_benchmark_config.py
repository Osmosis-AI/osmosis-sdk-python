from __future__ import annotations

from pathlib import Path

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.cli.benchmark_config import load_benchmark_submit_config


def _write_config(path: Path, body: str) -> Path:
    path.write_text(body.strip() + "\n", encoding="utf-8")
    return path


def test_load_benchmark_submit_config_accepts_provider_and_endpoint_agents(
    tmp_path: Path,
) -> None:
    path = _write_config(
        tmp_path / "benchmark.toml",
        """
[experiment]
benchmark = "Terminal-Bench 2.1"

[tasks]
task_names = ["git-multibranch"]

[[agents]]
harness = "codex"

[agents.model]
type = "provider"
model = "openai/gpt-5"
api_key_secret = "OPENAI_API_KEY"

[agents.env]
AGENT_MODE = "strict"

[[agents]]
harness = "claude-code"

[agents.model]
type = "endpoint"
base_url = "https://models.example.com/v1"
model = "custom-model"
api_key_secret = "CUSTOM_API_KEY"

[execution]
attempts_per_task = 2
max_concurrent_attempts = 8

[env]
LOG_LEVEL = "info"
""",
    )

    config = load_benchmark_submit_config(path)

    assert config.experiment_config == {"benchmark": "Terminal-Bench 2.1"}
    assert config.tasks_config == {"task_names": ["git-multibranch"]}
    assert config.execution_config == {
        "attempts_per_task": 2,
        "max_concurrent_attempts": 8,
    }
    assert config.env == {"LOG_LEVEL": "info"}
    assert config.required_secrets == ["OPENAI_API_KEY", "CUSTOM_API_KEY"]
    assert config.agents_config[0] == {
        "harness": "codex",
        "model": {
            "type": "provider",
            "model": "openai/gpt-5",
            "api_key_secret": "OPENAI_API_KEY",
        },
        "env": {"AGENT_MODE": "strict"},
    }


def test_load_benchmark_submit_config_accepts_hosted_model(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path / "benchmark.toml",
        """
[experiment]
benchmark = "Terminal-Bench 2.1"

[[agents]]
harness = "codex"

[agents.model]
type = "hosted"
base_model = "Qwen/Qwen3-8B"
checkpoint_name = "terminal-agent"
""",
    )

    config = load_benchmark_submit_config(path)

    assert config.tasks_config == {}
    assert config.execution_config == {}
    assert config.required_secrets == []


def test_load_benchmark_submit_config_rejects_unknown_section(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path / "benchmark.toml",
        """
[experiment]
benchmark = "Terminal-Bench 2.1"

[[agents]]
[agents.model]
type = "provider"
model = "openai/gpt-5"
api_key_secret = "OPENAI_API_KEY"

[harbor]
n_attempts = 3
""",
    )

    with pytest.raises(CLIError) as exc_info:
        load_benchmark_submit_config(path)

    assert "harbor: Unrecognized key" in str(exc_info.value)


def test_load_benchmark_submit_config_rejects_secret_env_collision(
    tmp_path: Path,
) -> None:
    path = _write_config(
        tmp_path / "benchmark.toml",
        """
[experiment]
benchmark = "Terminal-Bench 2.1"

[[agents]]
[agents.model]
type = "provider"
model = "openai/gpt-5"
api_key_secret = "OPENAI_API_KEY"

[env]
OPENAI_API_KEY = "literal-secret"
""",
    )

    with pytest.raises(CLIError, match=r"agent 1's api_key_secret"):
        load_benchmark_submit_config(path)


def test_load_benchmark_submit_config_allows_cross_agent_secret_env_names(
    tmp_path: Path,
) -> None:
    """Agent 2 may use agent 1's secret name as a literal env var.

    The platform injects each secret only into the env of the agent that
    references it, so this configuration is valid server-side.
    """
    path = _write_config(
        tmp_path / "benchmark.toml",
        """
[experiment]
benchmark = "Terminal-Bench 2.1"

[[agents]]
[agents.model]
type = "provider"
model = "openai/gpt-5"
api_key_secret = "OPENAI_API_KEY"

[[agents]]
[agents.model]
type = "endpoint"
base_url = "https://models.example.com/v1"
model = "custom-model"
api_key_secret = "CUSTOM_API_KEY"

[agents.env]
OPENAI_API_KEY = "placeholder-for-harness"
""",
    )

    config = load_benchmark_submit_config(path)

    assert config.required_secrets == ["OPENAI_API_KEY", "CUSTOM_API_KEY"]
    assert config.agents[1].env == {"OPENAI_API_KEY": "placeholder-for-harness"}


def test_load_benchmark_submit_config_rejects_judge_secret_env_collision(
    tmp_path: Path,
) -> None:
    path = _write_config(
        tmp_path / "benchmark.toml",
        """
[experiment]
benchmark = "Terminal-Bench 2.1"

[[agents]]
[agents.model]
type = "provider"
model = "openai/gpt-5"
api_key_secret = "OPENAI_API_KEY"

[agents.env]
JUDGE_KEY = "literal"

[execution]
judge_model = "openai/gpt-5"
judge_api_key_secret = "JUDGE_KEY"
""",
    )

    with pytest.raises(CLIError, match=r"judge_api_key_secret"):
        load_benchmark_submit_config(path)
