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
benchmark = "DeepSWE"

[tasks]
task_names = ["abs-module-cache-flags"]

[[agents]]
harness = "cursor-cli"
harness_api_key_secret = "CURSOR_API_KEY"

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

    assert config.experiment_config == {"benchmark": "DeepSWE"}
    assert config.tasks_config == {"task_names": ["abs-module-cache-flags"]}
    assert config.execution_config == {
        "attempts_per_task": 2,
        "max_concurrent_attempts": 8,
    }
    assert config.env == {"LOG_LEVEL": "info"}
    assert config.required_secrets == [
        "OPENAI_API_KEY",
        "CUSTOM_API_KEY",
        "CURSOR_API_KEY",
    ]
    assert config.agents_config[0] == {
        "harness": "cursor-cli",
        "harness_api_key_secret": "CURSOR_API_KEY",
        "model": {
            "type": "provider",
            "model": "openai/gpt-5",
            "api_key_secret": "OPENAI_API_KEY",
        },
        "env": {"AGENT_MODE": "strict"},
    }


@pytest.mark.parametrize(
    "header_name",
    ["Authorization", "authorization", "AUTHORIZATION", "aUtHoRiZaTiOn"],
)
def test_load_benchmark_submit_config_rejects_endpoint_authorization_header(
    tmp_path: Path,
    header_name: str,
) -> None:
    path = _write_config(
        tmp_path / "benchmark.toml",
        f"""
[experiment]
benchmark = "DeepSWE"

[[agents]]
[agents.model]
type = "endpoint"
base_url = "https://models.example.com/v1"
model = "custom-model"
api_key_secret = "CUSTOM_API_KEY"

[agents.model.extra_headers]
"{header_name}" = "Bearer literal-token"
""",
    )

    with pytest.raises(
        CLIError,
        match=r"use api_key_secret for endpoint authentication",
    ):
        load_benchmark_submit_config(path)


def test_load_benchmark_submit_config_accepts_non_authorization_endpoint_header(
    tmp_path: Path,
) -> None:
    path = _write_config(
        tmp_path / "benchmark.toml",
        """
[experiment]
benchmark = "DeepSWE"

[[agents]]
[agents.model]
type = "endpoint"
base_url = "https://models.example.com/v1"
model = "custom-model"
api_key_secret = "CUSTOM_API_KEY"

[agents.model.extra_headers]
"X-Request-ID" = "benchmark-run"
""",
    )

    config = load_benchmark_submit_config(path)

    assert config.agents_config[0]["model"]["extra_headers"] == {
        "X-Request-ID": "benchmark-run"
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
lora_model_name = "terminal-agent"
""",
    )

    config = load_benchmark_submit_config(path)

    assert config.tasks_config == {}
    assert config.execution_config == {}
    assert config.required_secrets == []


def test_load_benchmark_submit_config_accepts_hle_parity_with_explicit_filters(
    tmp_path: Path,
) -> None:
    """The platform gives task_set precedence over explicit filters."""
    path = _write_config(
        tmp_path / "benchmark.toml",
        """
[experiment]
benchmark = "HLE"

[tasks]
task_set = "parity"
task_names = ["hle__sample"]
categories = ["Math"]

[[agents]]
harness = "codex"

[agents.model]
type = "hosted"
base_model = "Qwen/Qwen3-8B"
lora_model_name = "hle-agent"

[execution]
judge_model = "openai/gpt-5"
judge_api_key_secret = "OPENAI_API_KEY"
""",
    )

    config = load_benchmark_submit_config(path)

    assert config.tasks_config == {
        "categories": ["Math"],
        "task_names": ["hle__sample"],
        "task_set": "parity",
    }
    assert config.required_secrets == ["OPENAI_API_KEY"]


def test_load_benchmark_submit_config_rejects_unknown_task_set(
    tmp_path: Path,
) -> None:
    """The route expands an unrecognized task_set to every task rather than
    failing, so the typo has to be caught here."""
    path = _write_config(
        tmp_path / "benchmark.toml",
        """
[experiment]
benchmark = "HLE"

[tasks]
task_set = "full"

[[agents]]
harness = "codex"

[agents.model]
type = "hosted"
base_model = "Qwen/Qwen3-8B"
lora_model_name = "hle-agent"
""",
    )

    with pytest.raises(CLIError, match=r"tasks.task_set"):
        load_benchmark_submit_config(path)


@pytest.mark.parametrize(
    "tasks_body, field_name",
    [
        ('task_names = "hle__sample"', "task_names"),
        ('task_names = [""]', "task_names"),
        ('task_names = ["   "]', "task_names"),
        ('categories = "Math"', "categories"),
        ('categories = [""]', "categories"),
        ('categories = ["   "]', "categories"),
    ],
)
def test_load_benchmark_submit_config_rejects_invalid_explicit_task_filters(
    tmp_path: Path,
    tasks_body: str,
    field_name: str,
) -> None:
    path = _write_config(
        tmp_path / "benchmark.toml",
        f"""
[experiment]
benchmark = "HLE"

[tasks]
{tasks_body}

[[agents]]
harness = "codex"

[agents.model]
type = "hosted"
base_model = "Qwen/Qwen3-8B"
lora_model_name = "hle-agent"
""",
    )

    with pytest.raises(CLIError, match=rf"tasks.{field_name}"):
        load_benchmark_submit_config(path)


def test_load_benchmark_submit_config_rejects_unknown_section(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path / "benchmark.toml",
        """
[experiment]
benchmark = "DeepSWE"

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


@pytest.mark.parametrize("env_key", ["bad-name", "_OSMOSIS_INTERNAL"])
def test_load_benchmark_submit_config_labels_invalid_agent_env(
    tmp_path: Path,
    env_key: str,
) -> None:
    path = _write_config(
        tmp_path / "benchmark.toml",
        f"""
[experiment]
benchmark = "DeepSWE"

[[agents]]
[agents.model]
type = "provider"
model = "openai/gpt-5"
api_key_secret = "OPENAI_API_KEY"

[agents.env]
{env_key} = "value"
""",
    )

    with pytest.raises(CLIError) as exc_info:
        load_benchmark_submit_config(path)

    assert "agent 1's [agents.env]" in str(exc_info.value)


def test_load_benchmark_submit_config_rejects_secret_env_collision(
    tmp_path: Path,
) -> None:
    path = _write_config(
        tmp_path / "benchmark.toml",
        """
[experiment]
benchmark = "DeepSWE"

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
benchmark = "DeepSWE"

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
benchmark = "DeepSWE"

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


def test_load_benchmark_submit_config_rejects_non_string_judge_secret(
    tmp_path: Path,
) -> None:
    path = _write_config(
        tmp_path / "benchmark.toml",
        """
[experiment]
benchmark = "DeepSWE"

[[agents]]
[agents.model]
type = "hosted"
base_model = "Qwen/Qwen3-8B"
lora_model_name = "deep-swe-agent"

[execution]
judge_api_key_secret = 42
""",
    )

    with pytest.raises(CLIError, match=r"execution.judge_api_key_secret"):
        load_benchmark_submit_config(path)


@pytest.mark.parametrize(
    "secret_name",
    [
        "DAYTONA_API_KEY",
        "DAYTONA_API_URL",
        "SKYPILOT_SERVICE_ACCOUNT_TOKEN",
        "SKYPILOT_API_SERVER_ENDPOINT",
    ],
)
@pytest.mark.parametrize("model_type", ["provider", "endpoint"])
def test_load_benchmark_submit_config_rejects_reserved_model_secret_names(
    tmp_path: Path,
    secret_name: str,
    model_type: str,
) -> None:
    model_fields = (
        'model = "openai/gpt-5"'
        if model_type == "provider"
        else 'base_url = "https://models.example.com/v1"\nmodel = "custom-model"'
    )
    path = _write_config(
        tmp_path / "benchmark.toml",
        f"""
[experiment]
benchmark = "Terminal-Bench 2.1"

[[agents]]
harness = "codex"

[agents.model]
type = "{model_type}"
{model_fields}
api_key_secret = "{secret_name}"
""",
    )

    with pytest.raises(CLIError) as exc_info:
        load_benchmark_submit_config(path)

    assert secret_name in str(exc_info.value)
    assert "reserved by the benchmark runner" in str(exc_info.value)


@pytest.mark.parametrize("benchmark", ["Terminal-Bench 2.1", "HLE"])
@pytest.mark.parametrize(
    "env_section",
    [
        '[env]\nHF_TOKEN = "agent-token"',
        '[agents.env]\nHF_TOKEN = "agent-token"',
    ],
)
def test_load_benchmark_submit_config_accepts_an_agents_own_hf_token(
    tmp_path: Path,
    benchmark: str,
    env_section: str,
) -> None:
    path = _write_config(
        tmp_path / "benchmark.toml",
        f"""
[experiment]
benchmark = "{benchmark}"

[[agents]]
harness = "codex"

[agents.model]
type = "hosted"
base_model = "Qwen/Qwen3-8B"
lora_model_name = "benchmark-agent"

{env_section}
""",
    )

    config = load_benchmark_submit_config(path)

    assert {**config.env, **config.agents[0].env}["HF_TOKEN"] == "agent-token"


def test_load_benchmark_submit_config_validates_harness_secret_name(
    tmp_path: Path,
) -> None:
    path = _write_config(
        tmp_path / "benchmark.toml",
        """
[experiment]
benchmark = "DeepSWE"

[[agents]]
harness = "cursor-cli"
harness_api_key_secret = "invalid-secret"

[agents.model]
type = "provider"
model = "openai/gpt-5"
api_key_secret = "OPENAI_API_KEY"
""",
    )

    with pytest.raises(CLIError, match=r"Invalid secret name 'invalid-secret'"):
        load_benchmark_submit_config(path)


@pytest.mark.parametrize(
    "harness, destination_env",
    [
        ("cursor-cli", "CURSOR_API_KEY"),
    ],
)
def test_load_benchmark_submit_config_accepts_pinned_harness_secret_name(
    tmp_path: Path,
    harness: str,
    destination_env: str,
) -> None:
    path = _write_config(
        tmp_path / "benchmark.toml",
        f"""
[experiment]
benchmark = "DeepSWE"

[[agents]]
harness = "{harness}"
harness_api_key_secret = "{destination_env}"

[agents.model]
type = "provider"
model = "openai/gpt-5"
api_key_secret = "OPENAI_API_KEY"
""",
    )

    config = load_benchmark_submit_config(path)

    assert config.required_secrets == ["OPENAI_API_KEY", destination_env]


@pytest.mark.parametrize(
    "harness, destination_env",
    [
        ("cursor-cli", "CURSOR_API_KEY"),
    ],
)
def test_load_benchmark_submit_config_rejects_unpinned_harness_secret_name(
    tmp_path: Path,
    harness: str,
    destination_env: str,
) -> None:
    """The record must be named for the variable the harness actually reads."""
    path = _write_config(
        tmp_path / "benchmark.toml",
        f"""
[experiment]
benchmark = "DeepSWE"

[[agents]]
harness = "{harness}"
harness_api_key_secret = "MY_HARNESS_TOKEN"

[agents.model]
type = "provider"
model = "openai/gpt-5"
api_key_secret = "OPENAI_API_KEY"
""",
    )

    with pytest.raises(CLIError, match=rf"named exactly {destination_env}"):
        load_benchmark_submit_config(path)


@pytest.mark.parametrize(
    "harness, destination_env",
    [
        ("cursor-cli", "CURSOR_API_KEY"),
        ("mini-swe-agent", "MSWEA_API_KEY"),
    ],
)
def test_load_benchmark_submit_config_rejects_harness_destination_env_collision(
    tmp_path: Path,
    harness: str,
    destination_env: str,
) -> None:
    secret_line = (
        f'harness_api_key_secret = "{destination_env}"'
        if harness == "cursor-cli"
        else ""
    )
    path = _write_config(
        tmp_path / "benchmark.toml",
        f"""
[experiment]
benchmark = "DeepSWE"

[[agents]]
harness = "{harness}"
{secret_line}

[agents.model]
type = "hosted"
base_model = "Qwen/Qwen3-8B"
lora_model_name = "deep-swe-agent"

[agents.env]
{destination_env} = "literal-for-the-agent"
""",
    )

    with pytest.raises(CLIError, match=destination_env):
        load_benchmark_submit_config(path)


def test_load_benchmark_submit_config_rejects_harness_destination_env_without_secret(
    tmp_path: Path,
) -> None:
    path = _write_config(
        tmp_path / "benchmark.toml",
        """
[experiment]
benchmark = "DeepSWE"

[[agents]]
harness = "cursor-cli"

[agents.model]
type = "hosted"
base_model = "Qwen/Qwen3-8B"
lora_model_name = "deep-swe-agent"

[agents.env]
CURSOR_API_KEY = "literal-for-the-agent"
""",
    )

    with pytest.raises(CLIError, match=r"CURSOR_API_KEY"):
        load_benchmark_submit_config(path)


def test_load_benchmark_submit_config_accepts_mini_swe_without_harness_secret(
    tmp_path: Path,
) -> None:
    path = _write_config(
        tmp_path / "benchmark.toml",
        """
[experiment]
benchmark = "DeepSWE"

[[agents]]
harness = "mini-swe-agent"

[agents.model]
type = "hosted"
base_model = "Qwen/Qwen3-8B"
lora_model_name = "deep-swe-agent"
""",
    )

    config = load_benchmark_submit_config(path)

    assert config.required_secrets == []
    assert config.agents[0].harness_api_key_secret is None


def test_load_benchmark_submit_config_rejects_mini_swe_harness_secret(
    tmp_path: Path,
) -> None:
    path = _write_config(
        tmp_path / "benchmark.toml",
        """
[experiment]
benchmark = "DeepSWE"

[[agents]]
harness = "mini-swe-agent"
harness_api_key_secret = "MSWEA_API_KEY"

[agents.model]
type = "provider"
model = "openai/gpt-5"
api_key_secret = "OPENAI_API_KEY"
""",
    )

    with pytest.raises(CLIError, match=r"does not use harness_api_key_secret"):
        load_benchmark_submit_config(path)


@pytest.mark.parametrize(
    "secret_field, section",
    [
        ('harness_api_key_secret = ""', ""),
        ("", '[execution]\njudge_api_key_secret = ""'),
    ],
)
def test_load_benchmark_submit_config_rejects_empty_secret_references(
    tmp_path: Path,
    secret_field: str,
    section: str,
) -> None:
    path = _write_config(
        tmp_path / "benchmark.toml",
        f"""
[experiment]
benchmark = "DeepSWE"

[[agents]]
harness = "cursor-cli"
{secret_field}

[agents.model]
type = "hosted"
base_model = "Qwen/Qwen3-8B"
lora_model_name = "deep-swe-agent"

{section}
""",
    )

    with pytest.raises(CLIError, match=r"Invalid secret name ''"):
        load_benchmark_submit_config(path)


_VERIFIER_BASE = """
[experiment]
benchmark = "acme/custom@1.0"

[[agents]]
harness = "terminus"

[agents.model]
type = "provider"
model = "openai/gpt-5"
api_key_secret = "OPENAI_API_KEY"
"""


def test_verifier_required_becomes_execution_verifier_secrets(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path / "benchmark.toml",
        _VERIFIER_BASE
        + """
[verifier]
required = ["VLM_API_KEY"]
""",
    )

    config = load_benchmark_submit_config(path)

    assert config.execution_config["verifier_secrets"] == ["VLM_API_KEY"]
    assert "VLM_API_KEY" in config.required_secrets


def test_config_without_verifier_secrets_omits_the_key(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "benchmark.toml", _VERIFIER_BASE)

    config = load_benchmark_submit_config(path)

    assert "verifier_secrets" not in config.execution_config


def test_verifier_secret_colliding_with_agent_env_is_rejected(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path / "benchmark.toml",
        """
[experiment]
benchmark = "acme/custom@1.0"

[[agents]]
harness = "terminus"

[agents.model]
type = "provider"
model = "openai/gpt-5"
api_key_secret = "OPENAI_API_KEY"

[agents.env]
VLM_API_KEY = "literal-value"

[verifier]
required = ["VLM_API_KEY"]
""",
    )

    with pytest.raises(CLIError, match=r"'VLM_API_KEY' appears in \[agents\.env\]"):
        load_benchmark_submit_config(path)


def test_verifier_required_rejects_an_invalid_secret_name(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path / "benchmark.toml",
        _VERIFIER_BASE
        + """
[verifier]
required = ["not-a-secret"]
""",
    )

    with pytest.raises(CLIError, match=r"Invalid secret name 'not-a-secret'"):
        load_benchmark_submit_config(path)
