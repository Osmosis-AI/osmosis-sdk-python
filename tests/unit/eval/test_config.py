from pathlib import Path

import pytest

from osmosis_ai.eval.config import load_eval_config


@pytest.fixture
def tmp_toml(tmp_path):
    """Helper to write TOML content to a temp file and return the path."""

    def _write(content: str) -> Path:
        p = tmp_path / "test.toml"
        p.write_text(content)
        return p

    return _write


def test_load_minimal_config(tmp_toml):
    path = tmp_toml("""
[eval]
rollout = "my_rollout"
entrypoint = "workflow.py"
dataset = "data.jsonl"

[llm]
model = "openai/gpt-5.4"
""")
    config = load_eval_config(path)
    assert config.eval_rollout == "my_rollout"
    assert config.eval_entrypoint == "workflow.py"
    assert config.eval_dataset == "data.jsonl"
    assert config.llm_model == "openai/gpt-5.4"
    assert config.has_grader is False
    assert config.runs_n == 1
    assert config.runs_batch_size == 1
    # Defaults for all CLI-overridable flags
    assert config.eval_limit is None
    assert config.eval_offset == 0
    assert config.eval_fresh is False
    assert config.eval_retry_failed is False
    assert config.output_log_samples is False
    assert config.output_path is None
    assert config.output_quiet is False
    assert config.output_debug is False


def test_load_full_config(tmp_toml):
    path = tmp_toml("""
[eval]
rollout = "my_rollout"
entrypoint = "workflow.py"
dataset = "data.jsonl"
limit = 10
offset = 5
fresh = true
retry_failed = true

[llm]
model = "openai/gpt-5.4"
base_url = "http://localhost:8080"
api_key_env = "MY_KEY"

[grader]

[runs]
n = 4
batch_size = 2
pass_threshold = 0.8

[output]
log_samples = true
output_path = "/tmp/eval_output"
quiet = true
debug = true

[baseline]
model = "openai/gpt-3.5-turbo"
""")
    config = load_eval_config(path)
    assert config.llm_base_url == "http://localhost:8080"
    assert config.llm_api_key_env == "MY_KEY"
    assert config.has_grader is True
    assert config.runs_n == 4
    assert config.runs_batch_size == 2
    assert config.runs_pass_threshold == 0.8
    assert config.output_log_samples is True
    assert config.baseline_model == "openai/gpt-3.5-turbo"
    # CLI-overridable flags from TOML
    assert config.eval_limit == 10
    assert config.eval_offset == 5
    assert config.eval_fresh is True
    assert config.eval_retry_failed is True
    assert config.output_path == "/tmp/eval_output"
    assert config.output_quiet is True
    assert config.output_debug is True


def test_load_config_missing_file():
    from osmosis_ai.cli.errors import CLIError

    with pytest.raises(CLIError, match="Config file not found"):
        load_eval_config(Path("/nonexistent/config.toml"))


def test_load_config_invalid_toml(tmp_toml):
    from osmosis_ai.cli.errors import CLIError

    path = tmp_toml("this is not [valid toml")
    with pytest.raises(CLIError, match="Invalid TOML"):
        load_eval_config(path)


def test_load_config_missing_eval_section(tmp_toml):
    from osmosis_ai.cli.errors import CLIError

    path = tmp_toml("""
[llm]
model = "openai/gpt-5.4"
""")
    with pytest.raises(CLIError, match="Missing \\[eval\\] section"):
        load_eval_config(path)


def test_load_config_missing_rollout(tmp_toml):
    from osmosis_ai.cli.errors import CLIError

    path = tmp_toml("""
[eval]
entrypoint = "workflow.py"
dataset = "data.jsonl"

[llm]
model = "openai/gpt-5.4"
""")
    with pytest.raises(CLIError, match="Missing 'rollout'"):
        load_eval_config(path)


def test_load_config_missing_entrypoint(tmp_toml):
    from osmosis_ai.cli.errors import CLIError

    path = tmp_toml("""
[eval]
rollout = "my_rollout"
dataset = "data.jsonl"

[llm]
model = "openai/gpt-5.4"
""")
    with pytest.raises(CLIError, match="Missing 'entrypoint'"):
        load_eval_config(path)


def test_load_config_missing_dataset(tmp_toml):
    from osmosis_ai.cli.errors import CLIError

    path = tmp_toml("""
[eval]
rollout = "my_rollout"
entrypoint = "workflow.py"

[llm]
model = "openai/gpt-5.4"
""")
    with pytest.raises(CLIError, match="Missing 'dataset'"):
        load_eval_config(path)


def test_load_config_missing_llm_model(tmp_toml):
    from osmosis_ai.cli.errors import CLIError

    path = tmp_toml("""
[eval]
rollout = "my_rollout"
entrypoint = "workflow.py"
dataset = "data.jsonl"
""")
    with pytest.raises(CLIError, match="Missing \\[llm\\]"):
        load_eval_config(path)


def test_grader_with_explicit_module(tmp_toml):
    path = tmp_toml("""
[eval]
rollout = "my_rollout"
entrypoint = "workflow.py"
dataset = "data.jsonl"

[llm]
model = "openai/gpt-5.4"

[grader]
module = "my_rollout:MyGrader"
config = "my_rollout:grader_config"
""")
    config = load_eval_config(path)
    assert config.has_grader is True
    assert config.grader_module == "my_rollout:MyGrader"
    assert config.grader_config == "my_rollout:grader_config"


def test_smoke_test_mode_no_grader(tmp_toml):
    """No [grader] section = smoke test mode."""
    path = tmp_toml("""
[eval]
rollout = "my_rollout"
entrypoint = "workflow.py"
dataset = "data.jsonl"

[llm]
model = "openai/gpt-5.4"
""")
    config = load_eval_config(path)
    assert config.has_grader is False
    assert config.grader_module is None


def test_load_config_invalid_runs_n_type(tmp_toml):
    """runs.n must be an integer, not a string."""
    from osmosis_ai.cli.errors import CLIError

    path = tmp_toml("""
[eval]
rollout = "my_rollout"
entrypoint = "workflow.py"
dataset = "data.jsonl"

[llm]
model = "openai/gpt-5.4"

[runs]
n = "four"
""")
    with pytest.raises(CLIError, match="Invalid config"):
        load_eval_config(path)


def test_load_config_invalid_batch_size_zero(tmp_toml):
    """runs.batch_size must be >= 1."""
    from osmosis_ai.cli.errors import CLIError

    path = tmp_toml("""
[eval]
rollout = "my_rollout"
entrypoint = "workflow.py"
dataset = "data.jsonl"

[llm]
model = "openai/gpt-5.4"

[runs]
batch_size = 0
""")
    with pytest.raises(CLIError, match="Invalid config"):
        load_eval_config(path)


def test_load_config_invalid_pass_threshold_range(tmp_toml):
    """runs.pass_threshold must be 0.0..1.0."""
    from osmosis_ai.cli.errors import CLIError

    path = tmp_toml("""
[eval]
rollout = "my_rollout"
entrypoint = "workflow.py"
dataset = "data.jsonl"

[llm]
model = "openai/gpt-5.4"

[runs]
pass_threshold = 2.0
""")
    with pytest.raises(CLIError, match="Invalid config"):
        load_eval_config(path)
