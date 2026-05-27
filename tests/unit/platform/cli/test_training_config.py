"""Tests for training TOML config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.cli.training_config import (
    TrainingConfig,
    load_training_config,
    validate_training_context_paths,
)

# ---------------------------------------------------------------------------
# Valid configs
# ---------------------------------------------------------------------------


def test_load_full_config(tmp_path: Path) -> None:
    path = tmp_path / "train.toml"
    path.write_text(
        """
[experiment]
rollout = "calculator"
entrypoint = "main.py"
model_path = "Qwen/Qwen3.6-35B-A3B"
dataset = "abc-123"
commit_sha = "deadbeef"

[training]
lr = 1e-6
total_epochs = 2
n_samples_per_prompt = 8
rollout_batch_size = 64
max_prompt_length = 4096
max_response_length = 8192

[sampling]
rollout_temperature = 0.8
rollout_top_p = 0.95

[checkpoints]
eval_interval = 10
checkpoint_save_freq = 20

[advanced]
optimizer = "adam"
custom_flag = true
""".strip(),
        encoding="utf-8",
    )

    cfg = load_training_config(path)
    assert isinstance(cfg, TrainingConfig)
    assert cfg.experiment_rollout == "calculator"
    assert cfg.experiment_entrypoint == "main.py"
    assert cfg.experiment_model_path == "Qwen/Qwen3.6-35B-A3B"
    assert cfg.experiment_dataset == "abc-123"
    assert cfg.experiment_commit_sha == "deadbeef"
    assert cfg.experiment_config == {
        "rollout": "calculator",
        "entrypoint": "main.py",
        "model_path": "Qwen/Qwen3.6-35B-A3B",
        "dataset": "abc-123",
        "commit_sha": "deadbeef",
    }
    assert cfg.training_lr == 1e-6
    assert cfg.training_total_epochs == 2
    assert cfg.training_n_samples_per_prompt == 8
    assert cfg.training_rollout_batch_size == 64
    assert cfg.training_max_prompt_length == 4096
    assert cfg.training_max_response_length == 8192
    assert cfg.sampling_rollout_temperature == 0.8
    assert cfg.sampling_rollout_top_p == 0.95
    assert cfg.checkpoints_eval_interval == 10
    assert cfg.checkpoints_checkpoint_save_freq == 20
    assert cfg.advanced_config == {"optimizer": "adam", "custom_flag": True}


def test_load_minimal_config(tmp_path: Path) -> None:
    path = tmp_path / "minimal.toml"
    path.write_text(
        """
[experiment]
rollout = "r"
entrypoint = "e.py"
model_path = "Qwen/Qwen3.6-35B-A3B"
dataset = "id-1"
""".strip(),
        encoding="utf-8",
    )

    cfg = load_training_config(path)
    assert cfg.experiment_rollout == "r"
    assert cfg.experiment_commit_sha is None
    assert cfg.training_lr is None
    assert cfg.training_total_epochs is None
    assert cfg.training_n_samples_per_prompt is None
    assert cfg.training_rollout_batch_size is None
    assert cfg.sampling_rollout_temperature is None
    assert cfg.checkpoints_checkpoint_save_freq is None
    assert cfg.checkpoints_eval_interval is None


def test_load_config_accepts_top_level_env_and_secrets(tmp_path: Path) -> None:
    path = tmp_path / "env_secrets.toml"
    path.write_text(
        """
[experiment]
rollout = "r"
entrypoint = "e.py"
model_path = "m"
dataset = "d"

[env]
LOG_LEVEL = "INFO"

[secrets]
OPENAI_API_KEY = "openai-api-key"
""".strip(),
        encoding="utf-8",
    )

    cfg = load_training_config(path)

    assert cfg.env == {"LOG_LEVEL": "INFO"}
    assert cfg.secrets == {"OPENAI_API_KEY": "openai-api-key"}


# ---------------------------------------------------------------------------
# API config sections
# ---------------------------------------------------------------------------


def test_api_config_sections_all_fields(tmp_path: Path) -> None:
    path = tmp_path / "full.toml"
    path.write_text(
        """
[experiment]
rollout = "r"
entrypoint = "e.py"
model_path = "m"
dataset = "d"

[training]
lr = 1e-6
total_epochs = 1
n_samples_per_prompt = 8
rollout_batch_size = 64

[sampling]
rollout_temperature = 0.9

[checkpoints]
checkpoint_save_freq = 10

[advanced]
optimizer = "adam"
""".strip(),
        encoding="utf-8",
    )

    cfg = load_training_config(path)
    assert cfg.training_config == {
        "lr": 1e-6,
        "total_epochs": 1,
        "n_samples_per_prompt": 8,
        "rollout_batch_size": 64,
    }
    assert cfg.sampling_config == {
        "rollout_temperature": 0.9,
    }
    assert cfg.checkpoints_config == {
        "checkpoint_save_freq": 10,
    }
    assert cfg.advanced_config == {"optimizer": "adam"}


def test_api_config_sections_empty_optional_sections(tmp_path: Path) -> None:
    path = tmp_path / "minimal.toml"
    path.write_text(
        """
[experiment]
rollout = "r"
entrypoint = "e.py"
model_path = "m"
dataset = "d"
""".strip(),
        encoding="utf-8",
    )

    cfg = load_training_config(path)
    assert cfg.training_config == {}
    assert cfg.sampling_config == {}
    assert cfg.checkpoints_config == {}
    assert cfg.advanced_config == {}
    assert not hasattr(cfg, "to_api_config")


# ---------------------------------------------------------------------------
# Context path validation
# ---------------------------------------------------------------------------


def _training_config(
    *,
    rollout: str = "calculator",
    entrypoint: str = "workers/main.py",
) -> TrainingConfig:
    return TrainingConfig(
        experiment={
            "rollout": rollout,
            "entrypoint": entrypoint,
            "model_path": "m",
            "dataset": "d",
        },
        params={},
    )


def test_validate_training_context_paths_allows_entrypoint_under_rollout(
    tmp_path: Path,
) -> None:
    cfg = _training_config()

    validate_training_context_paths(cfg, tmp_path)


def test_validate_training_context_paths_rejects_entrypoint_escape(
    tmp_path: Path,
) -> None:
    cfg = _training_config(entrypoint="../outside.py")

    with pytest.raises(CLIError, match="under rollouts/<rollout>"):
        validate_training_context_paths(cfg, tmp_path)


def test_validate_training_context_paths_rejects_rollout_escape(
    tmp_path: Path,
) -> None:
    cfg = _training_config(rollout="../outside", entrypoint="main.py")

    with pytest.raises(CLIError, match="current workspace directory's rollouts"):
        validate_training_context_paths(cfg, tmp_path)


def test_validate_training_context_paths_rejects_absolute_rollout(
    tmp_path: Path,
) -> None:
    cfg = _training_config(rollout=str(tmp_path / "outside"), entrypoint="main.py")

    with pytest.raises(CLIError, match="logical rollout name"):
        validate_training_context_paths(cfg, tmp_path)


def test_validate_training_context_paths_rejects_absolute_rollout_under_project(
    tmp_path: Path,
) -> None:
    cfg = _training_config(
        rollout=str(tmp_path / "rollouts" / "calculator"),
        entrypoint="main.py",
    )

    with pytest.raises(CLIError, match="logical rollout name"):
        validate_training_context_paths(cfg, tmp_path)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_missing_experiment_section(tmp_path: Path) -> None:
    path = tmp_path / "no_experiment.toml"
    path.write_text("[training]\nlr = 1e-6\n", encoding="utf-8")

    with pytest.raises(CLIError) as exc_info:
        load_training_config(path)
    assert "[experiment]" in str(exc_info.value)


def test_experiment_must_be_table(tmp_path: Path) -> None:
    path = tmp_path / "bad_experiment.toml"
    path.write_text("experiment = 123\n", encoding="utf-8")

    with pytest.raises(CLIError) as exc_info:
        load_training_config(path)
    assert "[experiment]" in str(exc_info.value)


@pytest.mark.parametrize(
    "missing_key", ["rollout", "entrypoint", "model_path", "dataset"]
)
def test_missing_required_experiment_field(tmp_path: Path, missing_key: str) -> None:
    fields = {
        "rollout": '"r"',
        "entrypoint": '"e.py"',
        "model_path": '"m"',
        "dataset": '"d"',
    }
    del fields[missing_key]
    body = "\n".join(f"{k} = {v}" for k, v in fields.items())
    path = tmp_path / "missing_field.toml"
    path.write_text(f"[experiment]\n{body}\n", encoding="utf-8")

    with pytest.raises(CLIError) as exc_info:
        load_training_config(path)
    assert missing_key in str(exc_info.value)


def test_invalid_toml(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[[[not valid", encoding="utf-8")

    with pytest.raises(CLIError) as exc_info:
        load_training_config(path)
    assert "Invalid TOML" in str(exc_info.value)


def test_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(CLIError) as exc_info:
        load_training_config(tmp_path / "missing.toml")
    assert "not found" in str(exc_info.value)


def test_directory_path_raises(tmp_path: Path) -> None:
    dir_path = tmp_path / "config_dir"
    dir_path.mkdir()

    with pytest.raises(CLIError) as exc_info:
        load_training_config(dir_path)
    assert "Cannot read config file" in str(exc_info.value)


@pytest.mark.parametrize(
    ("training_body", "expected_config"),
    [
        ("rollout_batch_size = 32", {"rollout_batch_size": 32}),
        ("n_samples_per_prompt = 4", {"n_samples_per_prompt": 4}),
        ("", {}),
    ],
)
def test_optional_training_fields_only_emit_explicit_values(
    tmp_path: Path, training_body: str, expected_config: dict[str, int | float]
) -> None:
    path = tmp_path / "optional_training_fields.toml"
    path.write_text(
        f"""
[experiment]
rollout = "r"
entrypoint = "e.py"
model_path = "m"
dataset = "d"

[training]
{training_body}
""".strip(),
        encoding="utf-8",
    )

    cfg = load_training_config(path)
    assert cfg.training_config == expected_config


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("training", "lr", "0"),
        ("training", "lr", "1.1"),
        ("training", "total_epochs", "0"),
        ("training", "total_epochs", "10001"),
        ("training", "n_samples_per_prompt", "0"),
        ("training", "n_samples_per_prompt", "1025"),
        ("training", "rollout_batch_size", "0"),
        ("training", "rollout_batch_size", "1000001"),
        ("training", "max_prompt_length", "0"),
        ("training", "max_prompt_length", "262145"),
        ("training", "max_response_length", "0"),
        ("training", "max_response_length", "262145"),
        ("training", "agent_workflow_timeout_s", "0"),
        ("training", "agent_workflow_timeout_s", "86401"),
        ("training", "grader_timeout_s", "0"),
        ("training", "grader_timeout_s", "86401"),
        ("sampling", "rollout_temperature", "-0.1"),
        ("sampling", "rollout_temperature", "2.1"),
        ("sampling", "rollout_top_p", "-0.1"),
        ("sampling", "rollout_top_p", "1.1"),
        ("checkpoints", "eval_interval", "0"),
        ("checkpoints", "eval_interval", "1000001"),
        ("checkpoints", "checkpoint_save_freq", "0"),
        ("checkpoints", "checkpoint_save_freq", "1000001"),
    ],
)
def test_training_param_bounds_are_delegated_to_backend(
    tmp_path: Path, section: str, field: str, value: str
) -> None:
    path = tmp_path / "bad_bounds.toml"
    path.write_text(
        f"""
[experiment]
rollout = "r"
entrypoint = "e.py"
model_path = "m"
dataset = "d"

[{section}]
{field} = {value}
""".strip(),
        encoding="utf-8",
    )

    cfg = load_training_config(path)
    expected = float(value) if "." in value else int(value)
    section_configs = {
        "training": cfg.training_config,
        "sampling": cfg.sampling_config,
        "checkpoints": cfg.checkpoints_config,
    }
    assert section_configs[section][field] == expected


def test_rollout_batch_size_not_required_to_be_divisible(tmp_path: Path) -> None:
    path = tmp_path / "ok_batch.toml"
    path.write_text(
        """
[experiment]
rollout = "r"
entrypoint = "e.py"
model_path = "m"
dataset = "d"

[training]
n_samples_per_prompt = 8
rollout_batch_size = 65
""".strip(),
        encoding="utf-8",
    )

    cfg = load_training_config(path)
    assert cfg.training_rollout_batch_size == 65
    assert cfg.training_n_samples_per_prompt == 8


def test_invalid_training_field_type_is_delegated_to_backend(tmp_path: Path) -> None:
    path = tmp_path / "bad_type.toml"
    path.write_text(
        """
[experiment]
rollout = "r"
entrypoint = "e.py"
model_path = "m"
dataset = "d"

[training]
total_epochs = "not-a-number"
""".strip(),
        encoding="utf-8",
    )

    cfg = load_training_config(path)
    assert cfg.training_config["total_epochs"] == "not-a-number"


def test_advanced_section_preserves_backend_params(tmp_path: Path) -> None:
    path = tmp_path / "unknown_field.toml"
    path.write_text(
        """
[experiment]
rollout = "r"
entrypoint = "e.py"
model_path = "m"
dataset = "d"

[advanced]
dummy = 1
rollout_bach_size = 32
""".strip(),
        encoding="utf-8",
    )

    cfg = load_training_config(path)
    assert cfg.training_config == {}
    assert cfg.advanced_config["dummy"] == 1
    assert cfg.advanced_config["rollout_bach_size"] == 32


def test_extra_param_fields_outside_advanced_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unknown_field.toml"
    path.write_text(
        """
[experiment]
rollout = "r"
entrypoint = "e.py"
model_path = "m"
dataset = "d"

[training]
dummy = 1
rollout_bach_size = 32
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(CLIError) as exc_info:
        load_training_config(path)

    message = str(exc_info.value)
    assert message.startswith("Invalid training config:")
    assert "training.dummy: Unrecognized key" in message
    assert "training.rollout_bach_size: Unrecognized key" in message


def test_sdk_detected_errors_ignore_advanced_backend_params(tmp_path: Path) -> None:
    path = tmp_path / "multiple_sdk_errors.toml"
    path.write_text(
        """
[experiment]
rollout = "r"
entrypoint = "e.py"
model_path = "m"
dataset = "d"
extra = 1

[advanced]
dummy = 1
rollout_bach_size = 32
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(CLIError) as exc_info:
        load_training_config(path)

    message = str(exc_info.value)
    assert message.startswith("Invalid training config:")
    assert "experiment.extra: Unrecognized key" in message
    assert "dummy" not in message
    assert "rollout_bach_size" not in message
    assert exc_info.value.details == {
        "error": "Invalid training config",
        "issues": [
            {"key": "experiment.extra", "message": "Unrecognized key"},
        ],
    }


def test_unknown_top_level_sections_are_reported_together(tmp_path: Path) -> None:
    path = tmp_path / "unknown_sections.toml"
    path.write_text(
        """
[experiment]
rollout = "r"
entrypoint = "e.py"
model_path = "m"
dataset = "d"
extra = 1

[section]
foo = "bar"

[another_section]
enabled = true

[training]
dummy = 1
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(CLIError) as exc_info:
        load_training_config(path)

    message = str(exc_info.value)
    assert message.startswith("Invalid training config:")
    assert "experiment.extra: Unrecognized key" in message
    assert "section: Unrecognized section" in message
    assert "another_section: Unrecognized section" in message
    assert "training.dummy: Unrecognized key" in message
    assert exc_info.value.details == {
        "error": "Invalid training config",
        "issues": [
            {"key": "section", "message": "Unrecognized section"},
            {"key": "another_section", "message": "Unrecognized section"},
            {"key": "experiment.extra", "message": "Unrecognized key"},
            {"key": "training.dummy", "message": "Unrecognized key"},
        ],
    }


def test_known_training_field_in_wrong_section_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "misplaced_field.toml"
    path.write_text(
        """
[experiment]
rollout = "r"
entrypoint = "e.py"
model_path = "m"
dataset = "d"

[sampling]
lr = 1e-6
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(CLIError) as exc_info:
        load_training_config(path)

    message = str(exc_info.value)
    assert message.startswith("Invalid training config:")
    assert str(path) not in message
    assert "sampling.lr: Unrecognized key" in message


def test_param_section_errors_are_reported_together(tmp_path: Path) -> None:
    path = tmp_path / "multiple_section_errors.toml"
    path.write_text(
        """
[experiment]
rollout = "r"
entrypoint = "e.py"
model_path = "m"
dataset = "d"

[training]
lr = 1e-6

[sampling]
lr = 2e-6
checkpoint_save_freq = 10
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(CLIError) as exc_info:
        load_training_config(path)

    message = str(exc_info.value)
    assert message.startswith("Invalid training config:")
    assert str(path) not in message
    assert "sampling.lr: Unrecognized key" in message
    assert "sampling.checkpoint_save_freq: Unrecognized key" in message
