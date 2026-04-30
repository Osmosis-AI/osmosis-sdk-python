"""Tests for training TOML config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.cli.training_config import TrainingConfig, load_training_config

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
global_batch_size = 64
max_prompt_length = 4096
max_response_length = 8192

[sampling]
rollout_temperature = 0.8
rollout_top_p = 0.95

[checkpoints]
eval_interval = 10
checkpoint_save_freq = 20
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
    assert cfg.training_lr == 1e-6
    assert cfg.training_total_epochs == 2
    assert cfg.training_n_samples_per_prompt == 8
    assert cfg.training_global_batch_size == 64
    assert cfg.training_max_prompt_length == 4096
    assert cfg.training_max_response_length == 8192
    assert cfg.sampling_rollout_temperature == 0.8
    assert cfg.sampling_rollout_top_p == 0.95
    assert cfg.checkpoints_eval_interval == 10
    assert cfg.checkpoints_checkpoint_save_freq == 20


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
    assert cfg.sampling_rollout_temperature is None
    assert cfg.checkpoints_eval_interval is None


# ---------------------------------------------------------------------------
# to_api_config
# ---------------------------------------------------------------------------


def test_to_api_config_all_fields(tmp_path: Path) -> None:
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

[sampling]
rollout_temperature = 0.9

[checkpoints]
checkpoint_save_freq = 10
""".strip(),
        encoding="utf-8",
    )

    cfg = load_training_config(path)
    api = cfg.to_api_config()
    assert api == {
        "lr": 1e-6,
        "total_epochs": 1,
        "rollout_temperature": 0.9,
        "checkpoint_save_freq": 10,
    }


def test_to_api_config_empty_optional_sections(tmp_path: Path) -> None:
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
    assert cfg.to_api_config() == {}


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


def test_batch_size_not_divisible(tmp_path: Path) -> None:
    path = tmp_path / "bad_batch.toml"
    path.write_text(
        """
[experiment]
rollout = "r"
entrypoint = "e.py"
model_path = "m"
dataset = "d"

[training]
n_samples_per_prompt = 8
global_batch_size = 65
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(CLIError) as exc_info:
        load_training_config(path)
    assert "divisible" in str(exc_info.value)


def test_batch_size_divisible_ok(tmp_path: Path) -> None:
    """Batch size exactly divisible by n_samples_per_prompt should succeed."""
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
global_batch_size = 64
""".strip(),
        encoding="utf-8",
    )

    cfg = load_training_config(path)
    assert cfg.training_global_batch_size == 64
    assert cfg.training_n_samples_per_prompt == 8


def test_invalid_training_field_type(tmp_path: Path) -> None:
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

    with pytest.raises(CLIError) as exc_info:
        load_training_config(path)
    assert "Invalid config" in str(exc_info.value)
