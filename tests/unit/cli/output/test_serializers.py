"""Tests for public JSON serializers and golden snapshots."""

from __future__ import annotations

import json
from pathlib import Path

from osmosis_ai.cli.output.serializers import (
    serialize_checkpoint,
    serialize_dataset,
    serialize_deployment,
    serialize_eval_cache_entry,
    serialize_model,
    serialize_rollout,
    serialize_training_run,
    serialize_workspace,
)
from osmosis_ai.platform.api.models import (
    BaseModelInfo,
    DatasetFile,
    DeploymentInfo,
    LoraCheckpointInfo,
    RolloutInfo,
    TrainingRun,
)

GOLDEN_DIR = Path(__file__).resolve().parents[3] / "golden" / "cli_output"


def _assert_keys_match_golden(payload: dict, golden_name: str) -> None:
    expected = json.loads((GOLDEN_DIR / golden_name).read_text(encoding="utf-8"))
    assert sorted(payload.keys()) == sorted(expected["keys"])


def test_serialize_dataset_keys() -> None:
    dataset = DatasetFile.from_dict(
        {
            "id": "ds_1",
            "file_name": "train.jsonl",
            "file_size": 12345,
            "status": "uploaded",
            "processing_step": None,
            "processing_percent": None,
            "error": None,
            "created_at": "2026-04-26T00:00:00Z",
            "updated_at": "2026-04-26T00:00:01Z",
        }
    )
    payload = serialize_dataset(dataset)
    _assert_keys_match_golden(payload, "dataset_serializer.json")
    assert payload["id"] == "ds_1"
    assert payload["file_size"] == 12345
    assert "upload" not in payload


def test_serialize_dataset_keeps_none_optional_contract_fields() -> None:
    dataset = DatasetFile.from_dict(
        {
            "id": "ds_1",
            "file_name": "x.jsonl",
            "file_size": 0,
            "status": "pending",
        }
    )
    payload = serialize_dataset(dataset)
    assert payload["error"] is None
    assert payload["processing_step"] is None


def test_serialize_training_run_keys() -> None:
    run = TrainingRun.from_dict(
        {
            "id": "run_1",
            "name": "qwen3-run1",
            "status": "running",
            "model_id": "model_1",
            "model": {"model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"},
            "eval_accuracy": 0.4,
            "reward_increase_delta": 0.02,
            "processing_step": "training",
            "processing_percent": 50.0,
            "error_message": None,
            "creator_name": "brian",
            "creator_email": "b@example.com",
            "created_at": "2026-04-26T00:00:00Z",
            "started_at": "2026-04-26T00:00:01Z",
            "completed_at": None,
        }
    )
    payload = serialize_training_run(run)
    _assert_keys_match_golden(payload, "training_run_serializer.json")
    assert payload["model_name"] == "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"


def test_serialize_checkpoint_keys() -> None:
    checkpoint = LoraCheckpointInfo.from_dict(
        {
            "id": "ckpt_1",
            "checkpoint_step": 100,
            "checkpoint_name": "qwen3-run1-step-100",
            "status": "uploaded",
            "created_at": "2026-04-26T00:00:00Z",
        }
    )
    payload = serialize_checkpoint(checkpoint)
    _assert_keys_match_golden(payload, "checkpoint_serializer.json")


def test_serialize_deployment_keys() -> None:
    deployment = DeploymentInfo.from_dict(
        {
            "id": "dep_1",
            "checkpoint_name": "qwen3-run1-step-100",
            "status": "active",
            "checkpoint_step": 100,
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
            "training_run_id": "run_1",
            "training_run_name": "qwen3-run1",
            "creator_name": "brian",
            "created_at": "2026-04-26T00:00:00Z",
        }
    )
    payload = serialize_deployment(deployment)
    _assert_keys_match_golden(payload, "deployment_serializer.json")


def test_serialize_model_keys() -> None:
    model = BaseModelInfo.from_dict(
        {
            "id": "model_1",
            "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
            "base_model": "Qwen/Qwen3",
            "status": "ready",
            "creator_name": "brian",
            "created_at": "2026-04-26T00:00:00Z",
            "updated_at": "2026-04-26T00:00:01Z",
        }
    )
    payload = serialize_model(model)
    _assert_keys_match_golden(payload, "model_serializer.json")


def test_serialize_workspace_minimal() -> None:
    payload = serialize_workspace({"id": "ws_1", "name": "default"})
    _assert_keys_match_golden(payload, "workspace_serializer.json")
    assert payload == {"id": "ws_1", "name": "default"}


def test_serialize_rollout_keys() -> None:
    rollout = RolloutInfo.from_dict(
        {
            "id": "ro_1",
            "name": "my-rollout",
            "is_active": True,
            "repo_full_name": "org/repo",
            "last_synced_commit_sha": "abc123",
            "created_at": "2026-04-26T00:00:00Z",
        }
    )
    payload = serialize_rollout(rollout)
    _assert_keys_match_golden(payload, "rollout_serializer.json")


def test_serialize_eval_cache_entry_keys() -> None:
    entry = {
        "task_id": "task_1",
        "config": {
            "llm_model": "openai/gpt-5.4",
            "eval_dataset": "data.jsonl",
        },
        "status": "completed",
        "runs_count": 3,
        "created_at": "2026-04-26T00:00:00Z",
    }
    payload = serialize_eval_cache_entry(entry)
    _assert_keys_match_golden(payload, "eval_cache_entry_serializer.json")
    assert payload["model"] == "openai/gpt-5.4"
    assert payload["dataset"] == "data.jsonl"
