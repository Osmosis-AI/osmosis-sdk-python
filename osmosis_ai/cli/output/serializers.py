"""Public JSON serializers for CLI output."""

from __future__ import annotations

from typing import Any

from osmosis_ai.platform.api.models import (
    BaseModelInfo,
    DatasetFile,
    DeploymentInfo,
    LoraCheckpointInfo,
    RolloutInfo,
    TrainingRun,
)


def serialize_dataset(df: DatasetFile) -> dict[str, Any]:
    """Serialize a dataset for the public JSON contract."""
    data = {
        "id": df.id,
        "file_name": df.file_name,
        "file_size": df.file_size,
        "status": df.status,
        "processing_step": df.processing_step,
        "processing_percent": df.processing_percent,
        "error": df.error,
        "created_at": df.created_at,
        "updated_at": df.updated_at,
    }
    if df.platform_url:
        data["platform_url"] = df.platform_url
    return data


def serialize_training_run(run: TrainingRun) -> dict[str, Any]:
    """Serialize a training run for the public JSON contract."""
    data = {
        "id": run.id,
        "name": run.name,
        "status": run.status,
        "model_id": run.model_id,
        "model_name": run.model_name,
        "eval_accuracy": run.eval_accuracy,
        "reward": run.reward,
        "reward_increase_delta": run.reward_increase_delta,
        "processing_step": run.processing_step,
        "processing_percent": run.processing_percent,
        "error_message": run.error_message,
        "creator_name": run.creator_name,
        "creator_email": run.creator_email,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
    }
    if run.platform_url:
        data["platform_url"] = run.platform_url
    return data


def serialize_checkpoint(ckpt: LoraCheckpointInfo) -> dict[str, Any]:
    """Serialize a LoRA checkpoint for the public JSON contract."""
    return {
        "id": ckpt.id,
        "checkpoint_name": ckpt.checkpoint_name,
        "checkpoint_step": ckpt.checkpoint_step,
        "status": ckpt.status,
        "created_at": ckpt.created_at,
    }


def serialize_deployment(dep: DeploymentInfo) -> dict[str, Any]:
    """Serialize a deployment for the public JSON contract."""
    return {
        "id": dep.id,
        "checkpoint_name": dep.checkpoint_name,
        "status": dep.status,
        "base_model": dep.base_model,
        "checkpoint_step": dep.checkpoint_step,
        "training_run_id": dep.training_run_id,
        "training_run_name": dep.training_run_name,
        "creator_name": dep.creator_name,
        "created_at": dep.created_at,
    }


def serialize_model(model: BaseModelInfo) -> dict[str, Any]:
    """Serialize a base model for the public JSON contract."""
    return {
        "id": model.id,
        "model_name": model.model_name,
        "base_model": model.base_model,
        "creator_name": model.creator_name,
        "created_at": model.created_at,
        "updated_at": model.updated_at,
    }


def serialize_rollout(rollout: RolloutInfo) -> dict[str, Any]:
    """Serialize a rollout for the public JSON contract."""
    return {
        "id": rollout.id,
        "name": rollout.name,
        "is_active": rollout.is_active,
        "repo_full_name": rollout.repo_full_name,
        "last_synced_commit_sha": rollout.last_synced_commit_sha,
        "created_at": rollout.created_at,
    }


def serialize_eval_cache_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Serialize an eval cache entry dict."""
    config = entry.get("config", {}) or {}
    return {
        "task_id": entry.get("task_id", ""),
        "model": config.get("llm_model") or config.get("model", ""),
        "dataset": config.get("eval_dataset") or config.get("dataset", ""),
        "status": entry.get("status", ""),
        "runs_count": entry.get("runs_count", 0),
        "created_at": entry.get("created_at", ""),
    }
