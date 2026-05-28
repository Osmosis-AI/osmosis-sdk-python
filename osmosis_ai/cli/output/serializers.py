"""Public JSON serializers for CLI output."""

from __future__ import annotations

from typing import Any

from osmosis_ai.platform.api.models import (
    BaseModelInfo,
    DatasetFile,
    DeploymentInfo,
    EnvironmentSecretInfo,
    EvaluationRun,
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
        "file_format": df.file_format,
        "original_file_format": df.original_file_format,
        "row_count": df.row_count,
        "original_file_size": df.original_file_size,
        "creator_name": df.creator_name,
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
        "dataset_id": run.dataset_id,
        "dataset_name": run.dataset_name,
        "rollout_id": run.rollout_id,
        "rollout_name": run.rollout_name,
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


def serialize_environment_secret(secret: EnvironmentSecretInfo) -> dict[str, Any]:
    """Serialize an environment secret for the public JSON contract.

    Intentionally never includes the secret value — the platform does not
    return it, and there must be no path by which a value reaches output.
    """
    return {
        "id": secret.id,
        "name": secret.name,
        "created_at": secret.created_at,
        "updated_at": secret.updated_at,
        "creator_name": secret.creator_name,
    }


def serialize_eval_run(run: EvaluationRun) -> dict[str, Any]:
    """Serialize an evaluation run for the public JSON contract."""
    data: dict[str, Any] = {
        "id": run.id,
        "name": run.name,
        "status": run.status,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "model": run.model.get("name") if run.model else None,
        "dataset": run.dataset.get("name") if run.dataset else None,
        "rollout": run.rollout.get("name") if run.rollout else None,
        "creator_name": run.creator_name,
        "creator_email": run.creator_email,
    }
    if run.platform_url:
        data["platform_url"] = run.platform_url
    return data
