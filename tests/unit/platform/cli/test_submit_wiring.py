"""Tests that train/eval submit helpers forward secrets as a list."""

from __future__ import annotations

from unittest.mock import MagicMock

from osmosis_ai.platform.cli import eval as eval_mod
from osmosis_ai.platform.cli import train as train_mod
from osmosis_ai.platform.cli.eval_config import EvalSubmitConfig
from osmosis_ai.platform.cli.training_config import TrainSubmitConfig


def _train_config(secrets: list[str]) -> TrainSubmitConfig:
    return TrainSubmitConfig.model_validate(
        {
            "experiment": {
                "rollout": "r",
                "entrypoint": "rollouts/main.py",
                "model_path": "m",
                "dataset": "d",
            },
            "secrets": secrets,
        }
    )


def _eval_config(secrets: list[str]) -> EvalSubmitConfig:
    return EvalSubmitConfig.model_validate(
        {
            "experiment": {
                "rollout": "r",
                "entrypoint": "rollouts/main.py",
                "model_path": "m",
                "dataset": "d",
            },
            "secrets": secrets,
        }
    )


def test_submit_training_forwards_secrets_list() -> None:
    client = MagicMock()
    train_mod._submit_training(
        client, _train_config(["OPENAI_API_KEY"]), credentials=None, git_identity="g"
    )
    kwargs = client.submit_training_run.call_args.kwargs
    assert kwargs["secrets"] == ["OPENAI_API_KEY"]
    assert "secret_refs_config" not in kwargs


def test_submit_training_passes_none_when_empty() -> None:
    client = MagicMock()
    train_mod._submit_training(
        client, _train_config([]), credentials=None, git_identity="g"
    )
    assert client.submit_training_run.call_args.kwargs["secrets"] is None


def test_submit_eval_forwards_secrets_list() -> None:
    client = MagicMock()
    eval_mod._submit_eval(
        client, _eval_config(["GITHUB_TOKEN"]), credentials=None, git_identity="g"
    )
    kwargs = client.submit_evaluation_run.call_args.kwargs
    assert kwargs["secrets"] == ["GITHUB_TOKEN"]
    assert "secret_refs_config" not in kwargs


def test_submit_eval_passes_none_when_empty() -> None:
    client = MagicMock()
    eval_mod._submit_eval(client, _eval_config([]), credentials=None, git_identity="g")
    assert client.submit_evaluation_run.call_args.kwargs["secrets"] is None
