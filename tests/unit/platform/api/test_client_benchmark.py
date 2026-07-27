from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from osmosis_ai.platform.api.client import OsmosisClient


@patch("osmosis_ai.platform.api.client.platform_request")
def test_submit_benchmark_run_posts_config_sections(mock_request: MagicMock) -> None:
    mock_request.return_value = {
        "id": "benchmark-run-1",
        "name": "bright-otter",
        "status": "pending",
        "workflow_id": "benchmark-run/benchmark-run-1",
        "task_count": 12,
        "created_at": "2026-07-25T00:00:00Z",
        "platform_url": "https://platform.osmosis.ai/acme/benchmarks/benchmark-run-1",
    }
    agent: dict[str, Any] = {
        "harness": "codex",
        "model": {
            "type": "provider",
            "model": "openai/gpt-5",
            "api_key_secret": "OPENAI_API_KEY",
        },
    }

    result = OsmosisClient().submit_benchmark_run(
        experiment_config={"benchmark": "Terminal-Bench 2.1"},
        tasks_config={"task_set": "parity"},
        agents=[agent],
        execution_config={"attempts_per_task": 2},
        env_config={"LOG_LEVEL": "info"},
        git_identity="acme/workspace",
    )

    assert result.id == "benchmark-run-1"
    assert result.task_count == 12
    assert mock_request.call_args.args[0] == "/api/cli/benchmark-runs"
    assert mock_request.call_args.kwargs == {
        "method": "POST",
        "data": {
            "experiment_config": {"benchmark": "Terminal-Bench 2.1"},
            "tasks_config": {"task_set": "parity"},
            "agents": [agent],
            "execution_config": {"attempts_per_task": 2},
            "env_config": {"LOG_LEVEL": "info"},
        },
        "credentials": None,
        "git_identity": "acme/workspace",
    }


@patch("osmosis_ai.platform.api.client.platform_request")
def test_submit_benchmark_run_omits_empty_optional_sections(
    mock_request: MagicMock,
) -> None:
    mock_request.return_value = {
        "id": "benchmark-run-1",
        "name": "bright-otter",
        "status": "pending",
        "created_at": "2026-07-25T00:00:00Z",
    }

    OsmosisClient().submit_benchmark_run(
        experiment_config={"benchmark": "Terminal-Bench 2.1"},
        agents=[{"model": {"type": "hosted"}}],
        git_identity="acme/workspace",
    )

    assert mock_request.call_args.kwargs["data"] == {
        "experiment_config": {"benchmark": "Terminal-Bench 2.1"},
        "agents": [{"model": {"type": "hosted"}}],
    }
