from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from osmosis_ai.platform.api.client import OsmosisClient


@patch("osmosis_ai.platform.api.client.platform_request")
def test_list_benchmarks_gets_paginated_catalog(mock_request: MagicMock) -> None:
    mock_request.return_value = {
        "benchmarks": [
            {
                "id": "benchmark-1",
                "name": "HLE",
                "description": "Humanity's Last Exam",
                "source_type": "osmosis_managed",
                "source_ref": "hle",
                "task_count": 2_500,
                "category_count": 30,
                "task_sets": [
                    {
                        "name": "parity",
                        "task_count": 249,
                        "recommended": True,
                        "description": "Published comparison sample.",
                    }
                ],
            }
        ],
        "total_count": 7,
        "has_more": True,
        "next_offset": 1,
    }

    result = OsmosisClient().list_benchmarks(
        limit=1,
        offset=0,
        git_identity="acme/workspace",
    )

    assert result.total_count == 7
    assert result.has_more is True
    assert result.next_offset == 1
    assert result.benchmarks[0].name == "HLE"
    assert result.benchmarks[0].task_sets[0].name == "parity"
    assert result.benchmarks[0].task_sets[0].recommended is True
    assert mock_request.call_args.args[0] == "/api/cli/benchmarks?limit=1&offset=0"
    assert mock_request.call_args.kwargs == {
        "credentials": None,
        "git_identity": "acme/workspace",
    }


@patch("osmosis_ai.platform.api.client.platform_request")
def test_get_benchmark_encodes_name_and_parses_detail(mock_request: MagicMock) -> None:
    mock_request.return_value = {
        "benchmark": {
            "id": "benchmark-2",
            "name": "Terminal-Bench 2.1",
            "description": "Terminal benchmark",
            "source_type": "osmosis_managed",
            "source_ref": "terminal-bench-2-1",
            "task_count": 89,
            "category_count": 1,
            "task_sets": [],
            "runner_family": "harbor",
            "supports_harness": True,
            "requires_harness": True,
            "requires_judge_model": False,
            "judge_model_default": None,
            "pass_threshold": 1,
            "categories": [{"name": "terminal", "task_count": 89}],
            "tasks": [{"name": "task-1", "category": "terminal"}],
            "unavailable_tasks": None,
        }
    }

    result = OsmosisClient().get_benchmark(
        "Terminal-Bench 2.1",
        git_identity="acme/workspace",
    )

    assert result.name == "Terminal-Bench 2.1"
    assert result.categories[0].name == "terminal"
    assert result.tasks == [{"name": "task-1", "category": "terminal"}]
    assert mock_request.call_args.args[0] == (
        "/api/cli/benchmarks/Terminal-Bench%202.1"
    )
    assert mock_request.call_args.kwargs == {
        "credentials": None,
        "git_identity": "acme/workspace",
    }


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
        experiment_config={"benchmark": "HLE"},
        tasks_config={"task_set": "parity"},
        agents=[agent],
        execution_config={
            "attempts_per_task": 2,
            "judge_api_key_secret": "OPENAI_API_KEY",
        },
        env_config={"LOG_LEVEL": "info"},
        git_identity="acme/workspace",
    )

    assert result.id == "benchmark-run-1"
    assert result.workflow_id == "benchmark-run/benchmark-run-1"
    assert result.task_count == 12
    assert result.platform_url == (
        "https://platform.osmosis.ai/acme/benchmarks/benchmark-run-1"
    )
    assert mock_request.call_args.args[0] == "/api/cli/benchmark-runs"
    assert mock_request.call_args.kwargs == {
        "method": "POST",
        "data": {
            "experiment_config": {"benchmark": "HLE"},
            "tasks_config": {"task_set": "parity"},
            "agents": [agent],
            "execution_config": {
                "attempts_per_task": 2,
                "judge_api_key_secret": "OPENAI_API_KEY",
            },
            "env_config": {"LOG_LEVEL": "info"},
        },
        "credentials": None,
        "git_identity": "acme/workspace",
    }


@patch("osmosis_ai.platform.api.client.platform_request")
def test_submit_benchmark_run_forwards_harness_api_key_secret(
    mock_request: MagicMock,
) -> None:
    mock_request.return_value = {
        "id": "benchmark-run-1",
        "name": "bright-otter",
        "status": "pending",
        "workflow_id": "benchmark-run/benchmark-run-1",
        "task_count": 120,
        "created_at": "2026-07-25T00:00:00Z",
        "platform_url": None,
    }
    agent: dict[str, Any] = {
        "harness": "cursor-cli",
        "harness_api_key_secret": "CURSOR_API_KEY",
        "model": {
            "type": "provider",
            "model": "openai/gpt-5",
            "api_key_secret": "OPENAI_API_KEY",
        },
    }

    OsmosisClient().submit_benchmark_run(
        experiment_config={"benchmark": "DeepSWE"},
        agents=[agent],
        git_identity="acme/workspace",
    )

    assert mock_request.call_args.kwargs["data"] == {
        "experiment_config": {"benchmark": "DeepSWE"},
        "agents": [agent],
    }


@patch("osmosis_ai.platform.api.client.platform_request")
def test_submit_benchmark_run_omits_empty_optional_sections(
    mock_request: MagicMock,
) -> None:
    mock_request.return_value = {
        "id": "benchmark-run-1",
        "name": "bright-otter",
        "status": "pending",
        "workflow_id": "benchmark-run/benchmark-run-1",
        "task_count": 12,
        "created_at": "2026-07-25T00:00:00Z",
    }

    OsmosisClient().submit_benchmark_run(
        experiment_config={"benchmark": "Terminal-Bench 2.1"},
        agents=[
            {
                "harness": "codex",
                "model": {
                    "type": "hosted",
                    "base_model": "Qwen/Qwen3-8B",
                    "checkpoint_name": "terminal-agent",
                },
            }
        ],
        git_identity="acme/workspace",
    )

    assert mock_request.call_args.kwargs["data"] == {
        "experiment_config": {"benchmark": "Terminal-Bench 2.1"},
        "agents": [
            {
                "harness": "codex",
                "model": {
                    "type": "hosted",
                    "base_model": "Qwen/Qwen3-8B",
                    "checkpoint_name": "terminal-agent",
                },
            }
        ],
    }
