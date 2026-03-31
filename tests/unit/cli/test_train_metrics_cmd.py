"""Tests for the osmosis train metrics CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from osmosis_ai.platform.api.models import (
    MetricDataPoint,
    MetricHistory,
    TrainingRunDetail,
    TrainingRunMetrics,
    TrainingRunMetricsOverview,
)


def _make_run_detail(**overrides) -> TrainingRunDetail:
    defaults = dict(
        id="550e8400-e29b-41d4-a716-446655440000",
        name="reward-tuning-v3",
        status="finished",
        model_name="Qwen/Qwen3-8B",
        started_at="2026-03-28T10:00:00Z",
        completed_at="2026-03-28T11:05:30Z",
        examples_processed_count=5000,
    )
    defaults.update(overrides)
    return TrainingRunDetail(**defaults)


def _make_metrics(**overrides) -> TrainingRunMetrics:
    defaults = dict(
        training_run_id="550e8400-e29b-41d4-a716-446655440000",
        status="finished",
        overview=TrainingRunMetricsOverview(
            mlflow_run_id="mlflow-abc",
            mlflow_status="FINISHED",
            duration_ms=3600000,
            duration_formatted="1h",
            reward=0.85,
            reward_delta=0.15,
            examples_processed_count=5000,
        ),
        metrics=[
            MetricHistory(
                metric_key="rollout/raw_reward",
                title="Training Reward",
                data_points=[
                    MetricDataPoint(step=0, value=0.5, timestamp=1711800000000),
                    MetricDataPoint(step=100, value=0.85, timestamp=1711803600000),
                ],
            ),
        ],
    )
    defaults.update(overrides)
    return TrainingRunMetrics(**defaults)


# Patch at source modules since train.py uses function-level lazy imports.
_PATCH_AUTH = "osmosis_ai.platform.cli.project._require_auth"
_PATCH_CLIENT = "osmosis_ai.platform.api.client.OsmosisClient"


class TestMetricsCommandWritesFile:
    """Test that the metrics command writes the correct JSON file."""

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_writes_json_to_explicit_output(
        self, mock_auth: MagicMock, mock_client_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_auth.return_value = ("ws", MagicMock())
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail()
        client.get_training_run_metrics.return_value = _make_metrics()

        output = tmp_path / "metrics.json"

        from osmosis_ai.cli.commands.train import metrics

        metrics(
            id="550e8400-e29b-41d4-a716-446655440000",
            output=str(output),
        )

        assert output.exists()
        data = json.loads(output.read_text())
        assert data["training_run"]["name"] == "reward-tuning-v3"
        assert data["summary"]["final_reward"] == 0.85
        assert data["metrics"][0]["key"] == "training_reward"

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_writes_to_default_path_in_workspace(
        self,
        mock_auth: MagicMock,
        mock_client_cls: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_auth.return_value = ("ws", MagicMock())
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail()
        client.get_training_run_metrics.return_value = _make_metrics()

        # Create workspace marker
        osmosis_dir = tmp_path / ".osmosis"
        osmosis_dir.mkdir()
        (osmosis_dir / "workspace.toml").write_text("[workspace]\n")
        monkeypatch.chdir(tmp_path)

        from osmosis_ai.cli.commands.train import metrics

        metrics(
            id="550e8400-e29b-41d4-a716-446655440000",
            output=None,
        )

        expected = osmosis_dir / "metrics" / "reward-tuning-v3_550e8400.json"
        assert expected.exists()
        data = json.loads(expected.read_text())
        assert data["training_run"]["id"] == "550e8400-e29b-41d4-a716-446655440000"


class TestMetricsCommandErrors:
    """Test error handling in the metrics command."""

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_non_terminal_run_raises(
        self, mock_auth: MagicMock, mock_client_cls: MagicMock
    ) -> None:
        mock_auth.return_value = ("ws", MagicMock())
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail(status="running")

        from osmosis_ai.cli.commands.train import metrics
        from osmosis_ai.cli.errors import CLIError

        with pytest.raises(CLIError, match="only available for terminal"):
            metrics(
                id="550e8400-e29b-41d4-a716-446655440000",
                output="/tmp/out.json",
            )

    def test_no_workspace_no_output_raises(self, tmp_path: Path) -> None:
        """Without .osmosis/workspace.toml and no -o flag, should error."""
        from osmosis_ai.cli.commands.train import _resolve_default_output
        from osmosis_ai.cli.errors import CLIError

        with pytest.raises(CLIError, match="Not in an Osmosis workspace"):
            _resolve_default_output("my-run", "abc12345", cwd=tmp_path)

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_output_parent_dir_missing_raises(
        self, mock_auth: MagicMock, mock_client_cls: MagicMock
    ) -> None:
        mock_auth.return_value = ("ws", MagicMock())
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail()
        client.get_training_run_metrics.return_value = _make_metrics()

        from osmosis_ai.cli.commands.train import metrics
        from osmosis_ai.cli.errors import CLIError

        with pytest.raises(CLIError, match="Output directory does not exist"):
            metrics(
                id="550e8400-e29b-41d4-a716-446655440000",
                output="/nonexistent/dir/metrics.json",
            )
