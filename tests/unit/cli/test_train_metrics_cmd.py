"""Tests for the osmosis train metrics CLI command."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from osmosis_ai.cli.console import Console
from osmosis_ai.cli.metrics_graph import MIN_TREND_TERMINAL_WIDTH, SPARKLINE_BLOCKS
from osmosis_ai.platform.api.models import (
    MetricDataPoint,
    MetricHistory,
    ProjectDetail,
    TrainingRunDetail,
    TrainingRunMetrics,
    TrainingRunMetricsOverview,
)

_PROJECT_ID = "proj-0001-0001-0001-000000000001"
_PROJECT_NAME = "my-project"


def _make_run_detail(**overrides) -> TrainingRunDetail:
    defaults = dict(
        id="550e8400-e29b-41d4-a716-446655440000",
        name="reward-tuning-v3",
        status="finished",
        model_name="Qwen/Qwen3-8B",
        started_at="2026-03-28T10:00:00Z",
        completed_at="2026-03-28T11:05:30Z",
        examples_processed_count=5000,
        project_id=_PROJECT_ID,
    )
    defaults.update(overrides)
    return TrainingRunDetail(**defaults)


def _make_project_detail() -> ProjectDetail:
    return ProjectDetail(
        id=_PROJECT_ID,
        project_name=_PROJECT_NAME,
        role="owner",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )


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


def _patch_train_console(
    buf: io.StringIO,
    *,
    force_terminal: bool = True,
    width: int = 120,
):
    """Patch ``train`` module ``console`` for deterministic metrics output tests."""
    import osmosis_ai.cli.commands.train as train_module

    return patch.object(
        train_module,
        "console",
        Console(
            file=buf,
            force_terminal=force_terminal,
            no_color=True,
            width=width,
        ),
    )


class TestMetricsCommandPlatformUrl:
    """Platform URL is printed at the top of output."""

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_platform_url_printed(
        self, mock_auth: MagicMock, mock_client_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_auth.return_value = ("ws", MagicMock())
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail()
        client.get_project.return_value = _make_project_detail()
        client.get_training_run_metrics.return_value = _make_metrics()

        output = tmp_path / "m.json"
        buf = io.StringIO()
        with _patch_train_console(buf, force_terminal=False, width=120):
            from osmosis_ai.cli.commands.train import metrics

            metrics(
                id="550e8400-e29b-41d4-a716-446655440000",
                output=str(output),
            )

        text = buf.getvalue()
        assert "View full details:" in text
        assert f"ws/{_PROJECT_NAME}/training/" in text
        assert "550e8400-e29b-41d4-a716-446655440000" in text

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_platform_url_skipped_when_project_lookup_fails(
        self, mock_auth: MagicMock, mock_client_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_auth.return_value = ("ws", MagicMock())
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail()
        from osmosis_ai.platform.auth.platform_client import PlatformAPIError

        client.get_project.side_effect = PlatformAPIError("network error")
        client.get_training_run_metrics.return_value = _make_metrics()

        output = tmp_path / "m.json"
        buf = io.StringIO()
        with _patch_train_console(buf, force_terminal=False, width=120):
            from osmosis_ai.cli.commands.train import metrics

            metrics(
                id="550e8400-e29b-41d4-a716-446655440000",
                output=str(output),
            )

        text = buf.getvalue()
        # URL line should be absent, but the rest should work
        assert "/training/" not in text
        assert "Training Run Metrics" in text

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_platform_url_skipped_when_no_project_id(
        self, mock_auth: MagicMock, mock_client_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_auth.return_value = ("ws", MagicMock())
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail(project_id=None)
        client.get_training_run_metrics.return_value = _make_metrics()

        output = tmp_path / "m.json"
        buf = io.StringIO()
        with _patch_train_console(buf, force_terminal=False, width=120):
            from osmosis_ai.cli.commands.train import metrics

            metrics(
                id="550e8400-e29b-41d4-a716-446655440000",
                output=str(output),
            )

        text = buf.getvalue()
        assert "/training/" not in text
        assert "Training Run Metrics" in text


class TestMetricsCommandTrendGraphs:
    """Trend graphs after the summary table when TTY + width allow."""

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_graphs_render_after_summary_when_tty_and_width_sufficient(
        self, mock_auth: MagicMock, mock_client_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_auth.return_value = ("ws", MagicMock())
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail()
        client.get_project.return_value = _make_project_detail()
        client.get_training_run_metrics.return_value = _make_metrics()

        output = tmp_path / "m.json"
        buf = io.StringIO()
        with _patch_train_console(
            buf, force_terminal=True, width=MIN_TREND_TERMINAL_WIDTH
        ):
            from osmosis_ai.cli.commands.train import metrics

            metrics(
                id="550e8400-e29b-41d4-a716-446655440000",
                output=str(output),
            )

        text = buf.getvalue()
        assert "Training Run Metrics" in text
        assert "Metric Trends" in text  # Rule separator header
        assert "Training Reward" in text
        assert any(c in text for c in SPARKLINE_BLOCKS)

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_trend_block_prints_metric_title_with_brackets_literally(
        self, mock_auth: MagicMock, mock_client_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Bracket characters in metric titles must not be interpreted as Rich markup."""
        bracket_title = "Loss [eval]"
        mock_auth.return_value = ("ws", MagicMock())
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail()
        client.get_project.return_value = _make_project_detail()
        client.get_training_run_metrics.return_value = _make_metrics(
            metrics=[
                MetricHistory(
                    metric_key="loss/eval",
                    title=bracket_title,
                    data_points=[
                        MetricDataPoint(step=0, value=1.0, timestamp=0),
                        MetricDataPoint(step=1, value=0.5, timestamp=1),
                    ],
                ),
            ],
        )

        output = tmp_path / "m.json"
        buf = io.StringIO()
        with _patch_train_console(
            buf, force_terminal=True, width=MIN_TREND_TERMINAL_WIDTH
        ):
            from osmosis_ai.cli.commands.train import metrics

            metrics(
                id="550e8400-e29b-41d4-a716-446655440000",
                output=str(output),
            )

        text = buf.getvalue()
        assert bracket_title in text
        assert "Metric Trends" in text

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_graphs_skipped_when_not_tty(
        self, mock_auth: MagicMock, mock_client_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_auth.return_value = ("ws", MagicMock())
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail()
        client.get_project.return_value = _make_project_detail()
        client.get_training_run_metrics.return_value = _make_metrics()

        output = tmp_path / "m.json"
        buf = io.StringIO()
        with _patch_train_console(
            buf, force_terminal=False, width=MIN_TREND_TERMINAL_WIDTH
        ):
            from osmosis_ai.cli.commands.train import metrics

            metrics(
                id="550e8400-e29b-41d4-a716-446655440000",
                output=str(output),
            )

        assert "Metric Trends" not in buf.getvalue()

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_graphs_skipped_when_terminal_narrow(
        self, mock_auth: MagicMock, mock_client_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_auth.return_value = ("ws", MagicMock())
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail()
        client.get_project.return_value = _make_project_detail()
        client.get_training_run_metrics.return_value = _make_metrics()

        output = tmp_path / "m.json"
        buf = io.StringIO()
        with _patch_train_console(
            buf, force_terminal=True, width=MIN_TREND_TERMINAL_WIDTH - 1
        ):
            from osmosis_ai.cli.commands.train import metrics

            metrics(
                id="550e8400-e29b-41d4-a716-446655440000",
                output=str(output),
            )

        assert "Metric Trends" not in buf.getvalue()

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_no_metric_data_unchanged_no_graphs(
        self, mock_auth: MagicMock, mock_client_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_auth.return_value = ("ws", MagicMock())
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail()
        client.get_project.return_value = _make_project_detail()
        client.get_training_run_metrics.return_value = _make_metrics(metrics=[])

        output = tmp_path / "m.json"
        buf = io.StringIO()
        with _patch_train_console(
            buf, force_terminal=True, width=MIN_TREND_TERMINAL_WIDTH
        ):
            from osmosis_ai.cli.commands.train import metrics

            metrics(
                id="550e8400-e29b-41d4-a716-446655440000",
                output=str(output),
            )

        text = buf.getvalue()
        assert "No metric data found." in text
        assert "Metric Trends" not in text
        # Summary table is always shown (run metadata is valuable even without metrics)
        assert "Training Run Metrics" in text


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
        client.get_project.return_value = _make_project_detail()
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
        client.get_project.return_value = _make_project_detail()
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


class TestResolveOutputPath:
    """Test _resolve_output_path smart path resolution."""

    def test_explicit_json_extension_used_as_is(self, tmp_path: Path) -> None:
        from osmosis_ai.cli.commands.train import _resolve_output_path

        result = _resolve_output_path(
            str(tmp_path / "my_metrics.json"), "run-name", "abcd1234"
        )
        assert result == tmp_path / "my_metrics.json"

    def test_no_extension_appends_json(self, tmp_path: Path) -> None:
        from osmosis_ai.cli.commands.train import _resolve_output_path

        result = _resolve_output_path(
            str(tmp_path / "my_metrics"), "run-name", "abcd1234"
        )
        assert result == tmp_path / "my_metrics.json"

    def test_non_json_extension_replaced_with_json(self, tmp_path: Path) -> None:
        from osmosis_ai.cli.commands.train import _resolve_output_path

        result = _resolve_output_path(
            str(tmp_path / "my_metrics.csv"), "run-name", "abcd1234"
        )
        assert result == tmp_path / "my_metrics.json"

    def test_trailing_slash_uses_directory_mode(self, tmp_path: Path) -> None:
        from osmosis_ai.cli.commands.train import _resolve_output_path

        dir_path = tmp_path / "output"
        result = _resolve_output_path(
            str(dir_path) + "/", "reward-tuning", "abcd1234efgh5678"
        )
        assert result == dir_path / "reward-tuning_abcd1234.json"
        assert dir_path.is_dir()

    def test_existing_directory_uses_directory_mode(self, tmp_path: Path) -> None:
        from osmosis_ai.cli.commands.train import _resolve_output_path

        dir_path = tmp_path / "output"
        dir_path.mkdir()
        result = _resolve_output_path(str(dir_path), None, "abcd1234efgh5678")
        assert result == dir_path / "abcd1234.json"

    def test_auto_creates_parent_directories(self, tmp_path: Path) -> None:
        from osmosis_ai.cli.commands.train import _resolve_output_path

        result = _resolve_output_path(
            str(tmp_path / "nested" / "deep" / "metrics"), "run", "abcd1234"
        )
        assert result == tmp_path / "nested" / "deep" / "metrics.json"
        assert result.parent.is_dir()

    def test_trailing_slash_auto_creates_directory(self, tmp_path: Path) -> None:
        from osmosis_ai.cli.commands.train import _resolve_output_path

        dir_path = tmp_path / "new_dir"
        result = _resolve_output_path(str(dir_path) + "/", "my-run", "abcd1234efgh5678")
        assert dir_path.is_dir()
        assert result.parent == dir_path


class TestMetricsCommandErrors:
    """Test error handling in the metrics command."""

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_pending_run_raises(
        self, mock_auth: MagicMock, mock_client_cls: MagicMock
    ) -> None:
        mock_auth.return_value = ("ws", MagicMock())
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail(status="pending")

        from osmosis_ai.cli.commands.train import metrics
        from osmosis_ai.cli.errors import CLIError

        with pytest.raises(CLIError, match="not yet available for pending"):
            metrics(
                id="550e8400-e29b-41d4-a716-446655440000",
                output="/tmp/out.json",
            )

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_running_run_shows_snapshot_note(
        self, mock_auth: MagicMock, mock_client_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_auth.return_value = ("ws", MagicMock())
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail(status="running")
        client.get_project.return_value = _make_project_detail()
        client.get_training_run_metrics.return_value = _make_metrics(status="running")

        output = tmp_path / "m.json"
        buf = io.StringIO()
        with _patch_train_console(buf, force_terminal=False, width=120):
            from osmosis_ai.cli.commands.train import metrics

            metrics(
                id="550e8400-e29b-41d4-a716-446655440000",
                output=str(output),
            )

        text = buf.getvalue()
        assert "training is in progress" in text
        assert output.exists()

    def test_no_workspace_no_output_raises(self, tmp_path: Path) -> None:
        """Without .osmosis/workspace.toml and no -o flag, should error."""
        from osmosis_ai.cli.commands.train import _resolve_default_output
        from osmosis_ai.cli.errors import CLIError

        with pytest.raises(CLIError, match="Not in an Osmosis workspace"):
            _resolve_default_output("my-run", "abc12345", cwd=tmp_path)

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_output_unreachable_path_prints_warning(
        self, mock_auth: MagicMock, mock_client_cls: MagicMock
    ) -> None:
        """Unreachable save path prints warning instead of crashing."""
        mock_auth.return_value = ("ws", MagicMock())
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail()
        client.get_project.return_value = _make_project_detail()
        client.get_training_run_metrics.return_value = _make_metrics()

        buf = io.StringIO()
        with _patch_train_console(buf, force_terminal=False, width=80):
            from osmosis_ai.cli.commands.train import metrics

            metrics(
                id="550e8400-e29b-41d4-a716-446655440000",
                output="/nonexistent/dir/metrics.json",
            )

        text = buf.getvalue()
        assert "Could not save metrics" in text

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_no_output_outside_workspace_prints_warning(
        self,
        mock_auth: MagicMock,
        mock_client_cls: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without -o and outside a workspace dir, metrics still print."""
        mock_auth.return_value = ("ws", MagicMock())
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail()
        client.get_project.return_value = _make_project_detail()
        client.get_training_run_metrics.return_value = _make_metrics()

        monkeypatch.chdir(tmp_path)  # no .osmosis/workspace.toml here

        buf = io.StringIO()
        with _patch_train_console(buf, force_terminal=False, width=80):
            from osmosis_ai.cli.commands.train import metrics

            metrics(
                id="550e8400-e29b-41d4-a716-446655440000",
                output=None,
            )

        text = buf.getvalue()
        assert "Training Run Metrics" in text
        assert "Could not save metrics" in text
