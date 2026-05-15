"""Tests for the osmosis train metrics CLI command."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console as RichConsole

from osmosis_ai.cli.console import Console
from osmosis_ai.cli.metrics_graph import MIN_TREND_TERMINAL_WIDTH, SPARKLINE_BLOCKS
from osmosis_ai.cli.output import DetailResult
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
        platform_url="https://platform.osmosis.ai/acme/rollouts/training/550e8400-e29b-41d4-a716-446655440000",
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
_PATCH_AUTH = "osmosis_ai.platform.cli.utils.require_git_workspace_directory_context"
_PATCH_CLIENT = "osmosis_ai.platform.api.client.OsmosisClient"


def _make_git_context(
    *, workspace_directory: Path | None = None, credentials: object | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        workspace_directory=workspace_directory or Path.cwd(),
        git_identity="acme/rollouts",
        repo_url="https://github.com/acme/rollouts.git",
        credentials=credentials or MagicMock(),
    )


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


def _field_value(result: DetailResult, label: str) -> str:
    for field in result.fields:
        if field.label == label:
            return field.value
    raise AssertionError(f"Missing field {label!r}")


def _render_rich_text(value: object) -> str:
    buffer = io.StringIO()
    rich = RichConsole(file=buffer, force_terminal=False, no_color=True, width=120)
    rich.print(value)
    return buffer.getvalue()


def _render_section_text(result: DetailResult) -> str:
    assert result.sections
    return _render_rich_text(result.sections[0].rich)


class TestMetricsCommandPlatformUrl:
    """Platform URL is printed after the summary table."""

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_platform_url_printed(
        self, mock_auth: MagicMock, mock_client_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_auth.return_value = _make_git_context(workspace_directory=tmp_path)
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail()
        client.get_training_run_metrics.return_value = _make_metrics()

        output = tmp_path / "m.json"
        buf = io.StringIO()
        with _patch_train_console(buf, force_terminal=False, width=120):
            from osmosis_ai.cli.commands.train import metrics

            result = metrics(
                name="reward-tuning-v3",
                output=str(output),
            )

        assert isinstance(result, DetailResult)
        assert all(field.label != "View" for field in result.fields)
        assert result.display_hints
        assert "acme/rollouts/training/" in result.display_hints[0]
        assert "550e8400-e29b-41d4-a716-446655440000" in result.display_hints[0]
        client.get_training_run.assert_called_once_with(
            "reward-tuning-v3",
            credentials=mock_auth.return_value.credentials,
            git_identity="acme/rollouts",
        )
        client.get_training_run_metrics.assert_called_once_with(
            "550e8400-e29b-41d4-a716-446655440000",
            credentials=mock_auth.return_value.credentials,
            git_identity="acme/rollouts",
        )


class TestMetricsCommandTrendGraphs:
    """Trend graphs after the summary table when TTY + width allow."""

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_graphs_render_after_summary_when_tty_and_width_sufficient(
        self, mock_auth: MagicMock, mock_client_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_auth.return_value = _make_git_context(workspace_directory=tmp_path)
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail()
        client.get_training_run_metrics.return_value = _make_metrics()

        output = tmp_path / "m.json"
        buf = io.StringIO()
        with _patch_train_console(
            buf, force_terminal=True, width=MIN_TREND_TERMINAL_WIDTH
        ):
            from osmosis_ai.cli.commands.train import metrics

            result = metrics(
                name="reward-tuning-v3",
                output=str(output),
            )

        assert isinstance(result, DetailResult)
        assert result.title == "Training Run Metrics"
        assert all(field.label != "Metric Trends" for field in result.fields)
        assert result.sections
        trends = _render_section_text(result)
        assert "Training Reward" in trends
        assert any(c in trends for c in SPARKLINE_BLOCKS)

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_trend_block_prints_metric_title_with_brackets_literally(
        self, mock_auth: MagicMock, mock_client_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Bracket characters in metric titles must not be interpreted as Rich markup."""
        bracket_title = "Loss [eval]"
        mock_auth.return_value = _make_git_context(workspace_directory=tmp_path)
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail()
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

            result = metrics(
                name="reward-tuning-v3",
                output=str(output),
            )

        assert isinstance(result, DetailResult)
        assert all(field.label != "Metric Trends" for field in result.fields)
        trends = _render_section_text(result)
        assert bracket_title in trends

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_graphs_skipped_when_not_tty(
        self, mock_auth: MagicMock, mock_client_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_auth.return_value = _make_git_context(workspace_directory=tmp_path)
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail()
        client.get_training_run_metrics.return_value = _make_metrics()

        output = tmp_path / "m.json"
        buf = io.StringIO()
        with _patch_train_console(
            buf, force_terminal=False, width=MIN_TREND_TERMINAL_WIDTH
        ):
            from osmosis_ai.cli.commands.train import metrics

            metrics(
                name="reward-tuning-v3",
                output=str(output),
            )

        assert "Metric Trends" not in buf.getvalue()

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_graphs_skipped_when_terminal_narrow(
        self, mock_auth: MagicMock, mock_client_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_auth.return_value = _make_git_context(workspace_directory=tmp_path)
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail()
        client.get_training_run_metrics.return_value = _make_metrics()

        output = tmp_path / "m.json"
        buf = io.StringIO()
        with _patch_train_console(
            buf, force_terminal=True, width=MIN_TREND_TERMINAL_WIDTH - 1
        ):
            from osmosis_ai.cli.commands.train import metrics

            metrics(
                name="reward-tuning-v3",
                output=str(output),
            )

        assert "Metric Trends" not in buf.getvalue()

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_no_metric_data_unchanged_no_graphs(
        self, mock_auth: MagicMock, mock_client_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_auth.return_value = _make_git_context(workspace_directory=tmp_path)
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail()
        client.get_training_run_metrics.return_value = _make_metrics(metrics=[])

        output = tmp_path / "m.json"
        buf = io.StringIO()
        with _patch_train_console(
            buf, force_terminal=True, width=MIN_TREND_TERMINAL_WIDTH
        ):
            from osmosis_ai.cli.commands.train import metrics

            result = metrics(
                name="reward-tuning-v3",
                output=str(output),
            )

        assert isinstance(result, DetailResult)
        assert _field_value(result, "Metrics") == "No metric data found."
        assert all(field.label != "Metric Trends" for field in result.fields)
        # Summary table is always shown (run metadata is valuable even without metrics)
        assert result.title == "Training Run Metrics"


class TestMetricsCommandWritesFile:
    """Test that the metrics command writes the correct JSON file."""

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_writes_json_to_explicit_output(
        self, mock_auth: MagicMock, mock_client_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_auth.return_value = _make_git_context(workspace_directory=tmp_path)
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail()
        client.get_training_run_metrics.return_value = _make_metrics()

        output = tmp_path / "metrics.json"

        from osmosis_ai.cli.commands.train import metrics

        metrics(
            name="reward-tuning-v3",
            output=str(output),
        )

        assert output.exists()
        data = json.loads(output.read_text())
        assert data["training_run"]["name"] == "reward-tuning-v3"
        assert data["summary"]["final_reward"] == 0.85
        assert data["metrics"][0]["key"] == "training_reward"

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_writes_to_default_path_in_project(
        self,
        mock_auth: MagicMock,
        mock_client_cls: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_auth.return_value = _make_git_context(workspace_directory=tmp_path)
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail()
        client.get_training_run_metrics.return_value = _make_metrics()

        osmosis_dir = tmp_path / ".osmosis"
        monkeypatch.chdir(tmp_path)

        from osmosis_ai.cli.commands.train import metrics

        metrics(
            name="reward-tuning-v3",
            output=None,
        )

        expected = osmosis_dir / "metrics" / "reward-tuning-v3_550e8400.json"
        assert expected.exists()
        data = json.loads(expected.read_text())
        assert data["training_run"]["id"] == "550e8400-e29b-41d4-a716-446655440000"

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_writes_to_default_path_from_project_subdirectory(
        self,
        mock_auth: MagicMock,
        mock_client_cls: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_auth.return_value = _make_git_context(workspace_directory=tmp_path)
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail()
        client.get_training_run_metrics.return_value = _make_metrics()

        nested = tmp_path / "configs" / "training"
        nested.mkdir(parents=True)
        monkeypatch.chdir(nested)

        from osmosis_ai.cli.commands.train import metrics

        result = metrics(
            name="reward-tuning-v3",
            output=None,
        )

        expected = tmp_path / ".osmosis" / "metrics" / "reward-tuning-v3_550e8400.json"
        assert expected.exists()
        assert result.data["output_path"] == str(expected)
        assert f"Saved metrics to {expected}" in result.display_hints
        assert result.data["save_warning"] is None
        assert all(field.label != "Saved" for field in result.fields)
        assert any(str(expected) in hint for hint in result.display_hints)


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
        mock_auth.return_value = _make_git_context()
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail(status="pending")

        from osmosis_ai.cli.commands.train import metrics
        from osmosis_ai.cli.errors import CLIError

        with pytest.raises(CLIError, match="not yet available for pending"):
            metrics(
                name="reward-tuning-v3",
                output="/tmp/out.json",
            )

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_running_run_shows_snapshot_note(
        self, mock_auth: MagicMock, mock_client_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_auth.return_value = _make_git_context(workspace_directory=tmp_path)
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail(status="running")
        client.get_training_run_metrics.return_value = _make_metrics(status="running")

        output = tmp_path / "m.json"
        buf = io.StringIO()
        with _patch_train_console(buf, force_terminal=False, width=120):
            from osmosis_ai.cli.commands.train import metrics

            result = metrics(
                name="reward-tuning-v3",
                output=str(output),
            )

        assert isinstance(result, DetailResult)
        assert "Training is in progress" in _field_value(result, "Note")
        assert output.exists()

    def test_default_output_does_not_require_project_marker(
        self, tmp_path: Path
    ) -> None:
        from osmosis_ai.cli.commands.train import _resolve_default_output

        result = _resolve_default_output(
            "my-run", "abc12345", workspace_directory=tmp_path
        )

        assert result == tmp_path / ".osmosis" / "metrics" / "my-run_abc12345.json"
        assert result.parent.is_dir()

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_output_unreachable_path_prints_warning(
        self, mock_auth: MagicMock, mock_client_cls: MagicMock
    ) -> None:
        """Unreachable save path prints warning instead of crashing."""
        mock_auth.return_value = _make_git_context()
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail()
        client.get_training_run_metrics.return_value = _make_metrics()

        buf = io.StringIO()
        with _patch_train_console(buf, force_terminal=False, width=80):
            from osmosis_ai.cli.commands.train import metrics

            result = metrics(
                name="reward-tuning-v3",
                output="/nonexistent/dir/metrics.json",
            )

        assert isinstance(result, DetailResult)
        assert all(field.label != "Warning" for field in result.fields)
        assert result.data["save_warning"] is not None
        assert "Could not save metrics" in result.data["save_warning"]
        assert any("Could not save metrics" in hint for hint in result.display_hints)

    @patch(_PATCH_CLIENT)
    @patch(_PATCH_AUTH)
    def test_no_output_without_project_marker_writes_under_git_root(
        self,
        mock_auth: MagicMock,
        mock_client_cls: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_auth.return_value = _make_git_context(workspace_directory=tmp_path)
        client = mock_client_cls.return_value
        client.get_training_run.return_value = _make_run_detail()
        client.get_training_run_metrics.return_value = _make_metrics()

        nested = tmp_path / "rollouts" / "demo"
        nested.mkdir(parents=True)
        monkeypatch.chdir(nested)

        buf = io.StringIO()
        with _patch_train_console(buf, force_terminal=False, width=80):
            from osmosis_ai.cli.commands.train import metrics

            result = metrics(
                name="reward-tuning-v3",
                output=None,
            )

        expected = tmp_path / ".osmosis" / "metrics" / "reward-tuning-v3_550e8400.json"
        assert isinstance(result, DetailResult)
        assert result.title == "Training Run Metrics"
        assert expected.exists()
        assert result.data["output_path"] == str(expected)
        assert f"Saved metrics to {expected}" in result.display_hints
