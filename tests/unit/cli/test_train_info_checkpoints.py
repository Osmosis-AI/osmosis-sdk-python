"""Tests for 'osmosis train info' checkpoints section."""

from __future__ import annotations

from io import StringIO

import pytest

import osmosis_ai.cli.commands.train as train_module
import osmosis_ai.platform.api.client as api_client_module
import osmosis_ai.platform.cli.utils as utils_module
from osmosis_ai.cli.console import Console
from osmosis_ai.platform.api.models import (
    LoraCheckpointInfo,
    TrainingRunCheckpoints,
    TrainingRunDetail,
)


@pytest.fixture()
def console_capture(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    output = StringIO()
    console = Console(file=output, force_terminal=False)
    monkeypatch.setattr(train_module, "console", console)
    monkeypatch.setattr(utils_module, "console", console)
    return output


@pytest.fixture(autouse=True)
def _mock_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.utils._require_auth",
        lambda: ("ws-test", object()),
    )


def _make_finished_run() -> TrainingRunDetail:
    return TrainingRunDetail(
        id="run_1",
        name="qwen3-run1",
        status="finished",
        model_name="Qwen/Qwen3",
        created_at="2026-04-01T00:00:00Z",
    )


def _make_running_run() -> TrainingRunDetail:
    return TrainingRunDetail(
        id="run_1",
        name="qwen3-run1",
        status="running",
        model_name="Qwen/Qwen3",
        created_at="2026-04-01T00:00:00Z",
    )


def _make_stopped_run() -> TrainingRunDetail:
    return TrainingRunDetail(
        id="run_1",
        name="qwen3-run1",
        status="stopped",
        model_name="Qwen/Qwen3",
        created_at="2026-04-01T00:00:00Z",
    )


class TestInfoCheckpoints:
    def test_finished_run_shows_checkpoints_section(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        class FakeClient:
            def get_training_run(self, name, *, credentials=None):
                return _make_finished_run()

            def list_training_run_checkpoints(self, name, *, credentials=None):
                return TrainingRunCheckpoints(
                    training_run_id="run_1",
                    training_run_name="qwen3-run1",
                    checkpoints=[
                        LoraCheckpointInfo(
                            id="cp_1",
                            checkpoint_step=100,
                            status="uploaded",
                            checkpoint_name="qwen3-run1-step-100",
                            created_at="2026-04-20T00:00:00Z",
                        )
                    ],
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.info(name="qwen3-run1")
        out = console_capture.getvalue()
        assert "Checkpoints" in out
        assert "qwen3-run1-step-100" in out
        assert "step 100" in out
        assert "cp_1"[:8] in out
        assert "osmosis deploy <checkpoint-name>" in out

    def test_running_run_skips_checkpoints(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        called = {"ckpts": False}

        class FakeClient:
            def get_training_run(self, name, *, credentials=None):
                return _make_running_run()

            def list_training_run_checkpoints(
                self, name, *, credentials=None
            ):  # pragma: no cover
                called["ckpts"] = True
                raise AssertionError("should not call for running run")

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.info(name="qwen3-run1")
        out = console_capture.getvalue()
        assert "Checkpoints" not in out
        assert called["ckpts"] is False

    def test_stopped_run_shows_uploaded_checkpoints(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        class FakeClient:
            def get_training_run(self, name, *, credentials=None):
                return _make_stopped_run()

            def list_training_run_checkpoints(self, name, *, credentials=None):
                return TrainingRunCheckpoints(
                    training_run_id="run_1",
                    training_run_name="qwen3-run1",
                    checkpoints=[
                        LoraCheckpointInfo(
                            id="cp_stopped_1",
                            checkpoint_step=100,
                            status="uploaded",
                            checkpoint_name="qwen3-run1-step-100",
                            created_at="2026-04-20T00:00:00Z",
                        )
                    ],
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.info(name="qwen3-run1")
        out = console_capture.getvalue()
        assert "stopped" in out
        assert "Checkpoints" in out
        assert "qwen3-run1-step-100" in out

    def test_endpoint_error_is_non_fatal(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        from osmosis_ai.platform.auth.platform_client import PlatformAPIError

        class FakeClient:
            def get_training_run(self, name, *, credentials=None):
                return _make_finished_run()

            def list_training_run_checkpoints(self, name, *, credentials=None):
                raise PlatformAPIError("Internal server error", 500)

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.info(name="qwen3-run1")
        out = console_capture.getvalue()
        assert "Checkpoints" not in out
        assert "Training Run" in out

    def test_finished_but_no_checkpoints(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        class FakeClient:
            def get_training_run(self, name, *, credentials=None):
                return _make_finished_run()

            def list_training_run_checkpoints(self, name, *, credentials=None):
                return TrainingRunCheckpoints(
                    training_run_id="run_1",
                    training_run_name="qwen3-run1",
                    checkpoints=[],
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.info(name="qwen3-run1")
        out = console_capture.getvalue()
        assert "Checkpoints" not in out
        assert "Deploy with:" not in out
