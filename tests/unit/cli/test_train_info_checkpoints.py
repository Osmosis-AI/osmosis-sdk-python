"""Tests for 'osmosis train status' checkpoints section."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

import osmosis_ai.cli.commands.train as train_module
import osmosis_ai.platform.api.client as api_client_module
import osmosis_ai.platform.cli.utils as utils_module
from osmosis_ai.cli.console import Console
from osmosis_ai.cli.output import DetailResult
from osmosis_ai.platform.api.models import (
    LoraCheckpointInfo,
    TrainingRunCheckpoints,
    TrainingRunDetail,
)

GIT_IDENTITY = "acme/rollouts"
REPO_URL = "https://github.com/acme/rollouts.git"
FAKE_CREDENTIALS = object()


@pytest.fixture()
def console_capture(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    output = StringIO()
    console = Console(file=output, force_terminal=False)
    monkeypatch.setattr(train_module, "console", console)
    monkeypatch.setattr(utils_module, "console", console)
    return output


@pytest.fixture(autouse=True)
def _mock_git_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.utils.require_git_project_context",
        lambda: SimpleNamespace(
            project_root=Path("/repo"),
            git_identity=GIT_IDENTITY,
            repo_url=REPO_URL,
            credentials=FAKE_CREDENTIALS,
        ),
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


class TestStatusCheckpoints:
    def test_finished_run_shows_checkpoints_section(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        class FakeClient:
            def get_training_run(self, name, *, git_identity, credentials=None):
                assert credentials is FAKE_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                return _make_finished_run()

            def list_training_run_checkpoints(
                self, name, *, git_identity, credentials=None
            ):
                assert credentials is FAKE_CREDENTIALS
                assert git_identity == GIT_IDENTITY
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
        result = train_module.status(name="qwen3-run1")

        assert isinstance(result, DetailResult)
        checkpoint = result.data["checkpoints"][0]
        assert checkpoint["checkpoint_name"] == "qwen3-run1-step-100"
        assert checkpoint["checkpoint_step"] == 100
        assert checkpoint["id"] == "cp_1"
        checkpoint_fields = [
            field.value for field in result.fields if field.label == "Checkpoint"
        ]
        assert checkpoint_fields == [
            "qwen3-run1-step-100  step 100  [uploaded]  cp_1  2026-04-20"
        ]
        assert ("Deploy", "osmosis deploy <checkpoint-name>") in [
            (field.label, field.value) for field in result.fields
        ]

    def test_running_run_skips_checkpoints(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        called = {"ckpts": False}

        class FakeClient:
            def get_training_run(self, name, *, git_identity, credentials=None):
                assert credentials is FAKE_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                return _make_running_run()

            def list_training_run_checkpoints(
                self, name, *, git_identity, credentials=None
            ):  # pragma: no cover
                assert credentials is FAKE_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                called["ckpts"] = True
                raise AssertionError("should not call for running run")

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        result = train_module.status(name="qwen3-run1")

        assert isinstance(result, DetailResult)
        assert result.data["checkpoints"] == []
        assert all(field.label != "Checkpoint" for field in result.fields)
        assert called["ckpts"] is False

    def test_stopped_run_shows_uploaded_checkpoints(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        class FakeClient:
            def get_training_run(self, name, *, git_identity, credentials=None):
                assert credentials is FAKE_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                return _make_stopped_run()

            def list_training_run_checkpoints(
                self, name, *, git_identity, credentials=None
            ):
                assert credentials is FAKE_CREDENTIALS
                assert git_identity == GIT_IDENTITY
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
        result = train_module.status(name="qwen3-run1")

        assert isinstance(result, DetailResult)
        assert result.data["status"] == "stopped"
        assert result.data["checkpoints"][0]["checkpoint_name"] == (
            "qwen3-run1-step-100"
        )

    def test_endpoint_error_is_non_fatal(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        from osmosis_ai.platform.auth.platform_client import PlatformAPIError

        class FakeClient:
            def get_training_run(self, name, *, git_identity, credentials=None):
                assert credentials is FAKE_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                return _make_finished_run()

            def list_training_run_checkpoints(
                self, name, *, git_identity, credentials=None
            ):
                assert credentials is FAKE_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                raise PlatformAPIError("Internal server error", 500)

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        result = train_module.status(name="qwen3-run1")

        assert isinstance(result, DetailResult)
        assert result.title == "Training Run"
        assert result.data["checkpoints"] == []
        assert all(field.label != "Checkpoint" for field in result.fields)

    def test_finished_but_no_checkpoints(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        class FakeClient:
            def get_training_run(self, name, *, git_identity, credentials=None):
                assert credentials is FAKE_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                return _make_finished_run()

            def list_training_run_checkpoints(
                self, name, *, git_identity, credentials=None
            ):
                assert credentials is FAKE_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                return TrainingRunCheckpoints(
                    training_run_id="run_1",
                    training_run_name="qwen3-run1",
                    checkpoints=[],
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        result = train_module.status(name="qwen3-run1")

        assert isinstance(result, DetailResult)
        assert result.data["checkpoints"] == []
        assert all(
            field.label not in {"Checkpoint", "Deploy"} for field in result.fields
        )
