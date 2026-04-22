"""Tests for osmosis_ai.cli.commands.model."""

from __future__ import annotations

from io import StringIO

import pytest

import osmosis_ai.cli.commands.model as model_module
import osmosis_ai.platform.api.client as api_client_module
import osmosis_ai.platform.cli.utils as utils_module
from osmosis_ai.cli.console import Console
from osmosis_ai.platform.api.models import (
    AffectedTrainingRun,
    BaseModelInfo,
    ModelAffectedResources,
    PaginatedBaseModels,
)


@pytest.fixture()
def console_capture(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    output = StringIO()
    console = Console(file=output, force_terminal=False)
    monkeypatch.setattr(model_module, "console", console)
    monkeypatch.setattr(utils_module, "console", console)
    return output


@pytest.fixture(autouse=True)
def _mock_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.utils._require_auth",
        lambda: ("ws-test", object()),
    )


class TestListModels:
    def test_empty_list(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        class FakeClient:
            def list_base_models(self, limit=30, offset=0, *, credentials=None):
                return PaginatedBaseModels(models=[], total_count=0, has_more=False)

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        model_module.list_models(limit=30, all_=False)
        assert "No models found" in console_capture.getvalue()

    def test_list_with_base_models(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        base = BaseModelInfo(
            id="model_base_12345678901234567890",
            model_name="gpt-2",
            status="available",
            creator_name="openai",
            created_at="2025-06-01T00:00:00Z",
        )

        class FakeClient:
            def list_base_models(self, limit=30, offset=0, *, credentials=None):
                return PaginatedBaseModels(models=[base], total_count=1, has_more=False)

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        model_module.list_models(limit=30, all_=False)
        out = console_capture.getvalue()
        assert "gpt-2" in out
        assert "Base Models" in out

    def test_list_has_more_truncation(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        base = BaseModelInfo(
            id="model_base_12345678901234567890",
            model_name="model-a",
            status="available",
            created_at="2025-01-01",
        )

        class FakeClient:
            def list_base_models(self, limit=1, offset=0, *, credentials=None):
                return PaginatedBaseModels(models=[base], total_count=5, has_more=True)

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        model_module.list_models(limit=1, all_=False)
        out = console_capture.getvalue()
        assert "Showing 1 of 5 base models" in out
        assert "--all" in out


class TestDeleteModel:
    def test_happy_path(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured: dict[str, object] = {}

        class FakeClient:
            def get_model_affected_resources(self, name, *, credentials=None):
                return ModelAffectedResources(training_runs_using_model=[])

            def delete_model(self, name, *, credentials=None):
                captured["deleted"] = name
                return True

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        model_module.delete(name="Qwen/Qwen3", yes=True)
        assert captured["deleted"] == "Qwen/Qwen3"
        assert 'Model "Qwen/Qwen3" deleted' in console_capture.getvalue()

    def test_blocked_by_training_runs(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        import typer

        class FakeClient:
            def get_model_affected_resources(self, name, *, credentials=None):
                return ModelAffectedResources(
                    training_runs_using_model=[
                        AffectedTrainingRun(id="run_abcdef12", training_run_name="r1"),
                    ]
                )

            def delete_model(
                self, name, *, credentials=None
            ):  # pragma: no cover — blocked
                raise AssertionError("should not delete when blocked")

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        with pytest.raises(typer.Exit):
            model_module.delete(name="Qwen/Qwen3", yes=True)
        out = console_capture.getvalue()
        assert "Cannot delete this model" in out
        assert "r1" in out

    def test_affected_resources_api_error(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        from osmosis_ai.cli.errors import CLIError
        from osmosis_ai.platform.auth.platform_client import PlatformAPIError

        class FakeClient:
            def get_model_affected_resources(self, name, *, credentials=None):
                raise PlatformAPIError("boom", 500)

            def delete_model(self, name, *, credentials=None):  # pragma: no cover
                raise AssertionError("should not delete")

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        with pytest.raises(CLIError, match="verify model dependencies"):
            model_module.delete(name="Qwen/Qwen3", yes=True)
