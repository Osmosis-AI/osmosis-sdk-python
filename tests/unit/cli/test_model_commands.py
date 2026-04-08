"""Tests for osmosis_ai.cli.commands.model."""

from __future__ import annotations

from io import StringIO

import pytest

import osmosis_ai.cli.commands.model as model_module
import osmosis_ai.platform.api.client as api_client_module
import osmosis_ai.platform.cli.utils as utils_module
from osmosis_ai.cli.console import Console
from osmosis_ai.platform.api.models import (
    BaseModelInfo,
    OutputModelInfo,
    PaginatedBaseModels,
    PaginatedOutputModels,
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
            def fetch_all_models(self, *, limit=30, credentials=None):
                return (
                    PaginatedBaseModels(models=[], total_count=0, has_more=False),
                    PaginatedOutputModels(models=[], total_count=0, has_more=False),
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        model_module.list_models(limit=30, all_=False)
        assert "No models found" in console_capture.getvalue()

    def test_list_with_output_models(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        out_model = OutputModelInfo(
            id="model_out_12345678901234567890",
            model_name="my-finetuned",
            status="uploaded",
            training_run_name="run-1",
            created_at="2026-01-15T00:00:00Z",
        )

        class FakeClient:
            def fetch_all_models(self, *, limit=30, credentials=None):
                return (
                    PaginatedBaseModels(models=[], total_count=0, has_more=False),
                    PaginatedOutputModels(
                        models=[out_model], total_count=1, has_more=False
                    ),
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        model_module.list_models(limit=30, all_=False)
        out = console_capture.getvalue()
        assert "my-finetuned" in out
        assert "Output Models" in out

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
            def fetch_all_models(self, *, limit=30, credentials=None):
                return (
                    PaginatedBaseModels(models=[base], total_count=1, has_more=False),
                    PaginatedOutputModels(models=[], total_count=0, has_more=False),
                )

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
            def fetch_all_models(self, *, limit=1, credentials=None):
                return (
                    PaginatedBaseModels(models=[base], total_count=5, has_more=True),
                    PaginatedOutputModels(models=[], total_count=0, has_more=False),
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        model_module.list_models(limit=1, all_=False)
        out = console_capture.getvalue()
        assert "Showing 1 of 5 models" in out
        assert "--all" in out
