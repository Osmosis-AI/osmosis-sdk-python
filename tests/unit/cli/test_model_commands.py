"""Tests for osmosis_ai.cli.commands.model."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

import osmosis_ai.cli.commands.model as model_module
import osmosis_ai.platform.api.client as api_client_module
import osmosis_ai.platform.cli.utils as utils_module
from osmosis_ai.cli.console import Console
from osmosis_ai.cli.output import ListResult, OperationResult
from osmosis_ai.platform.api.models import (
    AffectedTrainingRun,
    BaseModelInfo,
    ModelAffectedResources,
    PaginatedBaseModels,
)

AUTH_CREDENTIALS = object()
WORKSPACE_ID = "ws-test"
WORKSPACE_NAME = "team-test"
PROJECT_ROOT = Path("/tmp/osmosis-project")


@pytest.fixture()
def console_capture(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    output = StringIO()
    console = Console(file=output, force_terminal=False)
    monkeypatch.setattr(model_module, "console", console)
    monkeypatch.setattr(utils_module, "console", console)
    return output


@pytest.fixture()
def mock_workspace_context(monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = SimpleNamespace(
        project_root=PROJECT_ROOT,
        workspace_id=WORKSPACE_ID,
        workspace_name=WORKSPACE_NAME,
        credentials=AUTH_CREDENTIALS,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.utils.require_workspace_context",
        lambda: workspace,
    )


def test_model_list_requires_linked_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from osmosis_ai.cli.main import main

    monkeypatch.chdir(tmp_path)

    rc = main(["--json", "model", "list"])

    assert rc == 1
    assert "Not in an Osmosis project" in capsys.readouterr().err


@pytest.mark.usefixtures("mock_workspace_context")
class TestListModels:
    def test_empty_list(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        class FakeClient:
            def list_base_models(
                self, limit=30, offset=0, *, workspace_id, credentials=None
            ):
                assert credentials is AUTH_CREDENTIALS
                assert workspace_id == WORKSPACE_ID
                return PaginatedBaseModels(models=[], total_count=0, has_more=False)

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        result = model_module.list_models(limit=30, all_=False)

        assert isinstance(result, ListResult)
        assert result.title == "Base Models"
        assert result.items == []
        assert result.total_count == 0
        assert result.has_more is False
        assert result.extra == {
            "workspace": {"id": WORKSPACE_ID, "name": WORKSPACE_NAME},
            "project_root": str(PROJECT_ROOT),
        }

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
            def list_base_models(
                self, limit=30, offset=0, *, workspace_id, credentials=None
            ):
                assert credentials is AUTH_CREDENTIALS
                assert workspace_id == WORKSPACE_ID
                return PaginatedBaseModels(models=[base], total_count=1, has_more=False)

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        result = model_module.list_models(limit=30, all_=False)

        assert isinstance(result, ListResult)
        assert result.title == "Base Models"
        assert result.items[0]["model_name"] == "gpt-2"
        assert result.items[0]["status"] == "available"

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
            def list_base_models(
                self, limit=1, offset=0, *, workspace_id, credentials=None
            ):
                assert workspace_id == WORKSPACE_ID
                return PaginatedBaseModels(models=[base], total_count=5, has_more=True)

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        result = model_module.list_models(limit=1, all_=False)

        assert isinstance(result, ListResult)
        assert len(result.items) == 1
        assert result.total_count == 5
        assert result.has_more is True


@pytest.mark.usefixtures("mock_workspace_context")
class TestDeleteModel:
    def test_happy_path(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured: dict[str, object] = {}

        class FakeClient:
            def get_model_affected_resources(
                self, name, *, workspace_id, credentials=None
            ):
                assert credentials is AUTH_CREDENTIALS
                assert workspace_id == WORKSPACE_ID
                return ModelAffectedResources(training_runs_using_model=[])

            def delete_model(self, name, *, workspace_id, credentials=None):
                assert credentials is AUTH_CREDENTIALS
                assert workspace_id == WORKSPACE_ID
                captured["deleted"] = name
                return True

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        result = model_module.delete(name="Qwen/Qwen3", yes=True)

        assert captured["deleted"] == "Qwen/Qwen3"
        assert isinstance(result, OperationResult)
        assert result.operation == "model.delete"
        assert result.status == "success"
        assert result.resource == {
            "name": "Qwen/Qwen3",
            "workspace": {"id": WORKSPACE_ID, "name": WORKSPACE_NAME},
            "project_root": str(PROJECT_ROOT),
        }
        assert result.message == 'Model "Qwen/Qwen3" deleted.'

    def test_blocked_by_training_runs(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        import typer

        class FakeClient:
            def get_model_affected_resources(
                self, name, *, workspace_id, credentials=None
            ):
                assert workspace_id == WORKSPACE_ID
                return ModelAffectedResources(
                    training_runs_using_model=[
                        AffectedTrainingRun(id="run_abcdef12", training_run_name="r1"),
                    ]
                )

            def delete_model(
                self, name, *, workspace_id, credentials=None
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
            def get_model_affected_resources(
                self, name, *, workspace_id, credentials=None
            ):
                assert workspace_id == WORKSPACE_ID
                raise PlatformAPIError("boom", 500)

            def delete_model(  # pragma: no cover
                self, name, *, workspace_id, credentials=None
            ):
                raise AssertionError("should not delete")

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        with pytest.raises(CLIError, match="verify model dependencies"):
            model_module.delete(name="Qwen/Qwen3", yes=True)
