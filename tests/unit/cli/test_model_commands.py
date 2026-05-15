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
from osmosis_ai.cli.output import ListResult
from osmosis_ai.platform.api.models import (
    BaseModelInfo,
    PaginatedBaseModels,
)

AUTH_CREDENTIALS = object()
GIT_IDENTITY = "acme/rollouts"
REPO_URL = "https://github.com/acme/rollouts.git"
PROJECT_ROOT = Path("/repo")


def assert_git_context(data: dict[str, object]) -> None:
    assert data["workspace_directory"] == "/repo"
    assert data["git"] == {
        "identity": GIT_IDENTITY,
        "remote_url": REPO_URL,
    }
    assert "workspace" not in data


@pytest.fixture()
def console_capture(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    output = StringIO()
    console = Console(file=output, force_terminal=False)
    monkeypatch.setattr(utils_module, "console", console)
    return output


@pytest.fixture()
def mock_git_context(monkeypatch: pytest.MonkeyPatch) -> None:
    context = SimpleNamespace(
        workspace_directory=PROJECT_ROOT,
        git_identity=GIT_IDENTITY,
        repo_url=REPO_URL,
        credentials=AUTH_CREDENTIALS,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.utils.require_git_workspace_directory_context",
        lambda: context,
    )


def test_model_list_requires_linked_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from osmosis_ai.cli.main import main

    monkeypatch.chdir(tmp_path)

    rc = main(["--json", "model", "list"])

    assert rc == 1
    assert "cloned Osmosis repository" in capsys.readouterr().err


@pytest.mark.usefixtures("mock_git_context")
class TestListModels:
    def test_empty_list(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        class FakeClient:
            def list_base_models(
                self, limit=30, offset=0, *, git_identity, credentials=None
            ):
                assert credentials is AUTH_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                return PaginatedBaseModels(models=[], total_count=0, has_more=False)

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        result = model_module.list_models(limit=30, all_=False)

        assert isinstance(result, ListResult)
        assert result.title == "Base Models"
        assert result.items == []
        assert result.total_count == 0
        assert result.has_more is False
        assert_git_context(result.extra)

    def test_list_with_base_models(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        base = BaseModelInfo(
            id="model_base_12345678901234567890",
            model_name="gpt-2",
            creator_name="openai",
            created_at="2025-06-01T00:00:00Z",
        )

        class FakeClient:
            def list_base_models(
                self, limit=30, offset=0, *, git_identity, credentials=None
            ):
                assert credentials is AUTH_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                return PaginatedBaseModels(models=[base], total_count=1, has_more=False)

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        result = model_module.list_models(limit=30, all_=False)

        assert isinstance(result, ListResult)
        assert result.title == "Base Models"
        assert result.items[0]["model_name"] == "gpt-2"
        assert "status" not in result.items[0]

    def test_list_display_columns_use_name_label_and_hide_id(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        base = BaseModelInfo(
            id="model_base_12345678901234567890",
            model_name="Qwen/Qwen3-8B",
            base_model="Qwen/Qwen3",
            created_at="2025-06-01T00:00:00Z",
        )

        class FakeClient:
            def list_base_models(
                self, limit=30, offset=0, *, git_identity, credentials=None
            ):
                assert git_identity == GIT_IDENTITY
                return PaginatedBaseModels(models=[base], total_count=1, has_more=False)

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        result = model_module.list_models(limit=30, all_=False)

        assert [column.label for column in result.columns] == [
            "Name",
            "Base",
            result.columns[2].label,
        ]
        assert result.columns[0].key == "model_name"
        assert result.columns[0].ratio == 4
        assert result.columns[0].overflow == "fold"
        assert result.columns[2].label.startswith("Created (")
        assert all(column.key != "id" for column in result.columns)
        assert all(column.key != "status" for column in result.columns)

    def test_list_has_more_truncation(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        base = BaseModelInfo(
            id="model_base_12345678901234567890",
            model_name="model-a",
            created_at="2025-01-01",
        )

        class FakeClient:
            def list_base_models(
                self, limit=1, offset=0, *, git_identity, credentials=None
            ):
                assert git_identity == GIT_IDENTITY
                return PaginatedBaseModels(models=[base], total_count=5, has_more=True)

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        result = model_module.list_models(limit=1, all_=False)

        assert isinstance(result, ListResult)
        assert len(result.items) == 1
        assert result.total_count == 5
        assert result.has_more is True
