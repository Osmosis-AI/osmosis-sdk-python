"""Regression tests for dataset status rendering in Rich mode."""

from __future__ import annotations

import re
from io import StringIO
from types import SimpleNamespace

import pytest

import osmosis_ai.platform.api.client as api_client_module
import osmosis_ai.platform.cli.dataset as dataset_module
import osmosis_ai.platform.cli.workspace as workspace_module
from osmosis_ai.cli.console import Console
from osmosis_ai.platform.api.models import DatasetFile, PaginatedDatasets

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _make_rich_console() -> tuple[Console, StringIO]:
    output = StringIO()
    console = Console(file=output, force_terminal=True)
    assert console.use_rich is True
    return console, output


def _strip_ansi(text: str) -> str:
    """Normalize Rich-rendered output for version-independent assertions."""
    return ANSI_ESCAPE_RE.sub("", text)


@pytest.mark.parametrize("status", ["cancelled", "deleted"])
def test_list_datasets_preserves_uncategorized_status_brackets(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    console, output = _make_rich_console()
    fake_credentials = object()

    monkeypatch.setattr(dataset_module, "console", console)
    monkeypatch.setattr(
        dataset_module, "_require_auth", lambda: ("ws-b", fake_credentials)
    )
    monkeypatch.setattr(
        dataset_module,
        "_resolve_project_id",
        lambda _project, *, workspace_name: "proj_123",
    )

    class FakeClient:
        def list_datasets(
            self,
            project_id: str,
            *,
            credentials=None,
        ) -> PaginatedDatasets:
            assert project_id == "proj_123"
            assert credentials is fake_credentials
            return PaginatedDatasets(
                datasets=[
                    DatasetFile(
                        id="abcdef123456",
                        file_name="data.jsonl",
                        file_size=123,
                        status=status,
                    )
                ],
                total_count=1,
                has_more=False,
            )

    monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)

    dataset_module.list_datasets(project=None)

    assert f"[{status}]" in _strip_ansi(output.getvalue())


@pytest.mark.parametrize("status", ["cancelled", "deleted"])
def test_workspace_status_format_preserves_uncategorized_status_brackets(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    console, output = _make_rich_console()
    monkeypatch.setattr(workspace_module, "console", console)

    dataset = SimpleNamespace(
        status=status,
        processing_step=None,
        processing_percent=None,
    )

    console.print(workspace_module._format_dataset_status(dataset))

    assert f"[{status}]" in _strip_ansi(output.getvalue())


def test_workspace_status_format_preserves_plain_text_brackets() -> None:
    console = Console(file=StringIO(), force_terminal=False)
    dataset = SimpleNamespace(
        status="deleted",
        processing_step=None,
        processing_percent=None,
    )

    original_console = workspace_module.console
    workspace_module.console = console
    try:
        assert workspace_module._format_dataset_status(dataset) == "[deleted]"
    finally:
        workspace_module.console = original_console
