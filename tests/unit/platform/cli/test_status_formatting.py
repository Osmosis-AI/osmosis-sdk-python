"""Regression tests for dataset status rendering in Rich mode."""

from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

import pytest

import osmosis_ai.platform.api.client as api_client_module
import osmosis_ai.platform.cli.dataset as dataset_module
import osmosis_ai.platform.cli.utils as utils_module
from osmosis_ai.cli.console import Console
from osmosis_ai.platform.api.models import DatasetFile, PaginatedDatasets
from tests.unit.platform.cli.conftest import strip_ansi


def _make_rich_console() -> tuple[Console, StringIO]:
    output = StringIO()
    console = Console(file=output, force_terminal=True)
    assert console.is_tty is True
    return console, output


@pytest.mark.parametrize("status", ["cancelled", "deleted"])
def test_list_datasets_preserves_uncategorized_status_brackets(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    console, output = _make_rich_console()
    fake_credentials = object()

    monkeypatch.setattr(dataset_module, "console", console)
    monkeypatch.setattr(utils_module, "console", console)
    monkeypatch.setattr(
        dataset_module, "_require_auth", lambda: ("ws-b", fake_credentials)
    )

    class FakeClient:
        def list_datasets(
            self,
            *,
            credentials=None,
        ) -> PaginatedDatasets:
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

    dataset_module.list_datasets()

    assert f"[{status}]" in strip_ansi(output.getvalue())


@pytest.mark.parametrize("status", ["cancelled", "deleted"])
def test_workspace_status_format_preserves_uncategorized_status_brackets(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    console, output = _make_rich_console()
    monkeypatch.setattr(utils_module, "console", console)

    dataset = SimpleNamespace(
        status=status,
        processing_step=None,
        processing_percent=None,
    )

    console.print(utils_module.format_dataset_status(dataset))

    assert f"[{status}]" in strip_ansi(output.getvalue())


def test_workspace_status_format_preserves_plain_text_brackets() -> None:
    console = Console(file=StringIO(), force_terminal=False)
    dataset = SimpleNamespace(
        status="deleted",
        processing_step=None,
        processing_percent=None,
    )

    original_console = utils_module.console
    utils_module.console = console
    try:
        assert utils_module.format_dataset_status(dataset) == "\\[deleted]"
    finally:
        utils_module.console = original_console
