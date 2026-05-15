"""Regression tests for dataset status rendering in Rich mode."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

import osmosis_ai.platform.api.client as api_client_module
import osmosis_ai.platform.cli.dataset as dataset_module
import osmosis_ai.platform.cli.utils as utils_module
from osmosis_ai.cli.console import Console
from osmosis_ai.platform.api.models import DatasetFile, PaginatedDatasets
from tests.unit.platform.cli.conftest import strip_ansi

GIT_IDENTITY = "acme/rollouts"
REPO_URL = "https://github.com/acme/rollouts.git"


def _make_rich_console() -> tuple[Console, StringIO]:
    output = StringIO()
    console = Console(file=output, force_terminal=True)
    assert console.is_tty is True
    return console, output


@pytest.mark.parametrize("status", ["cancelled", "deleted"])
def test_list_datasets_preserves_uncategorized_status_brackets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: str,
) -> None:
    console, _output = _make_rich_console()
    fake_credentials = object()

    monkeypatch.setattr(dataset_module, "console", console)
    monkeypatch.setattr(utils_module, "console", console)
    monkeypatch.setattr(
        dataset_module,
        "require_git_workspace_directory_context",
        lambda: SimpleNamespace(
            credentials=fake_credentials,
            git_identity=GIT_IDENTITY,
            repo_url=REPO_URL,
            workspace_directory=tmp_path,
        ),
    )

    class FakeClient:
        def list_datasets(
            self,
            limit: int = 50,
            offset: int = 0,
            *,
            credentials=None,
            git_identity: str,
        ) -> PaginatedDatasets:
            assert credentials is fake_credentials
            assert git_identity == GIT_IDENTITY
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

    result = dataset_module.list_datasets()

    assert result.display_items is not None
    assert f"\\[{status}]" == result.display_items[0]["status"]


@pytest.mark.parametrize("status", ["cancelled", "deleted"])
def test_dataset_status_format_preserves_uncategorized_status_brackets(
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


def test_dataset_status_format_preserves_plain_text_brackets() -> None:
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
