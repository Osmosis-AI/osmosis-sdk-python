"""Tests for platform CLI utility functions."""

from __future__ import annotations

from contextlib import contextmanager
from io import StringIO
from types import SimpleNamespace

import pytest

from osmosis_ai.cli.console import Console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.cli.utils import (
    fetch_all_pages,
    format_dataset_status,
    format_processing_step,
    format_run_status,
    format_size,
    platform_call,
    validate_list_options,
)
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

# ── format_size ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "size_bytes, expected",
    [
        (0, "0 B"),
        (512, "512 B"),
        (1024, "1.0 KB"),
        (1536, "1.5 KB"),
        (1048576, "1.0 MB"),
        (1073741824, "1.0 GB"),
        (1099511627776, "1.0 TB"),
        (5497558138880, "5.0 TB"),
    ],
)
def test_format_size(size_bytes: int, expected: str) -> None:
    assert format_size(size_bytes) == expected


# ── format_processing_step ───────────────────────────────────────────


def test_format_processing_step_none_when_no_step() -> None:
    obj = SimpleNamespace(processing_step=None, processing_percent=None)
    assert format_processing_step(obj) is None


def test_format_processing_step_with_step_only() -> None:
    obj = SimpleNamespace(processing_step="training", processing_percent=None)
    assert format_processing_step(obj) == "training"


def test_format_processing_step_with_percent() -> None:
    obj = SimpleNamespace(processing_step="training", processing_percent=42.7)
    assert format_processing_step(obj) == "training (43%)"


# ── format_dataset_status ────────────────────────────────────────────


def _dataset(status: str, step: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(status=status, processing_step=step, processing_percent=None)


def test_format_dataset_status_for_prompt() -> None:
    assert format_dataset_status(_dataset("uploaded"), for_prompt=True) == "[uploaded]"


def test_format_dataset_status_for_prompt_with_step() -> None:
    """processing_step is not shown in list display — only [status]."""
    assert (
        format_dataset_status(_dataset("processing", "validating"), for_prompt=True)
        == "[processing]"
    )


def test_format_dataset_status_uploaded_is_green(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import osmosis_ai.platform.cli.utils as mod

    c = Console(file=StringIO(), force_terminal=True)
    monkeypatch.setattr(mod, "console", c)
    result = format_dataset_status(_dataset("uploaded"))
    assert "uploaded" in result


def test_format_dataset_status_in_progress_styled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import osmosis_ai.platform.cli.utils as mod

    c = Console(file=StringIO(), force_terminal=True)
    monkeypatch.setattr(mod, "console", c)
    result = format_dataset_status(_dataset("processing"))
    assert "processing" in result


def test_format_dataset_status_error_styled(monkeypatch: pytest.MonkeyPatch) -> None:
    import osmosis_ai.platform.cli.utils as mod

    c = Console(file=StringIO(), force_terminal=True)
    monkeypatch.setattr(mod, "console", c)
    result = format_dataset_status(_dataset("error"))
    assert "error" in result


def test_format_dataset_status_unknown_uses_escape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import osmosis_ai.platform.cli.utils as mod

    c = Console(file=StringIO(), force_terminal=True)
    monkeypatch.setattr(mod, "console", c)
    result = format_dataset_status(_dataset("unknown_status"))
    assert "[unknown_status]" in result


# ── format_run_status ────────────────────────────────────────────────


def _run(
    status: str, step: str | None = None, pct: float | None = None
) -> SimpleNamespace:
    return SimpleNamespace(status=status, processing_step=step, processing_percent=pct)


def test_format_run_status_for_prompt() -> None:
    assert format_run_status(_run("completed"), for_prompt=True) == "[completed]"


def test_format_run_status_for_prompt_with_step_and_percent() -> None:
    result = format_run_status(_run("running", "training", 55.3), for_prompt=True)
    assert result == "[running: training 55%]"


def test_format_run_status_success(monkeypatch: pytest.MonkeyPatch) -> None:
    import osmosis_ai.platform.cli.utils as mod

    c = Console(file=StringIO(), force_terminal=True)
    monkeypatch.setattr(mod, "console", c)
    result = format_run_status(_run("completed"))
    assert "completed" in result


def test_format_run_status_in_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    import osmosis_ai.platform.cli.utils as mod

    c = Console(file=StringIO(), force_terminal=True)
    monkeypatch.setattr(mod, "console", c)
    result = format_run_status(_run("running"))
    assert "running" in result


def test_format_run_status_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import osmosis_ai.platform.cli.utils as mod

    c = Console(file=StringIO(), force_terminal=True)
    monkeypatch.setattr(mod, "console", c)
    result = format_run_status(_run("failed"))
    assert "failed" in result


def test_format_run_status_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    import osmosis_ai.platform.cli.utils as mod

    c = Console(file=StringIO(), force_terminal=True)
    monkeypatch.setattr(mod, "console", c)
    result = format_run_status(_run("stopped"))
    assert "stopped" in result


def test_format_run_status_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    import osmosis_ai.platform.cli.utils as mod

    c = Console(file=StringIO(), force_terminal=True)
    monkeypatch.setattr(mod, "console", c)
    result = format_run_status(_run("weird"))
    assert "[weird]" in result


# ── fetch_all_pages ─────────────────────────────────────────────────


def _make_page(
    items: list[str],
    *,
    total_count: int,
    next_offset: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        things=[SimpleNamespace(id=i) for i in items],
        total_count=total_count,
        next_offset=next_offset,
    )


def test_fetch_all_pages_single_page() -> None:
    page = _make_page(["a", "b"], total_count=2, next_offset=None)
    calls: list[tuple[int, int]] = []

    def fetch(limit: int, offset: int) -> SimpleNamespace:
        calls.append((limit, offset))
        return page

    items, total = fetch_all_pages(fetch, items_attr="things")
    assert len(items) == 2
    assert total == 2
    assert len(calls) == 1
    assert calls[0] == (50, 0)


def test_fetch_all_pages_multiple_pages() -> None:
    pages = [
        _make_page(["a", "b", "c"], total_count=5, next_offset=3),
        _make_page(["d", "e"], total_count=5, next_offset=None),
    ]
    call_idx = 0

    def fetch(limit: int, offset: int) -> SimpleNamespace:
        nonlocal call_idx
        page = pages[call_idx]
        call_idx += 1
        return page

    items, total = fetch_all_pages(fetch, items_attr="things")
    assert [x.id for x in items] == ["a", "b", "c", "d", "e"]
    assert total == 5


def test_fetch_all_pages_uses_server_next_offset() -> None:
    """Verify that fetch_all_pages uses next_offset from the server response,
    not a client-computed value based on item count."""
    pages = [
        _make_page(["a", "b"], total_count=4, next_offset=10),
        _make_page(["c", "d"], total_count=4, next_offset=None),
    ]
    calls: list[tuple[int, int]] = []
    call_idx = 0

    def fetch(limit: int, offset: int) -> SimpleNamespace:
        nonlocal call_idx
        calls.append((limit, offset))
        page = pages[call_idx]
        call_idx += 1
        return page

    fetch_all_pages(fetch, items_attr="things", page_size=10)
    assert calls == [(10, 0), (10, 10)]


# ── validate_list_options ───────────────────────────────────────────


def test_validate_list_options_default() -> None:
    limit, fetch_all = validate_list_options(limit=DEFAULT_PAGE_SIZE, all_=False)
    assert limit == DEFAULT_PAGE_SIZE
    assert fetch_all is False


def test_validate_list_options_all_flag() -> None:
    _limit, fetch_all = validate_list_options(limit=DEFAULT_PAGE_SIZE, all_=True)
    assert fetch_all is True


def test_validate_list_options_custom_limit() -> None:
    limit, fetch_all = validate_list_options(limit=10, all_=False)
    assert limit == 10
    assert fetch_all is False


def test_validate_list_options_mutual_exclusion() -> None:
    with pytest.raises(CLIError, match="mutually exclusive"):
        validate_list_options(limit=10, all_=True)


# ── platform_call ────────────────────────────────────────────────────


def test_platform_call_uses_injected_console() -> None:
    output = StringIO()
    status_console = Console(file=output, force_terminal=False)
    messages: list[str] = []

    @contextmanager
    def record_spinner(message: str):
        messages.append(message)
        yield

    status_console.spinner = record_spinner  # type: ignore[method-assign]

    result = platform_call(
        "Fetching things...",
        lambda: "ok",
        output_console=status_console,
    )

    assert result == "ok"
    assert messages == ["Fetching things..."]
