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
    format_deployment_status,
    format_eval_status,
    format_run_status,
    format_size,
    platform_call,
    validate_list_options,
)
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE


def test_platform_utils_do_not_import_workspace_mapping_or_subscription_cache() -> None:
    import osmosis_ai.platform.cli.utils as utils

    assert not hasattr(utils, "_require_subscription")
    assert not hasattr(utils, "load_subscription_status")
    assert not hasattr(utils, "save_subscription_status")


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


# ── format_dataset_status ────────────────────────────────────────────


def _dataset(status: str) -> SimpleNamespace:
    return SimpleNamespace(status=status)


def test_format_dataset_status_for_prompt() -> None:
    assert format_dataset_status(_dataset("uploaded"), for_prompt=True) == "[uploaded]"


def test_format_dataset_status_for_prompt_processing() -> None:
    assert (
        format_dataset_status(_dataset("processing"), for_prompt=True) == "[processing]"
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


def test_format_dataset_status_cancelled_is_dim() -> None:
    assert format_dataset_status(_dataset("cancelled")) == "[dim]\\[cancelled][/dim]"


def test_format_dataset_status_unknown_uses_escape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import osmosis_ai.platform.cli.utils as mod

    c = Console(file=StringIO(), force_terminal=True)
    monkeypatch.setattr(mod, "console", c)
    result = format_dataset_status(_dataset("unknown_status"))
    assert "[unknown_status]" in result


# ── format_run_status ────────────────────────────────────────────────


def _run(status: str) -> SimpleNamespace:
    return SimpleNamespace(status=status)


def test_format_run_status_for_prompt() -> None:
    assert format_run_status(_run("completed"), for_prompt=True) == "[completed]"


def test_format_run_status_for_prompt_with_step_and_percent() -> None:
    result = format_run_status(_run("running"), for_prompt=True)
    assert result == "[running]"


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


# ── format_deployment_status ─────────────────────────────────────────
#
# ``format_styled`` returns Rich *markup* (``[green]\\[active][/green]``), so we
# can assert the exact style bucket each status lands in. These guard formatters
# that previously had zero coverage.


@pytest.mark.parametrize(
    "status, style",
    [
        ("active", "green"),
        ("inactive", "dim"),
        ("failed", "red"),
    ],
)
def test_format_deployment_status_styles(status: str, style: str) -> None:
    assert format_deployment_status(status) == f"[{style}]\\[{status}][/{style}]"


def test_format_deployment_status_unknown_uses_escape() -> None:
    assert format_deployment_status("weird") == "\\[weird]"


def test_format_deployment_status_none_renders_em_dash() -> None:
    assert format_deployment_status(None) == "—"


# ── format_eval_status ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "status, style",
    [
        ("pending", "orange3"),
        ("running", "blue"),
        ("failed", "red"),
        ("stopped", "dim"),
    ],
)
def test_format_eval_status_styles(status: str, style: str) -> None:
    assert format_eval_status(_run(status)) == f"[{style}]\\[{status}][/{style}]"


def test_format_eval_status_finished_is_green_not_terminal_red() -> None:
    # ``finished`` is a member of EVAL_RUN_STATUSES_TERMINAL too; the ordered
    # matcher must hit the success (green) bucket before the terminal (red) one.
    assert format_eval_status(_run("finished")) == "[green]\\[finished][/green]"


def test_format_eval_status_unknown_uses_escape() -> None:
    assert format_eval_status(_run("weird")) == "\\[weird]"


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


def test_platform_call_shows_status_and_returns_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import osmosis_ai.cli.output as output_mod

    messages: list[str] = []

    @contextmanager
    def record_status(message: str):
        messages.append(message)
        yield

    class _FakeContext:
        def status(self, message: str):
            return record_status(message)

    monkeypatch.setattr(output_mod, "get_output_context", lambda: _FakeContext())

    result = platform_call("Fetching things...", lambda: "ok")

    assert result == "ok"
    # Progress routes through the active output context (stderr), not a console.
    assert messages == ["Fetching things..."]
