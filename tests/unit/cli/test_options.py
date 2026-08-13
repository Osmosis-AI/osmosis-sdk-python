"""Tests for shared CLI Option factories."""

from __future__ import annotations

from osmosis_ai.cli.options import (
    all_option,
    cursor_option,
    limit_option,
    log_limit_option,
)
from osmosis_ai.platform.constants import MAX_LOG_PAGE_SIZE, MAX_PAGE_SIZE


def test_limit_option_uses_page_size_bounds() -> None:
    opt = limit_option("Maximum number of datasets to show.")
    assert opt.min == 1
    assert opt.max == MAX_PAGE_SIZE


def test_log_limit_option_uses_log_page_size() -> None:
    opt = log_limit_option()
    assert opt.min == 1
    assert opt.max == MAX_LOG_PAGE_SIZE


def test_all_and_cursor_option_flags() -> None:
    assert "--all" in all_option("Show all datasets.").param_decls
    assert "--cursor" in cursor_option().param_decls
