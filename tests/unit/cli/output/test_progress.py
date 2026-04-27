"""Output-aware make_progress_bar tests."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

from osmosis_ai.cli.output.context import OutputFormat, override_output_context
from osmosis_ai.platform.api.upload import make_progress_bar


def test_json_mode_emits_no_progress_anywhere() -> None:
    out, err = io.StringIO(), io.StringIO()
    with override_output_context(format=OutputFormat.json):
        with redirect_stdout(out), redirect_stderr(err):
            ctx, callback = make_progress_bar(1024)
            with ctx:
                callback(256, 1024)
                callback(1024, 1024)
    assert out.getvalue() == ""
    assert err.getvalue() == ""


def test_plain_mode_emits_only_to_stderr() -> None:
    out, err = io.StringIO(), io.StringIO()
    with override_output_context(format=OutputFormat.plain):
        with redirect_stdout(out), redirect_stderr(err):
            ctx, callback = make_progress_bar(1024)
            with ctx:
                callback(512, 1024)
                callback(1024, 1024)
                callback(1024, 1024)
    assert out.getvalue() == ""
    assert err.getvalue() != ""
    assert err.getvalue().count("100%") == 1


def test_rich_mode_emits_to_stderr_only() -> None:
    out, err = io.StringIO(), io.StringIO()
    with override_output_context(format=OutputFormat.rich):
        with redirect_stdout(out), redirect_stderr(err):
            ctx, callback = make_progress_bar(1024)
            with ctx:
                callback(1024, 1024)
    assert out.getvalue() == ""
