"""Tests for the Console class (Rich-based TTY and non-TTY modes)."""

from __future__ import annotations

import sys
from io import StringIO

from osmosis_ai.cli.console import Console

# ── Helpers ──────────────────────────────────────────────────────────


def _plain_console() -> tuple[Console, StringIO]:
    buf = StringIO()
    c = Console(file=buf, force_terminal=False)
    return c, buf


def _rich_console() -> tuple[Console, StringIO]:
    buf = StringIO()
    c = Console(file=buf, force_terminal=True)
    return c, buf


def _no_color_console() -> tuple[Console, StringIO]:
    buf = StringIO()
    c = Console(file=buf, force_terminal=True, no_color=True)
    return c, buf


# ── Properties ───────────────────────────────────────────────────────


def test_plain_console_properties() -> None:
    c, _ = _plain_console()
    assert c.is_tty is False


def test_rich_console_properties() -> None:
    c, _ = _rich_console()
    assert c.is_tty is True


def test_no_color_console_properties() -> None:
    c, _ = _no_color_console()
    assert c.is_tty is True


def test_rich_console_width_override() -> None:
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=120)
    assert console.is_tty is True
    assert console.rich.width == 120
    assert console._rich_stderr.width == 120


def test_print_error_stderr_wraps_using_width_override(monkeypatch) -> None:
    """print_error writes via stderr; wrapping must follow the width override."""
    # Stabilize Rich dumb-terminal + env so explicit width+height is the variable under test.
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.setenv("TTY_COMPATIBLE", "1")
    for key in (
        "NO_COLOR",
        "FORCE_COLOR",
        "COLUMNS",
        "LINES",
        "TTY_INTERACTIVE",
        "JUPYTER_COLUMNS",
        "JUPYTER_LINES",
    ):
        monkeypatch.delenv(key, raising=False)

    err_narrow = StringIO()
    monkeypatch.setattr(sys, "stderr", err_narrow)
    narrow = Console(file=StringIO(), force_terminal=True, width=40, no_color=True)
    narrow.print_error("x" * 200)

    err_wide = StringIO()
    monkeypatch.setattr(sys, "stderr", err_wide)
    wide = Console(file=StringIO(), force_terminal=True, width=120, no_color=True)
    wide.print_error("x" * 200)

    assert err_narrow.getvalue().count("\n") > err_wide.getvalue().count("\n")


# ── print ───────────────────────────────────────────────────────────


def test_plain_print_basic() -> None:
    c, buf = _plain_console()
    c.print("hello world")
    assert "hello world" in buf.getvalue()


def test_plain_print_no_ansi_when_not_tty() -> None:
    c, buf = _plain_console()
    c.print("styled", style="green")
    output = buf.getvalue()
    assert "styled" in output
    assert "\033[" not in output  # no ANSI codes in non-TTY


def test_print_passes_rich_kwargs_through() -> None:
    """Console.print forwards kwargs like markup and highlight to Rich."""
    output = StringIO()
    console = Console(file=output, force_terminal=True)

    console.print(
        "hello",
        "world",
        style="green",
        markup=True,
        highlight=False,
    )

    text = output.getvalue()
    assert "hello" in text
    assert "world" in text


def test_print_respects_sep() -> None:
    """Console.print preserves separator behavior via Rich."""
    output = StringIO()
    console = Console(file=output, force_terminal=True)

    console.print("a", "b", "c", sep="|")

    assert "a|b|c" in output.getvalue()


def test_print_soft_wrap_preserves_url_without_rich_line_breaks() -> None:
    """URL output should not get hard line breaks inserted by Rich wrapping."""
    output = StringIO()
    console = Console(file=output, force_terminal=True, no_color=True, width=92)
    url = (
        "https://platform.osmosis.ai/osmosis-shared/training/"
        "328be61c-ef39-45e1-9b33-1e3c7c482e97"
    )

    console.print("  View: ", console.format_url(url), sep="", soft_wrap=True)

    assert output.getvalue() == f"  View: {url}\n"


def test_format_url_emits_terminal_hyperlink(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=120)
    url = "https://platform.osmosis.ai/osmosis-shared/training/run-1"

    console.print("View: ", console.format_url(url), sep="")

    rendered = output.getvalue()
    assert "\x1b]8;" in rendered
    assert url in rendered


def test_format_url_handles_brackets_in_url(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=120)
    url = "https://platform.osmosis.ai/osmosis-shared/training/run-1?filter[]=ok"

    console.print("View: ", console.format_url(url), sep="")

    rendered = output.getvalue()
    assert "\x1b]8;" in rendered
    assert url in rendered


def test_print_error_preserves_url_without_rich_line_breaks(monkeypatch) -> None:
    """URL-bearing error messages should stay copyable in captured output."""
    error_output = StringIO()
    monkeypatch.setattr(sys, "stderr", error_output)
    console = Console(file=StringIO(), force_terminal=True, no_color=True, width=92)
    url = (
        "https://platform.osmosis.ai/osmosis-shared/settings/billing/"
        "328be61c-ef39-45e1-9b33-1e3c7c482e97"
    )

    console.print_error(f"Upgrade at: {url}", soft_wrap=True)

    assert error_output.getvalue() == f"Upgrade at: {url}\n"


def test_print_error_does_not_parse_rich_markup(monkeypatch) -> None:
    error_output = StringIO()
    monkeypatch.setattr(sys, "stderr", error_output)
    console = Console(file=StringIO(), force_terminal=True, no_color=True, width=120)

    console.print_error("Missing [experiment] section", soft_wrap=True)

    assert error_output.getvalue() == "Missing [experiment] section\n"


def test_table_url_emits_terminal_hyperlink(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=120)
    url = "https://platform.osmosis.ai/osmosis-shared/datasets/dataset-1"

    console.table([("URL", console.format_url(url))], title="Dataset")

    rendered = output.getvalue()
    assert "\x1b]8;" in rendered
    assert url in rendered


def test_format_url_uses_label_for_visible_text(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=120)
    url = "https://platform.osmosis.ai/osmosis-shared/datasets/dataset-1"

    console.print("Open ", console.format_url(url, label="Dataset"), sep="")

    rendered = output.getvalue()
    assert "\x1b]8;" in rendered
    assert url in rendered
    assert "Dataset" in rendered


def test_non_tty_output_has_no_ansi() -> None:
    """Non-TTY output should not contain ANSI escape codes."""
    output = StringIO()
    console = Console(file=output, force_terminal=False)

    console.print("styled text", style="bold green")

    text = output.getvalue()
    assert "styled text" in text
    assert "\033[" not in text


# ── separator ───────────────────────────────────────────────────────


def test_separator_plain_no_title() -> None:
    c, buf = _plain_console()
    c.separator()
    output = buf.getvalue().strip()
    assert "─" in output


def test_separator_plain_with_title() -> None:
    c, buf = _plain_console()
    c.separator("Test")
    output = buf.getvalue().strip()
    assert "Test" in output
    assert "─" in output


def test_separator_rich() -> None:
    c, buf = _rich_console()
    c.separator("Title")
    assert "Title" in buf.getvalue()


# ── panel ───────────────────────────────────────────────────────────


def test_panel_plain_with_title() -> None:
    c, buf = _plain_console()
    c.panel("My Title", "line1\nline2")
    output = buf.getvalue()
    assert "My Title" in output
    assert "line1" in output
    assert "line2" in output
    assert "╭" in output
    assert "╰" in output


def test_panel_plain_no_title() -> None:
    c, buf = _plain_console()
    c.panel("", "content")
    output = buf.getvalue()
    assert "content" in output
    assert "╭" in output


def test_panel_rich() -> None:
    c, buf = _rich_console()
    c.panel("Title", "content", style="green")
    output = buf.getvalue()
    assert "Title" in output
    assert "content" in output


# ── table ───────────────────────────────────────────────────────────


def test_table_plain_basic() -> None:
    c, buf = _plain_console()
    c.table([("Key", "Value"), ("Name", "Alice")])
    output = buf.getvalue()
    assert "Key" in output
    assert "Alice" in output


def test_table_plain_with_title() -> None:
    c, buf = _plain_console()
    c.table([("A", "B")], title="My Table")
    assert "My Table" in buf.getvalue()


def test_table_rich_with_headers() -> None:
    c, buf = _rich_console()
    c.table([("A", "B")], headers=("Col1", "Col2"))
    output = buf.getvalue()
    assert "Col1" in output


def test_table_rich_without_headers() -> None:
    c, buf = _rich_console()
    c.table([("Key", "Val")])
    output = buf.getvalue()
    assert output, "Expected non-empty table output from rich console"
    assert "Key" in output


# ── escape ───────────────────────────────────────────────────────────


def test_escape_plain() -> None:
    c, _ = _plain_console()
    result = c.escape("[bold]text[/bold]")
    assert "\\[" in result  # brackets are escaped via rich_escape


def test_escape_rich() -> None:
    c, _ = _rich_console()
    result = c.escape("[bold]text[/bold]")
    assert "[bold]" not in result or "\\[" in result


def test_escape_none() -> None:
    c, _ = _plain_console()
    assert c.escape(None) == ""


# ── format_styled ────────────────────────────────────────────────────


def test_format_styled_returns_rich_markup() -> None:
    c, _ = _plain_console()
    result = c.format_styled("hello", "green")
    assert result == "[green]hello[/green]"


def test_format_styled_escapes_brackets() -> None:
    c, _ = _plain_console()
    result = c.format_styled("[bold]text", "cyan")
    assert "\\[bold]" in result  # opening bracket escaped
    assert "cyan" in result


# ── format_text / print_url ──────────────────────────────────────────


def test_format_text_does_not_parse_markup() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=True, no_color=True, width=120)

    console.print(console.format_text("[red]model-123[/red]", style="green"))

    assert output.getvalue() == "[red]model-123[/red]\n"


def test_print_url_preserves_url_without_rich_line_breaks() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=True, no_color=True, width=92)
    url = (
        "https://platform.osmosis.ai/osmosis-shared/training/"
        "328be61c-ef39-45e1-9b33-1e3c7c482e97?filter[]=ok"
    )

    console.print_url("  View: ", url)

    assert output.getvalue() == f"  View: {url}\n"
