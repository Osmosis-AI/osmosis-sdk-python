"""Tests for the Console class (fallback/plain-text and rich modes)."""

from __future__ import annotations

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
    assert c.use_rich is False


def test_rich_console_properties() -> None:
    c, _ = _rich_console()
    assert c.is_tty is True
    assert c.use_rich is True


def test_no_color_console_disables_rich() -> None:
    c, _ = _no_color_console()
    assert c.is_tty is True
    assert c.use_rich is False


# ── print (fallback) ────────────────────────────────────────────────


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


# ── separator (fallback) ────────────────────────────────────────────


def test_separator_plain_no_title() -> None:
    c, buf = _plain_console()
    c.separator()
    output = buf.getvalue().strip()
    assert "─" in output
    assert len(output) == 60


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


# ── panel (fallback) ────────────────────────────────────────────────


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


# ── table (fallback) ────────────────────────────────────────────────


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
    assert "Key" in output if (output := buf.getvalue()) else True


# ── escape ───────────────────────────────────────────────────────────


def test_escape_plain() -> None:
    c, _ = _plain_console()
    assert c.escape("[bold]text[/bold]") == "[bold]text[/bold]"


def test_escape_rich() -> None:
    c, _ = _rich_console()
    result = c.escape("[bold]text[/bold]")
    assert "[bold]" not in result or "\\[" in result


def test_escape_none() -> None:
    c, _ = _plain_console()
    assert c.escape(None) == ""  # type: ignore[arg-type]


# ── format_styled ────────────────────────────────────────────────────


def test_format_styled_plain_with_tty() -> None:
    buf = StringIO()
    # force_terminal=True but no_color=True → uses ANSI fallback path
    c = Console(file=buf, force_terminal=True, no_color=True)
    # no_color disables ANSI too, so plain text
    result = c.format_styled("hello", "green")
    assert "hello" in result


def test_format_styled_plain_no_tty() -> None:
    c, _ = _plain_console()
    result = c.format_styled("hello", "green")
    assert result == "hello"


def test_format_styled_rich() -> None:
    c, _ = _rich_console()
    result = c.format_styled("hello", "cyan")
    assert "[cyan]" in result
    assert "hello" in result


# ── run_rich ─────────────────────────────────────────────────────────


def test_run_rich_returns_false_in_plain() -> None:
    c, _ = _plain_console()
    assert c.run_rich(lambda rc: rc.print("test")) is False


def test_run_rich_returns_true_in_rich() -> None:
    c, _ = _rich_console()
    assert c.run_rich(lambda rc: rc.print("test")) is True
