"""Tests for Console behavior in TTY and non-TTY modes."""

from __future__ import annotations

from io import StringIO

from osmosis_ai.cli.console import Console


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


def test_non_tty_output_has_no_ansi() -> None:
    """Non-TTY output should not contain ANSI escape codes."""
    output = StringIO()
    console = Console(file=output, force_terminal=False)

    console.print("styled text", style="bold green")

    text = output.getvalue()
    assert "styled text" in text
    assert "\033[" not in text
