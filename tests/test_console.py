"""Tests for rollout console fallback behavior."""

from __future__ import annotations

from io import StringIO

import osmosis_ai.rollout.console as console_module


def test_print_fallback_ignores_rich_only_kwargs(monkeypatch) -> None:
    """Fallback print should ignore rich-only kwargs and still render output."""
    monkeypatch.setattr(console_module, "RICH_AVAILABLE", False)
    output = StringIO()
    console = console_module.Console(file=output, force_terminal=True)

    assert console.use_rich is False

    console.print(
        "hello",
        "world",
        style="green",
        markup=True,
        highlight=False,
        justify="center",
        overflow="fold",
    )

    text = output.getvalue()
    assert "hello world" in text
    assert "\033[32m" in text


def test_print_fallback_respects_sep(monkeypatch) -> None:
    """Fallback print should preserve built-in print separator behavior."""
    monkeypatch.setattr(console_module, "RICH_AVAILABLE", False)
    output = StringIO()
    console = console_module.Console(file=output, force_terminal=False)

    console.print("a", "b", "c", sep="|", markup=True, highlight=False)

    assert output.getvalue() == "a|b|c\n"


def test_run_rich_executes_renderer_when_available() -> None:
    """run_rich should execute renderer and return True when rich backend exists."""
    output = StringIO()
    console = console_module.Console(file=output, force_terminal=False)
    sentinel = object()
    console._use_rich = True
    console._rich = sentinel

    captured = {"value": None}

    def _renderer(rich_console: object) -> None:
        captured["value"] = rich_console

    assert console.run_rich(_renderer) is True
    assert captured["value"] is sentinel


def test_run_rich_returns_false_on_import_error() -> None:
    """run_rich should return False when renderer raises ImportError."""
    output = StringIO()
    console = console_module.Console(file=output, force_terminal=False)
    console._use_rich = True
    console._rich = object()

    def _renderer(_rich_console: object) -> None:
        raise ImportError("rich unavailable")

    assert console.run_rich(_renderer) is False
