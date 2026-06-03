"""Console output facade with Rich and output-context aware rendering.

Rich automatically strips ANSI control codes when output is not directed
to a terminal (e.g., piped to a file), and respects the NO_COLOR
environment variable.

Usage:
    from osmosis_ai.cli.console import console

    console.print("Hello", style="green")
    console.print_error("Something went wrong")
    console.panel("Server Info", content)
    console.separator("Section Title")
"""

from __future__ import annotations

import sys
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from rich import box
from rich.console import Console as RichConsole
from rich.markup import escape as rich_escape
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text


def _url_link_text(
    url: str, label: str | None = None, style: str | None = None
) -> Text:
    link_style = Style(link=url)
    if style:
        link_style = Style.parse(style) + link_style
    return Text(label or url, style=link_style)


class Console:
    """Console output using Rich with automatic TTY-aware rendering.

    Rich handles terminal detection, color stripping for non-TTY output,
    and NO_COLOR environment variable support natively.
    """

    def __init__(
        self,
        *,
        file: Any = None,
        force_terminal: bool | None = None,
        no_color: bool = False,
        width: int | None = None,
    ) -> None:
        """Initialize the console.

        Args:
            file: Output file. Defaults to sys.stdout.
            force_terminal: Force terminal mode (for testing). None = auto-detect.
            no_color: Disable all colors, even in TTY mode.
            width: Fixed terminal width (for testing). None = auto-detect.
        """
        file = file or sys.stdout
        self._no_color = no_color

        # When TERM=dumb, Rich's Console.size short-circuits to 80x25 unless both width and
        # height are set; width alone is ignored (see rich.console.Console.size).
        rich_size: dict[str, Any] = {}
        if width is not None:
            rich_size["width"] = width
            rich_size["height"] = 25

        # Disable Rich's default auto-highlighter: ReprHighlighter recolors
        # numbers, hex strings, UUIDs, URLs, etc. inside printed strings,
        # which clashes with explicit `style=` passed to Console.print (e.g.,
        # a "green" message would show mixed colors on tokens like
        # "step-40-lora" or "2d7a22"). Callers can opt back in per-call via
        # `console.print(..., highlight=True)`.
        self._rich = RichConsole(
            file=file,
            force_terminal=force_terminal,
            no_color=no_color,
            highlight=False,
            **rich_size,
        )
        self._rich_stderr = RichConsole(
            file=sys.stderr,
            no_color=no_color,
            highlight=False,
            **rich_size,
        )
        # Rich Status of a currently-live spinner, if any. print_warning stops
        # it before printing and restarts it after, so the transient spinner
        # line is not left stranded on screen mid-spin.
        self._active_status: Any = None

    @property
    def is_tty(self) -> bool:
        """Whether output is to a TTY."""
        return self._rich.is_terminal

    @property
    def width(self) -> int:
        """Terminal width in characters."""
        return self._rich.width

    @property
    def rich(self) -> RichConsole:
        """The underlying Rich Console instance."""
        return self._rich

    @staticmethod
    def _output_format() -> Any:
        from osmosis_ai.cli.output.context import get_output_context

        return get_output_context().format

    def _is_rich_mode(self) -> bool:
        from osmosis_ai.cli.output.context import OutputFormat

        return self._output_format() is OutputFormat.rich

    def print(
        self,
        *args: Any,
        style: str | None = None,
        end: str = "\n",
        **kwargs: Any,
    ) -> None:
        """Print text with optional styling.

        Args:
            *args: Values to print.
            style: Style name (e.g., "green", "bold red", "dim").
            end: String to print at end. Defaults to newline.
            **kwargs: Additional arguments passed to rich.print.
        """
        if not self._is_rich_mode():
            return
        self._rich.print(*args, style=style, end=end, **kwargs)

    def print_error(
        self,
        message: str,
        *,
        soft_wrap: bool | None = None,
        markup: bool = False,
    ) -> None:
        """Print an error message to stderr.

        Args:
            message: Error message to print.
            soft_wrap: Whether Rich should avoid inserting hard line breaks.
            markup: Whether to interpret Rich markup in the message.
        """
        kwargs: dict[str, Any] = {"markup": markup}
        if soft_wrap is not None:
            kwargs["soft_wrap"] = soft_wrap
        self._rich_stderr.print(message, style="bold red", **kwargs)

    def print_warning(
        self,
        message: str,
        *,
        code: str | None = None,
        soft_wrap: bool | None = None,
        markup: bool = False,
    ) -> None:
        """Print a non-fatal warning to stderr, honoring the output format.

        Warnings are emitted automatically deep in the request path (e.g. from a
        deprecation response header), so they are routed per output format to
        avoid corrupting the machine contract:

        - JSON: a one-line structured warning envelope on stderr, distinguished
          from the error envelope by the ``warning`` key so the stream stays
          parseable as JSON Lines.
        - plain: unstyled ``warning: <message>`` text on stderr.
        - rich: a yellow ``⚠`` line on stderr.

        Args:
            message: Warning message to print.
            code: Optional machine-readable code (e.g. ``"DEPRECATION"``) carried
                in the JSON envelope. Ignored in plain/rich modes.
            soft_wrap: Whether Rich should avoid inserting hard line breaks.
            markup: Whether to interpret Rich markup in the message.
        """
        from osmosis_ai.cli.output.context import OutputFormat

        fmt = self._output_format()
        if fmt is OutputFormat.json:
            from osmosis_ai.cli.output.error import emit_structured_warning_to_stderr

            emit_structured_warning_to_stderr(message, code=code)
            return
        if fmt is OutputFormat.plain:
            sys.stderr.write(f"warning: {message}\n")
            return
        kwargs: dict[str, Any] = {"markup": markup}
        if soft_wrap is not None:
            kwargs["soft_wrap"] = soft_wrap
        # A live spinner owns the terminal via a Rich Live region; writing to the
        # separate stderr console mid-spin would strand the spinner line. Pause
        # it (a clean transient erase), print, then resume so it redraws below.
        status = self._active_status
        if status is not None:
            status.stop()
        self._rich_stderr.print(f"⚠ {message}", style="yellow", **kwargs)
        if status is not None:
            status.start()

    def separator(self, title: str = "") -> None:
        """Print a separator line with optional title.

        Args:
            title: Optional title to display in the separator.
        """
        if not self._is_rich_mode():
            return
        from rich.rule import Rule

        self._rich.print(Rule(title, style="dim"))

    def panel(
        self,
        title: str,
        content: str,
        *,
        style: str = "blue",
        padding: tuple[int, int] = (0, 1),
    ) -> None:
        """Print content in a panel/box.

        Args:
            title: Panel title.
            content: Panel content.
            style: Border style color.
            padding: Padding (vertical, horizontal).
        """
        if not self._is_rich_mode():
            return
        panel = Panel(content, title=title, border_style=style, padding=padding)
        self._rich.print(panel)

    def table(
        self,
        rows: list[tuple[Any, Any]],
        *,
        title: str | None = None,
        headers: tuple[str, str] | None = None,
    ) -> None:
        """Print a simple two-column table.

        Args:
            rows: List of (key, value) tuples.
            title: Optional table title.
            headers: Optional column headers.
        """
        if not self._is_rich_mode():
            return
        table = Table(
            title=title,
            box=box.ROUNDED,
            show_header=headers is not None,
        )
        if headers:
            table.add_column(headers[0], style="cyan")
            table.add_column(headers[1])
        else:
            table.add_column("", style="cyan")
            table.add_column("")
        for key, value in rows:
            table.add_row(key, value)
        self._rich.print(table)

    def escape(self, text: str | None) -> str:
        """Escape text so it is not interpreted as Rich markup.

        Args:
            text: Text to escape. Returns empty string for None.

        Returns:
            Escaped text safe for embedding in Rich markup strings.
        """
        if text is None:
            return ""
        return rich_escape(str(text))

    def format_styled(self, text: str, style: str) -> str:
        """Return text with inline Rich markup.

        This is useful for building complex strings with mixed styles.

        Args:
            text: Text to style.
            style: Style name.

        Returns:
            Styled text string with Rich markup.
        """
        return f"[{style}]{rich_escape(text)}[/{style}]"

    def format_text(self, text: Any, style: str | None = None) -> Text:
        """Return plain text with optional Rich styling.

        Use this for dynamic values that should never be parsed as Rich markup.
        """
        value = "" if text is None else str(text)
        if style is None:
            return Text(value)
        return Text(value, style=style)

    def format_url(
        self,
        url: str,
        *,
        label: str | None = None,
        style: str | None = None,
    ) -> Text:
        """Return a Rich terminal hyperlink for a URL."""
        if self._no_color:
            return self.format_text(label or url, style=style)
        return _url_link_text(url, label=label, style=style)

    def print_url(
        self,
        prefix: str,
        url: str,
        *,
        label: str | None = None,
        style: str | None = None,
    ) -> None:
        """Print a URL without inserting hard line breaks into the link target."""
        if not self._is_rich_mode():
            return
        self._rich.print(
            self.format_text(prefix),
            self.format_url(url, label=label, style=style),
            sep="",
            soft_wrap=True,
        )

    @contextmanager
    def spinner(self, message: str) -> Generator[None, None, None]:
        """Show a spinner while work is in progress (alias for :meth:`status`).

        Historically this rendered the spinner to *stdout*, which leaked the
        progress text into redirected output — ``osmosis dataset list > out.txt``
        used to write ``Fetching datasets...`` into ``out.txt``. It now delegates
        to :meth:`status` so progress goes to stderr and stays output-mode aware,
        keeping stdout clean for piping/redirection. Kept as a named alias so the
        existing call sites (dataset, auth) read naturally.

        Usage::

            with console.spinner("Loading workspaces..."):
                result = api_call()
        """
        with self.status(message):
            yield

    @contextmanager
    def track_spinner(self, status: Any) -> Generator[None, None, None]:
        """Register ``status`` as the process-wide active spinner while live.

        A terminal can only show one spinner at a time, so the active Rich
        ``Status`` is tracked on this (singleton) console regardless of which
        spinner implementation created it — ``console.spinner`` here, or
        ``OutputContext.status`` on its own stderr console. ``print_warning``
        consults it to pause/resume the spinner around a stderr write, so a
        warning fired mid-spin (e.g. the upgrade nudge from deep in the request
        path) neither glues onto the spinner line nor strands it on screen.

        Saves and restores any previously-active status so nested spinners
        behave; the innermost live spinner is the one a warning pauses.
        """
        prev = self._active_status
        self._active_status = status
        try:
            yield
        finally:
            self._active_status = prev

    @contextmanager
    def status(self, message: str) -> Generator[None, None, None]:
        """Alias for spinner with output-context aware routing."""
        from osmosis_ai.cli.output.context import get_output_context

        with get_output_context().status(message):
            yield

    def input(self, prompt: str = "", style: str | None = None) -> str:
        """Get user input with optional styled prompt.

        Args:
            prompt: Prompt text.
            style: Optional style for the prompt.

        Returns:
            User input string.
        """
        if style:
            self._rich.print(prompt, style=style, end="")
            return input()
        return input(prompt)


# Default console instance for convenient access
console: Console = Console()


__all__ = [
    "Console",
    "console",
]
