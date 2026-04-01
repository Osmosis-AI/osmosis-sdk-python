"""Console output with Rich and automatic TTY-aware rendering.

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
from rich.table import Table


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
    ):
        """Initialize the console.

        Args:
            file: Output file. Defaults to sys.stdout.
            force_terminal: Force terminal mode (for testing). None = auto-detect.
            no_color: Disable all colors, even in TTY mode.
            width: Fixed terminal width (for testing). None = auto-detect.
        """
        file = file or sys.stdout

        # When TERM=dumb, Rich's Console.size short-circuits to 80x25 unless both width and
        # height are set; width alone is ignored (see rich.console.Console.size).
        rich_size: dict[str, Any] = {}
        if width is not None:
            rich_size["width"] = width
            rich_size["height"] = 25

        self._rich = RichConsole(
            file=file,
            force_terminal=force_terminal,
            no_color=no_color,
            **rich_size,
        )
        self._rich_stderr = RichConsole(
            file=sys.stderr,
            no_color=no_color,
            **rich_size,
        )

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
        self._rich.print(*args, style=style, end=end, **kwargs)

    def print_error(self, message: str) -> None:
        """Print an error message to stderr.

        Args:
            message: Error message to print.
        """
        self._rich_stderr.print(message, style="bold red")

    def separator(self, title: str = "") -> None:
        """Print a separator line with optional title.

        Args:
            title: Optional title to display in the separator.
        """
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
        panel = Panel(content, title=title, border_style=style, padding=padding)
        self._rich.print(panel)

    def table(
        self,
        rows: list[tuple[str, str]],
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

    @contextmanager
    def spinner(self, message: str) -> Generator[None, None, None]:
        """Show a spinner animation while work is in progress.

        Usage::

            with console.spinner("Loading workspaces..."):
                result = api_call()
        """
        if self.is_tty:
            from rich.status import Status

            with Status(message, console=self._rich, spinner="dots"):
                yield
        else:
            self._rich.print(message)
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
