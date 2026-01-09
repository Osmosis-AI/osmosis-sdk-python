"""Unified console output with rich support and graceful degradation.

This module provides a Console class that uses rich for beautiful terminal output
when available, with automatic fallback to plain text when rich is not installed
or when output is redirected to a non-TTY.

Usage:
    from osmosis_ai.rollout.console import console

    console.print("Hello", style="green")
    console.print_error("Something went wrong")
    console.panel("Server Info", content)
    console.separator("Section Title")
"""

from __future__ import annotations

import sys
from typing import Any, List, Optional

# Try to import rich, gracefully degrade if not available
try:
    from rich.console import Console as RichConsole
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.style import Style
    from rich import box

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    RichConsole = None  # type: ignore
    Panel = None  # type: ignore
    Table = None  # type: ignore
    Text = None  # type: ignore
    Style = None  # type: ignore
    box = None  # type: ignore


class _AnsiColors:
    """ANSI color codes for fallback mode."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    # Style name to ANSI code mapping
    STYLE_MAP = {
        "bold": BOLD,
        "dim": DIM,
        "red": RED,
        "green": GREEN,
        "yellow": YELLOW,
        "blue": BLUE,
        "magenta": MAGENTA,
        "cyan": CYAN,
        "bold red": BOLD + RED,
        "bold green": BOLD + GREEN,
        "bold yellow": BOLD + YELLOW,
        "bold blue": BOLD + BLUE,
        "bold magenta": BOLD + MAGENTA,
        "bold cyan": BOLD + CYAN,
    }


class Console:
    """Unified console output with rich support and graceful degradation.

    When rich is available and output is to a TTY, uses rich for beautiful
    formatted output. Otherwise, falls back to plain text with optional
    ANSI colors (only when writing to a TTY).
    """

    def __init__(
        self,
        *,
        file: Any = None,
        force_terminal: Optional[bool] = None,
        no_color: bool = False,
    ):
        """Initialize the console.

        Args:
            file: Output file. Defaults to sys.stdout.
            force_terminal: Force terminal mode (for testing). None = auto-detect.
            no_color: Disable all colors, even in TTY mode.
        """
        self._file = file or sys.stdout
        self._no_color = no_color

        # Determine if we're writing to a TTY
        if force_terminal is not None:
            self._is_tty = force_terminal
        else:
            self._is_tty = getattr(self._file, "isatty", lambda: False)()

        # Use rich if available and writing to TTY
        self._use_rich = RICH_AVAILABLE and self._is_tty and not no_color

        if self._use_rich:
            self._rich = RichConsole(file=self._file, force_terminal=force_terminal)
            self._rich_stderr = RichConsole(file=sys.stderr)
        else:
            self._rich = None
            self._rich_stderr = None

    @property
    def is_tty(self) -> bool:
        """Whether output is to a TTY."""
        return self._is_tty

    @property
    def use_rich(self) -> bool:
        """Whether rich formatting is being used."""
        return self._use_rich

    def _get_ansi_style(self, style: Optional[str]) -> str:
        """Get ANSI escape code for a style name."""
        if not style or self._no_color or not self._is_tty:
            return ""
        return _AnsiColors.STYLE_MAP.get(style.lower(), "")

    def print(
        self,
        *args: Any,
        style: Optional[str] = None,
        end: str = "\n",
        **kwargs: Any,
    ) -> None:
        """Print text with optional styling.

        Args:
            *args: Values to print.
            style: Style name (e.g., "green", "bold red", "dim").
            end: String to print at end. Defaults to newline.
            **kwargs: Additional arguments passed to print/rich.print.
        """
        if self._use_rich and self._rich:
            self._rich.print(*args, style=style, end=end, **kwargs)
        else:
            text = " ".join(str(arg) for arg in args)
            ansi_style = self._get_ansi_style(style)
            if ansi_style:
                text = f"{ansi_style}{text}{_AnsiColors.RESET}"
            print(text, end=end, file=self._file, **kwargs)

    def print_error(self, message: str) -> None:
        """Print an error message to stderr.

        Args:
            message: Error message to print.
        """
        if self._use_rich and self._rich_stderr:
            self._rich_stderr.print(message, style="bold red")
        else:
            is_stderr_tty = getattr(sys.stderr, "isatty", lambda: False)()
            if is_stderr_tty and not self._no_color:
                print(
                    f"{_AnsiColors.BOLD}{_AnsiColors.RED}{message}{_AnsiColors.RESET}",
                    file=sys.stderr,
                )
            else:
                print(message, file=sys.stderr)

    def separator(self, title: str = "", width: int = 60) -> None:
        """Print a separator line with optional title.

        Args:
            title: Optional title to display in the separator.
            width: Width of the separator line.
        """
        if self._use_rich and self._rich:
            from rich.rule import Rule

            self._rich.print(Rule(title, style="dim"))
        else:
            if title:
                padding = (width - len(title) - 2) // 2
                line = "─" * padding + f" {title} " + "─" * padding
                if len(line) < width:
                    line += "─"
            else:
                line = "─" * width

            style = self._get_ansi_style("dim")
            if style:
                line = f"{style}{line}{_AnsiColors.RESET}"
            print(line, file=self._file)

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
        if self._use_rich and self._rich:
            panel = Panel(content, title=title, border_style=style, padding=padding)
            self._rich.print(panel)
        else:
            # Fallback: simple bordered box
            lines = content.split("\n")
            max_width = max(len(line) for line in lines) if lines else 0
            title_width = len(title) + 2 if title else 0
            box_width = max(max_width + 4, title_width + 4)

            style_code = self._get_ansi_style(style)
            reset = _AnsiColors.RESET if style_code else ""

            # Top border with title
            if title:
                padding_left = (box_width - len(title) - 2) // 2
                padding_right = box_width - len(title) - 2 - padding_left
                top = f"╭{'─' * padding_left} {title} {'─' * padding_right}╮"
            else:
                top = f"╭{'─' * (box_width - 2)}╮"

            print(f"{style_code}{top}{reset}", file=self._file)

            # Content lines
            for line in lines:
                padded = line.ljust(box_width - 4)
                print(f"{style_code}│{reset}  {padded}  {style_code}│{reset}", file=self._file)

            # Bottom border
            bottom = f"╰{'─' * (box_width - 2)}╯"
            print(f"{style_code}{bottom}{reset}", file=self._file)

    def table(
        self,
        rows: List[tuple[str, str]],
        *,
        title: Optional[str] = None,
        headers: Optional[tuple[str, str]] = None,
    ) -> None:
        """Print a simple two-column table.

        Args:
            rows: List of (key, value) tuples.
            title: Optional table title.
            headers: Optional column headers.
        """
        if self._use_rich and self._rich:
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
        else:
            # Fallback: aligned text
            if title:
                print(f"\n{title}:", file=self._file)
            key_width = max(len(row[0]) for row in rows) if rows else 0
            for key, value in rows:
                print(f"  {key.ljust(key_width)}  {value}", file=self._file)

    def format_styled(self, text: str, style: str) -> str:
        """Return text with inline style markup (for rich) or ANSI codes.

        This is useful for building complex strings with mixed styles.

        Args:
            text: Text to style.
            style: Style name.

        Returns:
            Styled text string.
        """
        if self._use_rich:
            return f"[{style}]{text}[/{style}]"
        else:
            ansi_style = self._get_ansi_style(style)
            if ansi_style:
                return f"{ansi_style}{text}{_AnsiColors.RESET}"
            return text

    def input(self, prompt: str = "", style: Optional[str] = None) -> str:
        """Get user input with optional styled prompt.

        Args:
            prompt: Prompt text.
            style: Optional style for the prompt.

        Returns:
            User input string.
        """
        if self._use_rich and self._rich and style:
            self._rich.print(prompt, style=style, end="")
            return input()
        else:
            if style:
                ansi_style = self._get_ansi_style(style)
                if ansi_style:
                    prompt = f"{ansi_style}{prompt}{_AnsiColors.RESET}"
            return input(prompt)


# Default console instance for convenient access
console = Console()


__all__ = [
    "Console",
    "console",
    "RICH_AVAILABLE",
]
