"""Shared interactive prompt utilities built on questionary.

Provides a consistent visual style and helper functions for all
platform CLI commands that need interactive user input.

Usage:
    from osmosis_ai.cli.prompts import select, confirm, text

    choice = select("Pick a workspace:", choices=["ws-a", "ws-b"])
    ok = confirm("Proceed?")
    name = text("Project name:", validate=my_validator)
"""

from __future__ import annotations

import sys
from typing import Any

import questionary
from questionary import Choice, Separator, Style

# ── Osmosis brand style ──────────────────────────────────────────

OSMOSIS_STYLE = Style(
    [
        ("qmark", "fg:#a78bfa bold"),  # Purple question mark
        ("question", "bold"),  # Bold question text
        ("answer", "fg:#06b6d4 bold"),  # Cyan submitted answer
        ("pointer", "fg:#a78bfa bold"),  # Purple pointer (»)
        ("highlighted", "fg:#a78bfa bold"),  # Purple highlighted option
        ("selected", "fg:#06b6d4"),  # Cyan selected (checkbox)
        ("separator", "fg:#6b7280"),  # Gray separator
        ("instruction", "fg:#6b7280"),  # Gray instruction text
        ("text", ""),  # Default text
        ("disabled", "fg:#6b7280 italic"),  # Gray italic disabled
    ]
)


def is_interactive() -> bool:
    """Return True if stdin is a TTY (interactive terminal)."""
    return sys.stdin.isatty()


def select(
    message: str,
    choices: list[str | Choice | Separator],
    *,
    default: str | None = None,
    instruction: str | None = None,
) -> str | None:
    """Interactive single-selection prompt with arrow-key navigation.

    Returns the selected value, or None if the user cancels (Ctrl+C).
    """
    return questionary.select(
        message,
        choices=choices,
        default=default,
        style=OSMOSIS_STYLE,
        qmark="?",
        pointer="\u276f",
        instruction=instruction or "(arrow keys to navigate)",
        use_shortcuts=False,
    ).ask()


def confirm(
    message: str,
    *,
    default: bool = True,
) -> bool | None:
    """Interactive yes/no confirmation prompt.

    Returns True/False, or None if the user cancels (Ctrl+C).
    """
    return questionary.confirm(
        message,
        default=default,
        style=OSMOSIS_STYLE,
        qmark="?",
    ).ask()


def text(
    message: str,
    *,
    default: str = "",
    validate: Any = None,
    instruction: str | None = None,
) -> str | None:
    """Interactive text input prompt with optional validation.

    The validate callable receives the input string and should return
    True if valid, or an error message string if invalid.

    Returns the entered text, or None if the user cancels (Ctrl+C).
    """
    return questionary.text(
        message,
        default=default,
        validate=validate,
        style=OSMOSIS_STYLE,
        qmark="?",
        instruction=instruction,
    ).ask()


def autocomplete(
    message: str,
    choices: list[str],
    *,
    default: str = "",
    validate: Any = None,
) -> str | None:
    """Interactive text input with autocomplete suggestions.

    Returns the entered text, or None if the user cancels (Ctrl+C).
    """
    return questionary.autocomplete(
        message,
        choices=choices,
        default=default,
        validate=validate,
        style=OSMOSIS_STYLE,
        qmark="?",
    ).ask()


__all__ = [
    "OSMOSIS_STYLE",
    "Choice",
    "Separator",
    "autocomplete",
    "confirm",
    "is_interactive",
    "select",
    "text",
]
