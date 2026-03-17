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
from collections.abc import Sequence
from typing import Any

import questionary
from prompt_toolkit.data_structures import Point
from prompt_toolkit.key_binding import KeyBindings, KeyBindingsBase, merge_key_bindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import HSplit as _HSplit
from prompt_toolkit.layout.containers import Window as _Window
from prompt_toolkit.layout.controls import UIContent, UIControl
from prompt_toolkit.layout.dimension import Dimension
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


def _add_extra_keys(
    question: questionary.Question,
    extra: KeyBindings,
) -> questionary.Question:
    """Merge *extra* key bindings into a questionary Question."""
    app = question.application
    kb = app.key_bindings

    # select() gives a mutable KeyBindings; text()/autocomplete() give
    # an immutable _MergedKeyBindings — handle both cases.
    if isinstance(kb, KeyBindings):
        for binding in extra.bindings:
            kb.bindings.append(binding)
    else:
        bindings: list[KeyBindingsBase] = [extra]
        if kb is not None:
            bindings.insert(0, kb)
        app.key_bindings = merge_key_bindings(bindings)

    return question


def _add_escape_binding(question: questionary.Question) -> questionary.Question:
    """Add ESC key binding to cancel/go back (same as Ctrl+C)."""
    extra = KeyBindings()
    extra.add(Keys.Escape, eager=True)(
        lambda event: event.app.exit(
            exception=KeyboardInterrupt, style="class:aborting"
        )
    )
    return _add_extra_keys(question, extra)


def _get_choices_container(question: questionary.Question):
    """Return the ConditionalContainer holding the choices window.

    Questionary builds its select layout as:
    ``HSplit → [prompt_row, ConditionalContainer(Window(InquirerControl)), ...]``
    """
    children = question.application.layout.container.children  # type: ignore[union-attr]
    if len(children) < 2 or not hasattr(children[1], "content"):
        raise RuntimeError(
            "Unexpected questionary select layout. "
            "The internal structure may have changed — update prompts.py."
        )
    return children[1]


def _apply_max_height(
    question: questionary.Question,
    max_height: int,
) -> questionary.Question:
    """Cap the visible choice list to *max_height* rows with scrolling."""
    _get_choices_container(question).content.height = Dimension(max=max_height)
    return question


# ── Split-scroll layout ─────────────────────────────────────────


class _SlicedView(UIControl):
    """Renders a contiguous vertical slice of another UIControl's output.

    Used to split a single InquirerControl into independently-scrollable
    sections while preserving its state management and key bindings.
    """

    def __init__(self, source: UIControl, start: int, end: int) -> None:
        self._source = source
        self._start = start
        self._end = end
        # Track last cursor position so the Window holds its scroll offset
        # when the cursor leaves this slice (UIContent coerces None → (0,0)).
        self._last_cursor: Point | None = None

    def create_content(self, width: int, height: int) -> UIContent:
        full = self._source.create_content(width, height)
        count = min(self._end, full.line_count) - self._start
        if count <= 0:
            return UIContent()

        cursor = None
        if full.cursor_position is not None:
            y = full.cursor_position.y
            if self._start <= y < self._end:
                cursor = Point(x=full.cursor_position.x, y=y - self._start)
                self._last_cursor = cursor

        return UIContent(
            get_line=lambda i: full.get_line(self._start + i),
            line_count=count,
            cursor_position=cursor if cursor is not None else self._last_cursor,
        )

    def preferred_height(
        self,
        width: int,
        max_available_height: int,
        wrap_lines: bool,
        get_line_prefix: Any | None,
    ) -> int | None:
        return self._end - self._start

    def is_focusable(self) -> bool:
        return True


def _add_tab_jump(
    question: questionary.Question,
    split_at: int,
    total: int,
) -> questionary.Question:
    """Add Tab key to jump between data items and action items.

    Must be called **before** ``_apply_split_scroll`` because it navigates
    the original questionary layout to find the InquirerControl.
    """
    conditional = _get_choices_container(question)
    ic = conditional.content.content  # Window → InquirerControl

    first_action = None
    for i in range(split_at, total):
        if not ic.choices[i].disabled:
            first_action = i
            break

    if first_action is not None:
        first_data = next((i for i in range(split_at) if not ic.choices[i].disabled), 0)
        saved_data_pos = first_data

        extra = KeyBindings()

        @extra.add(Keys.Tab, eager=True)
        def _toggle_section(event: Any) -> None:
            nonlocal saved_data_pos
            if ic.pointed_at < split_at:
                saved_data_pos = ic.pointed_at
                ic.pointed_at = first_action
            else:
                ic.pointed_at = saved_data_pos

        _add_extra_keys(question, extra)

    return question


def _apply_split_scroll(
    question: questionary.Question,
    split_at: int,
    total: int,
    max_visible: int,
) -> questionary.Question:
    """Split the choices into a scrollable data area and a fixed action area.

    Replaces the single ``Window(InquirerControl)`` in questionary's layout
    with two Windows: a height-limited data view that scrolls, and an
    unconstrained action view that is always visible below it.
    """
    conditional = _get_choices_container(question)
    ic = conditional.content.content  # Window → InquirerControl

    conditional.content = _HSplit(
        [
            _Window(_SlicedView(ic, 0, split_at), height=Dimension(max=max_visible)),
            _Window(_SlicedView(ic, split_at, total), dont_extend_height=True),
        ]
    )

    return question


def _create_select_question(
    message: str,
    choices: list[str | Choice | Separator],
    *,
    default: Any = None,
    instruction: str | None = None,
) -> questionary.Question:
    """Create a styled select Question with ESC binding."""
    return _add_escape_binding(
        questionary.select(
            message,
            choices=choices,
            default=default,
            style=OSMOSIS_STYLE,
            qmark="?",
            pointer="\u276f",
            instruction=instruction or "(↑↓ select, ESC back)",
            use_shortcuts=False,
        )
    )


def select(
    message: str,
    choices: list[str | Choice | Separator],
    *,
    default: Any = None,
    instruction: str | None = None,
    max_height: int | None = None,
) -> Any | None:
    """Interactive single-selection prompt with arrow-key navigation.

    Args:
        max_height: Maximum visible choices before scrolling. None = no limit.

    Returns the selected value, or None if the user cancels (Ctrl+C / ESC).
    """
    question = _create_select_question(
        message,
        choices,
        default=default,
        instruction=instruction,
    )
    if max_height is not None:
        _apply_max_height(question, max_height)
    return question.ask()


def select_list(
    message: str,
    items: list[str | Choice | Separator],
    *,
    actions: Sequence[str | Choice | Separator] | None = None,
    default: Any = None,
    instruction: str | None = None,
    max_visible: int | None = None,
) -> Any | None:
    """Select prompt with scrollable data items and pinned action items.

    Actions are placed below the data list with a separator.  When scrolling
    is active the data area scrolls independently while the action area
    remains fixed at the bottom.

    Args:
        message: Question text.
        items: Data choices (scrollable).
        actions: Action choices pinned below the list (e.g. Create, Back).
        default: Initial selection value.  Defaults to the first selectable
            item in *items*.
        instruction: Override the default navigation hint.
        max_visible: Cap visible data rows before scrolling kicks in.
            ``None`` means no cap.

    Returns the selected value, or None if the user cancels.
    """
    choices: list[str | Choice | Separator] = list(items)
    action_start = len(items)
    if actions:
        choices.append(Separator())
        choices.extend(actions)

    # Auto-default to first selectable data item
    if default is None:
        for c in items:
            if isinstance(c, str):
                default = c
                break
            if isinstance(c, Choice) and not isinstance(c, Separator):
                default = c.value
                break

    if instruction is None and actions:
        instruction = "(↑↓ select, Tab jump to actions, ESC back)"

    question = _create_select_question(
        message,
        choices,
        default=default,
        instruction=instruction,
    )

    # Tab to jump between data and action sections (must precede split scroll)
    if actions:
        _add_tab_jump(question, action_start, len(choices))

    # Split layout: scrollable data + fixed actions
    if max_visible is not None and actions and len(items) > max_visible:
        _apply_split_scroll(question, action_start, len(choices), max_visible)

    return question.ask()


def confirm(
    message: str,
    *,
    default: bool = True,
) -> bool | None:
    """Interactive yes/no confirmation prompt.

    Returns True/False, or None if the user cancels (Ctrl+C / ESC).
    """
    return _add_escape_binding(
        questionary.confirm(
            message,
            default=default,
            style=OSMOSIS_STYLE,
            qmark="?",
        )
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

    Returns the entered text, or None if the user cancels (Ctrl+C / ESC).
    """
    return _add_escape_binding(
        questionary.text(
            message,
            default=default,
            validate=validate,
            style=OSMOSIS_STYLE,
            qmark="?",
            instruction=instruction,
        )
    ).ask()


__all__ = [
    "OSMOSIS_STYLE",
    "Choice",
    "Separator",
    "confirm",
    "is_interactive",
    "select",
    "select_list",
    "text",
]
