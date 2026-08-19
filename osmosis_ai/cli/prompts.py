"""Shared interactive prompt utilities built on questionary.

Provides a consistent visual style and helper functions for all
platform CLI commands that need interactive user input.

Usage:
    from osmosis_ai.cli.prompts import select_list, confirm, password

    choice = select_list("Pick a workspace:", items=["ws-a", "ws-b"])
    ok = confirm("Proceed?")
    secret = password("API key:")
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import questionary
from prompt_toolkit.key_binding import KeyBindings, KeyBindingsBase, merge_key_bindings
from prompt_toolkit.keys import Keys
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


# prompt_toolkit's 1.0s default + 0.5s input flush makes bare ESC feel dead;
# chords arrive in one read so they never wait this out.
_ESCAPE_TIMEOUT = 0.1


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
    """Cancel on bare ESC (like Ctrl+C); option/alt chords arrive as ESC+key.

    Escape+Any keeps ESC a prefix so prompt_toolkit waits ``timeoutlen`` before
    cancelling and hands complete chords to their own bindings.
    """
    extra = KeyBindings()
    extra.add(Keys.Escape, Keys.Any)(lambda event: None)
    extra.add(Keys.Escape)(
        lambda event: event.app.exit(
            exception=KeyboardInterrupt, style="class:aborting"
        )
    )
    question.application.timeoutlen = _ESCAPE_TIMEOUT
    return _add_extra_keys(question, extra)


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


def select_list(
    message: str,
    items: list[str | Choice | Separator],
    *,
    default: Any = None,
    instruction: str | None = None,
) -> Any | None:
    """Select prompt over a list of choices.

    Args:
        message: Question text.
        items: Choices to select from.
        default: Initial selection value.  Defaults to the first selectable
            item in *items*.
        instruction: Override the default navigation hint.

    Returns the selected value, or None if the user cancels.
    """
    question = _create_select_question(
        message,
        list(items),
        default=default,
        instruction=instruction,
    )
    return question.ask()


def _confirm_question(message: str, *, default: bool) -> questionary.Question:
    return _add_escape_binding(
        questionary.confirm(
            message,
            default=default,
            style=OSMOSIS_STYLE,
            qmark="?",
        )
    )


def confirm(
    message: str,
    *,
    default: bool = True,
) -> bool | None:
    """Interactive yes/no confirmation prompt.

    Returns True/False, or None if the user cancels (Ctrl+C / ESC).

    Only for callers with no event loop of their own: ``ask()`` reaches
    ``Application.run()``, which starts one. Inside a coroutine use
    :func:`confirm_async`.
    """
    return _confirm_question(message, default=default).ask()


async def confirm_async(
    message: str,
    *,
    default: bool = True,
) -> bool | None:
    """:func:`confirm` for callers already running inside an event loop.

    ``ask()`` would raise ``RuntimeError: asyncio.run() cannot be called from
    a running event loop`` there; ``ask_async()`` awaits the same prompt on the
    caller's loop instead (prompt_toolkit 3.0: ``run_async`` from a coroutine).
    """
    return await _confirm_question(message, default=default).ask_async()


def text_input(
    message: str,
    *,
    default: str = "",
    validate: Callable[[str], bool | str] | None = None,
    instruction: str | None = None,
) -> str | None:
    """Interactive free-text prompt with an optional pre-filled, editable default.

    ``validate`` returns True when the input is valid, or the error message to
    show when it is not.

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


def pause(message: str) -> bool:
    """Wait for Enter; False on Ctrl+C/ESC.

    Uses the text prompt because questionary's press-any-key prompt swallows ESC.
    """
    answer = _add_escape_binding(
        questionary.text(
            message,
            default="",
            style=OSMOSIS_STYLE,
            qmark="?",
        )
    ).ask()
    return answer is not None


def password(
    message: str,
    *,
    validate: Any = None,
    instruction: str | None = None,
) -> str | None:
    """Interactive masked input prompt (no echo) for secret values.

    Built on ``questionary.password`` (prompt_toolkit ``is_password=True``),
    so the typed value is read straight from the terminal: it is never
    echoed, never written to shell history, and never placed on the process
    command line. Use this for any sensitive value so it is never echoed,
    written to history, or placed on the command line.

    The validate callable receives the input string and should return True
    if valid, or an error message string if invalid.

    Returns the entered text, or None if the user cancels (Ctrl+C / ESC).
    """
    return _add_escape_binding(
        questionary.password(
            message,
            validate=validate,
            style=OSMOSIS_STYLE,
            qmark="?",
            instruction=instruction,
        )
    ).ask()


def _confirmation_needs_prompt(
    message: str,
    *,
    yes: bool,
    summary: list[tuple[str, str]] | None,
    notes: list[str] | None,
    warnings: list[str] | None,
) -> bool:
    """Whether the caller still has to ask a human, shared by both guards.

    False when ``--yes`` already answered. Non-interactive sessions never reach
    a prompt at all: they raise here, after JSON mode has emitted the
    structured ``INTERACTIVE_REQUIRED`` envelope and plain mode has written the
    same context to stderr.
    """
    if yes:
        return False

    from osmosis_ai.cli.output import OutputFormat, get_output_context

    output = get_output_context()
    if output.format is OutputFormat.rich and output.interactive:
        return True

    from osmosis_ai.cli.errors import CLIError

    details: dict[str, Any] = {"prompt": message}
    if summary:
        details["summary"] = {label: value for label, value in summary}
    if notes:
        details["notes"] = list(notes)
    if warnings:
        details["warnings"] = list(warnings)

    if output.format is OutputFormat.plain:
        lines: list[str] = [f"Confirmation required: {message}"]
        if summary:
            for label, value in summary:
                lines.append(f"  {label}: {value}")
        if notes:
            lines.append("Notes:")
            for note in notes:
                lines.append(f"  - {note}")
        if warnings:
            lines.append("Warnings:")
            for warning in warnings:
                lines.append(f"  - {warning}")
        sys.stderr.write("\n".join(lines) + "\n")
        sys.stderr.flush()

    err = CLIError(
        "Use --yes to confirm in non-interactive mode.",
        code="INTERACTIVE_REQUIRED",
        details=details,
    )
    if output.format is OutputFormat.json:
        import typer

        from osmosis_ai.cli.output import emit_structured_error_to_stderr

        emit_structured_error_to_stderr(err)
        raise typer.Exit(1)
    raise err


def _exit_on_decline() -> None:
    import typer

    from osmosis_ai.cli.console import console

    console.print("Cancelled.", style="dim")
    raise typer.Exit(0)


def require_confirmation(
    message: str,
    *,
    yes: bool = False,
    default: bool = True,
    summary: list[tuple[str, str]] | None = None,
    notes: list[str] | None = None,
    warnings: list[str] | None = None,
) -> None:
    """Guard for destructive CLI commands that need user confirmation.

    Does nothing when *yes* is True (``--yes`` flag). In rich + interactive
    sessions prompts the user with questionary and exits cleanly on decline.
    In JSON mode emits a structured ``INTERACTIVE_REQUIRED`` error envelope
    (so agents/CI can see exactly what they are being asked to confirm) and
    exits 1. In plain mode writes the prompt + context to stderr and raises
    :class:`CLIError`.

    The optional *summary*, *notes*, and *warnings* carry the same context
    the rich panel showed: the JSON envelope embeds them as structured
    fields, and the plain-mode stderr output prints them inline.

    Callers inside an event loop must use :func:`require_confirmation_async`.
    """
    if not _confirmation_needs_prompt(
        message, yes=yes, summary=summary, notes=notes, warnings=warnings
    ):
        return
    if not confirm(message, default=default):
        _exit_on_decline()


async def require_confirmation_async(
    message: str,
    *,
    yes: bool = False,
    default: bool = True,
    summary: list[tuple[str, str]] | None = None,
    notes: list[str] | None = None,
    warnings: list[str] | None = None,
) -> None:
    """:func:`require_confirmation` for callers inside an event loop.

    Same contract, same non-interactive envelopes; only the prompt itself is
    awaited rather than run on a nested loop.
    """
    if not _confirmation_needs_prompt(
        message, yes=yes, summary=summary, notes=notes, warnings=warnings
    ):
        return
    if not await confirm_async(message, default=default):
        _exit_on_decline()


__all__ = [
    "OSMOSIS_STYLE",
    "Choice",
    "Separator",
    "confirm",
    "confirm_async",
    "password",
    "pause",
    "require_confirmation",
    "require_confirmation_async",
    "select_list",
    "text_input",
]
