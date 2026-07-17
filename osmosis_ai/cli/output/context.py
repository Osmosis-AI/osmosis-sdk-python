"""Output format/context plumbing for the CLI."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import StrEnum

from osmosis_ai.cli._click_compat import Context


class OutputFormat(StrEnum):
    """Top-level CLI output format."""

    rich = "rich"
    json = "json"
    plain = "plain"


@dataclass
class OutputContext:
    """Per-invocation output state."""

    format: OutputFormat
    interactive: bool
    schema_version: int = 1
    output_emitted: bool = False

    @contextmanager
    def status(self, message: str) -> Iterator[None]:
        """Spinner-equivalent that respects the active output format."""
        if self.format is OutputFormat.json:
            yield
            return

        if self.format is OutputFormat.plain or not self.interactive:
            sys.stderr.write(message + "\n")
            sys.stderr.flush()
            yield
            return

        from rich.console import Console as RichConsole
        from rich.status import Status

        from osmosis_ai.cli.console import console

        rich_stderr = RichConsole(stderr=True)
        status = Status(message, console=rich_stderr, spinner="dots")
        # Register on the shared console so a warning emitted mid-spin (e.g. the
        # upgrade nudge) can pause/resume this spinner instead of gluing onto or
        # stranding its line. This is the dominant spinner path for the CLI.
        with console.track_spinner(status), status:
            yield


_output_context_var: ContextVar[OutputContext | None] = ContextVar(
    "osmosis_output_context",
    default=None,
)


def default_output_context() -> OutputContext:
    """Default rich + stdin-derived interactivity."""
    return OutputContext(
        format=OutputFormat.rich,
        interactive=sys.stdin.isatty() if sys.stdin else False,
    )


def _argv_format_prescan(argv: list[str]) -> OutputFormat | None:
    """Tolerantly scan argv for output format selectors at any position.

    Tokens after a literal ``--`` are arguments, never format selectors.
    """
    for token in argv:
        if token == "--":
            return None
        if token == "--json":
            return OutputFormat.json
        if token == "--plain":
            return OutputFormat.plain
    return None


def hoist_format_selectors(argv: list[str]) -> list[str]:
    """Move ``--json``/``--plain`` to the front of argv.

    The selectors are root-level Typer options, so a postfix spelling like
    ``osmosis dataset list --json`` would otherwise be rejected by the
    subcommand. Tokens after a literal ``--`` are left in place. No command
    has an option whose value can start with ``-``, so a bare ``--json``/
    ``--plain`` token is always a format selector.
    """
    selectors: list[str] = []
    rest: list[str] = []
    for index, token in enumerate(argv):
        if token == "--":
            rest.extend(argv[index:])
            break
        if token in ("--json", "--plain"):
            selectors.append(token)
        else:
            rest.append(token)
    return selectors + rest


def get_output_context() -> OutputContext:
    """Resolve the active OutputContext through the fallback stack.

    The ContextVar is the source of truth: install_output_context() sets it
    for the lifetime of the root Click context (reset via call_on_close), so
    it is populated exactly while a CLI invocation is in flight.
    """
    stored = _output_context_var.get()
    if stored is not None:
        return stored

    pre = _argv_format_prescan(sys.argv[1:])
    if pre is not None:
        return OutputContext(format=pre, interactive=False)

    return default_output_context()


def install_output_context(ctx: Context, output: OutputContext) -> None:
    """Mirror `output` to Click.Context.obj and the ContextVar."""
    ctx.obj = output
    token: Token[OutputContext | None] = _output_context_var.set(output)
    ctx.call_on_close(lambda: _output_context_var.reset(token))


@contextmanager
def override_output_context(
    *,
    format: OutputFormat = OutputFormat.rich,
    interactive: bool = False,
) -> Iterator[OutputContext]:
    """Test helper. Installs an OutputContext for the duration of the block."""
    output = OutputContext(format=format, interactive=interactive)
    token = _output_context_var.set(output)
    try:
        yield output
    finally:
        _output_context_var.reset(token)


def resolve_format_selectors(
    *,
    json_alias: bool,
    plain_alias: bool,
) -> OutputFormat:
    """Reconcile the global output selector flags."""
    from osmosis_ai.cli.errors import CLIError

    selected: set[OutputFormat] = set()
    if json_alias:
        selected.add(OutputFormat.json)
    if plain_alias:
        selected.add(OutputFormat.plain)

    if len(selected) > 1:
        raise CLIError(
            "Conflicting output format selectors: "
            f"{', '.join(sorted(f.value for f in selected))}. "
            "Choose at most one of --json or --plain.",
            code="VALIDATION",
        )

    return next(iter(selected), OutputFormat.rich)
