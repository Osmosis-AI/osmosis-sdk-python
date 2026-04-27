"""Output format/context plumbing for the CLI."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import StrEnum

import click


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
    quiet: bool = False
    schema_version: int = 1
    output_emitted: bool = False
    _close_callbacks: list[Callable[[], None]] = field(default_factory=list, repr=False)

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

        rich_stderr = RichConsole(stderr=True)
        with Status(message, console=rich_stderr, spinner="dots"):
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
    """Tolerantly scan root-level argv for output format selectors."""
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--json":
            return OutputFormat.json
        if token == "--plain":
            return OutputFormat.plain
        if token == "--format":
            if i + 1 >= len(argv):
                return None
            try:
                return OutputFormat(argv[i + 1])
            except ValueError:
                return None
        if token.startswith("--format="):
            try:
                return OutputFormat(token.split("=", 1)[1])
            except ValueError:
                return None
        if not token.startswith("-"):
            return None
        i += 1
    return None


def get_output_context() -> OutputContext:
    """Resolve the active OutputContext through the fallback stack."""
    try:
        ctx = click.get_current_context(silent=True)
    except RuntimeError:
        ctx = None

    if ctx is not None:
        root_obj = ctx.find_root().obj
        if isinstance(root_obj, OutputContext):
            return root_obj

    stored = _output_context_var.get()
    if stored is not None:
        return stored

    pre = _argv_format_prescan(sys.argv[1:])
    if pre is not None:
        return OutputContext(format=pre, interactive=False)

    return default_output_context()


def install_output_context(ctx: click.Context, output: OutputContext) -> None:
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
    format_value: OutputFormat | None,
    *,
    json_alias: bool,
    plain_alias: bool,
) -> OutputFormat:
    """Reconcile --format / --json / --plain."""
    from osmosis_ai.cli.errors import CLIError

    selected: set[OutputFormat] = set()
    if format_value is not None:
        selected.add(format_value)
    if json_alias:
        selected.add(OutputFormat.json)
    if plain_alias:
        selected.add(OutputFormat.plain)

    if len(selected) > 1:
        raise CLIError(
            "Conflicting output format selectors: "
            f"{', '.join(sorted(f.value for f in selected))}. "
            "Choose at most one of --format, --json, --plain.",
            code="VALIDATION",
        )

    return next(iter(selected), OutputFormat.rich)
