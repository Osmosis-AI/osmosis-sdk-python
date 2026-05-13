"""Rich, JSON, and plain renderers + migration safety hooks."""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from .context import OutputContext, OutputFormat
from .result import (
    CommandResult,
    DetailResult,
    ListResult,
    MessageResult,
    OperationResult,
)

_RICH_STYLE_TAG_RE = re.compile(
    r"\[/?(?:bold|dim|italic|underline|blink|reverse|strike|"
    r"black|red|green|yellow|blue|magenta|cyan|white)"
    r"(?: [a-zA-Z0-9_#./ -]+)?\]"
)


def _merge_extra(
    payload: dict[str, Any],
    extra: dict[str, Any],
    *,
    reserved: set[str],
) -> dict[str, Any]:
    for key, value in extra.items():
        if key not in reserved:
            payload[key] = value
    return payload


def _envelope_list(result: ListResult, *, schema_version: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "items": result.items,
        "total_count": result.total_count,
        "has_more": result.has_more,
        "next_offset": result.next_offset,
    }
    return _merge_extra(
        payload,
        result.extra,
        reserved={"schema_version", "items", "total_count", "has_more", "next_offset"},
    )


def _envelope_detail(result: DetailResult, *, schema_version: int) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "data": result.data,
    }


def _envelope_operation(
    result: OperationResult, *, schema_version: int
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "status": result.status,
        "operation": result.operation,
    }
    if result.resource is not None:
        payload["resource"] = result.resource
    if result.message is not None:
        payload["message"] = result.message
    if result.next_steps_structured:
        payload["next_steps_structured"] = result.next_steps_structured
    return _merge_extra(
        payload,
        result.extra,
        reserved={
            "schema_version",
            "status",
            "operation",
            "resource",
            "message",
            "next_steps_structured",
        },
    )


def _envelope_message(result: MessageResult, *, schema_version: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "message": result.message,
    }
    return _merge_extra(payload, result.extra, reserved={"schema_version", "message"})


def _build_json_envelope(
    result: CommandResult, *, schema_version: int
) -> dict[str, Any]:
    if isinstance(result, ListResult):
        return _envelope_list(result, schema_version=schema_version)
    if isinstance(result, DetailResult):
        return _envelope_detail(result, schema_version=schema_version)
    if isinstance(result, OperationResult):
        return _envelope_operation(result, schema_version=schema_version)
    if isinstance(result, MessageResult):
        return _envelope_message(result, schema_version=schema_version)
    raise TypeError(f"Unsupported CommandResult: {type(result).__name__}")


def _render_json(result: CommandResult, output: OutputContext) -> None:
    envelope = _build_json_envelope(result, schema_version=output.schema_version)
    sys.stdout.write(json.dumps(envelope, ensure_ascii=False))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _normalise_plain_value(value: Any, *, rich_display: bool = False) -> str:
    text = "" if value is None else str(value)
    if rich_display:
        text = _RICH_STYLE_TAG_RE.sub("", text)
        text = text.replace("\\[", "[").replace("\\]", "]")
    return text.replace("\t", " ").replace("\n", " ").replace("\r", " ")


def _display_items(result: ListResult) -> list[dict[str, Any]]:
    return result.display_items if result.display_items is not None else result.items


def _rich_text(value: Any, *, style: str | None = None) -> Any:
    from rich.text import Text

    if isinstance(value, Text):
        if style is not None:
            value.stylize(style)
        return value
    text = "" if value is None else str(value)
    text = text.replace("\\[", "[").replace("\\]", "]")
    if style is None:
        return Text(text)
    return Text(text, style=style)


def _render_plain(result: CommandResult, output: OutputContext) -> None:
    if isinstance(result, DetailResult):
        for field in result.fields:
            value = _normalise_plain_value(field.value, rich_display=True)
            sys.stdout.write(f"{field.label}: {value}\n")
        for section in result.sections:
            for line in section.plain_lines:
                sys.stdout.write(_normalise_plain_value(line, rich_display=True) + "\n")
        for hint in result.display_hints:
            sys.stdout.write(_normalise_plain_value(hint, rich_display=True) + "\n")
        return

    if isinstance(result, ListResult):
        cols = [column for column in result.columns if column.plain]
        display_items = _display_items(result)
        for idx, item in enumerate(display_items):
            raw_item = result.items[idx] if idx < len(result.items) else {}
            row = "\t".join(
                _normalise_plain_value(
                    item.get(c.key),
                    rich_display=(
                        result.display_items is not None
                        and item.get(c.key) != raw_item.get(c.key)
                    ),
                )
                for c in cols
            )
            sys.stdout.write(row + "\n")
        return

    if isinstance(result, OperationResult):
        line = result.message or f"{result.operation}: {result.status}"
        sys.stdout.write(_normalise_plain_value(line) + "\n")
        for hint in result.display_next_steps:
            sys.stdout.write(_normalise_plain_value(hint) + "\n")
        return

    if isinstance(result, MessageResult):
        sys.stdout.write(_normalise_plain_value(result.message) + "\n")
        return

    raise TypeError(f"Unsupported CommandResult: {type(result).__name__}")


def _render_rich(result: CommandResult, output: OutputContext) -> None:
    from osmosis_ai.cli.console import Console

    console = Console()

    if isinstance(result, DetailResult):
        if result.title:
            console.separator(result.title)
        console.table(
            [(field.label, _rich_text(field.value)) for field in result.fields]
        )
        for section in result.sections:
            if section.rich is not None:
                console.rich.print(section.rich)
        for hint in result.display_hints:
            console.rich.print(_rich_text(hint, style="dim"))
        return

    if isinstance(result, ListResult):
        from rich.table import Table

        table = Table(show_header=True, header_style="bold")
        for column in result.columns:
            table.add_column(
                column.label,
                no_wrap=column.no_wrap,
                overflow=column.overflow,
                ratio=column.ratio,
                min_width=column.min_width,
                max_width=column.max_width,
            )
        for idx, item in enumerate(_display_items(result)):
            raw_item = result.items[idx] if idx < len(result.items) else {}
            table.add_row(
                *[
                    ("" if item.get(column.key) is None else str(item.get(column.key)))
                    if result.display_items is not None
                    and item.get(column.key) != raw_item.get(column.key)
                    else _rich_text(item.get(column.key))
                    for column in result.columns
                ]
            )
        console.rich.print(table)
        if result.has_more:
            console.print(
                f"\nShowing {len(result.items)} of {result.total_count}. "
                "Use --all to show all, or --limit to adjust.",
                style="dim",
            )
        for hint in result.display_hints:
            console.print(hint, style="dim")
        return

    if isinstance(result, OperationResult):
        if result.message:
            style = "green" if result.status == "success" else "yellow"
            console.rich.print(_rich_text(result.message, style=style))
        for hint in result.display_next_steps:
            console.rich.print(_rich_text(hint, style="dim"))
        return

    if isinstance(result, MessageResult):
        console.rich.print(_rich_text(result.message))
        return

    raise TypeError(f"Unsupported CommandResult: {type(result).__name__}")


def render(result: CommandResult, output: OutputContext) -> None:
    """Dispatch to the format-specific renderer and mark output as emitted."""
    if output.format is OutputFormat.json:
        _render_json(result, output)
    elif output.format is OutputFormat.plain:
        _render_plain(result, output)
    else:
        _render_rich(result, output)
    output.output_emitted = True


def render_command_result(result: Any, **_: Any) -> None:
    """Typer result callback that renders converted command results."""
    from .context import get_output_context

    if isinstance(result, CommandResult):
        render(result, get_output_context())
        exit_code = getattr(result, "exit_code", 0)
        if exit_code:
            import click

            raise click.exceptions.Exit(exit_code)
        return

    output = get_output_context()
    if result is None and output.format is OutputFormat.rich:
        return

    from osmosis_ai.cli.errors import CLIError

    if result is None:
        raise CLIError(
            "Command exited without producing structured output. This usually "
            "means the command has not been converted to return a CommandResult.",
            code="INTERNAL",
        )

    raise CLIError(
        f"Handler returned unsupported result type: {type(result).__name__}",
        code="INTERNAL",
    )


def verify_output_emitted() -> None:
    """Catch successful JSON/plain invocations that produced no structured output."""
    from .context import _output_context_var
    from .error import emit_structured_error_to_stderr

    output = _output_context_var.get()
    if output is None or output.format is OutputFormat.rich:
        return
    if output.output_emitted:
        return
    if sys.exc_info()[0] is not None:
        return

    import click

    from osmosis_ai.cli.errors import CLIError

    emit_structured_error_to_stderr(
        CLIError(
            "Command exited without producing structured output.",
            code="INTERNAL",
        )
    )
    raise click.exceptions.Exit(1)
