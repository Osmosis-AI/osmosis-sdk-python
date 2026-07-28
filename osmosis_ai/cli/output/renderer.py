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
    ListColumn,
    ListResult,
    ListSection,
    MessageResult,
    OperationResult,
    SectionedListResult,
)

_RICH_STYLE_TAG_RE = re.compile(
    r"\[/?(?:bold|dim|italic|underline|blink|reverse|strike|"
    r"black|red|green|yellow|blue|magenta|cyan|white|orange3)"
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


def _list_page_payload(source: ListResult | ListSection) -> dict[str, Any]:
    return {
        "items": source.items,
        "total_count": source.total_count,
        "has_more": source.has_more,
        "next_offset": source.next_offset,
    }


def _envelope_list(result: ListResult, *, schema_version: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": schema_version,
        **_list_page_payload(result),
    }
    return _merge_extra(
        payload,
        result.extra,
        reserved={"schema_version", "items", "total_count", "has_more", "next_offset"},
    )


def _envelope_sectioned_list(
    result: SectionedListResult, *, schema_version: int
) -> dict[str, Any]:
    payload: dict[str, Any] = {"schema_version": schema_version}
    for section in result.sections:
        payload[section.key] = _list_page_payload(section)
    return _merge_extra(
        payload,
        result.extra,
        reserved={"schema_version", *(section.key for section in result.sections)},
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
    if isinstance(result, SectionedListResult):
        return _envelope_sectioned_list(result, schema_version=schema_version)
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


def _display_items(result: ListResult | ListSection) -> list[dict[str, Any]]:
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


def _display_width(value: Any) -> int:
    from rich.cells import cell_len

    text = "" if value is None else str(value)
    return max((cell_len(line) for line in text.splitlines()), default=0)


def _is_primary_column(column: ListColumn) -> bool:
    return column.label.lower() == "name" or column.key in {
        "name",
        "file_name",
        "model_name",
        "checkpoint_name",
    }


def _primary_column_index(columns: list[ListColumn]) -> int | None:
    for index, column in enumerate(columns):
        if _is_primary_column(column):
            return index
    return None


def _max_column_display_width(
    column: ListColumn,
    display_items: list[dict[str, Any]],
) -> int:
    width = _display_width(column.label)
    for item in display_items:
        width = max(width, _display_width(item.get(column.key)))
    return max(width, 1)


def _can_protect_primary_column(
    columns: list[ListColumn],
    display_items: list[dict[str, Any]],
    *,
    console_width: int,
) -> bool:
    primary_index = _primary_column_index(columns)
    if primary_index is None:
        return False

    # Rich collapses wrapable columns first. Protect the primary column only
    # when it can fit on one line after other columns shrink to one cell.
    content_budget = console_width - (3 * len(columns)) - 1
    other_column_budget = sum(
        max(column.min_width or 1, 1)
        for index, column in enumerate(columns)
        if index != primary_index
    )
    primary_column = columns[primary_index]
    primary_width = max(
        _max_column_display_width(primary_column, display_items),
        primary_column.min_width or 1,
    )
    return primary_width + other_column_budget <= content_budget


def _write_plain_list_rows(source: ListResult | ListSection) -> None:
    cols = [column for column in source.columns if column.plain]
    display_items = _display_items(source)
    for idx, item in enumerate(display_items):
        raw_item = source.items[idx] if idx < len(source.items) else {}
        row = "\t".join(
            _normalise_plain_value(
                item.get(c.key),
                rich_display=(
                    source.display_items is not None
                    and item.get(c.key) != raw_item.get(c.key)
                ),
            )
            for c in cols
        )
        sys.stdout.write(row + "\n")


def _render_plain(result: CommandResult, output: OutputContext) -> None:
    if isinstance(result, DetailResult):
        for field in result.fields:
            value = _normalise_plain_value(field.value, rich_display=True)
            sys.stdout.write(f"{field.label}: {value}\n")
        for section in result.sections:
            for line in section.plain_lines:
                sys.stdout.write(_normalise_plain_value(line) + "\n")
        for hint in result.display_hints:
            sys.stdout.write(_normalise_plain_value(hint) + "\n")
        return

    if isinstance(result, ListResult):
        _write_plain_list_rows(result)
        return

    if isinstance(result, SectionedListResult):
        for section in result.sections:
            sys.stdout.write(f"{_normalise_plain_value(section.title)}:\n")
            _write_plain_list_rows(section)
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


def _print_rich_list(
    console: Any,
    source: ListResult | ListSection,
    *,
    title: str | None = None,
) -> None:
    from rich.table import Table

    display_items = _display_items(source)
    protect_primary_column = _can_protect_primary_column(
        source.columns,
        display_items,
        console_width=console.width,
    )
    table = Table(
        title=title,
        title_justify="left",
        show_header=True,
        header_style="bold",
        expand=False,
    )
    for column in source.columns:
        overflow = column.overflow if column.overflow is not None else "ellipsis"
        no_wrap = column.no_wrap
        if protect_primary_column and _is_primary_column(column):
            no_wrap = True

        table.add_column(
            column.label,
            no_wrap=no_wrap,
            overflow=overflow,
            ratio=column.ratio,
            min_width=column.min_width,
            max_width=column.max_width,
        )
    for idx, item in enumerate(display_items):
        raw_item = source.items[idx] if idx < len(source.items) else {}
        table.add_row(
            *[
                ("" if item.get(column.key) is None else str(item.get(column.key)))
                if source.display_items is not None
                and item.get(column.key) != raw_item.get(column.key)
                else _rich_text(item.get(column.key))
                for column in source.columns
            ]
        )
    console.rich.print(table)
    if source.has_more:
        console.print(
            f"\nShowing {len(source.items)} of {source.total_count}. "
            "Use --all to show all, or --limit to adjust.",
            style="dim",
        )


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
        _print_rich_list(console, result)
        for hint in result.display_hints:
            console.rich.print(_rich_text(hint, style="dim"))
        return

    if isinstance(result, SectionedListResult):
        for section in result.sections:
            _print_rich_list(console, section, title=section.title)
        for hint in result.display_hints:
            console.rich.print(_rich_text(hint, style="dim"))
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
            import typer

            raise typer.Exit(exit_code)
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

    import typer

    from osmosis_ai.cli.errors import CLIError

    emit_structured_error_to_stderr(
        CLIError(
            "Command exited without producing structured output.",
            code="INTERNAL",
        )
    )
    raise typer.Exit(1)
