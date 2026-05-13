"""CommandResult types returned by command handlers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class DetailField:
    """Single label/value row for Rich + plain detail output."""

    label: str
    value: Any


@dataclass(frozen=True)
class ListColumn:
    """A column in a list result."""

    key: str
    label: str
    plain: bool = True
    no_wrap: bool = False
    align: str | None = None
    overflow: Literal["fold", "crop", "ellipsis", "ignore"] | None = None
    ratio: int | None = None
    min_width: int | None = None
    max_width: int | None = None


@dataclass(frozen=True)
class DetailSection:
    """Post-table detail content with Rich and plain representations."""

    rich: Any | None = None
    plain_lines: list[str] = field(default_factory=list)


class CommandResult:
    """Marker base class for renderer-dispatchable results."""


@dataclass
class DetailResult(CommandResult):
    """Single-resource detail output."""

    title: str
    data: dict[str, Any]
    fields: list[DetailField] = field(default_factory=list)
    exit_code: int = 0
    sections: list[DetailSection] = field(default_factory=list)
    display_hints: list[str] = field(default_factory=list)


@dataclass
class ListResult(CommandResult):
    """Paginated list output."""

    title: str
    items: list[dict[str, Any]]
    total_count: int
    has_more: bool
    next_offset: int | None
    columns: list[ListColumn] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    display_items: list[dict[str, Any]] | None = None
    exit_code: int = 0
    display_hints: list[str] = field(default_factory=list)


@dataclass
class OperationResult(CommandResult):
    """Mutation output."""

    operation: str
    status: str
    resource: dict[str, Any] | None = None
    message: str | None = None
    display_next_steps: list[str] = field(default_factory=list)
    next_steps_structured: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    exit_code: int = 0


@dataclass
class MessageResult(CommandResult):
    """Free-form message that does not map cleanly to a resource."""

    message: str
    extra: dict[str, Any] = field(default_factory=dict)
    exit_code: int = 0
