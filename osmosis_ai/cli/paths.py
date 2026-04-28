"""Shared helpers for parsing CLI path arguments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParsedCliPath:
    path: Path
    has_trailing_separator: bool


def parse_cli_path(value: str, *, expand_user: bool = False) -> ParsedCliPath:
    """Parse a raw CLI path while preserving trailing-separator intent."""
    separators = {os.sep}
    if os.altsep is not None:
        separators.add(os.altsep)

    path = Path(value)
    if expand_user:
        path = path.expanduser()

    return ParsedCliPath(
        path=path,
        has_trailing_separator=value.endswith(tuple(separators)),
    )
