"""Shared test helpers for platform CLI tests."""

from __future__ import annotations

import re

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove ANSI SGR escape codes from text for assertion-friendly output."""
    return ANSI_ESCAPE_RE.sub("", text)
