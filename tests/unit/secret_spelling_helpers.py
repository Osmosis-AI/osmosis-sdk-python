"""Independent spellings and decoding for known-secret redaction tests.

Decoding goes through the standard JSON decoder, never the SDK's own escape
handling, so a blind spot in the SDK cannot also hide inside an assertion.
"""

from __future__ import annotations

import json
import re

_ESCAPE_RUN = re.compile(r'(?:\\(?:u[0-9a-fA-F]{4}|["\\/bfnrt]))+')


def every_unit_escaped(text: str) -> str:
    """*text* as uppercase ``\\uXXXX`` escapes, one per UTF-16 code unit.

    Valid JSON that no standard encoder emits; astral characters become
    surrogate pairs such as ``\\uD83D\\uDD11``.
    """
    units = text.encode("utf-16-be")
    return "".join(
        f"\\u{int.from_bytes(units[index : index + 2], 'big'):04X}"
        for index in range(0, len(units), 2)
    )


def recoverable(text: str, *, levels: int = 2) -> str:
    """What a reader recovers: each run of JSON escapes decoded by ``json.loads``.

    Whole runs go to the decoder so surrogate pairs combine as JSON defines.
    Two levels cover a JSON error envelope whose message itself holds escapes.
    """
    for _ in range(levels):
        text = _ESCAPE_RUN.sub(lambda match: json.loads(f'"{match.group()}"'), text)
    return text
