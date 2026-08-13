"""JSON encoding for CLI envelopes.

The machine contract is RFC 8259 JSON. Python's default ``json.dumps`` emits
``NaN`` / ``Infinity`` tokens; ``allow_nan=False`` refuses those so a command
fails instead of writing unparseable stdout or stderr.
"""

from __future__ import annotations

import json
from typing import Any


def dump_cli_json(payload: dict[str, Any], *, indent: int | None = None) -> str:
    """Serialize *payload* with ``allow_nan=False``.

    Raises:
        CLIError: If the payload contains a non-finite float or a value that
            is not JSON-serializable.
    """
    try:
        return json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=indent)
    except (TypeError, ValueError) as exc:
        from osmosis_ai.cli.errors import CLIError, CLIErrorCode

        raise CLIError(
            "Refusing to emit non-JSON output (non-finite float or non-serializable value).",
            code=CLIErrorCode.INTERNAL,
        ) from exc
