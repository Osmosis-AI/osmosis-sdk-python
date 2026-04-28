from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osmosis_ai.errors import CLIError


def extract_assistant_content(messages: list[dict[str, Any]]) -> str:
    """Extract the content of the last assistant message from a conversation.

    Iterates in reverse so the *most recent* assistant turn is used.  Handles
    both plain string content and structured content arrays (multimodal
    messages with ``{"type": "text", "text": "..."}`` parts).

    Returns an empty string when no assistant message is found.
    """
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                text_parts = [
                    part["text"].strip()
                    for part in content
                    if isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                    and part["text"].strip()
                ]
                if text_parts:
                    return "\n".join(text_parts)
    return ""


@dataclass(frozen=True)
class RubricRecord:
    """A single record for rubric evaluation."""

    solution_str: str
    ground_truth: str | None
    original_input: str | None
    metadata: dict[str, Any] | None
    record_id: str | None

    def label(self, index: int) -> str:
        if self.record_id:
            return self.record_id
        return f"record[{index}]"


def load_rubric_dataset(path: Path) -> list[RubricRecord]:
    """Load JSONL file into RubricRecord list.

    Each line must have 'messages' (list of chat messages) OR 'solution_str' (string, legacy).
    Optional: 'ground_truth', 'metadata', 'original_input', 'id', 'conversation_id'.
    """
    records: list[RubricRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise CLIError(
                    f"Invalid JSON on line {line_number} of '{path}': {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise CLIError(
                    f"Expected JSON object on line {line_number} of '{path}'."
                )
            records.append(_parse_record(payload, line_number, path))

    if not records:
        raise CLIError(f"No JSON records found in '{path}'.")
    return records


def _parse_record(
    payload: dict[str, Any], line_number: int, path: Path
) -> RubricRecord:
    messages = payload.get("messages")
    solution_str = payload.get("solution_str")

    if isinstance(messages, list) and messages:
        extracted = extract_assistant_content(messages)
        if not extracted:
            raise CLIError(
                f"Line {line_number} of '{path}': 'messages' list must contain "
                f"at least one assistant message with non-empty content."
            )
        resolved_solution = extracted
    elif isinstance(solution_str, str) and solution_str.strip():
        resolved_solution = solution_str.strip()
    else:
        raise CLIError(
            f"Line {line_number} of '{path}' must include 'messages' (list) "
            f"or 'solution_str' (string)."
        )

    ground_truth = payload.get("ground_truth")
    original_input = payload.get("original_input")
    metadata = payload.get("metadata")
    record_id_raw = payload.get("id") or payload.get("conversation_id")

    return RubricRecord(
        solution_str=resolved_solution,
        ground_truth=ground_truth if isinstance(ground_truth, str) else None,
        original_input=original_input if isinstance(original_input, str) else None,
        metadata=metadata if isinstance(metadata, dict) else None,
        record_id=str(record_id_raw).strip() if record_id_raw else None,
    )
