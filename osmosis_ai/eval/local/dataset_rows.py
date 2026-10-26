"""``dataset_rows.jsonl``: the prompt and label text the platform searches per row.

Line format, truncation, and encoding match what a managed evaluation writes.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from osmosis_ai.eval.local.dataset import EvalDatasetRow

DATASET_ROWS_FILENAME = "dataset_rows.jsonl"
DATASET_ROW_PROMPT_MAX_CHARS = 2_000
DATASET_ROW_LABEL_MAX_CHARS = 500


def _build_messages_from_row(row: Mapping[str, Any]) -> list[Any]:
    if isinstance(row.get("messages"), list):
        return row["messages"]
    system_prompt = row.get("system_prompt")
    user_prompt = row.get("user_prompt") or row.get("prompt")
    if user_prompt is not None:
        messages: list[Any] = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": str(user_prompt)})
        return messages
    return [{"role": "user", "content": str(row)}]


def _dataset_row_prompt(row: Mapping[str, Any]) -> str:
    if (
        not isinstance(row.get("messages"), list)
        and (row.get("user_prompt") or row.get("prompt")) is None
        and isinstance(row.get("metadata"), dict)
    ):
        text = json.dumps(
            row["metadata"], ensure_ascii=False, sort_keys=True, default=str
        )
    else:
        parts: list[str] = []
        for message in _build_messages_from_row(row):
            if not isinstance(message, dict) or message.get("role") == "system":
                continue
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(
                    block["text"]
                    for block in content
                    if isinstance(block, dict) and isinstance(block.get("text"), str)
                )
        text = " → ".join(part for part in parts if part.strip())
    return text[:DATASET_ROW_PROMPT_MAX_CHARS]


def _dataset_row_label(row: Mapping[str, Any]) -> str | None:
    for key in ("ground_truth", "label"):
        raw = row.get(key)
        if raw is not None and raw != "":
            return str(raw)[:DATASET_ROW_LABEL_MAX_CHARS]
    return None


def _raw_row(row: EvalDatasetRow) -> dict[str, Any]:
    """Rebuild the dataset columns a managed run reads, so both write the same line."""
    raw: dict[str, Any] = {}
    for message in row.initial_messages:
        column = "system_prompt" if message["role"] == "system" else "user_prompt"
        raw[column] = message["content"]
    if row.label is not None:
        raw["ground_truth"] = row.label
    if row.metadata is not None:
        raw["metadata"] = row.metadata
    return raw


def render_dataset_rows(rows: Iterable[EvalDatasetRow]) -> bytes:
    lines: list[str] = []
    for row in rows:
        raw = _raw_row(row)
        # A row with only a system prompt sends no user text, so it has no prompt.
        has_prompt = "user_prompt" in raw or "metadata" in raw
        line: dict[str, Any] = {
            "row_index": row.row_index,
            "prompt": _dataset_row_prompt(raw) if has_prompt else "",
        }
        label = _dataset_row_label(raw)
        if label is not None:
            line["label"] = label
        lines.append(json.dumps(line, ensure_ascii=False))
    # A lone surrogate cannot be encoded; replace it instead of failing the run.
    return "\n".join(lines).encode("utf-8", errors="replace")
