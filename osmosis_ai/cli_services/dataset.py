from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from .errors import CLIError
from .shared import coerce_optional_float, gather_text_fragments


@dataclass(frozen=True)
class ConversationMessage:
    """Normalized conversation message with preserved raw payload fields."""

    role: str
    content: Any
    metadata: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = copy.deepcopy(self.metadata)
        payload["role"] = self.role
        if self.content is None:
            payload.pop("content", None)
        else:
            payload["content"] = copy.deepcopy(self.content)
        return payload

    def text_fragments(self) -> list[str]:
        fragments: list[str] = []
        seen: set[int] = set()
        gather_text_fragments(self.content, fragments, allow_free_strings=True, seen=seen)
        for value in self.metadata.values():
            gather_text_fragments(value, fragments, seen=seen)
        return fragments

    @classmethod
    def from_raw(cls, raw: dict[str, Any], *, source_label: str, index: int) -> "ConversationMessage":
        role_value = raw.get("role")
        if not isinstance(role_value, str) or not role_value.strip():
            raise CLIError(
                f"Message {index} in {source_label} must include a non-empty string 'role'."
            )
        content_value = copy.deepcopy(raw.get("content"))
        metadata: dict[str, Any] = {}
        for key, value in raw.items():
            if key in {"role", "content"}:
                continue
            metadata[str(key)] = copy.deepcopy(value)
        return cls(role=role_value.strip().lower(), content=content_value, metadata=metadata)


@dataclass(frozen=True)
class DatasetRecord:
    payload: dict[str, Any]
    rubric_id: str
    conversation_id: Optional[str]
    record_id: Optional[str]
    messages: tuple[ConversationMessage, ...]
    ground_truth: Optional[str]
    system_message: Optional[str]
    original_input: Optional[str]
    metadata: Optional[dict[str, Any]]
    extra_info: Optional[dict[str, Any]]
    score_min: Optional[float]
    score_max: Optional[float]

    def message_payloads(self) -> list[dict[str, Any]]:
        """Return messages as provider-ready payloads."""
        return [message.to_payload() for message in self.messages]

    def merged_extra_info(self, config_extra: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        merged: dict[str, Any] = {}
        if isinstance(config_extra, dict):
            merged.update(copy.deepcopy(config_extra))
        if isinstance(self.extra_info, dict):
            merged.update(copy.deepcopy(self.extra_info))
        if isinstance(self.metadata, dict) and self.metadata:
            merged.setdefault("dataset_metadata", copy.deepcopy(self.metadata))
        return merged or None

    def assistant_preview(self, *, max_length: int = 140) -> Optional[str]:
        for message in reversed(self.messages):
            if message.role != "assistant":
                continue
            fragments = message.text_fragments()
            if not fragments:
                continue
            preview = " ".join(" ".join(fragments).split())
            if not preview:
                continue
            if len(preview) > max_length:
                preview = preview[: max_length - 3].rstrip() + "..."
            return preview
        return None

    def conversation_label(self, fallback_index: int) -> str:
        if isinstance(self.conversation_id, str) and self.conversation_id.strip():
            return self.conversation_id.strip()
        return f"record[{fallback_index}]"

    def record_identifier(self, conversation_label: str) -> str:
        if isinstance(self.record_id, str) and self.record_id.strip():
            return self.record_id.strip()
        raw_id = self.payload.get("id")
        if isinstance(raw_id, str) and raw_id.strip():
            return raw_id.strip()
        if raw_id is not None:
            return str(raw_id)
        return conversation_label


class DatasetLoader:
    """Loads dataset records from JSONL files."""

    def load(self, path: Path) -> list[DatasetRecord]:
        records: list[DatasetRecord] = []
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

                records.append(self._create_record(payload))

        if not records:
            raise CLIError(f"No JSON records found in '{path}'.")

        return records

    @staticmethod
    def _create_record(payload: dict[str, Any]) -> DatasetRecord:
        rubric_id = payload.get("rubric_id")
        rubric_id_str = str(rubric_id).strip() if isinstance(rubric_id, str) else ""

        conversation_id_raw = payload.get("conversation_id")
        conversation_id = None
        if isinstance(conversation_id_raw, str) and conversation_id_raw.strip():
            conversation_id = conversation_id_raw.strip()

        record_id_raw = payload.get("id")
        record_id = str(record_id_raw).strip() if isinstance(record_id_raw, str) else None

        score_min = coerce_optional_float(
            payload.get("score_min"), "score_min", f"record '{conversation_id or rubric_id or '<record>'}'"
        )
        score_max = coerce_optional_float(
            payload.get("score_max"), "score_max", f"record '{conversation_id or rubric_id or '<record>'}'"
        )

        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None
        extra_info = payload.get("extra_info") if isinstance(payload.get("extra_info"), dict) else None
        record_label = conversation_id or record_id or rubric_id_str or "<record>"
        messages = _parse_messages(payload.get("messages"), source_label=record_label)

        return DatasetRecord(
            payload=payload,
            rubric_id=rubric_id_str,
            conversation_id=conversation_id,
            record_id=record_id,
            messages=messages,
            ground_truth=payload.get("ground_truth") if isinstance(payload.get("ground_truth"), str) else None,
            system_message=payload.get("system_message") if isinstance(payload.get("system_message"), str) else None,
            original_input=payload.get("original_input") if isinstance(payload.get("original_input"), str) else None,
            metadata=metadata,
            extra_info=extra_info,
            score_min=score_min,
            score_max=score_max,
        )


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise CLIError(f"Invalid JSON on line {line_number} of '{path}': {exc.msg}") from exc
            if not isinstance(record, dict):
                raise CLIError(f"Expected JSON object on line {line_number} of '{path}'.")
            records.append(record)

    if not records:
        raise CLIError(f"No JSON records found in '{path}'.")

    return records


def render_json_records(records: Sequence[dict[str, Any]]) -> str:
    segments: list[str] = []
    total = len(records)

    for index, record in enumerate(records, start=1):
        body = json.dumps(record, indent=2, ensure_ascii=False)
        snippet = [f"JSONL record #{index}", body]
        if index != total:
            snippet.append("")
        segments.append("\n".join(snippet))

    return "\n".join(segments)


def _parse_messages(messages: Any, *, source_label: str) -> tuple[ConversationMessage, ...]:
    if not isinstance(messages, list) or not messages:
        raise CLIError(f"Record '{source_label}' must include a non-empty 'messages' list.")

    normalized: list[ConversationMessage] = []
    for index, entry in enumerate(messages):
        if not isinstance(entry, dict):
            raise CLIError(
                f"Message {index} in {source_label} must be an object, got {type(entry).__name__}."
            )
        normalized.append(ConversationMessage.from_raw(entry, source_label=source_label, index=index))
    return tuple(normalized)
