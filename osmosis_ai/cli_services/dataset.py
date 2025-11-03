from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from .errors import CLIError
from .shared import coerce_optional_float


@dataclass(frozen=True)
class DatasetRecord:
    payload: dict[str, Any]
    rubric_id: str
    conversation_id: Optional[str]
    record_id: Optional[str]
    solution_str: str
    ground_truth: Optional[str]
    original_input: Optional[str]
    metadata: Optional[dict[str, Any]]
    extra_info: Optional[dict[str, Any]]
    score_min: Optional[float]
    score_max: Optional[float]

    def merged_extra_info(self) -> Optional[dict[str, Any]]:
        merged: dict[str, Any] = {}
        if isinstance(self.extra_info, dict):
            merged.update(copy.deepcopy(self.extra_info))
        if isinstance(self.metadata, dict) and self.metadata:
            merged.setdefault("dataset_metadata", copy.deepcopy(self.metadata))
        return merged or None

    def assistant_preview(self, *, max_length: int = 140) -> Optional[str]:
        text = self.solution_str.strip()
        if not text:
            return None
        preview = " ".join(text.split())
        if not preview:
            return None
        if len(preview) > max_length:
            preview = preview[: max_length - 3].rstrip() + "..."
        return preview

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
        solution_raw = payload.get("solution_str")
        if not isinstance(solution_raw, str) or not solution_raw.strip():
            raise CLIError(f"Record '{record_label}' must include a non-empty 'solution_str' string.")

        original_input_raw = payload.get("original_input")
        if isinstance(original_input_raw, str):
            original_input = original_input_raw
        else:
            original_input = None

        if original_input is None and isinstance(extra_info, dict):
            extra_original_input = extra_info.get("original_input")
            if isinstance(extra_original_input, str):
                original_input = extra_original_input

        return DatasetRecord(
            payload=payload,
            rubric_id=rubric_id_str,
            conversation_id=conversation_id,
            record_id=record_id,
            solution_str=solution_raw,
            ground_truth=payload.get("ground_truth") if isinstance(payload.get("ground_truth"), str) else None,
            original_input=original_input,
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
