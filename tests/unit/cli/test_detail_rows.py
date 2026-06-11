"""ID rows in shared detail-row builders are internal-user only."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from osmosis_ai.platform.cli.utils import (
    build_dataset_detail_rows,
    build_run_detail_rows,
)


def _run(**overrides: Any) -> SimpleNamespace:
    base: dict[str, Any] = {
        "id": "run_1",
        "name": "qwen3-run1",
        "status": "finished",
        "created_at": None,
        "creator_name": None,
        "dataset_name": None,
        "model_name": None,
        "rollout_name": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _dataset(**overrides: Any) -> SimpleNamespace:
    base: dict[str, Any] = {
        "id": "ds_1",
        "file_name": "train.jsonl",
        "status": "ready",
        "file_size": 1024,
        "created_at": "",
        "creator_name": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class TestRunDetailRows:
    def test_includes_id_row_for_internal_user(self) -> None:
        rows = build_run_detail_rows(_run(), include_id=True)
        assert rows[:3] == [
            ("Name", "qwen3-run1"),
            ("ID", "run_1"),
            ("Status", "Finished"),
        ]

    def test_omits_id_row_for_external_user(self) -> None:
        rows = build_run_detail_rows(_run(), include_id=False)
        labels = [label for label, _value in rows]
        assert "ID" not in labels
        assert labels[:2] == ["Name", "Status"]


class TestDatasetDetailRows:
    def test_includes_id_row_for_internal_user(self) -> None:
        rows = build_dataset_detail_rows(_dataset(), include_id=True)
        assert rows[:3] == [
            ("File", "train.jsonl"),
            ("ID", "ds_1"),
            ("Status", "Ready"),
        ]

    def test_omits_id_row_for_external_user(self) -> None:
        rows = build_dataset_detail_rows(_dataset(), include_id=False)
        labels = [label for label, _value in rows]
        assert "ID" not in labels
        assert labels[:2] == ["File", "Status"]
