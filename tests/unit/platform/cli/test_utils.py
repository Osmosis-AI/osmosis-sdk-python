"""Tests for platform CLI utility functions."""

from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

import pytest

from osmosis_ai.cli.console import Console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.cli.utils import (
    format_dataset_status,
    format_processing_step,
    format_run_status,
    format_size,
    resolve_id_prefix,
)

# ── format_size ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "size_bytes, expected",
    [
        (0, "0 B"),
        (512, "512 B"),
        (1024, "1.0 KB"),
        (1536, "1.5 KB"),
        (1048576, "1.0 MB"),
        (1073741824, "1.0 GB"),
        (1099511627776, "1.0 TB"),
        (5497558138880, "5.0 TB"),
    ],
)
def test_format_size(size_bytes: int, expected: str) -> None:
    assert format_size(size_bytes) == expected


# ── resolve_id_prefix ────────────────────────────────────────────────


def _make_items(*ids: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(id=id_) for id_ in ids]


def test_resolve_id_prefix_full_id_returned_as_is() -> None:
    full_id = "a" * 32
    assert resolve_id_prefix(full_id, []) == full_id


def test_resolve_id_prefix_unique_match() -> None:
    items = _make_items("abc123def456", "xyz789ghi012")
    assert resolve_id_prefix("abc", items) == "abc123def456"


def test_resolve_id_prefix_no_match_raises() -> None:
    items = _make_items("abc123def456")
    with pytest.raises(CLIError, match="No item found"):
        resolve_id_prefix("zzz", items)


def test_resolve_id_prefix_no_match_with_has_more_hint() -> None:
    items = _make_items("abc123def456")
    with pytest.raises(CLIError, match="too large to search"):
        resolve_id_prefix("zzz", items, has_more=True)


def test_resolve_id_prefix_ambiguous_raises() -> None:
    items = _make_items("abc111", "abc222", "abc333")
    with pytest.raises(CLIError, match="Ambiguous"):
        resolve_id_prefix("abc", items)


def test_resolve_id_prefix_custom_entity_name() -> None:
    items = _make_items("abc123")
    with pytest.raises(CLIError, match="No dataset found"):
        resolve_id_prefix("zzz", items, entity_name="dataset")


# ── format_processing_step ───────────────────────────────────────────


def test_format_processing_step_none_when_no_step() -> None:
    obj = SimpleNamespace(processing_step=None, processing_percent=None)
    assert format_processing_step(obj) is None


def test_format_processing_step_with_step_only() -> None:
    obj = SimpleNamespace(processing_step="training", processing_percent=None)
    assert format_processing_step(obj) == "training"


def test_format_processing_step_with_percent() -> None:
    obj = SimpleNamespace(processing_step="training", processing_percent=42.7)
    assert format_processing_step(obj) == "training (43%)"


# ── format_dataset_status ────────────────────────────────────────────


def _dataset(status: str, step: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(status=status, processing_step=step, processing_percent=None)


def test_format_dataset_status_for_prompt() -> None:
    assert (
        format_dataset_status(_dataset("completed"), for_prompt=True) == "[completed]"
    )


def test_format_dataset_status_for_prompt_with_step() -> None:
    assert (
        format_dataset_status(_dataset("processing", "validating"), for_prompt=True)
        == "[processing: validating]"
    )


def test_format_dataset_status_success_styled(monkeypatch: pytest.MonkeyPatch) -> None:
    import osmosis_ai.platform.cli.utils as mod

    c = Console(file=StringIO(), force_terminal=True)
    monkeypatch.setattr(mod, "console", c)
    result = format_dataset_status(_dataset("completed"))
    assert "completed" in result


def test_format_dataset_status_in_progress_styled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import osmosis_ai.platform.cli.utils as mod

    c = Console(file=StringIO(), force_terminal=True)
    monkeypatch.setattr(mod, "console", c)
    result = format_dataset_status(_dataset("processing"))
    assert "processing" in result


def test_format_dataset_status_error_styled(monkeypatch: pytest.MonkeyPatch) -> None:
    import osmosis_ai.platform.cli.utils as mod

    c = Console(file=StringIO(), force_terminal=True)
    monkeypatch.setattr(mod, "console", c)
    result = format_dataset_status(_dataset("failed"))
    assert "failed" in result


def test_format_dataset_status_unknown_uses_escape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import osmosis_ai.platform.cli.utils as mod

    c = Console(file=StringIO(), force_terminal=True)
    monkeypatch.setattr(mod, "console", c)
    result = format_dataset_status(_dataset("unknown_status"))
    assert "[unknown_status]" in result


# ── format_run_status ────────────────────────────────────────────────


def _run(
    status: str, step: str | None = None, pct: float | None = None
) -> SimpleNamespace:
    return SimpleNamespace(status=status, processing_step=step, processing_percent=pct)


def test_format_run_status_for_prompt() -> None:
    assert format_run_status(_run("completed"), for_prompt=True) == "[completed]"


def test_format_run_status_for_prompt_with_step_and_percent() -> None:
    result = format_run_status(_run("running", "training", 55.3), for_prompt=True)
    assert result == "[running: training 55%]"


def test_format_run_status_success(monkeypatch: pytest.MonkeyPatch) -> None:
    import osmosis_ai.platform.cli.utils as mod

    c = Console(file=StringIO(), force_terminal=True)
    monkeypatch.setattr(mod, "console", c)
    result = format_run_status(_run("completed"))
    assert "completed" in result


def test_format_run_status_in_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    import osmosis_ai.platform.cli.utils as mod

    c = Console(file=StringIO(), force_terminal=True)
    monkeypatch.setattr(mod, "console", c)
    result = format_run_status(_run("running"))
    assert "running" in result


def test_format_run_status_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import osmosis_ai.platform.cli.utils as mod

    c = Console(file=StringIO(), force_terminal=True)
    monkeypatch.setattr(mod, "console", c)
    result = format_run_status(_run("failed"))
    assert "failed" in result


def test_format_run_status_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    import osmosis_ai.platform.cli.utils as mod

    c = Console(file=StringIO(), force_terminal=True)
    monkeypatch.setattr(mod, "console", c)
    result = format_run_status(_run("stopped"))
    assert "stopped" in result


def test_format_run_status_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    import osmosis_ai.platform.cli.utils as mod

    c = Console(file=StringIO(), force_terminal=True)
    monkeypatch.setattr(mod, "console", c)
    result = format_run_status(_run("weird"))
    assert "[weird]" in result
