"""The bounded grace a cancelled run gives the backend to unwind (§8).

The supervisor kills the rollout server as soon as this returns, so the wait
must honour its own budget and its verdict must reflect what it actually saw.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from osmosis_ai.eval.local import runner as runner_module
from osmosis_ai.eval.local.runner import (
    EvalRunSpec,
    LocalEvalOptions,
    LocalEvalRunner,
)

BASE_URL = "http://rollout.test"
UNWOUND = "cancelled work unwound"


class _RecordingLog:
    """Stands in for ``RunLog``: captures every level, not just warnings."""

    def __init__(self) -> None:
        self.entries: list[tuple[str, str, str]] = []

    def write(self, level: str, step: str, message: str, **details: Any) -> None:
        self.entries.append((level, step, message))

    def messages(self) -> list[str]:
        return [message for _, _, message in self.entries]


class _Hooks:
    def note(self, message: str) -> None: ...
    def stage(self, message: str) -> None: ...
    async def confirm_dispatch(self, *, pending: int, model_path: str) -> None: ...
    async def confirm_new_run(self, *, run_name: str, total: int) -> bool:
        return False

    def resolve_secrets(self, names: Sequence[str]) -> dict[str, str]:
        return {}

    def progress(self, snapshot: Any) -> None: ...


def _runner(
    tmp_path: Path, rollout_ids: Sequence[str]
) -> tuple[LocalEvalRunner, _RecordingLog]:
    from osmosis_ai.eval.local.dataset import resolve_explicit_dataset_file, select_rows

    data = tmp_path / "d.jsonl"
    data.write_text('{"user_prompt": "a", "ground_truth": "1"}\n')
    runner = LocalEvalRunner(
        spec=EvalRunSpec(
            rollout_name="echo-rollout",
            entrypoint="main.py",
            model_path="openai/gpt-5-mini",
            dataset_name="echo",
        ),
        options=LocalEvalOptions(name="run-1"),
        dataset=resolve_explicit_dataset_file(data),
        selection=select_rows(data),
        rollout_dir=tmp_path,
        output_root=tmp_path / "evals",
        hooks=_Hooks(),
    )
    log = _RecordingLog()
    runner._log = log  # type: ignore[assignment]
    # Settling reads only the dispatched ids, never the work items.
    runner._dispatch_context = dict.fromkeys(rollout_ids)  # type: ignore[assignment]
    return runner, log


def _shrink_grace(
    monkeypatch: pytest.MonkeyPatch, *, grace: float, poll: float = 0.01
) -> None:
    monkeypatch.setattr(runner_module, "_CANCEL_SETTLE_SEC", grace)
    monkeypatch.setattr(runner_module, "_CANCEL_POLL_INTERVAL_SEC", poll)


def _rollout_id_of(request: httpx.Request) -> str:
    return request.url.path.split("/")[2]


def _server_error(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, text="boom")


def _malformed_body(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text="<html>not json</html>")


def _unreachable(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def _unknown_status(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"status": "some-future-state"})


def _missing_status(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"rollout_id": _rollout_id_of(request)})


@pytest.mark.parametrize(
    "handler",
    [_server_error, _malformed_body, _unreachable, _unknown_status, _missing_status],
    ids=[
        "server-error",
        "malformed-body",
        "unreachable",
        "unknown-status",
        "missing-status",
    ],
)
async def test_an_unreadable_status_is_never_reported_as_unwound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    # Nothing was observed, so waiting is pointless -- but the run must not
    # claim the backend released its sandbox either.
    _shrink_grace(monkeypatch, grace=5.0)
    runner, log = _runner(tmp_path, ["id0", "id1"])

    started = time.monotonic()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await runner._settle_cancellations(client, BASE_URL)

    assert time.monotonic() - started < 5.0
    assert UNWOUND not in log.messages()
    assert log.entries[-1][0] == "warning"


async def test_a_settled_rollout_stops_being_polled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _shrink_grace(monkeypatch, grace=5.0)
    runner, log = _runner(tmp_path, ["settles-first", "settles-late"])
    calls: Counter[str] = Counter()

    def handler(request: httpx.Request) -> httpx.Response:
        rollout_id = _rollout_id_of(request)
        calls[rollout_id] += 1
        running = rollout_id == "settles-late" and calls[rollout_id] < 3
        status = "running" if running else "cancelled"
        return httpx.Response(200, json={"status": status})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await runner._settle_cancellations(client, BASE_URL)

    # Re-polling an id already seen terminal spends the grace on nothing.
    assert calls["settles-first"] == 1
    assert calls["settles-late"] == 3
    assert log.entries[-1] == ("info", "cancel", UNWOUND)
