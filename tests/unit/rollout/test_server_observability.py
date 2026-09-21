from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter

from osmosis_ai.rollout.backend.base import ExecutionBackend
from osmosis_ai.rollout.context import RolloutProgress, get_rollout_context
from osmosis_ai.rollout.server import app as app_module
from osmosis_ai.rollout.server.observability import RolloutObservability
from osmosis_ai.rollout.types import ExecutionOutcome, ExecutionResult, RolloutStatus


@pytest.fixture
def exported(monkeypatch):
    from opentelemetry.exporter.otlp.proto.http import _log_exporter

    exporter = InMemoryLogRecordExporter()
    monkeypatch.setattr(_log_exporter, "OTLPLogExporter", lambda **_: exporter)
    monkeypatch.setenv("OSMOSIS_ROLLOUT_OTLP_ENDPOINT", "http://collector")
    monkeypatch.setenv("OSMOSIS_ROLLOUT_SERVER_ID", "dev-owner")
    monkeypatch.setenv("_OSMOSIS_ROLLOUT_NAME", "remote-opencode")
    monkeypatch.setenv("OSMOSIS_ROLLOUT_NAMESPACE", "managed-dev-servers-staging")
    monkeypatch.setattr(app_module, "save_trajectory", AsyncMock())
    return exporter


@pytest.mark.parametrize("terminal", [RolloutStatus.SUCCESS, RolloutStatus.FAILURE])
async def test_ownership_survives_all_phases_without_exporting_inputs(
    exported, terminal
):
    class Backend(ExecutionBackend):
        async def execute(self, request):
            ctx = get_rollout_context()
            assert ctx is not None
            await ctx.set_status(RolloutStatus.GRADING)
            return ExecutionOutcome(workflow=ExecutionResult(status=terminal))

    app = app_module.create_rollout_server(backend=Backend())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://server"
        ) as client:
            admitted = await client.post(
                "/rollout",
                json={
                    "rollout_id": "episode-1",
                    "initial_messages": [{"role": "user", "content": "private prompt"}],
                    "chat_completions_url": "http://private-callback",
                    "llm_api_key": "private-key",
                    "grade": False,
                    "metadata": {
                        "secret": "private-metadata",
                        "osmosis_observability": {
                            "run_id": "train-123",
                            "run_name": "GLM training",
                            "server_id": "forged-owner",
                            "secret": "another-secret",
                        },
                    },
                },
            )
            assert admitted.status_code == 202
            entry = app.state.rollout_futures.entry("episode-1")
            await entry.result
            await asyncio.sleep(0)  # completed-future ownership callback
    records = [dict(row.log_record.attributes) for row in exported.get_finished_logs()]
    assert [row["status"] for row in records] == [
        "queued",
        "running",
        "grading",
        terminal.value,
    ]
    for sequence, row in enumerate(records, 1):
        assert row == {
            "event": "rollout.ownership",
            "rollout_id": "episode-1",
            "server_id": "dev-owner",
            "server_name": "remote-opencode",
            "namespace": "managed-dev-servers-staging",
            "run_id": "train-123",
            "run_name": "GLM training",
            "status": row["status"],
            "event_sequence": sequence,
        }


@pytest.mark.parametrize("expired", [False, True])
async def test_cancellation_and_lease_expiry_publish_terminal_ownership(
    exported, expired
):
    started = asyncio.Event()

    class Backend(ExecutionBackend):
        async def execute(self, request):
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    app = app_module.create_rollout_server(
        backend=Backend(),
        result_wait_timeout_sec=0.01,
        polling_lease_timeout_sec=0.08 if expired else 30,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://server"
        ) as client:
            payload = {
                "rollout_id": "episode",
                "initial_messages": [],
                "chat_completions_url": "http://llm",
            }
            assert (await client.post("/rollout", json=payload)).status_code == 202
            assert (await client.post("/rollout", json=payload)).status_code == 409
            await started.wait()
            if not expired:
                await client.post("/rollout/cancel", json={"ids": ["episode"]})
            result = await asyncio.wait_for(
                app.state.rollout_futures.entry("episode").result, 1
            )
            await asyncio.sleep(0)
    records = [dict(row.log_record.attributes) for row in exported.get_finished_logs()]
    assert [row["status"] for row in records].count("queued") == 1
    assert records[-1]["status"] == result.status.value
    assert records[-1]["status"] == ("failure" if expired else "cancelled")
    assert "run_name" not in records[-1]


def test_no_endpoint_leaves_app_telemetry_disabled(monkeypatch):
    monkeypatch.delenv("OSMOSIS_ROLLOUT_OTLP_ENDPOINT", raising=False)
    observer = RolloutObservability()
    observer.start()
    assert observer.provider is None
    observer.record(observer.fields("episode", {}), RolloutStatus.RUNNING)
    observer.close()


def test_metadata_is_bounded_and_cannot_override_owner(exported):
    observer = RolloutObservability()
    fields = observer.fields(
        "episode",
        {
            "osmosis_observability": {
                "run_id": ["bad"],
                "run_name": "x" * 513,
                "server_id": "fake",
            }
        },
    )
    assert fields is not None
    assert "run_id" not in fields and "run_name" not in fields
    assert fields["server_id"] == "dev-owner"


@pytest.mark.parametrize("value", ["x" * 513, "a\nb"])
@pytest.mark.parametrize(
    ("field", "env_var"),
    [
        ("server_id", "OSMOSIS_ROLLOUT_SERVER_ID"),
        ("server_name", "_OSMOSIS_ROLLOUT_NAME"),
        ("namespace", "OSMOSIS_ROLLOUT_NAMESPACE"),
        ("rollout_id", None),
        ("run_id", None),
        ("run_name", None),
    ],
)
def test_invalid_identity_never_exports_a_different_owner(
    exported, monkeypatch, field, env_var, value
):
    if env_var is not None:
        monkeypatch.setenv(env_var, value)
    observer = RolloutObservability()
    observer.start()
    fields = observer.fields(
        value if field == "rollout_id" else "episode",
        {"osmosis_observability": {field: value}},
    )
    observer.record(fields, RolloutStatus.RUNNING)
    observer.close()
    records = exported.get_finished_logs()
    if field in {"server_id", "rollout_id"}:
        assert not records
    else:
        assert len(records) == 1
        attributes = dict(records[0].log_record.attributes)
        assert field not in attributes
        assert attributes["server_id"] == "dev-owner"
        assert attributes["rollout_id"] == "episode"


async def test_status_observer_failure_does_not_hold_lock_or_fail_rollout():
    progress = RolloutProgress()
    lock_states = []

    def broken_observer(status):
        lock_states.append(progress.changed.locked())
        raise RuntimeError("broken observer")

    progress.on_status = broken_observer
    await progress.set_status(RolloutStatus.RUNNING)
    assert progress.status == RolloutStatus.RUNNING
    await progress.set_status(RolloutStatus.SUCCESS)
    assert await progress.wait_for_status_change() == RolloutStatus.SUCCESS
    assert lock_states == [False, False]
