from __future__ import annotations

import asyncio
import copy
import json
import threading
from pathlib import Path

import httpx
import pytest

from osmosis_ai.rollout.backend.base import ExecutionBackend
from osmosis_ai.rollout.context import (
    RolloutContext,
    RolloutProgress,
    SampleSource,
    get_rollout_context,
)
from osmosis_ai.rollout.server.app import create_rollout_server
from osmosis_ai.rollout.trajectory import preview
from osmosis_ai.rollout.trajectory.atif import Trajectory
from osmosis_ai.rollout.types import (
    ExecutionOutcome,
    ExecutionResult,
    RolloutResultResponse,
    RolloutSample,
    RolloutStatus,
)


def document():
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": "untrusted",
        "agent": {"name": "agent", "version": "1"},
        "steps": [{"step_id": 1, "source": "agent", "message": "first turn"}],
    }


async def wait_preview(path: Path, status: str | None = None):
    async with asyncio.timeout(3):
        while True:
            if path.exists():
                doc = json.loads(path.read_text())
                if status is None or doc["extra"]["osmosis"]["status"] == status:
                    return doc
            await asyncio.sleep(0.005)


def test_export_is_bounded_sanitized_valid_atif_without_mutating_source():
    source = document()
    source["extra"] = {"osmosis": {"rollout_id": "forged", "reward": 9}}
    source["agent"]["extra"] = {"llm_kwargs": {"api_key": "private"}}
    source["steps"] = [
        {
            "step_id": index,
            "source": "agent",
            "message": "provider-key authorization=hidden " + "x" * 20_000,
            "extra": {"reward": 123, "request_metadata": "private"},
            "metrics": {"prompt_token_ids": [42], "extra": {"secret": "hidden"}},
            "tool_calls": [
                {
                    "tool_call_id": "call",
                    "function_name": "shell",
                    "arguments": {"password": "hidden", "command": "echo ok"},
                    "extra": {"private": "hidden"},
                }
            ],
            "observation": {
                "results": [
                    {
                        "source_call_id": "omitted-call",
                        "content": "response",
                        "extra": {"private": "hidden"},
                        "subagent_trajectory_ref": [
                            {"trajectory_path": "/private/result"}
                        ],
                    }
                ]
            },
        }
        for index in range(1, 131)
    ]
    original = copy.deepcopy(source)
    clean = preview._preview_document(source, "trusted", "provider-key")
    encoded = json.dumps(clean)
    assert len(encoded) < preview.MAX_PREVIEW_BYTES
    assert all(
        value not in encoded
        for value in (
            "provider-key",
            "hidden",
            "private",
            "reward",
            "prompt_token_ids",
            "llm_kwargs",
        )
    )
    assert clean["session_id"] == clean["trajectory_id"] == "trusted"
    assert clean["extra"]["osmosis"] == {
        "rollout_id": "trusted",
        "preview": True,
        "turn": 130,
        "truncated": True,
    }
    assert clean["steps"][0]["extra"]["osmosis"]["original_step_id"] > 30
    assert clean["steps"][-1]["extra"]["osmosis"]["original_step_id"] == 130
    assert "source_call_id" not in clean["steps"][-1]["observation"]["results"][0]
    Trajectory.model_validate(clean)
    assert source == original


@pytest.mark.parametrize(
    "terminal", [RolloutStatus.SUCCESS, RolloutStatus.FAILURE, RolloutStatus.CANCELLED]
)
async def test_publisher_keeps_last_snapshot_and_canonical_status(
    tmp_path, monkeypatch, terminal
):
    monkeypatch.setattr(preview, "PREVIEW_INTERVAL_SEC", 0.01)
    conversions = []
    convert = preview._preview_document

    def counted(*args):
        conversions.append(args[0])
        return convert(*args)

    monkeypatch.setattr(preview, "_preview_document", counted)
    source = tmp_path / "private" / "snapshot.json"
    destination = tmp_path / "live" / "r1" / "preview.json"
    progress = RolloutProgress(status=RolloutStatus.RUNNING)
    result = asyncio.get_running_loop().create_future()
    stop = asyncio.Event()
    execution = asyncio.create_task(stop.wait())
    publisher = asyncio.create_task(
        preview.publish_previews(
            root=tmp_path / "live",
            source=source,
            rollout_id="r1",
            api_key=None,
            progress=progress,
            result=result,
            execution=execution,
        )
    )
    try:
        await asyncio.sleep(0.03)
        assert not destination.exists()
        preview.write_snapshot(source, document())
        first = await wait_preview(destination, "running")
        assert first["extra"]["osmosis"]["preview_version"] == 1
        await asyncio.sleep(0.04)
        assert json.loads(destination.read_text()) == first
        # Unchanged bytes are not converted again on later polls.
        assert len(conversions) == 1
        source.write_text("{incomplete")
        await progress.set_status(RolloutStatus.GRADING)
        grading = await wait_preview(destination, "grading")
        assert grading["steps"] == first["steps"]
        assert grading["extra"]["osmosis"]["preview_version"] == 2
        assert len(conversions) == 1
        stop.set()
        await execution
        unknown = await wait_preview(destination, "unknown")
        assert unknown["extra"]["osmosis"]["preview_version"] == 3
        result.set_result(RolloutResultResponse(rollout_id="r1", status=terminal))
        await asyncio.wait_for(publisher, 1)
        final = await wait_preview(destination, terminal.value)
        assert final["extra"]["osmosis"]["preview_version"] == 4
        preview.write_snapshot(
            source,
            {
                **document(),
                "steps": [{"step_id": 1, "source": "agent", "message": "late"}],
            },
        )
        await asyncio.sleep(0.03)
        assert json.loads(destination.read_text()) == final
    finally:
        stop.set()
        publisher.cancel()
        await asyncio.gather(publisher, execution, return_exceptions=True)


def test_snapshot_rejects_symlinks_oversize_and_invalid_root(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    preview.write_snapshot(source, document())
    link = tmp_path / "linked.json"
    link.symlink_to(source)
    assert preview.read_snapshot(link) is None
    with pytest.raises(ValueError, match="Linked"):
        preview.write_snapshot(link, document())
    source.write_bytes(b" " * (preview.MAX_SOURCE_BYTES + 1))
    assert preview.read_snapshot(source) is None
    monkeypatch.setenv("_OSMOSIS_ROLLOUT_PREVIEW_ROOT", str(tmp_path))
    assert preview.preview_root("../escape") is None
    assert preview.preview_root("run:1") == tmp_path
    monkeypatch.setenv("_OSMOSIS_ROLLOUT_PREVIEW_ROOT", "relative")
    assert preview.preview_root("r1") is None


def test_unchanged_snapshot_is_not_rewritten(tmp_path):
    path = tmp_path / "snapshot.json"
    data = preview.write_snapshot(path, document())
    inode = path.stat().st_ino
    assert preview.write_snapshot(path, document(), data) == data
    assert path.stat().st_ino == inode
    preview.write_snapshot(path, {**document(), "steps": []}, data)
    assert path.stat().st_ino != inode


async def test_redaction_precedes_clipping_in_native_and_sample_sources(tmp_path):
    key = "sk-preview-credential-1234567890"
    message = "x" * 16_380 + key
    source = document()
    source["steps"][0]["message"] = message
    clean = preview._preview_document(source, "r1", key)
    assert "sk-p" not in json.dumps(clean)

    class Source(SampleSource):
        async def get_sample(self):
            pytest.fail("must not collect final sample")

        async def get_preview(self, max_messages):
            return RolloutSample(
                messages=[
                    {
                        "role": "assistant",
                        "content": message,
                        "tool_calls": [
                            {
                                "id": "call",
                                "type": "function",
                                "function": {
                                    "name": "run",
                                    "arguments": json.dumps(
                                        {
                                            "password": "dummy-secret-field",
                                            "payload": "x" * 17_000,
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                ],
                extra_fields={"_preview_truncated": True, "_preview_turn": None},
            )

    path = tmp_path / "snapshot.json"
    await preview.capture_source(
        RolloutContext(
            rollout_id="r1",
            api_key=key,
            preview_path=path,
            sample_source=Source(),
        )
    )
    snapshot = preview.read_snapshot(path)
    assert snapshot is not None
    assert "sk-p" not in json.dumps(snapshot)
    clean = preview._preview_document(snapshot, "r1", key)
    assert "dummy-secret-field" not in json.dumps(clean)
    assert "_raw" not in json.dumps(clean)
    assert clean["extra"]["osmosis"]["turn"] is None
    assert clean["steps"][0]["extra"]["osmosis"]["original_step_id"] is None


async def test_repeated_sanitization_keeps_redactions_exact(tmp_path):
    message = "export PASSWORD=hunter2; curl -H x-api-key: abc"
    expected = "export PASSWORD=[REDACTED]; curl -H x-api-key: [REDACTED]"
    source = document()
    source["steps"][0]["message"] = message
    assert (
        preview._preview_document(source, "r1", None)["steps"][0]["message"] == expected
    )

    class Source(PreviewSource):
        async def get_preview(self, max_messages):
            return RolloutSample(messages=[{"role": "assistant", "content": message}])

    path = tmp_path / "snapshot.json"
    await preview.capture_source(
        RolloutContext(rollout_id="r1", preview_path=path, sample_source=Source())
    )
    snapshot = preview.read_snapshot(path)
    assert snapshot is not None
    clean = preview._preview_document(snapshot, "r1", None)
    assert clean["steps"][0]["message"] == expected


def test_lone_surrogates_do_not_reach_published_previews():
    source = document()
    source["steps"][0]["message"] = "tool output \ud83d"
    source["steps"][0]["tool_calls"] = [
        {
            "tool_call_id": "call",
            "function_name": "run",
            "arguments": {"\udc00key": "value \udfff"},
        }
    ]
    clean = preview._preview_document(source, "r1", None)
    published = json.loads(preview._json_bytes(clean))
    # Consumers re-encode previews as UTF-8 without ASCII escapes.
    encoded = json.dumps(published, ensure_ascii=False).encode()
    assert "tool output ?" in encoded.decode()
    assert published["steps"][0]["tool_calls"][0]["arguments"] == {"?key": "value ?"}


async def test_custom_window_without_flags_does_not_claim_source_ids(tmp_path):
    class Window(PreviewSource):
        def __init__(self, size):
            self.size = size

        async def get_preview(self, max_messages):
            history = [
                {"role": "assistant", "content": f"m{index}"} for index in range(1, 501)
            ]
            return RolloutSample(messages=history[-min(self.size, max_messages) :])

    for size, truncated in ((500, True), (3, False)):
        path = tmp_path / f"snapshot-{size}.json"
        await preview.capture_source(
            RolloutContext(
                rollout_id="r1", preview_path=path, sample_source=Window(size)
            )
        )
        snapshot = preview.read_snapshot(path)
        assert snapshot is not None
        clean = preview._preview_document(snapshot, "r1", None)
        assert clean["steps"][0]["message"] == f"m{501 - min(size, 100)}"
        assert clean["extra"]["osmosis"]["truncated"] is truncated
        assert clean["extra"]["osmosis"]["turn"] is None
        assert all(
            step["extra"]["osmosis"]["original_step_id"] is None
            for step in clean["steps"]
        )


def test_size_fit_serializes_each_step_once(monkeypatch):
    source = document()
    source["steps"] = [
        {"step_id": index, "source": "agent", "message": "x" * 16_000}
        for index in range(1, 101)
    ]
    serialized = []
    encode = preview._json_bytes

    def counted(value):
        data = encode(value)
        serialized.append(len(data))
        return data

    monkeypatch.setattr(preview, "_json_bytes", counted)
    clean = preview._preview_document(source, "r1", None)
    monkeypatch.undo()

    kept = clean["steps"]
    assert clean["extra"]["osmosis"]["truncated"] is True
    assert [step["step_id"] for step in kept] == list(range(1, len(kept) + 1))
    assert kept[-1]["extra"]["osmosis"]["original_step_id"] == 100
    assert len(preview._json_bytes(clean)) <= preview.MAX_PREVIEW_BYTES - 256
    with_next = {**clean, "steps": [source["steps"][-len(kept) - 1], *kept]}
    assert len(preview._json_bytes(with_next)) > preview.MAX_PREVIEW_BYTES - 256
    # Popping one step per whole-document pass serializes tens of megabytes.
    assert sum(serialized) < 4 * len(json.dumps(source))


async def test_oversized_source_keeps_last_snapshot_without_conversion(
    tmp_path, monkeypatch
):
    class Source(SampleSource):
        async def get_sample(self):
            pytest.fail("must not collect final sample")

        async def get_preview(self, max_messages):
            return RolloutSample(messages=[{"role": "assistant", "content": "x" * 256}])

    path = tmp_path / "snapshot.json"
    preview.write_snapshot(path, document())
    before = path.read_bytes()
    monkeypatch.setattr(preview, "MAX_SOURCE_BYTES", 128)
    monkeypatch.setattr(
        preview,
        "convert_sample_to_trajectory",
        lambda *a, **k: pytest.fail("oversized input must not be converted"),
    )
    await preview.capture_source(
        RolloutContext(rollout_id="r1", preview_path=path, sample_source=Source())
    )
    assert path.read_bytes() == before


async def test_unchanged_source_history_is_not_reconverted_or_rewritten(
    tmp_path, monkeypatch
):
    # Each poll returns (messages, turn); the loop stops after the last one.
    polls = [(["first"], 1)] * 3 + [(["first"], 2)] + [(["first", "next"], 2)] * 2
    conversions, writes = [], []
    convert, write = preview.convert_sample_to_trajectory, preview._atomic_write

    def counted_convert(*args, **kwargs):
        conversions.append(args[0])
        return convert(*args, **kwargs)

    def counted_write(*args):
        writes.append(args[1])
        write(*args)

    class Source(PreviewSource):
        async def get_preview(self, max_messages):
            if not polls:
                context.preview_enabled = False
                return None
            contents, turn = polls.pop(0)
            return RolloutSample(
                messages=[{"role": "assistant", "content": c} for c in contents],
                extra_fields={"_preview_truncated": False, "_preview_turn": turn},
            )

    monkeypatch.setattr(preview, "PREVIEW_INTERVAL_SEC", 0)
    monkeypatch.setattr(preview, "convert_sample_to_trajectory", counted_convert)
    monkeypatch.setattr(preview, "_atomic_write", counted_write)
    path = tmp_path / "snapshot.json"
    context = RolloutContext(rollout_id="r1", preview_path=path, sample_source=Source())
    await asyncio.wait_for(preview.collect_source(context), 5)

    # Only the first poll and the turn and history changes are converted.
    assert len(conversions) == len(writes) == 3
    snapshot = preview.read_snapshot(path)
    assert snapshot is not None
    assert snapshot["steps"][-1]["message"] == "next"
    assert snapshot["extra"]["osmosis"]["turn"] == 2


@pytest.mark.parametrize("completion", ["publish", "disable", "cancel"])
async def test_slow_conversion_keeps_loop_responsive_and_cannot_publish_after_stop(
    tmp_path, monkeypatch, completion
):
    loop = asyncio.get_running_loop()
    entered, converted = asyncio.Event(), asyncio.Event()
    release = threading.Event()
    original = preview.convert_sample_to_trajectory

    def paused_conversion(*args, **kwargs):
        loop.call_soon_threadsafe(entered.set)
        try:
            assert release.wait(timeout=5)
            return original(*args, **kwargs)
        finally:
            loop.call_soon_threadsafe(converted.set)

    monkeypatch.setattr(preview, "convert_sample_to_trajectory", paused_conversion)
    path = tmp_path / "snapshot.json"
    context = RolloutContext(
        rollout_id="r1", preview_path=path, sample_source=PreviewSource()
    )
    capture = asyncio.create_task(preview.capture_source(context))
    try:
        await entered.wait()
        assert not capture.done()
        assert not path.exists()
        if completion == "cancel":
            capture.cancel()
            with pytest.raises(asyncio.CancelledError):
                await capture
        elif completion == "disable":
            context.preview_enabled = False
        release.set()
        await converted.wait()
        await asyncio.gather(capture, return_exceptions=True)
        await loop.shutdown_default_executor()
        assert path.exists() is (completion == "publish")
    finally:
        release.set()
        await asyncio.gather(capture, return_exceptions=True)


class PreviewSource(SampleSource):
    async def get_sample(self):
        pytest.fail("preview must not collect the grading sample")

    async def get_preview(self, max_messages):
        return RolloutSample(messages=[{"role": "assistant", "content": "working"}])


@pytest.mark.parametrize("terminal", ["success", "failure", "cancelled", "lease"])
async def test_server_publishes_live_then_registry_terminal(
    tmp_path, monkeypatch, terminal
):
    root = tmp_path / "live"
    archive = tmp_path / "archive"
    monkeypatch.setenv("_OSMOSIS_ROLLOUT_PREVIEW_ROOT", str(root))
    monkeypatch.setattr(preview, "PREVIEW_INTERVAL_SEC", 0.01)
    monkeypatch.setattr(
        "osmosis_ai.rollout.trajectory.save.default_artifact_root", lambda: archive
    )
    ready, finish = asyncio.Event(), asyncio.Event()

    class Backend(ExecutionBackend):
        async def execute(self, request):
            context = get_rollout_context()
            assert context is not None
            context.set_sample_source(PreviewSource())
            ready.set()
            await finish.wait()
            if terminal == "failure":
                raise ValueError("workflow failed")
            return ExecutionOutcome(
                workflow=ExecutionResult(
                    status=RolloutStatus.SUCCESS,
                    sample=RolloutSample(
                        messages=[{"role": "assistant", "content": "canonical"}],
                        reward=0.75,
                    ),
                )
            )

    app = create_rollout_server(
        backend=Backend(),
        polling_lease_timeout_sec=0.25 if terminal == "lease" else 5,
        result_wait_timeout_sec=0.05,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as http:
            admission = await http.post(
                "/rollout",
                json={
                    "rollout_id": "r1",
                    "initial_messages": [],
                    "chat_completions_url": "https://model.example/v1",
                },
            )
            assert admission.status_code == 202
            await ready.wait()
            path = root / "r1" / "preview.json"
            running = await wait_preview(path, "running")
            assert running["steps"][0]["message"] == "working"
            assert not (archive / "r1" / "trajectory.json").exists()
            if terminal == "cancelled":
                await http.post("/rollout/cancel", json={"ids": ["r1"]})
            elif terminal != "lease":
                finish.set()
            status = "failure" if terminal == "lease" else terminal
            await wait_preview(path, status)
    if terminal == "success":
        canonical = json.loads((archive / "r1" / "trajectory.json").read_text())
        assert canonical["steps"][0]["message"] == "canonical"
        assert canonical["extra"]["osmosis"]["reward"] == 0.75
        assert "preview" not in canonical["extra"]["osmosis"]
    assert not (archive / "r1" / "preview.json").exists()


@pytest.mark.parametrize("opt_out", [False, True])
async def test_server_without_opt_in_does_not_read_sources(
    tmp_path, monkeypatch, opt_out
):
    if opt_out:
        monkeypatch.setenv("_OSMOSIS_ROLLOUT_PREVIEW_ROOT", str(tmp_path / "live"))
    else:
        monkeypatch.delenv("_OSMOSIS_ROLLOUT_PREVIEW_ROOT", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.rollout.trajectory.save.default_artifact_root", lambda: tmp_path
    )

    reads = []

    class Source(PreviewSource):
        async def get_preview(self, max_messages):
            reads.append(max_messages)
            return await super().get_preview(max_messages)

    class Backend(ExecutionBackend):
        async def execute(self, request):
            context = get_rollout_context()
            if opt_out:
                context.preview_enabled = False
            else:
                assert context.preview_path is None
            context.set_sample_source(Source())
            await asyncio.sleep(0.03)
            return ExecutionOutcome(
                workflow=ExecutionResult(status=RolloutStatus.SUCCESS)
            )

    app = create_rollout_server(backend=Backend())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as http:
            assert (
                await http.post(
                    "/rollout",
                    json={
                        "rollout_id": "r1",
                        "initial_messages": [],
                        "chat_completions_url": "https://model.example/v1",
                    },
                )
            ).status_code == 202
    assert reads == []
    assert not list(tmp_path.rglob("preview.json"))
