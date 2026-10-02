"""Opt-in file previews, isolated from samples and canonical trajectories."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from osmosis_ai.rollout.backend.harbor.evidence import sanitized_text
from osmosis_ai.rollout.context import RolloutContext, RolloutProgress
from osmosis_ai.rollout.trajectory.atif import Trajectory
from osmosis_ai.rollout.trajectory.converter import convert_sample_to_trajectory
from osmosis_ai.rollout.types import RolloutResultResponse, RolloutSample, RolloutStatus
from osmosis_ai.rollout.utils.evidence import _open_directory, _open_regular

logger: logging.Logger = logging.getLogger(__name__)
PREVIEW_INTERVAL_SEC = 3
MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_PREVIEW_BYTES = 900 * 1024
MAX_PREVIEW_MESSAGES = 100
_PREVIEW_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")


def preview_root(rollout_id: str) -> Path | None:
    configured = os.environ.get("_OSMOSIS_ROLLOUT_PREVIEW_ROOT")
    if not configured or not _PREVIEW_ID.fullmatch(rollout_id):
        return None
    root = Path(configured)
    if not root.is_absolute() or any(p.is_symlink() for p in (root, *root.parents)):
        logger.warning("Rollout preview root is not a safe absolute directory")
        return None
    return root


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).encode()


def _atomic_write(path: Path, data: bytes) -> None:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Linked preview destination")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".preview-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def write_snapshot(path: Path, document: dict[str, Any]) -> None:
    data = _json_bytes(document)
    if len(data) > MAX_SOURCE_BYTES:
        raise ValueError("Preview source exceeds size limit")
    _atomic_write(path, data)


def read_snapshot(path: Path) -> dict[str, Any] | None:
    try:
        directory = _open_directory(path.parent)
        try:
            with _open_regular(directory, path.name) as stream:
                data = stream.read(MAX_SOURCE_BYTES + 1)
        finally:
            os.close(directory)
        if len(data) > MAX_SOURCE_BYTES:
            return None
        document = json.loads(data)
        if (
            isinstance(document, dict)
            and isinstance(document.get("steps"), list)
            and document["steps"]
        ):
            return document
    except (OSError, ValueError, RecursionError):
        pass
    return None


def _clip(
    value: Any, truncated: list[bool], depth: int = 0, *, api_key: str | None = None
) -> Any:
    if depth > 20:
        truncated[0] = True
        return "[preview truncated]"
    if isinstance(value, str):
        value = json.loads(sanitized_text(_json_bytes(value), api_key))
        if len(value) > 16_384:
            truncated[0] = True
            return value[:16_384] + " [preview truncated]"
        return value
    if isinstance(value, dict):
        if len(value) > 1000:
            truncated[0] = True
        return {
            str(key): _clip(child, truncated, depth + 1, api_key=api_key)
            for key, child in list(value.items())[:1000]
            if key not in {"prompt_token_ids", "completion_token_ids", "logprobs"}
        }
    if isinstance(value, (list, tuple)):
        if len(value) > 1000:
            truncated[0] = True
        return [
            _clip(child, truncated, depth + 1, api_key=api_key)
            for child in value[:1000]
        ]
    return value


def _source_bytes(
    sample: RolloutSample, rollout_id: str, api_key: str | None
) -> bytes | None:
    size = 0
    for chunk in json.JSONEncoder(ensure_ascii=True, allow_nan=False).iterencode(
        sample.trajectory_messages
    ):
        size += len(chunk)
        if size > MAX_SOURCE_BYTES:
            return None
    truncated = [bool(sample.extra_fields.get("_preview_truncated"))]
    document = convert_sample_to_trajectory(
        sample, rollout_id=rollout_id
    ).to_json_dict()
    document = _clip(document, truncated, api_key=api_key)
    document["extra"] = {
        "osmosis": {
            "truncated": truncated[0],
            "turn": sample.extra_fields.get("_preview_turn"),
            "step_ids_known": not sample.extra_fields.get("_preview_truncated", False),
        }
    }
    data = _json_bytes(document)
    return data if len(data) <= MAX_SOURCE_BYTES else None


async def capture_source(context: RolloutContext) -> None:
    if (
        not context.preview_enabled
        or context.preview_path is None
        or context.sample_source is None
    ):
        return
    try:
        async with asyncio.timeout(1):
            sample = await context.sample_source.get_preview(MAX_PREVIEW_MESSAGES)
        if (
            not context.preview_enabled
            or sample is None
            or not sample.trajectory_messages
        ):
            return
        data = await asyncio.to_thread(
            _source_bytes, sample, context.rollout_id, context.api_key
        )
        # Only the collector commits; a cancelled conversion cannot publish late.
        if context.preview_enabled and data is not None:
            _atomic_write(context.preview_path, data)
    except Exception:
        logger.debug("Rollout preview snapshot unavailable")


async def collect_source(context: RolloutContext) -> None:
    while context.preview_enabled:
        await capture_source(context)
        await asyncio.sleep(PREVIEW_INTERVAL_SEC)


def _preview_steps(
    steps: list[dict[str, Any]], *, step_ids_known: bool = True
) -> list[dict[str, Any]]:
    selected = []
    for index, original in enumerate(steps, 1):
        step = {
            key: value
            for key, value in original.items()
            if key
            in {
                "timestamp",
                "source",
                "model_name",
                "message",
                "reasoning_content",
                "reasoning_effort",
                "llm_call_count",
                "is_copied_context",
            }
        }
        step["step_id"] = index
        original_id = original.get("step_id")
        step["extra"] = {
            "osmosis": {
                "original_step_id": original_id
                if step_ids_known and type(original_id) is int
                else None
            }
        }
        calls = original.get("tool_calls")
        if isinstance(calls, list):
            step["tool_calls"] = [
                {
                    key: value
                    for key, value in call.items()
                    if key
                    in {
                        "tool_call_id",
                        "function_name",
                        "arguments",
                    }
                }
                for call in calls
                if isinstance(call, dict)
            ]
        call_ids = {call.get("tool_call_id") for call in step.get("tool_calls", [])}
        observation = original.get("observation")
        if isinstance(observation, dict) and isinstance(
            observation.get("results"), list
        ):
            results = []
            for result in observation["results"]:
                if not isinstance(result, dict):
                    continue
                item = {"content": result.get("content")}
                if result.get("source_call_id") in call_ids:
                    item["source_call_id"] = result["source_call_id"]
                results.append(item)
            step["observation"] = {"results": results}
        selected.append(step)
    return selected


def _preview_document(
    document: dict[str, Any], rollout_id: str, api_key: str | None
) -> dict[str, Any]:
    source_metadata = (
        document.get("extra", {}).get("osmosis", {})
        if isinstance(document.get("extra"), dict)
        else {}
    )
    truncated = [
        bool(source_metadata.get("truncated"))
        if isinstance(source_metadata, dict)
        else False
    ]
    steps = document["steps"]
    if len(steps) > MAX_PREVIEW_MESSAGES:
        truncated[0] = True
    turn = sum(
        step.get("source") == "agent" for step in steps if isinstance(step, dict)
    )
    if isinstance(source_metadata, dict) and "turn" in source_metadata:
        turn = source_metadata["turn"]
        turn = min(max(0, turn), 1_000_000) if type(turn) is int else None
    # Discard source identity, request metadata and agent configuration entirely.
    agent = document.get("agent", {})
    clean = {
        "schema_version": document.get("schema_version", "ATIF-v1.7"),
        "session_id": rollout_id,
        "trajectory_id": rollout_id,
        "agent": {
            key: value
            for key, value in agent.items()
            if key in {"name", "version", "model_name"}
        },
        "steps": _preview_steps(
            _clip(steps[-MAX_PREVIEW_MESSAGES:], truncated, api_key=api_key),
            step_ids_known=not isinstance(source_metadata, dict)
            or source_metadata.get("step_ids_known", True),
        ),
    }
    clean = json.loads(sanitized_text(_json_bytes(clean), api_key))
    clean = Trajectory.model_validate(clean).to_json_dict()
    clean["extra"] = {
        "osmosis": {
            "rollout_id": rollout_id,
            "preview": True,
            "turn": turn,
            "truncated": truncated[0],
        }
    }
    # Reserve enough room for the version, timestamp and status added below.
    while len(_json_bytes(clean)) > MAX_PREVIEW_BYTES - 256:
        clean["extra"]["osmosis"]["truncated"] = True
        if len(clean["steps"]) == 1:
            clean["steps"] = [
                {
                    "step_id": 1,
                    "source": "agent",
                    "message": "This turn exceeds the live preview size limit.",
                }
            ]
            break
        clean["steps"].pop(0)
        for index, step in enumerate(clean["steps"], 1):
            step["step_id"] = index
    return Trajectory.model_validate(clean).to_json_dict()


async def publish_previews(
    *,
    root: Path,
    source: Path,
    rollout_id: str,
    api_key: str | None,
    progress: RolloutProgress,
    result: asyncio.Future[RolloutResultResponse],
    execution: asyncio.Task[None],
) -> None:
    previous = b""
    version = 0
    latest: dict[str, Any] | None = None
    destination = root / rollout_id / "preview.json"
    while True:
        terminal = result.done() and not result.cancelled()
        status = (
            result.result().status
            if terminal
            else (RolloutStatus.UNKNOWN if execution.done() else progress.status)
        )
        try:
            document = await asyncio.to_thread(read_snapshot, source)
            if document is not None:
                latest = await asyncio.to_thread(
                    _preview_document, document, rollout_id, api_key
                )
        except Exception:
            logger.debug("Rollout preview conversion unavailable")
        try:
            if latest is not None:
                metadata = latest["extra"]["osmosis"]
                metadata.pop("preview_version", None)
                metadata.pop("updated_at", None)
                metadata["status"] = status.value
                signature = _json_bytes(latest)
                if signature != previous:
                    metadata["preview_version"] = version + 1
                    metadata["updated_at"] = datetime.now(UTC).isoformat()
                    data = _json_bytes(latest)
                    if len(data) <= MAX_PREVIEW_BYTES:
                        await asyncio.to_thread(_atomic_write, destination, data)
                        previous = signature
                        version += 1
        except Exception:
            logger.debug("Rollout preview publication unavailable")
        if terminal:
            return
        try:
            async with asyncio.timeout(PREVIEW_INTERVAL_SEC):
                await progress.wait_for_status_change()
        except TimeoutError:
            pass
