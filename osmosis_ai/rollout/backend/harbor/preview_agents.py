"""Private, best-effort snapshots of native Harbor agents while they run."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import shlex
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.agents.installed.mini_swe_agent import (
    MiniSweAgent,
    convert_mini_swe_agent_to_atif,
)
from harbor.agents.installed.opencode import OpenCode
from harbor.agents.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

from osmosis_ai.rollout.trajectory.preview import (
    MAX_SOURCE_BYTES,
    PREVIEW_INTERVAL_SEC,
    read_snapshot,
    write_snapshot,
)
from osmosis_ai.rollout.utils.evidence import _open_directory, _open_regular


async def read_remote_snapshot(
    environment: Any, source: str, target: Path
) -> bytes | None:
    # Each poll's leading rm replaces one bounded staging file, so no cleanup
    # exec is needed; some sandboxes keep a session for every exec.
    staging_id = hashlib.sha256(str(target).encode()).hexdigest()[:32]
    remote = f"/tmp/osmosis-preview-{staging_id}"
    quoted_source = shlex.quote(source)
    try:
        # The sandbox can swap the staging file for a link or special file, so
        # every download lands in a new directory and is never followed.
        download = Path(tempfile.mkdtemp(dir=target.parent)).resolve()
    except OSError:
        return None
    try:
        async with asyncio.timeout(3):
            result = await environment.exec(
                f"umask 077; set -C; rm -f -- {remote} && "
                f"test -f {quoted_source} && test ! -L {quoted_source} && "
                f"head -c {MAX_SOURCE_BYTES + 1} -- {quoted_source} > {remote}",
                timeout_sec=3,
            )
            if result.return_code != 0:
                return None
            await environment.download_file(remote, download / "snapshot")
            directory = _open_directory(download)
            try:
                with _open_regular(directory, "snapshot") as snapshot:
                    raw = snapshot.read(MAX_SOURCE_BYTES + 1)
            finally:
                os.close(directory)
            return raw if len(raw) <= MAX_SOURCE_BYTES else None
    except Exception:
        return None
    finally:
        shutil.rmtree(download, ignore_errors=True)


class _PreviewRunMixin(BaseAgent):
    def __init__(self, *args: Any, _osmosis_preview_path: str, **kwargs: Any) -> None:
        self._preview_path = Path(_osmosis_preview_path)
        self._preview_session_id = str(uuid.uuid4())
        super().__init__(*args, **kwargs)

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        stopped = asyncio.Event()
        collector = asyncio.create_task(self._collect_previews(environment, stopped))
        collector.add_done_callback(
            lambda task: task.exception() if not task.cancelled() else None
        )
        try:
            await super().run(instruction, environment, context)
        finally:
            stopped.set()
            # A blocked remote read must not extend Harbor's agent timeout.
            collector.cancel()

    async def _collect_previews(
        self, environment: BaseEnvironment, stopped: asyncio.Event
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="osmosis-native-preview-") as temp:
            target = Path(temp).resolve() / "snapshot"
            written: bytes | None = None
            while not stopped.is_set():
                try:
                    document = await self._capture_preview(environment, target)
                    if stopped.is_set():
                        return
                    if document and document.get("steps"):
                        written = write_snapshot(self._preview_path, document, written)
                except Exception:
                    # Native files may be between writes; retain the previous snapshot.
                    pass
                await asyncio.sleep(PREVIEW_INTERVAL_SEC)

    async def _capture_preview(
        self, environment: BaseEnvironment, target: Path
    ) -> dict[str, Any] | None:
        raise NotImplementedError


class _PreviewTerminus2(_PreviewRunMixin, Terminus2):
    async def _capture_preview(
        self, environment: BaseEnvironment, target: Path
    ) -> dict[str, Any] | None:
        # Harbor 0.23 moved trajectory settings into its validated options.
        options = getattr(self, "options", None)
        linear_history = (
            (options.trajectory_config or {}).get("linear_history", False)
            if options is not None
            else getattr(self, "_linear_history", False)
        )
        filename = (
            f"trajectory.cont-{self._summarization_count}.json"
            if linear_history and self._summarization_count
            else "trajectory.json"
        )
        document = read_snapshot(self.logs_dir / filename)
        if document is not None and filename != "trajectory.json":
            document["extra"] = {"osmosis": {"turn": None, "truncated": True}}
        return document


class _PreviewMiniSweAgent(_PreviewRunMixin, MiniSweAgent):
    async def _capture_preview(
        self, environment: BaseEnvironment, target: Path
    ) -> dict[str, Any] | None:
        raw = await read_remote_snapshot(
            environment,
            str(
                getattr(self, "environment_logs_dir", EnvironmentPaths.agent_dir)
                / "mini-swe-agent.trajectory.json"
            ),
            target,
        )
        if raw is None:
            return None
        trajectory = convert_mini_swe_agent_to_atif(
            json.loads(raw), self._preview_session_id
        )
        return trajectory.to_json_dict()


class _PreviewOpenCode(_PreviewRunMixin, OpenCode):
    async def _capture_preview(
        self, environment: BaseEnvironment, target: Path
    ) -> dict[str, Any] | None:
        raw = await read_remote_snapshot(
            environment,
            str(
                getattr(self, "environment_logs_dir", EnvironmentPaths.agent_dir)
                / "opencode.txt"
            ),
            target,
        )
        if raw is None or b"\n" not in raw:
            return None
        events = []
        for line in raw.rsplit(b"\n", 1)[0].splitlines():
            try:
                event = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(event, dict):
                events.append(event)
        converter = copy.copy(self)
        # Upstream conversion diagnostics can include raw native events.
        converter.logger = logging.Logger(__name__, level=logging.CRITICAL + 1)
        converter.logger.disabled = True
        trajectory = converter._convert_events_to_trajectory(events)
        return trajectory.to_json_dict() if trajectory is not None else None
