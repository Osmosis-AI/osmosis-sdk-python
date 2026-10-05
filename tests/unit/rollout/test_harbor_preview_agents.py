import asyncio
import json
import shlex
import shutil
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import pytest

from osmosis_ai.rollout.trajectory.preview import read_snapshot


@pytest.fixture
def preview_agents(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from osmosis_ai.rollout.backend.harbor import preview_agents

    return preview_agents


@pytest.mark.parametrize("source_kind", ["regular", "oversized", "symlink"])
async def test_remote_snapshot_bounds_the_download(
    tmp_path, monkeypatch, source_kind, preview_agents
):
    monkeypatch.setattr(preview_agents, "MAX_SOURCE_BYTES", 16)
    source = tmp_path / "native.json"
    source.write_bytes(b"x" * (17 if source_kind == "oversized" else 16))
    if source_kind == "symlink":
        link = tmp_path / "link"
        link.symlink_to(source)
        source = link

    remote_paths = set()

    class Environment:
        async def exec(self, command, **kwargs):
            remote_paths.add(Path(command.rsplit(" > ", 1)[1]))
            process = await asyncio.create_subprocess_exec(
                "/bin/sh",
                "-c",
                command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            return SimpleNamespace(return_code=await process.wait())

        async def download_file(self, remote, target):
            shutil.copyfile(remote, target)

    try:
        raw = await preview_agents.read_remote_snapshot(
            Environment(), str(source), tmp_path / "snapshot"
        )

        assert raw == (b"x" * 16 if source_kind == "regular" else None)
        staged = [path for path in remote_paths if path.exists()]
        assert all(path.stat().st_size <= 17 for path in staged)
        assert len(staged) == int(source_kind != "symlink")
        assert not list(tmp_path.glob("tmp*"))
    finally:
        for path in remote_paths:
            path.unlink(missing_ok=True)


async def test_downloaded_links_are_never_followed_or_written_through(
    tmp_path, preview_agents
):
    source = tmp_path / "native.json"
    source.write_bytes(b"transcript")
    outside = tmp_path / "host-only"
    outside.write_bytes(b"host secret")
    private = tmp_path / "private"
    private.mkdir()
    downloads = 0

    class Environment:
        async def exec(self, command, **kwargs):
            return SimpleNamespace(return_code=0)

        async def download_file(self, remote, target):
            nonlocal downloads
            downloads += 1
            if downloads == 1:
                # Docker copies a link the sandbox swapped in as a link.
                Path(target).symlink_to(outside)
            else:
                # A later copy follows an existing destination link.
                Path(target).write_bytes(b"sandbox bytes")

    target = private / "snapshot"
    first = await preview_agents.read_remote_snapshot(
        Environment(), str(source), target
    )
    second = await preview_agents.read_remote_snapshot(
        Environment(), str(source), target
    )

    assert first is None
    assert second == b"sandbox bytes"
    assert outside.read_bytes() == b"host secret"
    assert not list(private.iterdir())


@pytest.mark.parametrize("source_exists", [False, True])
async def test_remote_read_costs_one_exec(tmp_path, preview_agents, source_exists):
    source = tmp_path / "native.json"
    if source_exists:
        source.write_bytes(b"transcript")
    commands = []
    downloads = []

    class Environment:
        async def exec(self, command, **kwargs):
            commands.append(command)
            process = await asyncio.create_subprocess_exec("/bin/sh", "-c", command)
            return SimpleNamespace(return_code=await process.wait())

        async def download_file(self, remote, target):
            downloads.append(remote)
            shutil.copyfile(remote, target)

    try:
        for _ in range(2):
            raw = await preview_agents.read_remote_snapshot(
                Environment(), str(source), tmp_path / "snapshot"
            )
            assert raw == (b"transcript" if source_exists else None)
        # Some sandboxes keep a session per exec; the next poll replaces staging.
        assert len(commands) == 2
        assert len(downloads) == (2 if source_exists else 0)
    finally:
        if commands:
            Path(commands[0].rsplit(" > ", 1)[1]).unlink(missing_ok=True)


async def test_remote_read_timeouts_keep_staging_bounded(
    tmp_path, monkeypatch, preview_agents
):
    source = tmp_path / "native.json"
    source.write_bytes(b"transcript")
    remote_paths = set()
    timeouts = []
    timeout = asyncio.timeout

    def record_timeout(delay):
        context = timeout(delay)
        timeouts.append(context)
        return context

    monkeypatch.setattr(preview_agents.asyncio, "timeout", record_timeout)

    class Environment:
        async def exec(self, command, **kwargs):
            remote_paths.add(Path(command.rsplit(" > ", 1)[1]))
            process = await asyncio.create_subprocess_exec("/bin/sh", "-c", command)
            return SimpleNamespace(return_code=await process.wait())

        async def download_file(self, remote, target):
            timeouts[-1].reschedule(asyncio.get_running_loop().time())
            await asyncio.Future()

    try:
        for _ in range(3):
            assert (
                await preview_agents.read_remote_snapshot(
                    Environment(), str(source), tmp_path / "snapshot"
                )
                is None
            )
        # Every timed-out poll reuses the same bounded staging file.
        assert len(remote_paths) == 1
        assert next(iter(remote_paths)).read_bytes() == b"transcript"
        assert not list(tmp_path.glob("tmp*"))
    finally:
        for path in remote_paths:
            path.unlink(missing_ok=True)


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
@pytest.mark.parametrize("race", [False, True])
async def test_remote_staging_cannot_overwrite_linked_file(
    tmp_path, link_kind, race, preview_agents
):
    source = tmp_path / "native.json"
    source.write_bytes(b"transcript")
    victim = tmp_path / "victim"
    victim.write_bytes(b"do not change")
    remote_paths = set()

    class Environment:
        async def exec(self, command, **kwargs):
            if command.startswith("umask"):
                remote = Path(command.rsplit(" > ", 1)[1])
                remote_paths.add(remote)
                flag = "-s " if link_kind == "symlink" else ""
                link = f"ln {flag}-- {shlex.quote(str(victim))} {remote}"
                if race:
                    command = f'rm() {{ command rm "$@" && {link}; }}; {command}'
                else:
                    command = f"{link} && {command}"
            process = await asyncio.create_subprocess_exec(
                "/bin/sh", "-c", command, stderr=asyncio.subprocess.DEVNULL
            )
            return SimpleNamespace(return_code=await process.wait())

        async def download_file(self, remote, target):
            shutil.copyfile(remote, target)

    try:
        snapshot = await preview_agents.read_remote_snapshot(
            Environment(), str(source), tmp_path / "snapshot"
        )
        assert victim.read_bytes() == b"do not change"
        assert snapshot == (None if race else b"transcript")
    finally:
        for path in remote_paths:
            path.unlink(missing_ok=True)


@pytest.mark.parametrize("agent_name", ["terminus", "mini", "opencode"])
async def test_native_preview_updates_during_run_and_keeps_complete_snapshot(
    tmp_path, monkeypatch, agent_name, preview_agents
):
    native, wrapper = {
        "terminus": (preview_agents.Terminus2, preview_agents._PreviewTerminus2),
        "mini": (preview_agents.MiniSweAgent, preview_agents._PreviewMiniSweAgent),
        "opencode": (preview_agents.OpenCode, preview_agents._PreviewOpenCode),
    }[agent_name]
    monkeypatch.setattr(
        preview_agents.Terminus2, "_init_llm", lambda *a, **kw: object()
    )
    monkeypatch.setattr(preview_agents, "PREVIEW_INTERVAL_SEC", 0)
    finished = asyncio.Event()
    published = asyncio.Event()
    sampled_partial = asyncio.Event()
    raw = json.dumps({"messages": [{"role": "assistant", "content": "first"}]}).encode()
    if agent_name == "opencode":
        raw = (
            b"\n".join(
                json.dumps(event).encode()
                for event in [
                    {"type": "step_start", "sessionID": "session"},
                    {"type": "text", "part": {"type": "text", "text": "first"}},
                    {"type": "step_finish", "part": {}},
                    {"type": "step_start"},
                    {"type": "text", "part": {"type": "text", "text": "unfinished"}},
                ]
            )
            + b"\n{"
        )

    async def remote_snapshot(*args):
        if raw == b"{":
            sampled_partial.set()
        return raw

    async def native_run(*args):
        await finished.wait()

    original_write = preview_agents.write_snapshot

    def publish(*args):
        written = original_write(*args)
        published.set()
        return written

    monkeypatch.setattr(preview_agents, "read_remote_snapshot", remote_snapshot)
    monkeypatch.setattr(preview_agents, "write_snapshot", publish)
    monkeypatch.setattr(native, "run", native_run)
    logs = tmp_path / "logs"
    logs.mkdir()
    if agent_name == "terminus":
        (logs / "trajectory.json").write_text(
            json.dumps(
                {"steps": [{"step_id": 1, "source": "agent", "message": "first"}]}
            )
        )
    destination = tmp_path / "private" / "snapshot.json"
    agent = wrapper(
        logs_dir=logs,
        model_name="openai/student",
        _osmosis_preview_path=str(destination),
        **(
            {"trajectory_config": {"linear_history": True}}
            if agent_name == "terminus"
            else {}
        ),
    )
    # Harbor 0.20 uses the shared EnvironmentPaths constant instead.
    if agent_name != "terminus" and hasattr(agent, "environment_logs_dir"):
        del agent.environment_logs_dir
    run = asyncio.create_task(agent.run("task", SimpleNamespace(), SimpleNamespace()))
    try:
        async with asyncio.timeout(2):
            await published.wait()
        first = read_snapshot(destination)
        assert first is not None
        assert first["steps"][-1]["message"] == "first"
        assert "unfinished" not in destination.read_text()
        assert not run.done()
        assert not (logs / "trajectory.preview.json").exists()

        published.clear()
        if agent_name == "terminus":
            agent._summarization_count = 1
            (logs / "trajectory.cont-1.json").write_text(
                json.dumps(
                    {
                        "steps": [
                            {"step_id": 1, "source": "agent", "message": "continued"}
                        ]
                    }
                )
            )
            async with asyncio.timeout(2):
                await published.wait()
            assert read_snapshot(destination)["steps"][-1]["message"] == "continued"
            assert read_snapshot(destination)["extra"]["osmosis"]["turn"] is None
        else:
            raw = b"{"
            async with asyncio.timeout(2):
                await sampled_partial.wait()
            assert read_snapshot(destination) == first
    finally:
        finished.set()
        await run


@pytest.mark.parametrize("agent_kind", ["native", "harness"])
@pytest.mark.parametrize("reuse", [False, True])
async def test_finishing_agent_does_not_wait_for_unresponsive_preview(
    tmp_path, monkeypatch, preview_agents, agent_kind, reuse
):
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    started = asyncio.Event()
    returned = asyncio.Event()
    samplers = []

    async def remote_snapshot(*args):
        samplers.append(asyncio.current_task())
        if len(samplers) > 1:
            await asyncio.Future()
        entered.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        returned.set()
        return json.dumps(
            {
                "messages": [{"role": "assistant", "content": "late"}],
                "steps": [{"step_id": 1, "source": "agent", "message": "late"}],
            }
        ).encode()

    async def native_run(*args):
        started.set()
        await finished.wait()

    monkeypatch.setattr(preview_agents, "read_remote_snapshot", remote_snapshot)
    destination = tmp_path / "private" / "snapshot.json"
    if agent_kind == "native":
        monkeypatch.setattr(preview_agents.MiniSweAgent, "run", native_run)
        agent = preview_agents._PreviewMiniSweAgent(
            logs_dir=tmp_path / "logs", _osmosis_preview_path=str(destination)
        )
    else:
        from osmosis_ai.rollout.backend.harbor.harness_agent import (
            OsmosisHarnessInstalledAgent,
        )
        from osmosis_ai.rollout.container.files import ContainerInput

        input_path = tmp_path / "input.json"
        ContainerInput(rollout_id="r1").write(input_path)
        (tmp_path / "logs").mkdir()
        agent = OsmosisHarnessInstalledAgent(
            logs_dir=tmp_path / "logs",
            bundle_path=str(tmp_path / "bundle.whl"),
            agent_script="test-agent",
            input_path=str(input_path),
            _osmosis_preview_path=str(destination),
        )
        monkeypatch.setattr(agent, "exec_as_agent", native_run)
    environment = SimpleNamespace(capabilities=SimpleNamespace(mounted=True))
    run = asyncio.create_task(agent.run("task", environment, SimpleNamespace()))
    next_run = None
    try:
        async with asyncio.timeout(2):
            await entered.wait()
            finished.set()
            await run
            await cancelled.wait()
        assert not destination.exists()
        if reuse:
            finished.clear()
            started.clear()
            next_run = asyncio.create_task(
                agent.run("next phase", environment, SimpleNamespace())
            )
            async with asyncio.timeout(2):
                await started.wait()
        release.set()
        async with asyncio.timeout(2):
            await returned.wait()
        assert not destination.exists()
        assert samplers[0].done()
    finally:
        release.set()
        finished.set()
        if next_run is not None:
            await next_run
        for sampler in samplers:
            sampler.cancel()
            with suppress(asyncio.CancelledError):
                await sampler
        run.cancel()
        with suppress(asyncio.CancelledError):
            await run
    assert not destination.exists()
