"""The bundled harness transfers previews without changing its agent result."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import osmosis_ai.rollout.backend.harbor.harness_agent as harness_module
from osmosis_ai.rollout.container.files import INPUT_FILENAME, ContainerInput


def make_agent(tmp_path, *, enabled=True):
    logs = tmp_path / "logs"
    logs.mkdir()
    input_path = tmp_path / INPUT_FILENAME
    ContainerInput(rollout_id="r1", label="answer").write(input_path)
    return harness_module.OsmosisHarnessInstalledAgent(
        logs_dir=logs,
        bundle_path=str(tmp_path / "bundle.whl"),
        agent_script="test-agent",
        input_path=str(input_path),
        _osmosis_preview_path=str(tmp_path / "private" / "snapshot.json")
        if enabled
        else None,
    )


async def test_harness_keeps_last_snapshot_and_does_not_wait_for_remote_read(
    monkeypatch, tmp_path
):
    agent = make_agent(tmp_path)
    monkeypatch.setattr(harness_module, "PREVIEW_INTERVAL_SEC", 0)
    blocked = asyncio.Event()
    cancelled = asyncio.Event()
    document = {"steps": [{"step_id": 1, "source": "agent", "message": "working"}]}

    class Environment:
        capabilities = SimpleNamespace(mounted=True)
        downloads = 0

        async def exec(self, command, timeout_sec):
            return SimpleNamespace(return_code=0)

        async def download_file(self, source, target):
            self.downloads += 1
            assert not Path(target).is_relative_to(agent.logs_dir)
            if self.downloads == 1:
                Path(target).write_text(json.dumps(document))
            elif self.downloads == 2:
                Path(target).write_text("{}")
            elif self.downloads == 3:
                Path(target).write_text('{"steps":')
            else:
                blocked.set()
                try:
                    await asyncio.Future()
                finally:
                    cancelled.set()

    env = Environment()

    async def run_agent(environment, command):
        await blocked.wait()
        assert json.loads(agent.preview_path.read_text()) == document

    monkeypatch.setattr(agent, "exec_as_agent", run_agent)
    await asyncio.wait_for(agent.run("task", env, None), timeout=2)
    await asyncio.wait_for(cancelled.wait(), timeout=1)

    staged = ContainerInput.read(agent.logs_dir / INPUT_FILENAME)
    assert staged.preview_path is not None
    assert staged.preview_path.startswith("/tmp/osmosis-preview/")
    assert str(tmp_path) not in staged.preview_path
    assert staged.label is None
    assert ContainerInput.read(agent.input_path).preview_path is None
    assert json.loads(agent.preview_path.read_text()) == document
    assert not (agent.logs_dir / "trajectory.json").exists()


@pytest.mark.parametrize("enabled", [False, True])
async def test_missing_preview_does_not_change_agent_completion(
    monkeypatch, tmp_path, enabled
):
    agent = make_agent(tmp_path, enabled=enabled)
    attempted = asyncio.Event()
    reads = 0

    async def remote_exec(command, timeout_sec):
        nonlocal reads
        reads += 1
        attempted.set()
        return SimpleNamespace(return_code=1)

    env = SimpleNamespace(capabilities=SimpleNamespace(mounted=True), exec=remote_exec)

    async def run_agent(environment, command):
        if enabled:
            await attempted.wait()
        else:
            await asyncio.sleep(0)

    monkeypatch.setattr(agent, "exec_as_agent", run_agent)
    await asyncio.wait_for(agent.run("task", env, None), timeout=2)
    await asyncio.sleep(0)

    staged = ContainerInput.read(agent.logs_dir / INPUT_FILENAME)
    assert bool(staged.preview_path) is enabled
    assert (reads > 0) is enabled
    assert not (tmp_path / "private" / "snapshot.json").exists()
