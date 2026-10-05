"""Exercise real Harbor trials while replacing only sandbox process/file I/O."""

import asyncio
import json

import pytest
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.trial.config import EnvironmentConfig
from harbor.trial.hooks import TrialEvent
from harbor.trial.queue import TrialQueue

from osmosis_ai.rollout.backend.harbor.backend import HarborBackend
from osmosis_ai.rollout.types import ExecutionRequest, RolloutStatus

SECRET = "harbor-contract-test-secret"


class ContractEnvironment(BaseEnvironment):
    """Mounted sandbox boundary; OracleAgent and Verifier still run normally."""

    def __init__(self, *args, block_agent=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_agent = block_agent
        self.stop_calls = 0
        self.agent_running = asyncio.Event()

    @staticmethod
    def type():
        return "contract"

    @property
    def capabilities(self):
        return EnvironmentCapabilities(mounted=True)

    def _validate_definition(self):
        pass

    async def start(self, force_build):
        pass

    async def stop(self, delete):
        assert delete
        self.stop_calls += 1

    async def upload_file(self, source_path, target_path):
        pass

    async def upload_dir(self, source_dir, target_dir):
        pass

    async def download_file(self, source_path, target_path):
        pass

    async def download_dir(self, source_dir, target_dir):
        pass

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if "/solution/solve.sh" in command and not command.startswith("chmod "):
            (self.trial_paths.agent_dir / "oracle.txt").write_text(SECRET)
            self.agent_running.set()
            if self.block_agent:
                await asyncio.Future()
        if "/tests/test.sh" in command and not command.startswith("chmod "):
            self.trial_paths.reward_text_path.write_text("1.0")
        return ExecResult(stdout="", stderr="", return_code=0)


@pytest.fixture
def contract_backend(tmp_path, monkeypatch):
    task = tmp_path / "task"
    for directory in ("environment", "solution", "tests"):
        (task / directory).mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (task / "instruction.md").write_text("Run the reference solution.\n")
    (task / "solution" / "solve.sh").write_text("#!/bin/sh\nexit 0\n")
    (task / "tests" / "test.sh").write_text(
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
    )
    (task / "task.toml").write_text(f'[verifier.env]\nTEST_API_KEY = "{SECRET}"\n')
    monkeypatch.setenv("_OSMOSIS_ROLLOUT_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    queue = TrialQueue(n_concurrent=1)
    backend = HarborBackend(
        orchestrator=queue,
        tasks_dir=task,
        agent="oracle",
        trials_dir=tmp_path / "trials",
        environment_config=EnvironmentConfig(
            import_path=f"{__name__}:ContractEnvironment"
        ),
    )
    backend.rollouts_dir = tmp_path / "rollouts"
    events = []
    environments = []
    results = []
    original_init = ContractEnvironment.__init__

    def capture_environment(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        environments.append(self)

    monkeypatch.setattr(ContractEnvironment, "__init__", capture_environment)

    async def capture(event):
        events.append(event.event)
        if event.event == TrialEvent.START:
            assert backend.running == 1
            assert backend.pending["contract"].started
        if event.event == TrialEvent.VERIFICATION_START:
            assert backend.pending["contract"].grading
        if event.event == TrialEvent.END:
            # Harbor emits END before its final secret scrub. SDK must wait for
            # submit() to finish before copying these bytes to durable storage.
            log = backend.trials_dir / "trial-contract" / "agent" / "oracle.txt"
            assert log.read_text() == SECRET
            results.append(event.result)

    for event in TrialEvent:
        queue.add_hook(event, capture)
    yield backend, events, environments, results
    backend.archive_executor.shutdown(wait=True)


@pytest.mark.parametrize("cancel", [False, True], ids=["success", "cancel"])
async def test_real_trial_lifecycle_and_archive(contract_backend, cancel):
    backend, events, environments, results = contract_backend
    backend.environment_config.kwargs["block_agent"] = cancel
    execution = asyncio.create_task(
        backend.execute(ExecutionRequest(id="contract", prompt=[], grade=True))
    )
    try:
        if cancel:
            # Wait for the real agent to reach the sandbox boundary; never guess
            # its state with a fixed sleep or manually invoke lifecycle hooks.
            async with asyncio.timeout(5):
                while not environments:
                    await asyncio.sleep(0)
                await environments[0].agent_running.wait()
            assert backend.cancel_rollouts(ids=["contract"]) == {
                "contract": "cancelled_running"
            }
        outcome = await asyncio.wait_for(execution, timeout=5)
    finally:
        if not execution.done():
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)

    assert environments[0].stop_calls == 1
    assert not backend.pending
    assert backend.running == 0
    assert not (backend.rollouts_dir / "contract").exists()
    assert not (backend.trials_dir / "trial-contract").exists()
    assert len(results) == 1
    assert events[:3] == [
        TrialEvent.START,
        TrialEvent.ENVIRONMENT_START,
        TrialEvent.AGENT_START,
    ]
    assert events[-1] == TrialEvent.END
    if cancel:
        assert outcome.result.status == RolloutStatus.CANCELLED
        assert TrialEvent.CANCEL in events
        assert TrialEvent.VERIFICATION_START not in events
        assert results[0].exception_info.exception_type == "CancelledError"
    else:
        assert events[3:] == [
            TrialEvent.AGENT_END,
            TrialEvent.VERIFICATION_START,
            TrialEvent.END,
        ]
        assert outcome.workflow.status == RolloutStatus.SUCCESS
        assert outcome.grader.status == RolloutStatus.SUCCESS
        assert outcome.result.sample.reward == 1.0
        assert results[0].verifier_result.rewards == {"reward": 1.0}
        assert results[0].exception_info is None

    archive = backend.artifact_root / "contract"
    manifest = json.loads((archive / "harbor" / "manifest.json").read_text())
    assert manifest["files"]
    archived_logs = list(archive.rglob("oracle.txt"))
    assert archived_logs
    assert all(path.read_text() == "[REDACTED]" for path in archived_logs)
    assert all(
        SECRET.encode() not in path.read_bytes()
        for path in archive.rglob("*")
        if path.is_file()
    )
