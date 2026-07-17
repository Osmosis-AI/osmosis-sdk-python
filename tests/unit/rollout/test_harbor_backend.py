import json
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from osmosis_ai.rollout.agent_workflow import AgentWorkflow
from osmosis_ai.rollout.backend.harbor.backend import (
    HarborBackend,
    PendingTrial,
    parse_sample,
)
from osmosis_ai.rollout.context import (
    AgentWorkflowContext,
    GraderContext,
    SampleSource,
)
from osmosis_ai.rollout.grader import Grader
from osmosis_ai.rollout.types import (
    ExecutionRequest,
    RolloutErrorCategory,
    RolloutSample,
    RolloutStatus,
)
from osmosis_ai.rollout.utils.file_artifacts import HARBOR_ARTIFACTS_DIR

# Module-level captures so importlib can resolve the workflow/grader by
# "<module>:<qualname>" and round-trip them through the runners.
_AGENT_CAPTURE: dict[str, Any] = {}
_GRADER_CAPTURE: dict[str, Any] = {}


class _StaticSampleSource(SampleSource):
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = messages

    async def get_sample(self) -> RolloutSample:
        return RolloutSample(messages=self.messages)


class MetadataCapturingWorkflow(AgentWorkflow):
    async def run(self, ctx: AgentWorkflowContext) -> Any:
        from osmosis_ai.rollout.context import get_rollout_context

        _AGENT_CAPTURE["metadata"] = ctx.metadata
        _AGENT_CAPTURE["artifacts_dir"] = ctx.artifacts_dir
        rollout_ctx = get_rollout_context()
        if rollout_ctx:
            rollout_ctx.set_sample_source(
                _StaticSampleSource([{"role": "assistant", "content": "done"}]),
            )


class MetadataCapturingGrader(Grader):
    async def grade(self, ctx: GraderContext) -> Any:
        _GRADER_CAPTURE["metadata"] = ctx.metadata
        _GRADER_CAPTURE["label"] = ctx.label
        _GRADER_CAPTURE["artifacts_dir"] = ctx.artifacts_dir
        ctx.set_reward(1.0)


def _make_backend_for_config(*, grader: bool = False) -> HarborBackend:
    """Build a HarborBackend skeleton sufficient for build_rollout_config."""
    backend = HarborBackend.__new__(HarborBackend)
    backend.workflow_path = (
        f"{MetadataCapturingWorkflow.__module__}:"
        f"{MetadataCapturingWorkflow.__qualname__}"
    )
    backend.workflow_config_path = None
    backend.grader_path = (
        f"{MetadataCapturingGrader.__module__}:{MetadataCapturingGrader.__qualname__}"
        if grader
        else None
    )
    backend.grader_config_path = None
    return backend


class TestHarborBackend:
    def test_sample_round_trip_preserves_native_and_trajectory_messages(self):
        sample = RolloutSample(
            messages=[{"type": "function_call", "name": "f"}],
            trajectory_messages=[{"role": "assistant", "content": "done"}],
        )

        parsed = parse_sample(json.loads(json.dumps(sample.model_dump(), default=str)))

        assert parsed is not None
        assert parsed.messages == sample.messages
        assert parsed.trajectory_messages == sample.trajectory_messages

    async def test_empty_verifier_rewards_logs_and_returns_validation_failure(
        self, caplog, tmp_path
    ):
        backend = HarborBackend.__new__(HarborBackend)
        backend.pending = {}
        backend.cleanup_successful_trials = False
        backend.trials_dir = tmp_path

        on_workflow = AsyncMock()
        on_grader = AsyncMock()
        pending = PendingTrial(on_workflow, on_grader)
        pending.workflow_complete_called = True
        backend.pending["r1"] = pending

        event = SimpleNamespace(
            config=SimpleNamespace(trial_name="trial-r1"),
            result=SimpleNamespace(
                agent_result=SimpleNamespace(
                    metadata={
                        "status": "success",
                        "sample": RolloutSample(messages=[]).model_dump(),
                    }
                ),
                verifier_result=SimpleNamespace(rewards={}),
                exception_info=None,
            ),
        )

        with caplog.at_level(
            logging.WARNING, logger="osmosis_ai.rollout.backend.harbor.backend"
        ):
            await backend.on_trial_end(event)

        on_grader.assert_awaited_once()
        result = on_grader.call_args.args[0]
        assert result.status == RolloutStatus.FAILURE
        assert result.err_category == RolloutErrorCategory.VALIDATION_ERROR
        assert "Harbor verifier returned empty rewards for rollout r1" in caplog.text

    def _make_trial_end_backend(self, tmp_path, *, cleanup: bool) -> HarborBackend:
        backend = HarborBackend.__new__(HarborBackend)
        backend.pending = {}
        backend.cleanup_successful_trials = cleanup
        backend.trials_dir = tmp_path / "trials"
        backend.rollouts_dir = tmp_path / "rollouts"
        backend.artifact_root = tmp_path / "collected"
        return backend

    def _seed_trial_artifacts(self, backend: HarborBackend) -> None:
        trial_artifacts = backend.trials_dir / "trial-r1" / "artifacts"
        convention = trial_artifacts / "logs" / "artifacts"
        convention.mkdir(parents=True)
        (convention / "output.txt").write_text("ok")
        (trial_artifacts / "manifest.json").write_text("[]")

    @staticmethod
    def _success_event() -> SimpleNamespace:
        return SimpleNamespace(
            config=SimpleNamespace(trial_name="trial-r1"),
            result=SimpleNamespace(
                agent_result=SimpleNamespace(
                    metadata={
                        "status": "success",
                        "sample": RolloutSample(messages=[]).model_dump(),
                    }
                ),
                verifier_result=SimpleNamespace(rewards={"reward": 1.0}),
                exception_info=None,
            ),
        )

    async def test_on_trial_end_relocates_artifacts_before_cleanup(self, tmp_path):
        backend = self._make_trial_end_backend(tmp_path, cleanup=True)
        self._seed_trial_artifacts(backend)

        on_grader = AsyncMock()
        pending = PendingTrial(AsyncMock(), on_grader)
        pending.workflow_complete_called = True
        backend.pending["r1"] = pending

        await backend.on_trial_end(self._success_event())

        result = on_grader.call_args.args[0]
        assert result.status == RolloutStatus.SUCCESS
        relocated = backend.artifact_root / "r1" / "artifacts"
        assert (relocated / "logs" / "artifacts" / "output.txt").read_text() == "ok"
        assert (relocated / "manifest.json").exists()
        assert not (backend.trials_dir / "trial-r1").exists()

    async def test_on_trial_end_copies_artifacts_when_trial_kept(self, tmp_path):
        backend = self._make_trial_end_backend(tmp_path, cleanup=False)
        self._seed_trial_artifacts(backend)

        pending = PendingTrial(AsyncMock(), AsyncMock())
        pending.workflow_complete_called = True
        backend.pending["r1"] = pending

        await backend.on_trial_end(self._success_event())

        relocated = backend.artifact_root / "r1" / "artifacts"
        assert (relocated / "logs" / "artifacts" / "output.txt").read_text() == "ok"
        source = backend.trials_dir / "trial-r1" / "artifacts"
        assert (source / "logs" / "artifacts" / "output.txt").exists()

    async def test_on_trial_end_keeps_trial_when_relocation_fails(
        self, tmp_path, monkeypatch, caplog
    ):
        backend = self._make_trial_end_backend(tmp_path, cleanup=True)
        self._seed_trial_artifacts(backend)

        def _boom(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(
            "osmosis_ai.rollout.backend.harbor.backend.shutil.move", _boom
        )

        pending = PendingTrial(AsyncMock(), AsyncMock())
        pending.workflow_complete_called = True
        backend.pending["r1"] = pending

        with caplog.at_level(
            logging.WARNING, logger="osmosis_ai.rollout.backend.harbor.backend"
        ):
            await backend.on_trial_end(self._success_event())

        # Relocation failed: the source artifacts must survive cleanup.
        source = backend.trials_dir / "trial-r1" / "artifacts"
        assert (source / "logs" / "artifacts" / "output.txt").read_text() == "ok"
        assert "Failed to relocate trial artifacts for rollout r1" in caplog.text


class TestBuildRolloutConfigMetadata:
    def test_metadata_written_when_present(self):
        backend = _make_backend_for_config()
        request = ExecutionRequest(
            id="r1",
            prompt=[{"role": "user", "content": "hi"}],
            metadata={"tools": ["search"], "difficulty": 3},
        )
        config = backend.build_rollout_config(request)
        assert config["metadata"] == {"tools": ["search"], "difficulty": 3}

    def test_metadata_omitted_when_none(self):
        backend = _make_backend_for_config()
        request = ExecutionRequest(
            id="r1",
            prompt=[{"role": "user", "content": "hi"}],
            metadata=None,
        )
        config = backend.build_rollout_config(request)
        assert "metadata" not in config


class TestAgentRunnerRoundTrip:
    async def test_metadata_surfaces_on_ctx(self, tmp_path, monkeypatch):
        import osmosis_ai.rollout.backend.harbor.agent_runner as agent_runner

        _AGENT_CAPTURE.clear()
        monkeypatch.setattr(agent_runner, "AGENT_LOGS_DIR", tmp_path)

        backend = _make_backend_for_config()
        metadata = {"tools": ["search"], "difficulty": 3}
        request = ExecutionRequest(
            id="r1",
            prompt=[{"role": "user", "content": "hi"}],
            metadata=metadata,
        )

        # Round-trip the config through JSON, as the on-disk file would.
        raw = json.dumps(backend.build_rollout_config(request), default=str)
        config = json.loads(raw)
        prompt = json.loads(json.dumps(request.prompt, default=str))

        meta = await agent_runner.run_workflow(config, prompt)

        assert meta["status"] == "success"
        assert _AGENT_CAPTURE["metadata"] == metadata
        assert _AGENT_CAPTURE["artifacts_dir"] == HARBOR_ARTIFACTS_DIR


class TestGraderRunnerRoundTrip:
    def _write_sample(self, path):
        sample = RolloutSample(messages=[])
        path.write_text(json.dumps(sample.model_dump(), default=str))

    def test_grader_ctx_receives_metadata(self, tmp_path, monkeypatch):
        import osmosis_ai.rollout.backend.harbor.grader_runner as grader_runner

        _GRADER_CAPTURE.clear()
        verifier_dir = tmp_path / "verifier"
        monkeypatch.setattr(grader_runner, "VERIFIER_LOGS_DIR", verifier_dir)

        backend = _make_backend_for_config(grader=True)
        metadata = {"tools": ["search"]}
        request = ExecutionRequest(
            id="r1",
            prompt=[{"role": "user", "content": "hi"}],
            label="test-label",
            metadata=metadata,
        )
        config_path = tmp_path / "rollout_config.json"
        config_path.write_text(
            json.dumps(backend.build_rollout_config(request), default=str)
        )
        sample_path = tmp_path / "sample.json"
        self._write_sample(sample_path)

        monkeypatch.setattr(
            grader_runner,
            "parse_args",
            lambda: SimpleNamespace(config=config_path, sample=sample_path),
        )

        grader_runner.main()

        assert _GRADER_CAPTURE["metadata"] == metadata
        assert _GRADER_CAPTURE["artifacts_dir"] == HARBOR_ARTIFACTS_DIR
        rewards = json.loads((verifier_dir / "reward.json").read_text())
        assert rewards == {"reward": 1.0}

    def test_metadata_only_config_still_grades(self, tmp_path, monkeypatch):
        """A config with metadata but no label still triggers grading."""
        import osmosis_ai.rollout.backend.harbor.grader_runner as grader_runner

        _GRADER_CAPTURE.clear()
        verifier_dir = tmp_path / "verifier"
        monkeypatch.setattr(grader_runner, "VERIFIER_LOGS_DIR", verifier_dir)

        backend = _make_backend_for_config(grader=True)
        request = ExecutionRequest(
            id="r1",
            prompt=[{"role": "user", "content": "hi"}],
            label=None,
            metadata={"tools": ["search"]},
        )
        config_path = tmp_path / "rollout_config.json"
        config_path.write_text(
            json.dumps(backend.build_rollout_config(request), default=str)
        )
        sample_path = tmp_path / "sample.json"
        self._write_sample(sample_path)

        monkeypatch.setattr(
            grader_runner,
            "parse_args",
            lambda: SimpleNamespace(config=config_path, sample=sample_path),
        )

        grader_runner.main()

        assert _GRADER_CAPTURE["label"] is None
        assert _GRADER_CAPTURE["metadata"] == {"tools": ["search"]}
        rewards = json.loads((verifier_dir / "reward.json").read_text())
        assert rewards == {"reward": 1.0}
