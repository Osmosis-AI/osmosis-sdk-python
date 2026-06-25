import json
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from osmosis_ai.rollout.agent_workflow import AgentWorkflow
from osmosis_ai.rollout.backend.harbor.backend import HarborBackend, PendingTrial
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
        rollout_ctx = get_rollout_context()
        if rollout_ctx:
            rollout_ctx.set_sample_source(
                _StaticSampleSource([{"role": "assistant", "content": "done"}]),
            )


class MetadataCapturingGrader(Grader):
    async def grade(self, ctx: GraderContext) -> Any:
        _GRADER_CAPTURE["metadata"] = ctx.metadata
        _GRADER_CAPTURE["label"] = ctx.label
        ctx.set_reward(1.0)


class ArtifactGrader(Grader):
    async def grade(self, ctx: GraderContext) -> Any:
        ctx.set_reward(1.0)
        ctx.set_artifacts({"judge": {"explanation": "ok"}})


class NonSerializableArtifactGrader(Grader):
    async def grade(self, ctx: GraderContext) -> Any:
        ctx.set_reward(1.0)
        ctx.set_artifacts({"bad": {1, 2, 3}})  # type: ignore[dict-item]


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
        reward = json.loads((verifier_dir / "reward.json").read_text())
        assert reward == {"reward": 1.0}

    def test_grader_writes_artifacts_file(self, tmp_path, monkeypatch):
        """A grader that calls set_artifacts produces grader_artifacts.json."""
        import osmosis_ai.rollout.backend.harbor.grader_runner as grader_runner

        verifier_dir = tmp_path / "verifier"
        monkeypatch.setattr(grader_runner, "VERIFIER_LOGS_DIR", verifier_dir)

        backend = _make_backend_for_config(grader=True)
        backend.grader_path = (
            f"{ArtifactGrader.__module__}:{ArtifactGrader.__qualname__}"
        )
        request = ExecutionRequest(
            id="r1",
            prompt=[{"role": "user", "content": "hi"}],
            label="test-label",
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

        artifacts = json.loads((verifier_dir / "grader_artifacts.json").read_text())
        assert artifacts == {"judge": {"explanation": "ok"}}

    def test_nonserializable_artifacts_never_block_rewards(self, tmp_path, monkeypatch):
        """A grader with a bad artifacts payload still persists rewards, and the
        artifacts file degrades to an _error marker instead of crashing."""
        import osmosis_ai.rollout.backend.harbor.grader_runner as grader_runner

        verifier_dir = tmp_path / "verifier"
        monkeypatch.setattr(grader_runner, "VERIFIER_LOGS_DIR", verifier_dir)

        backend = _make_backend_for_config(grader=True)
        backend.grader_path = (
            f"{NonSerializableArtifactGrader.__module__}:"
            f"{NonSerializableArtifactGrader.__qualname__}"
        )
        request = ExecutionRequest(
            id="r1",
            prompt=[{"role": "user", "content": "hi"}],
            label="test-label",
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

        reward = json.loads((verifier_dir / "reward.json").read_text())
        assert reward == {"reward": 1.0}
        artifacts = json.loads((verifier_dir / "grader_artifacts.json").read_text())
        assert artifacts["_error"]["code"] == "artifacts_not_serializable"

    def test_grader_without_artifacts_writes_no_file(self, tmp_path, monkeypatch):
        import osmosis_ai.rollout.backend.harbor.grader_runner as grader_runner

        verifier_dir = tmp_path / "verifier"
        monkeypatch.setattr(grader_runner, "VERIFIER_LOGS_DIR", verifier_dir)

        backend = _make_backend_for_config(grader=True)
        request = ExecutionRequest(
            id="r1",
            prompt=[{"role": "user", "content": "hi"}],
            label="test-label",
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

        assert not (verifier_dir / "grader_artifacts.json").exists()

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
        reward = json.loads((verifier_dir / "reward.json").read_text())
        assert reward == {"reward": 1.0}


class TestOnTrialEndArtifacts:
    """HarborBackend.on_trial_end reads grader_artifacts.json off the host
    trial dir and merges it into the grader result before cleanup."""

    def _make_backend(self, tmp_path) -> HarborBackend:
        backend = HarborBackend.__new__(HarborBackend)
        backend.pending = {}
        backend.cleanup_successful_trials = False
        backend.trials_dir = tmp_path
        return backend

    def _success_event(self) -> SimpleNamespace:
        return SimpleNamespace(
            config=SimpleNamespace(trial_name="trial-r1"),
            result=SimpleNamespace(
                agent_result=SimpleNamespace(
                    metadata={
                        "status": "success",
                        "sample": RolloutSample(messages=[]).model_dump(),
                    }
                ),
                verifier_result=SimpleNamespace(rewards={"sample-1": 1.0}),
                exception_info=None,
            ),
        )

    async def test_merges_artifacts_when_file_present(self, tmp_path):
        backend = self._make_backend(tmp_path)
        verifier = tmp_path / "trial-r1" / "verifier"
        verifier.mkdir(parents=True)
        (verifier / "grader_artifacts.json").write_text(
            json.dumps({"judge": {"explanation": "ok"}})
        )

        on_grader = AsyncMock()
        pending = PendingTrial(AsyncMock(), on_grader)
        pending.workflow_complete_called = True
        backend.pending["r1"] = pending

        await backend.on_trial_end(self._success_event())

        result = on_grader.call_args.args[0]
        assert result.status == RolloutStatus.SUCCESS
        assert result.artifacts == {"judge": {"explanation": "ok"}}

    async def test_artifacts_none_when_file_absent(self, tmp_path):
        backend = self._make_backend(tmp_path)

        on_grader = AsyncMock()
        pending = PendingTrial(AsyncMock(), on_grader)
        pending.workflow_complete_called = True
        backend.pending["r1"] = pending

        await backend.on_trial_end(self._success_event())

        result = on_grader.call_args.args[0]
        assert result.status == RolloutStatus.SUCCESS
        assert result.artifacts is None

    def test_read_grader_artifacts_tolerates_corrupt_file(self, tmp_path):
        backend = self._make_backend(tmp_path)
        verifier = tmp_path / "trial-r1" / "verifier"
        verifier.mkdir(parents=True)
        (verifier / "grader_artifacts.json").write_text("{not valid json")

        assert backend.read_grader_artifacts("r1") is None

    def test_read_grader_artifacts_tolerates_non_utf8(self, tmp_path):
        # Non-UTF-8 bytes raise UnicodeDecodeError (a ValueError), which must be
        # tolerated like any other corrupt artifacts file.
        backend = self._make_backend(tmp_path)
        verifier = tmp_path / "trial-r1" / "verifier"
        verifier.mkdir(parents=True)
        (verifier / "grader_artifacts.json").write_bytes(b"\xff\xfe\x00bad")

        assert backend.read_grader_artifacts("r1") is None
