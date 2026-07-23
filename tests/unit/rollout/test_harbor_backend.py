import json
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from osmosis_ai.rollout.agent_workflow import AgentWorkflow
from osmosis_ai.rollout.backend.harbor.backend import (
    HarborBackend,
    PendingTrial,
    parse_samples,
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
from osmosis_ai.rollout.utils.file_artifacts import (
    GRADER_ARTIFACTS_SNAPSHOT_DIRNAME,
    HARBOR_ARTIFACTS_DIR,
)
from osmosis_ai.rollout.utils.file_artifacts import (
    copy_artifact_tree as real_copy_artifact_tree,
)

# Module-level captures so importlib can resolve the workflow/grader by
# "<module>:<qualname>" and round-trip them through the runners.
_AGENT_CAPTURE: dict[str, Any] = {}
_GRADER_CAPTURE: dict[str, Any] = {}


class _StaticSampleSource(SampleSource):
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = messages

    async def get_sample(self, name: str) -> RolloutSample:
        return RolloutSample(id=name, messages=self.messages)


class MetadataCapturingWorkflow(AgentWorkflow):
    async def run(self, ctx: AgentWorkflowContext) -> Any:
        from osmosis_ai.rollout.context import get_rollout_context

        _AGENT_CAPTURE["metadata"] = ctx.metadata
        _AGENT_CAPTURE["artifacts_dir"] = ctx.artifacts_dir
        rollout_ctx = get_rollout_context()
        if rollout_ctx:
            rollout_ctx.register_sample_source(
                "sample-1",
                _StaticSampleSource([{"role": "assistant", "content": "done"}]),
            )


class MetadataCapturingGrader(Grader):
    async def grade(self, ctx: GraderContext) -> Any:
        _GRADER_CAPTURE["metadata"] = ctx.metadata
        _GRADER_CAPTURE["label"] = ctx.label
        _GRADER_CAPTURE["artifacts_dir"] = ctx.artifacts_dir
        if ctx.artifacts_dir:
            ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)
            (ctx.artifacts_dir / "grader.txt").write_text("grader")
        for sample_id in ctx.get_samples():
            ctx.set_sample_reward(sample_id, 1.0)


class ArtifactWritingFailingGrader(Grader):
    async def grade(self, ctx: GraderContext) -> Any:
        assert ctx.artifacts_dir is not None
        ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (ctx.artifacts_dir / "partial.txt").write_text("partial")
        raise RuntimeError("grader exploded")


class ArtifactWritingFailingConstructorGrader(Grader):
    def __init__(self, _config=None):
        import osmosis_ai.rollout.backend.harbor.grader_runner as grader_runner

        grader_runner.HARBOR_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        (grader_runner.HARBOR_ARTIFACTS_DIR / "constructor.txt").write_text(
            "constructor"
        )
        raise RuntimeError("grader constructor exploded")

    async def grade(self, _ctx: GraderContext) -> Any:
        raise AssertionError("grade should not run")


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
            id="s1",
            messages=[{"type": "function_call", "name": "f"}],
            trajectory_messages=[{"role": "assistant", "content": "done"}],
        )

        parsed = parse_samples(
            json.loads(json.dumps({"s1": sample.model_dump()}, default=str))
        )["s1"]

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
                        "samples": {
                            "sample-1": RolloutSample(
                                id="sample-1", messages=[]
                            ).model_dump()
                        },
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
                        "samples": {
                            "s1": RolloutSample(id="s1", messages=[]).model_dump()
                        },
                    }
                ),
                verifier_result=SimpleNamespace(rewards={"s1": 1.0}),
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

    async def test_on_trial_end_skips_symlinks_when_relocating(self, tmp_path):
        backend = self._make_trial_end_backend(tmp_path, cleanup=False)
        self._seed_trial_artifacts(backend)
        source = backend.trials_dir / "trial-r1" / "artifacts"
        host_file = tmp_path / "host-secret.txt"
        host_file.write_text("host secret")
        (source / "logs" / "artifacts" / "user-link.txt").symlink_to(host_file)

        pending = PendingTrial(AsyncMock(), AsyncMock())
        pending.workflow_complete_called = True
        backend.pending["r1"] = pending

        await backend.on_trial_end(self._success_event())

        relocated = backend.artifact_root / "r1" / "artifacts"
        assert not (relocated / "logs" / "artifacts" / "user-link.txt").exists()
        assert host_file.read_text() == "host secret"

    async def test_on_trial_end_merges_staged_grader_artifacts(self, tmp_path):
        backend = self._make_trial_end_backend(tmp_path, cleanup=True)
        self._seed_trial_artifacts(backend)
        staged = (
            backend.trials_dir
            / "trial-r1"
            / "verifier"
            / GRADER_ARTIFACTS_SNAPSHOT_DIRNAME
        )
        staged.mkdir(parents=True)
        (staged / "output.txt").write_text("updated by grader")
        (staged / "grader.json").write_text('{"reward": 1}')

        pending = PendingTrial(AsyncMock(), AsyncMock())
        pending.workflow_complete_called = True
        backend.pending["r1"] = pending

        await backend.on_trial_end(self._success_event())

        relocated = backend.artifact_root / "r1" / "artifacts" / "logs" / "artifacts"
        assert (relocated / "output.txt").read_text() == "updated by grader"
        assert (relocated / "grader.json").read_text() == '{"reward": 1}'

    async def test_on_trial_end_rejects_symlinked_grader_merge_parent(
        self, tmp_path, caplog
    ):
        backend = self._make_trial_end_backend(tmp_path, cleanup=False)
        trial_dir = backend.trials_dir / "trial-r1"
        trial_artifacts = trial_dir / "artifacts"
        outside = tmp_path / "outside"
        trial_artifacts.mkdir(parents=True)
        outside.mkdir()
        (trial_artifacts / "logs").symlink_to(outside, target_is_directory=True)
        staged = trial_dir / "verifier" / GRADER_ARTIFACTS_SNAPSHOT_DIRNAME
        staged.mkdir(parents=True)
        (staged / "grader.json").write_text("{}")

        pending = PendingTrial(AsyncMock(), AsyncMock())
        pending.workflow_complete_called = True
        backend.pending["r1"] = pending

        with caplog.at_level(
            logging.WARNING, logger="osmosis_ai.rollout.backend.harbor.backend"
        ):
            await backend.on_trial_end(self._success_event())

        assert not (outside / "artifacts" / "grader.json").exists()
        assert "Failed to merge grader artifacts for rollout r1" in caplog.text

    async def test_on_trial_end_keeps_result_when_grader_merge_fails(
        self, tmp_path, monkeypatch, caplog
    ):
        backend = self._make_trial_end_backend(tmp_path, cleanup=True)
        self._seed_trial_artifacts(backend)
        staged = (
            backend.trials_dir
            / "trial-r1"
            / "verifier"
            / GRADER_ARTIFACTS_SNAPSHOT_DIRNAME
        )
        staged.mkdir(parents=True)
        (staged / "grader.json").write_text("{}")

        def _boom_for_staged_snapshot(source, destination, **kwargs):
            if source == staged:
                raise RuntimeError("bad artifact tree")
            return real_copy_artifact_tree(source, destination, **kwargs)

        monkeypatch.setattr(
            "osmosis_ai.rollout.backend.harbor.backend.copy_artifact_tree",
            _boom_for_staged_snapshot,
        )
        on_grader = AsyncMock()
        pending = PendingTrial(AsyncMock(), on_grader)
        pending.workflow_complete_called = True
        backend.pending["r1"] = pending

        with caplog.at_level(
            logging.WARNING, logger="osmosis_ai.rollout.backend.harbor.backend"
        ):
            await backend.on_trial_end(self._success_event())

        assert on_grader.call_args.args[0].status == RolloutStatus.SUCCESS
        relocated = backend.artifact_root / "r1" / "artifacts"
        assert (relocated / "logs" / "artifacts" / "output.txt").read_text() == "ok"
        assert "Failed to merge grader artifacts for rollout r1" in caplog.text

    async def test_on_trial_end_keeps_trial_when_relocation_fails(
        self, tmp_path, monkeypatch, caplog
    ):
        backend = self._make_trial_end_backend(tmp_path, cleanup=True)
        self._seed_trial_artifacts(backend)

        def _boom(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(
            "osmosis_ai.rollout.backend.harbor.backend.copy_artifact_tree", _boom
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
    def _write_samples(self, path):
        sample = RolloutSample(id="sample-1", messages=[])
        path.write_text(json.dumps({"sample-1": sample.model_dump()}, default=str))

    def test_grader_ctx_receives_metadata(self, tmp_path, monkeypatch):
        import osmosis_ai.rollout.backend.harbor.grader_runner as grader_runner

        _GRADER_CAPTURE.clear()
        verifier_dir = tmp_path / "verifier"
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        (artifacts_dir / "workflow.txt").write_text("workflow")
        monkeypatch.setattr(grader_runner, "VERIFIER_LOGS_DIR", verifier_dir)
        monkeypatch.setattr(grader_runner, "HARBOR_ARTIFACTS_DIR", artifacts_dir)

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
        samples_path = tmp_path / "samples.json"
        self._write_samples(samples_path)

        monkeypatch.setattr(
            grader_runner,
            "parse_args",
            lambda: SimpleNamespace(config=config_path, samples=samples_path),
        )

        grader_runner.main()

        assert _GRADER_CAPTURE["metadata"] == metadata
        assert _GRADER_CAPTURE["artifacts_dir"] == artifacts_dir
        rewards = json.loads((verifier_dir / "reward.json").read_text())
        assert rewards == {"sample-1": 1.0}
        snapshot = verifier_dir / GRADER_ARTIFACTS_SNAPSHOT_DIRNAME
        assert not (snapshot / "workflow.txt").exists()
        assert (snapshot / "grader.txt").read_text() == "grader"

    def test_metadata_only_config_still_grades(self, tmp_path, monkeypatch):
        """A config with metadata but no label still triggers grading."""
        import osmosis_ai.rollout.backend.harbor.grader_runner as grader_runner

        _GRADER_CAPTURE.clear()
        verifier_dir = tmp_path / "verifier"
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        monkeypatch.setattr(grader_runner, "VERIFIER_LOGS_DIR", verifier_dir)
        monkeypatch.setattr(grader_runner, "HARBOR_ARTIFACTS_DIR", artifacts_dir)

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
        samples_path = tmp_path / "samples.json"
        self._write_samples(samples_path)

        monkeypatch.setattr(
            grader_runner,
            "parse_args",
            lambda: SimpleNamespace(config=config_path, samples=samples_path),
        )

        grader_runner.main()

        assert _GRADER_CAPTURE["label"] is None
        assert _GRADER_CAPTURE["metadata"] == {"tools": ["search"]}
        rewards = json.loads((verifier_dir / "reward.json").read_text())
        assert rewards == {"sample-1": 1.0}

    def test_grader_failure_still_stages_artifacts(self, tmp_path, monkeypatch):
        """The finally-block snapshot preserves files written before the crash."""
        import osmosis_ai.rollout.backend.harbor.grader_runner as grader_runner

        verifier_dir = tmp_path / "verifier"
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        monkeypatch.setattr(grader_runner, "VERIFIER_LOGS_DIR", verifier_dir)
        monkeypatch.setattr(grader_runner, "HARBOR_ARTIFACTS_DIR", artifacts_dir)

        backend = _make_backend_for_config(grader=True)
        backend.grader_path = (
            f"{ArtifactWritingFailingGrader.__module__}:"
            f"{ArtifactWritingFailingGrader.__qualname__}"
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
        samples_path = tmp_path / "samples.json"
        self._write_samples(samples_path)

        monkeypatch.setattr(
            grader_runner,
            "parse_args",
            lambda: SimpleNamespace(config=config_path, samples=samples_path),
        )

        with pytest.raises(RuntimeError, match="grader exploded"):
            grader_runner.main()

        snapshot = verifier_dir / GRADER_ARTIFACTS_SNAPSHOT_DIRNAME
        assert (snapshot / "partial.txt").read_text() == "partial"
        assert not (verifier_dir / "reward.json").exists()

    def test_grader_constructor_failure_still_stages_artifacts(
        self, tmp_path, monkeypatch
    ):
        import osmosis_ai.rollout.backend.harbor.grader_runner as grader_runner

        verifier_dir = tmp_path / "verifier"
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        monkeypatch.setattr(grader_runner, "VERIFIER_LOGS_DIR", verifier_dir)
        monkeypatch.setattr(grader_runner, "HARBOR_ARTIFACTS_DIR", artifacts_dir)

        backend = _make_backend_for_config(grader=True)
        backend.grader_path = (
            f"{ArtifactWritingFailingConstructorGrader.__module__}:"
            f"{ArtifactWritingFailingConstructorGrader.__qualname__}"
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
        samples_path = tmp_path / "samples.json"
        self._write_samples(samples_path)

        monkeypatch.setattr(
            grader_runner,
            "parse_args",
            lambda: SimpleNamespace(config=config_path, samples=samples_path),
        )

        with pytest.raises(RuntimeError, match="grader constructor exploded"):
            grader_runner.main()

        snapshot = verifier_dir / GRADER_ARTIFACTS_SNAPSHOT_DIRNAME
        assert (snapshot / "constructor.txt").read_text() == "constructor"
        assert not (verifier_dir / "reward.json").exists()


class TestStageGraderArtifacts:
    def test_drops_snapshot_when_grader_wrote_nothing(self, tmp_path, monkeypatch):
        import osmosis_ai.rollout.backend.harbor.grader_runner as grader_runner

        verifier_dir = tmp_path / "verifier"
        artifacts_dir = tmp_path / "artifacts"
        (artifacts_dir / "nested").mkdir(parents=True)
        (artifacts_dir / "nested" / "workflow.txt").write_text("workflow")
        monkeypatch.setattr(grader_runner, "VERIFIER_LOGS_DIR", verifier_dir)
        monkeypatch.setattr(grader_runner, "HARBOR_ARTIFACTS_DIR", artifacts_dir)

        baseline = grader_runner.capture_artifact_baseline()
        grader_runner.stage_grader_artifacts(baseline)

        assert not (verifier_dir / GRADER_ARTIFACTS_SNAPSHOT_DIRNAME).exists()

    def test_replaces_stale_snapshot_from_previous_run(self, tmp_path, monkeypatch):
        import osmosis_ai.rollout.backend.harbor.grader_runner as grader_runner

        verifier_dir = tmp_path / "verifier"
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        (artifacts_dir / "grader.txt").write_text("fresh")
        stale = verifier_dir / GRADER_ARTIFACTS_SNAPSHOT_DIRNAME
        stale.mkdir(parents=True)
        (stale / "stale.txt").write_text("stale")
        monkeypatch.setattr(grader_runner, "VERIFIER_LOGS_DIR", verifier_dir)
        monkeypatch.setattr(grader_runner, "HARBOR_ARTIFACTS_DIR", artifacts_dir)

        grader_runner.stage_grader_artifacts({})

        snapshot = verifier_dir / GRADER_ARTIFACTS_SNAPSHOT_DIRNAME
        assert (snapshot / "grader.txt").read_text() == "fresh"
        assert not (snapshot / "stale.txt").exists()
