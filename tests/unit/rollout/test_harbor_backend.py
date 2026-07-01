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

    async def get_sample(self, name: str) -> RolloutSample:
        return RolloutSample(id=name, messages=self.messages)


class MetadataCapturingWorkflow(AgentWorkflow):
    async def run(self, ctx: AgentWorkflowContext) -> Any:
        from osmosis_ai.rollout.context import get_rollout_context

        _AGENT_CAPTURE["metadata"] = ctx.metadata
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
        for sample_id in ctx.get_samples():
            ctx.set_sample_reward(sample_id, 1.0)


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
    def _write_samples(self, path):
        sample = RolloutSample(id="sample-1", messages=[])
        path.write_text(json.dumps({"sample-1": sample.model_dump()}, default=str))

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
        samples_path = tmp_path / "samples.json"
        self._write_samples(samples_path)

        monkeypatch.setattr(
            grader_runner,
            "parse_args",
            lambda: SimpleNamespace(config=config_path, samples=samples_path),
        )

        grader_runner.main()

        assert _GRADER_CAPTURE["metadata"] == metadata
        rewards = json.loads((verifier_dir / "reward.json").read_text())
        assert rewards == {"sample-1": 1.0}

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
