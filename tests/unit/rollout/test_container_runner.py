"""In-container runner: the sample must cross the container boundary whole."""

import json
from typing import Any

import pytest

import osmosis_ai.rollout.container.runner as runner
from osmosis_ai.rollout.container.files import (
    INPUT_FILENAME,
    RESULT_FILENAME,
    ContainerInput,
    ContainerResult,
)
from osmosis_ai.rollout.container.trajectories import messages_from_trajectory
from osmosis_ai.rollout.context import SampleSource, get_rollout_context
from osmosis_ai.rollout.types import RolloutSample, RolloutStatus
from osmosis_ai.rollout.types.output import AgentWorkflowOutput


class RecordedSource(SampleSource):
    def __init__(self, sample: RolloutSample):
        self.sample = sample

    async def get_sample(self) -> RolloutSample:
        return self.sample


def workflow_returning(value: Any, sample: RolloutSample | None = None):
    class Workflow:
        def __init__(self, config: Any) -> None:
            pass

        async def run(self, ctx: Any) -> Any:
            if sample is not None:
                context = get_rollout_context()
                assert context is not None
                context.set_sample_source(RecordedSource(sample))
            return value

    return Workflow


def stage_input(tmp_path) -> None:
    ContainerInput(
        rollout_id="r1", prompt=[{"role": "user", "content": "solve"}]
    ).write(tmp_path / INPUT_FILENAME)


class TestFallbackSampleBoundary:
    async def test_fallback_sample_round_trips_field_for_field(
        self, monkeypatch, tmp_path
    ):
        """run()->None: every sample field must survive the file boundary."""
        monkeypatch.setattr(runner, "AGENT_LOGS_DIR", tmp_path)
        stage_input(tmp_path)
        sample = RolloutSample(
            messages=[{"role": "assistant", "content": "done"}],
            trajectory_messages=None,
            label="workflow-label",
            reward=0.25,
            remove_sample=True,
            metrics={"turns": 2, "verdict": "pass"},
            extra_fields={"run": "a"},
        )

        result = await runner.run_agent(workflow_returning(None, sample), None)
        result.write(tmp_path / RESULT_FILENAME)
        loaded = ContainerResult.read(tmp_path / RESULT_FILENAME)

        assert loaded.status == RolloutStatus.SUCCESS
        assert loaded.sample is not None
        assert loaded.sample.trajectory_messages is None
        assert loaded.sample.label == "workflow-label"
        assert loaded.sample.reward == 0.25
        assert loaded.sample.remove_sample is True
        assert loaded.sample.metrics == {"turns": 2, "verdict": "pass"}
        assert loaded.sample.extra_fields == {"run": "a"}
        # The projection stays for older readers, numeric metrics only.
        assert loaded.output is not None
        assert loaded.output.metrics == {"turns": 2.0}

    async def test_fallback_sample_default_trajectory_survives(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(runner, "AGENT_LOGS_DIR", tmp_path)
        stage_input(tmp_path)
        messages = [{"role": "assistant", "content": "hi"}]
        sample = RolloutSample(messages=messages)

        result = await runner.run_agent(workflow_returning(None, sample), None)
        result.write(tmp_path / RESULT_FILENAME)
        loaded = ContainerResult.read(tmp_path / RESULT_FILENAME)

        assert loaded.sample is not None
        assert list(loaded.sample.trajectory_messages or []) == messages

    async def test_explicit_output_carries_no_sample(self, monkeypatch, tmp_path):
        """A workflow that returned its own output chose its own contract."""
        monkeypatch.setattr(runner, "AGENT_LOGS_DIR", tmp_path)
        stage_input(tmp_path)

        result = await runner.run_agent(
            workflow_returning([{"role": "assistant", "content": "y"}]), None
        )

        assert result.sample is None
        assert result.output is not None
        assert result.output.messages == [{"role": "assistant", "content": "y"}]

    async def test_non_finite_ambient_metrics_are_dropped_from_projection(
        self, monkeypatch, tmp_path
    ):
        """Bad ambient telemetry must not fail the rollout or lose the sample."""
        monkeypatch.setattr(runner, "AGENT_LOGS_DIR", tmp_path)
        stage_input(tmp_path)
        sample = RolloutSample(
            messages=[{"role": "assistant", "content": "done"}],
            metrics={"turns": 2, "loss": float("nan")},
        )

        result = await runner.run_agent(workflow_returning(None, sample), None)

        assert result.status == RolloutStatus.SUCCESS
        assert result.sample is not None
        assert result.sample.metrics["turns"] == 2
        assert result.output is not None
        assert result.output.metrics == {"turns": 2.0}


class TestUnsupportedOutputs:
    async def test_mutated_non_finite_metrics_rejected(self, monkeypatch, tmp_path):
        monkeypatch.setattr(runner, "AGENT_LOGS_DIR", tmp_path)
        stage_input(tmp_path)
        output = AgentWorkflowOutput(metrics={"score": 1.0})
        output.metrics["score"] = float("nan")

        with pytest.raises(ValueError, match="finite"):
            await runner.run_agent(workflow_returning(output), None)

    async def test_explicit_output_accepted(self, monkeypatch, tmp_path):
        monkeypatch.setattr(runner, "AGENT_LOGS_DIR", tmp_path)
        stage_input(tmp_path)
        output = AgentWorkflowOutput(messages=[{"role": "assistant", "content": "y"}])

        result = await runner.run_agent(workflow_returning(output), None)
        assert result.status == RolloutStatus.SUCCESS


class TestGraderSampleSource:
    def test_load_sample_prefers_full_sample(self, monkeypatch, tmp_path):
        monkeypatch.setattr(runner, "AGENT_LOGS_DIR", tmp_path)
        sample = RolloutSample(
            messages=[{"role": "assistant", "content": "full"}],
            label="from-sample",
            metrics={"verdict": "pass"},
        )
        ContainerResult(
            status=RolloutStatus.SUCCESS,
            sample=sample,
            output=AgentWorkflowOutput(
                messages=[{"role": "assistant", "content": "projected"}]
            ),
        ).write(tmp_path / RESULT_FILENAME)

        loaded = runner.load_sample()

        assert loaded is not None
        assert loaded.label == "from-sample"
        assert loaded.messages[0]["content"] == "full"

    def test_load_sample_falls_back_to_projection(self, monkeypatch, tmp_path):
        monkeypatch.setattr(runner, "AGENT_LOGS_DIR", tmp_path)
        ContainerResult(
            status=RolloutStatus.SUCCESS,
            output=AgentWorkflowOutput(
                messages=[{"role": "assistant", "content": "projected"}],
                metrics={"turns": 1.0},
            ),
        ).write(tmp_path / RESULT_FILENAME)

        loaded = runner.load_sample()

        assert loaded is not None
        assert loaded.messages[0]["content"] == "projected"
        assert loaded.metrics == {"turns": 1.0}

    def test_load_sample_falls_back_to_trajectory_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(runner, "AGENT_LOGS_DIR", tmp_path)
        (tmp_path / "trajectory.json").write_text(
            json.dumps({"steps": [{"source": "agent", "message": "from atif"}]})
        )

        loaded = runner.load_sample()

        assert loaded is not None
        assert loaded.messages == [{"role": "assistant", "content": "from atif"}]


class TestMessagesFromTrajectory:
    def test_raw_messages_document_passes_through(self):
        doc = {"messages": [{"role": "user", "content": "hi"}]}
        assert messages_from_trajectory(doc) == [{"role": "user", "content": "hi"}]

    def test_atif_steps_keep_tool_calls_and_observations(self):
        """Tool results must come back as tool turns."""
        doc = {
            "steps": [
                {"source": "user", "message": "fix it"},
                {
                    "source": "agent",
                    "message": "running ls",
                    "reasoning_content": "look around first",
                    "tool_calls": [
                        {
                            "tool_call_id": "c1",
                            "function_name": "bash",
                            "arguments": {"cmd": "ls"},
                        }
                    ],
                    "observation": {
                        "results": [{"source_call_id": "c1", "content": "file.txt"}]
                    },
                },
                {"source": "agent", "message": "done"},
            ]
        }

        messages = messages_from_trajectory(doc)

        assert messages == [
            {"role": "user", "content": "fix it"},
            {
                "role": "assistant",
                "content": "running ls",
                "reasoning_content": "look around first",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"cmd": "ls"}'},
                    }
                ],
            },
            {"role": "tool", "content": "file.txt", "tool_call_id": "c1"},
            {"role": "assistant", "content": "done"},
        ]

    def test_unmatched_observation_result_has_no_call_id(self):
        doc = {
            "steps": [
                {
                    "source": "agent",
                    "message": "checking",
                    "observation": {"results": [{"content": "ambient output"}]},
                }
            ]
        }
        messages = messages_from_trajectory(doc)
        assert messages[1] == {"role": "tool", "content": "ambient output"}


MESSAGES = [
    {"role": "system", "content": "multiply things"},
    {"role": "user", "content": "val1 = 2\nval2 = 3"},
    {"role": "assistant", "content": "#### 6"},
]


class TestWriteTrajectoryDocument:
    def test_writes_atif_from_sample(self, tmp_path, monkeypatch):
        monkeypatch.setattr(runner, "AGENT_LOGS_DIR", tmp_path)
        sample = RolloutSample(messages=MESSAGES)

        runner.write_trajectory_json(sample, AgentWorkflowOutput(), "r-1")

        doc = json.loads((tmp_path / "trajectory.json").read_text())
        assert doc["session_id"] == "r-1"
        assert doc["agent"]["name"] == "osmosis-rollout-sdk"
        sources = [step["source"] for step in doc["steps"]]
        assert sources == ["system", "user", "agent"]
        assert "#### 6" in doc["steps"][-1]["message"]

    def test_falls_back_to_output_messages(self, tmp_path, monkeypatch):
        monkeypatch.setattr(runner, "AGENT_LOGS_DIR", tmp_path)
        output = AgentWorkflowOutput(messages=MESSAGES)

        runner.write_trajectory_json(None, output, "r-2")

        doc = json.loads((tmp_path / "trajectory.json").read_text())
        assert doc["trajectory_id"] == "r-2"

    def test_no_messages_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(runner, "AGENT_LOGS_DIR", tmp_path)

        runner.write_trajectory_json(None, AgentWorkflowOutput(), "r-3")

        assert not (tmp_path / "trajectory.json").exists()

    def test_persistence_opt_out_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(runner, "AGENT_LOGS_DIR", tmp_path)
        sample = RolloutSample(messages=MESSAGES, trajectory_messages=None)

        runner.write_trajectory_json(sample, AgentWorkflowOutput(), "r-4")

        assert not (tmp_path / "trajectory.json").exists()

    def test_writes_atif_without_harbor_installed(self, tmp_path, monkeypatch):
        """The task container installs the bundle, never the harbor extra."""
        import builtins

        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.partition(".")[0] == "harbor":
                raise ModuleNotFoundError("No module named 'harbor'", name="harbor")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)
        monkeypatch.setattr(runner, "AGENT_LOGS_DIR", tmp_path)

        runner.write_trajectory_json(
            RolloutSample(messages=MESSAGES), AgentWorkflowOutput(), "r-5"
        )

        doc = json.loads((tmp_path / "trajectory.json").read_text())
        assert doc["session_id"] == "r-5"
