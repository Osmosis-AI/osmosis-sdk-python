"""Tests for the in-container runner's trajectory emission."""

import json

import osmosis_ai.rollout.container.runner as runner
from osmosis_ai.rollout.types import RolloutSample
from osmosis_ai.rollout.types.output import AgentWorkflowOutput

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
        output = AgentWorkflowOutput(samples={"default": MESSAGES})

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
