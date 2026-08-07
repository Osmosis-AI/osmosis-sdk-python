"""The two files exchanged with the rollout container.

ContainerInput is staged into the container before the agent phase;
ContainerResult comes back when it ends. Both ends are SDK code — user
harnesses and native agents never touch these. The verifier reward file is
Harbor's own contract and has one writer here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from osmosis_ai.rollout.types import RolloutSample, RolloutStatus
from osmosis_ai.rollout.types.output import AgentWorkflowOutput, Messages

AGENT_LOGS_DIR = Path("/logs/agent")
VERIFIER_LOGS_DIR = Path("/logs/verifier")
INPUT_FILENAME = "container_input.json"
RESULT_FILENAME = "container_result.json"


class ContainerInput(BaseModel):
    """Everything one agent run needs inside the container."""

    version: int = 1
    rollout_id: str
    prompt: Messages = []
    label: str | None = None
    metadata: dict[str, Any] | None = None
    chat_completions_url: str = ""
    api_key: str | None = None

    @classmethod
    def read(cls, path: Path) -> ContainerInput:
        return cls.model_validate_json(path.read_text())

    def write(self, path: Path) -> None:
        path.write_text(self.model_dump_json())


class ContainerResult(BaseModel):
    """How the agent phase went: status, error, and the workflow's output.

    ``sample`` round-trips the full RolloutSample across the container
    boundary; the lossy ``output`` projection is kept for readers that
    consult it before the full sample.
    """

    status: RolloutStatus
    output: AgentWorkflowOutput | None = None
    sample: RolloutSample | None = None
    err_message: str | None = None

    @classmethod
    def read(cls, path: Path) -> ContainerResult:
        return cls.model_validate_json(path.read_text())

    def write(self, path: Path) -> None:
        path.write_text(self.model_dump_json())


def write_reward(reward: float) -> None:
    VERIFIER_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (VERIFIER_LOGS_DIR / "reward.json").write_text(json.dumps({"reward": reward}))
