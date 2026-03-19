"""Osmosis installed agent adapter for Harbor.

Extends BaseInstalledAgent to run an AgentWorkflow inside the container.
User code and SDK are baked into the image at /workspace/ during Docker build.
Rollout config is passed via kwargs and copied to the mounted volume.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent, ExecInput
from harbor.models.agent.context import AgentContext


class OsmosisInstalledAgent(BaseInstalledAgent):
    def __init__(
        self, logs_dir: Path, *args: Any, rollout_config_path: str = "", **kwargs: Any
    ):
        super().__init__(logs_dir, *args, **kwargs)
        self.rollout_config_path = (
            Path(rollout_config_path) if rollout_config_path else None
        )

    @staticmethod
    def name() -> str:
        return "osmosis-rollout-agent"

    @property
    def _install_agent_template_path(self) -> Path:
        return Path("/dev/null")  # unused — setup() is overridden

    async def setup(self, environment) -> None:
        pass  # user code is baked into the image at /workspace/

    async def run(self, instruction, environment, context) -> None:
        # instruction is the prompt JSON from instruction.md
        (self.logs_dir / "prompt.json").write_text(instruction)

        # copy rollout config to mounted volume so it's at /logs/agent/rollout_config.json
        if self.rollout_config_path and self.rollout_config_path.exists():
            shutil.copy2(
                self.rollout_config_path, self.logs_dir / "rollout_config.json"
            )

        await super().run(instruction, environment, context)

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        return [
            ExecInput(
                command=(
                    "python -m osmosis_ai.rollout_v2.backend.harbor.agent_runner"
                    " --config /logs/agent/rollout_config.json"
                    " --prompt /logs/agent/prompt.json"
                ),
                cwd="/workspace",
                env={"PYTHONPATH": "/workspace"},
            )
        ]

    def populate_context_post_run(self, context: AgentContext) -> None:
        meta_path = self.logs_dir / "rollout_meta.json"
        if meta_path.exists():
            context.metadata = json.loads(meta_path.read_text())
