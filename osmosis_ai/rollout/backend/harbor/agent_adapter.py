"""Osmosis installed agent adapter for Harbor.

Extends BaseInstalledAgent to run an AgentWorkflow inside the container.
User code and SDK are baked into the image at /workspace/ during Docker build.
Rollout config is passed via kwargs and copied to the mounted volume for Docker.
For non-mounted environments, runner inputs are uploaded to Harbor's agent logs.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.models.agent.context import AgentContext


class OsmosisInstalledAgent(BaseInstalledAgent):
    def __init__(
        self, logs_dir: Path, *args: Any, rollout_config_path: str = "", **kwargs: Any
    ):
        super().__init__(logs_dir, *args, **kwargs)
        self.rollout_config_path: Path | None = (
            Path(rollout_config_path) if rollout_config_path else None
        )

    @staticmethod
    def name() -> str:
        return "osmosis-rollout-agent"

    async def install(self, environment: Any) -> None:
        pass  # user code is baked into the image

    async def setup(self, environment: Any) -> None:
        pass  # user code is baked into the image

    async def run(self, instruction: Any, environment: Any, context: Any) -> None:
        prompt_path = self.logs_dir / "prompt.json"
        config_path = self.logs_dir / "rollout_config.json"
        prompt_env_path = environment.env_paths.agent_dir / "prompt.json"
        config_env_path = environment.env_paths.agent_dir / "rollout_config.json"

        prompt_path.write_text(instruction)

        if self.rollout_config_path and self.rollout_config_path.exists():
            shutil.copy2(self.rollout_config_path, config_path)

        if not environment.capabilities.mounted:
            await environment.upload_file(
                prompt_path,
                prompt_env_path.as_posix(),
            )
            if config_path.exists():
                await environment.upload_file(
                    config_path,
                    config_env_path.as_posix(),
                )

        await self.exec_as_agent(
            environment,
            command=(
                "python -m osmosis_ai.rollout.backend.harbor.agent_runner"
                f" --config {config_env_path.as_posix()}"
                f" --prompt {prompt_env_path.as_posix()}"
            ),
            cwd="/workspace",
            env={"PYTHONPATH": "/workspace"},
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        meta_path = self.logs_dir / "rollout_meta.json"
        if meta_path.exists():
            context.metadata = json.loads(meta_path.read_text())
