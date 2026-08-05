"""Harbor installed agent that runs a bundled workflow.

install() bootstraps uv if the image lacks it, then installs the bundle wheel,
so any task image with python works unmodified. run() ships the ContainerInput
into the container and executes the bundle's agent console script; results come
back through the ContainerResult file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

from osmosis_ai.rollout.backend.harbor.tasks import (
    SDK_UV,
    SDK_VENV,
    venv_or_fallback_script,
)
from osmosis_ai.rollout.container.files import (
    INPUT_FILENAME,
    RESULT_FILENAME,
    ContainerInput,
)


class OsmosisHarnessInstalledAgent(BaseInstalledAgent):
    def __init__(
        self,
        logs_dir: Path,
        *args: Any,
        bundle_path: str,
        agent_script: str,
        input_path: str,
        **kwargs: Any,
    ):
        super().__init__(logs_dir, *args, **kwargs)
        self.bundle_path: Path = Path(bundle_path)
        self.agent_script = agent_script
        self.input_path: Path = Path(input_path)

    @staticmethod
    def name() -> str:
        return "osmosis-harness-agent"

    async def install(self, environment: Any) -> None:
        wheel = f"/tmp/{self.bundle_path.name}"
        await environment.upload_file(self.bundle_path, wheel)
        await self.exec_as_agent(
            environment,
            f"if [ -x {SDK_VENV}/bin/python ]; then "
            f"{SDK_UV} pip install --python {SDK_VENV}/bin/python --no-deps {wheel}; "
            f"else command -v uv >/dev/null || python3 -m pip install --quiet uv; "
            f"uv pip install --system {wheel}; fi",
        )

    async def run(self, instruction: Any, environment: Any, context: Any) -> None:
        container_input = ContainerInput.read(self.input_path)
        if not container_input.prompt:
            container_input.prompt = [{"role": "user", "content": instruction}]

        host_input = self.logs_dir / INPUT_FILENAME
        container_input.write(host_input)
        if not environment.capabilities.mounted:
            agent_dir = EnvironmentPaths.for_os(environment.os).agent_dir
            await environment.upload_file(
                host_input, (agent_dir / INPUT_FILENAME).as_posix()
            )

        await self.exec_as_agent(
            environment, venv_or_fallback_script(self.agent_script)
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        result_path = self.logs_dir / RESULT_FILENAME
        if result_path.exists():
            context.metadata = json.loads(result_path.read_text())
