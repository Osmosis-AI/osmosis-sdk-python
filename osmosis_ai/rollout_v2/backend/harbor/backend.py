from __future__ import annotations

import asyncio
import json
import logging
import platform
import shutil
from pathlib import Path
from typing import Any

import toml
import tomllib
from harbor.models.trial.config import (
    AgentConfig as HarborAgentConfig,
)
from harbor.models.trial.config import (
    EnvironmentConfig as HarborEnvironmentConfig,
)
from harbor.models.trial.config import (
    EnvironmentType,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)
from harbor.trial.hooks import TrialEvent, TrialHookEvent
from harbor.trial.queue import TrialQueue

from osmosis_ai.rollout_v2.agent_workflow import AgentWorkflow
from osmosis_ai.rollout_v2.backend.base import ExecutionBackend, ResultCallback
from osmosis_ai.rollout_v2.grader import Grader
from osmosis_ai.rollout_v2.types import (
    AgentWorkflowConfig,
    ExecutionRequest,
    ExecutionResult,
    GraderConfig,
    RolloutErrorCategory,
    RolloutSample,
    RolloutStatus,
)
from osmosis_ai.rollout_v2.utils.imports import to_import_path

logger = logging.getLogger(__name__)

AGENT_IMPORT_PATH = (
    "osmosis_ai.rollout_v2.backend.harbor.agent_adapter:OsmosisInstalledAgent"
)
TRIAL_NAME_PREFIX = "trial-"

TEST_SH_TEMPLATE = """\
#!/bin/bash
set -e
PYTHONPATH=/workspace python -m osmosis_ai.rollout_v2.backend.harbor.grader_runner \
    --config /logs/agent/rollout_config.json \
    --samples /logs/agent/samples.json
"""


class PendingTrial:
    def __init__(
        self,
        on_workflow_complete: ResultCallback,
        on_grader_complete: ResultCallback | None,
    ):
        self.on_workflow_complete = on_workflow_complete
        self.on_grader_complete = on_grader_complete
        self.done: asyncio.Future[None] = asyncio.get_event_loop().create_future()


class HarborBackend(ExecutionBackend):
    """Execution backend that runs workflows inside Harbor containers.

    The Docker image is built once at init (SDK + user code baked in).
    Per-rollout task dirs symlink to the shared environment dir so
    Harbor reuses the cached image.
    """

    def __init__(
        self,
        *,
        orchestrator: TrialQueue,
        task_dir: Path,
        user_code_dir: Path,
        workflow: type[AgentWorkflow] | str,
        workflow_config: AgentWorkflowConfig | str | None = None,
        grader: type[Grader] | str | None = None,
        grader_config: GraderConfig | str | None = None,
        trials_dir: Path = Path("trials"),
        custom_tests_dir: Path | None = None,
        environment_config: HarborEnvironmentConfig | None = None,
        cache_image: bool = True,
        _sdk_source_dir: Path | None = None,  # local dev only
    ) -> None:
        self.orchestrator = orchestrator
        self.task_dir = task_dir
        self.user_code_dir = user_code_dir
        self.workflow_path = ensure_import_path(workflow)
        self.workflow_config_path = (
            ensure_import_path(workflow_config) if workflow_config else None
        )
        self.grader_path = ensure_import_path(grader) if grader else None
        self.grader_config_path = (
            ensure_import_path(grader_config) if grader_config else None
        )
        self.grading = self.grader_path is not None
        self.custom_tests_dir = custom_tests_dir
        self.environment_config = environment_config or HarborEnvironmentConfig()
        self._sdk_source_dir = _sdk_source_dir
        self.cache_image = cache_image

        if self.cache_image:
            self.environment_config.delete = False

        self.root_dir = Path(f"/tmp/osmosis-harbor-{self.task_dir.name}")
        self.rollouts_dir = self.root_dir / "rollouts"
        self.rollouts_dir.mkdir(parents=True, exist_ok=True)
        self.shared_env_dir = self.root_dir / "shared-env"
        self.trials_dir = (
            trials_dir if trials_dir != Path("trials") else self.root_dir / "trials"
        )

        self.pending: dict[str, PendingTrial] = {}

        if self.cache_image:
            self.prepare_shared_env()

        self.orchestrator.add_hook(
            TrialEvent.VERIFICATION_START, self.on_verification_start
        )
        self.orchestrator.add_hook(TrialEvent.END, self.on_trial_end)

    def prepare_shared_env(self) -> None:
        """Build the shared environment dir with SDK + user code.
        Only done once — Docker caches the resulting image."""
        if self.shared_env_dir.exists():
            shutil.rmtree(self.shared_env_dir)
        self.shared_env_dir.mkdir(parents=True)

        base_env = self.task_dir / "environment"
        if base_env.exists():
            shutil.copytree(base_env, self.shared_env_dir, dirs_exist_ok=True)

        if self._sdk_source_dir and self._sdk_source_dir.exists():
            shutil.copytree(
                self._sdk_source_dir,
                self.shared_env_dir / self._sdk_source_dir.name,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    "__pycache__",
                    "*.pyc",
                    ".git",
                    ".venv",
                    "node_modules",
                ),
            )

        workspace = self.shared_env_dir / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            self.user_code_dir,
            workspace / self.user_code_dir.name,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                "__pycache__",
                "*.pyc",
                ".git",
                ".venv",
                "node_modules",
            ),
        )

    def prepare_env_dir(self, task_dir: Path) -> None:
        """Prepare the environment dir for a task dir.
        If cache_image is on, symlink to the shared env. Otherwise copy fresh."""
        if self.cache_image:
            (task_dir / "environment").symlink_to(self.shared_env_dir)
        else:
            env_dir = task_dir / "environment"
            base_env = self.task_dir / "environment"
            if base_env.exists():
                shutil.copytree(base_env, env_dir, dirs_exist_ok=True)

            if self._sdk_source_dir and self._sdk_source_dir.exists():
                shutil.copytree(
                    self._sdk_source_dir,
                    env_dir / self._sdk_source_dir.name,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(
                        "__pycache__",
                        "*.pyc",
                        ".git",
                        ".venv",
                        "node_modules",
                    ),
                )

            workspace = env_dir / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                self.user_code_dir,
                workspace / self.user_code_dir.name,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    "__pycache__",
                    "*.pyc",
                    ".git",
                    ".venv",
                    "node_modules",
                ),
            )

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "backend": "harbor",
            "pending_trials": len(self.pending),
        }

    async def execute(
        self,
        request: ExecutionRequest,
        on_workflow_complete: ResultCallback,
        on_grader_complete: ResultCallback | None = None,
    ) -> None:
        pending = PendingTrial(on_workflow_complete, on_grader_complete)
        self.pending[request.id] = pending

        try:
            task_dir = self.prepare_task_dir(request)
            trial_config = self.build_trial_config(task_dir, request)
            await self.orchestrator.submit(trial_config)
            await pending.done
        except Exception as e:
            self.pending.pop(request.id, None)
            logger.error("Failed trial %s: %s", request.id, e)
            await on_workflow_complete(
                ExecutionResult(
                    status=RolloutStatus.FAILURE,
                    err_message=str(e),
                    err_category=RolloutErrorCategory.AGENT_ERROR,
                )
            )

    def prepare_task_dir(self, request: ExecutionRequest) -> Path:
        rollout_dir = self.rollouts_dir / request.id
        task_dir = rollout_dir / self.task_dir.name
        task_dir.mkdir(parents=True, exist_ok=True)

        if self.task_dir.exists():
            for item in self.task_dir.iterdir():
                if item.name == "environment":
                    continue
                dest = task_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)

        self.prepare_env_dir(task_dir)
        self.write_instruction(task_dir, request)
        self.prepare_tests_dir(task_dir)
        self.inject_verifier_env(task_dir)

        return task_dir

    def write_instruction(self, task_dir: Path, request: ExecutionRequest) -> None:
        """Write instruction.md (prompt) and rollout_config.json as separate files."""
        (task_dir / "instruction.md").write_text(
            json.dumps(request.prompt, default=str)
        )

        rollout_config = self.build_rollout_config(request)
        (task_dir / "rollout_config.json").write_text(
            json.dumps(rollout_config, default=str)
        )

    def build_rollout_config(self, request: ExecutionRequest) -> dict[str, Any]:
        from osmosis_ai.rollout_v2.context import get_rollout_context

        config: dict[str, Any] = {
            "id": request.id,
            "workflow": self.workflow_path,
        }
        if self.workflow_config_path:
            config["workflow_config"] = self.workflow_config_path
        if self.grader_path:
            config["grader"] = self.grader_path
        if self.grader_config_path:
            config["grader_config"] = self.grader_config_path
        if request.label is not None:
            config["label"] = request.label

        ctx = get_rollout_context()
        if ctx:
            if ctx.chat_completions_url:
                url = ctx.chat_completions_url
                if self.environment_config.type == EnvironmentType.DOCKER:
                    url = rewrite_url_for_docker(url)
                config["chat_completions_url"] = url
            if ctx.api_key:
                config["api_key"] = ctx.api_key
            if ctx.rollout_id:
                config["rollout_id"] = ctx.rollout_id

        return config

    def prepare_tests_dir(self, task_dir: Path) -> None:
        if self.custom_tests_dir and self.custom_tests_dir.exists():
            shutil.copytree(
                self.custom_tests_dir, task_dir / "tests", dirs_exist_ok=True
            )
        elif self.grading:
            tests_dir = task_dir / "tests"
            tests_dir.mkdir(parents=True, exist_ok=True)
            test_sh = tests_dir / "test.sh"
            test_sh.write_text(TEST_SH_TEMPLATE)
            test_sh.chmod(0o755)

    def inject_verifier_env(self, task_dir: Path) -> None:
        task_toml_path = task_dir / "task.toml"
        if task_toml_path.exists():
            with open(task_toml_path, "rb") as f:
                config = tomllib.load(f)
        else:
            config = {}

        config.setdefault("verifier", {})
        config["verifier"].setdefault("env", {})
        config["verifier"]["env"]["PYTHONPATH"] = "/workspace"

        with open(task_toml_path, "w") as f:
            toml.dump(config, f)

    def build_trial_config(
        self, task_dir: Path, request: ExecutionRequest
    ) -> TrialConfig:
        agent_config = HarborAgentConfig(
            import_path=AGENT_IMPORT_PATH,
            kwargs={
                "rollout_config_path": str(task_dir / "rollout_config.json"),
            },
        )
        if request.agent_timeout_sec is not None:
            agent_config.override_timeout_sec = request.agent_timeout_sec

        verifier_config = VerifierConfig(disable=not self.grading)
        if request.grader_timeout_sec is not None:
            verifier_config.override_timeout_sec = request.grader_timeout_sec

        return TrialConfig(
            task=TaskConfig(path=task_dir),
            trial_name=f"{TRIAL_NAME_PREFIX}{request.id}",
            trials_dir=self.trials_dir,
            agent=agent_config,
            environment=self.environment_config,
            verifier=verifier_config,
        )

    async def on_verification_start(self, event: TrialHookEvent) -> None:
        rollout_id = parse_rollout_id(event)
        pending = self.pending.get(rollout_id)
        if not pending:
            logger.error("No pending trial found for rollout %s", rollout_id)
            return

        metadata = get_agent_metadata(event)
        if metadata and metadata.get("status") == "success":
            samples = parse_samples(metadata.get("samples", {}))
            result = ExecutionResult(status=RolloutStatus.SUCCESS, samples=samples)
        elif event.result and event.result.exception_info:
            err = event.result.exception_info
            result = ExecutionResult(
                status=RolloutStatus.FAILURE,
                err_message=err.exception_message,
                err_category=RolloutErrorCategory.AGENT_ERROR,
            )
        else:
            err_message = metadata.get("err_message") if metadata else "Unknown error"
            result = ExecutionResult(
                status=RolloutStatus.FAILURE,
                err_message=err_message,
                err_category=RolloutErrorCategory.AGENT_ERROR,
            )

        await pending.on_workflow_complete(result)

    async def on_trial_end(self, event: TrialHookEvent) -> None:
        rollout_id = parse_rollout_id(event)
        pending = self.pending.pop(rollout_id, None)
        if not pending:
            logger.error("No pending trial found for rollout %s", rollout_id)
            return

        if pending.on_grader_complete:
            metadata = get_agent_metadata(event)
            samples = parse_samples(metadata.get("samples", {})) if metadata else {}

            if event.result and event.result.verifier_result:
                rewards = event.result.verifier_result.rewards or {}
                for sid, reward in rewards.items():
                    if sid in samples:
                        samples[sid].reward = float(reward)
                result = ExecutionResult(status=RolloutStatus.SUCCESS, samples=samples)
            elif event.result and event.result.exception_info:
                err = event.result.exception_info
                result = ExecutionResult(
                    status=RolloutStatus.FAILURE,
                    samples=samples,
                    err_message=err.exception_message,
                    err_category=RolloutErrorCategory.AGENT_ERROR,
                )
            else:
                result = ExecutionResult(status=RolloutStatus.SUCCESS, samples=samples)

            await pending.on_grader_complete(result)

        if not pending.done.done():
            pending.done.set_result(None)


def ensure_import_path(ref: Any) -> str:
    """Convert a Python object or string to an import path string."""
    if isinstance(ref, str):
        return ref
    return to_import_path(ref)


def parse_rollout_id(event: TrialHookEvent) -> str:
    return event.config.trial_name.removeprefix(TRIAL_NAME_PREFIX)


def get_agent_metadata(event: TrialHookEvent) -> dict[str, Any] | None:
    if event.result and event.result.agent_result:
        return event.result.agent_result.metadata
    return None


def parse_samples(raw: dict[str, Any]) -> dict[str, RolloutSample]:
    return {
        sid: RolloutSample.model_validate(data) if isinstance(data, dict) else data
        for sid, data in raw.items()
    }


def rewrite_url_for_docker(url: str) -> str:
    if platform.system() != "Darwin":
        return url
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(url)
    if parsed.hostname in ("localhost", "127.0.0.1"):
        parsed = parsed._replace(
            netloc=parsed.netloc.replace(parsed.hostname, "host.docker.internal")
        )
    return urlunparse(parsed)
