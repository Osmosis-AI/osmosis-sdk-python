from __future__ import annotations

import asyncio
import secrets
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from osmosis_ai.eval.controller.litellm_bridge import LiteLLMBridge
from osmosis_ai.eval.controller.server import EvalControllerServer
from osmosis_ai.eval.controller.state import ControllerRolloutState
from osmosis_ai.rollout.driver import RolloutDriver, RolloutOutcome
from osmosis_ai.rollout.types import GraderStatus, RolloutInitResponse, RolloutStatus


@dataclass
class EvalControllerConfig:
    project_root: Path
    rollout_name: str
    rollout_dir: Path
    entrypoint: str
    llm_model: str
    api_key: str | None
    base_url: str | None
    agent_timeout_sec: float = 450.0
    grader_timeout_sec: float = 150.0
    controller_port: int | None = None


class EvalController(RolloutDriver):
    def __init__(self, *, config: EvalControllerConfig) -> None:
        self.config = config
        self.api_key = secrets.token_urlsafe(32)
        self.bridge = LiteLLMBridge(
            model=config.llm_model,
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.server = EvalControllerServer(api_key=self.api_key, bridge=self.bridge)
        self.controller_port = config.controller_port or 8067
        self._server_task: asyncio.Task[Any] | None = None
        self._uvicorn_server: Any | None = None

    @property
    def max_concurrency(self) -> int:
        return 0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.controller_port}"

    def _new_rollout_id(self) -> str:
        return uuid.uuid4().hex

    async def _post_rollout(
        self, url: str, json: dict[str, Any], timeout: float
    ) -> httpx.Response:
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.post(url, json=json)

    async def run(
        self,
        messages: list[dict[str, Any]],
        label: str | None = None,
        rollout_id: str = "",
    ) -> RolloutOutcome:
        del rollout_id
        start = time.monotonic()
        protocol_rollout_id = self._new_rollout_id()
        state = ControllerRolloutState(protocol_rollout_id)
        self.server.register_rollout_state(state)

        try:
            payload = self._rollout_init_payload(
                protocol_rollout_id, messages=messages, label=label
            )
            response: httpx.Response | None = None
            try:
                response = await self._post_rollout(
                    "http://127.0.0.1:8000/rollout",
                    json=payload,
                    timeout=30.0,
                )
            except (TimeoutError, httpx.TimeoutException):
                state.primary_error = (
                    "Timed out after 30s initializing rollout at 127.0.0.1:8000/rollout"
                )
            except Exception as exc:
                state.primary_error = str(exc)

            if state.primary_error is None and response is not None:
                if response.status_code < 200 or response.status_code >= 300:
                    state.primary_error = (
                        f"/rollout returned HTTP {response.status_code}: "
                        f"{response.text}"
                    )
                else:
                    try:
                        RolloutInitResponse.model_validate(response.json())
                    except Exception as exc:
                        state.primary_error = f"Invalid RolloutInitResponse JSON: {exc}"

            if state.primary_error is None:
                try:
                    rollout_callback = await asyncio.wait_for(
                        state.rollout_future, timeout=self.config.agent_timeout_sec
                    )
                    if rollout_callback.status is RolloutStatus.FAILURE:
                        state.primary_error = self._callback_error(
                            "Rollout callback failed",
                            rollout_callback.err_message,
                            rollout_callback.err_category,
                        )
                except TimeoutError:
                    state.primary_error = (
                        "Timed out waiting for rollout callback after "
                        f"{self.config.agent_timeout_sec}s"
                    )
                except Exception as exc:
                    state.primary_error = str(exc)

            if state.primary_error is None:
                try:
                    grader_callback = await asyncio.wait_for(
                        state.grader_future, timeout=self.config.grader_timeout_sec
                    )
                    if grader_callback.status is GraderStatus.FAILURE:
                        state.primary_error = self._callback_error(
                            "Grader callback failed",
                            grader_callback.err_message,
                            grader_callback.err_category,
                        )
                except TimeoutError:
                    state.primary_error = (
                        "Timed out waiting for grader callback after "
                        f"{self.config.grader_timeout_sec}s"
                    )
                except Exception as exc:
                    state.primary_error = str(exc)
        finally:
            state.cancel_pending()
            self._collect_bridge_state(state)
            self.server.pop_rollout_state(protocol_rollout_id)
        return self._outcome(state, start)

    def _rollout_init_payload(
        self,
        protocol_rollout_id: str,
        *,
        messages: list[dict[str, Any]],
        label: str | None,
    ) -> dict[str, Any]:
        return {
            "rollout_id": protocol_rollout_id,
            "initial_messages": messages,
            "label": label,
            "metadata": None,
            "chat_completions_url": self.base_url,
            "controller_api_key": self.api_key,
            "completion_callback_url": f"{self.base_url}/v1/rollout/completed",
            "grader_callback_url": f"{self.base_url}/v1/grader/completed",
            "agent_timeout_sec": self.config.agent_timeout_sec,
            "grader_timeout_sec": self.config.grader_timeout_sec,
            "extra_fields": None,
        }

    def _collect_bridge_state(self, state: ControllerRolloutState) -> None:
        systemic_error = self.bridge.collect_systemic_error(state.rollout_id)
        if systemic_error:
            state.mark_systemic_error(systemic_error)
        self.bridge.collect_tokens(state.rollout_id)

    def _callback_error(
        self, prefix: str, message: str | None, category: Any | None
    ) -> str:
        if message:
            return f"{prefix}: {message}"
        if category:
            return f"{prefix}: {category}"
        return prefix

    def _outcome(self, state: ControllerRolloutState, start: float) -> RolloutOutcome:
        return state.to_outcome(duration_ms=(time.monotonic() - start) * 1000)


__all__ = ["EvalController", "EvalControllerConfig"]
