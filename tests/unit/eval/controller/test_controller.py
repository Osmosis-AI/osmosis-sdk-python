from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

import osmosis_ai.eval.controller.controller as controller_mod
from osmosis_ai.eval.controller.controller import EvalController, EvalControllerConfig
from osmosis_ai.rollout.types import (
    GraderCompleteRequest,
    GraderStatus,
    RolloutCompleteRequest,
    RolloutSample,
    RolloutStatus,
)


class FakeBridge:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.tokens: dict[str, int] = {}
        self.systemic_errors: dict[str, str] = {}

    def collect_tokens(self, rollout_id: str) -> int:
        return self.tokens.pop(rollout_id, 0)

    def collect_systemic_error(self, rollout_id: str) -> str | None:
        return self.systemic_errors.pop(rollout_id, None)


@pytest.fixture
def config(tmp_path: Path) -> EvalControllerConfig:
    return EvalControllerConfig(
        project_root=tmp_path,
        rollout_name="demo",
        rollout_dir=tmp_path / "rollouts" / "demo",
        entrypoint="rollout.py",
        llm_model="openai/gpt-5-mini",
        api_key="llm-key",
        base_url="https://llm.example/v1",
        agent_timeout_sec=0.5,
        grader_timeout_sec=0.5,
        controller_port=8899,
    )


@pytest.fixture(autouse=True)
def fake_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(controller_mod, "LiteLLMBridge", FakeBridge)


def _complete_state(controller: EvalController, rollout_id: str) -> None:
    state = controller.server.get_rollout_state(rollout_id)
    assert state is not None
    state.register_chat_create_branch("s1", "multi_sample", [], None)
    state.mark_rollout_completed(
        RolloutCompleteRequest(rollout_id=rollout_id, status=RolloutStatus.SUCCESS)
    )
    state.mark_grader_completed(
        GraderCompleteRequest(
            rollout_id=rollout_id,
            status=GraderStatus.SUCCESS,
            samples={"s1": RolloutSample(id="s1", reward=1.0)},
        )
    )


def _complete_failed_state(controller: EvalController, rollout_id: str) -> None:
    state = controller.server.get_rollout_state(rollout_id)
    assert state is not None
    state.register_chat_create_branch("s1", "multi_sample", [], None)
    state.mark_rollout_completed(
        RolloutCompleteRequest(
            rollout_id=rollout_id,
            status=RolloutStatus.FAILURE,
            err_message="rollout failed",
        )
    )
    state.mark_grader_completed(
        GraderCompleteRequest(
            rollout_id=rollout_id,
            status=GraderStatus.FAILURE,
            err_message="grader failed",
            samples={"s1": RolloutSample(id="s1", reward=1.0)},
        )
    )


def test_controller_defers_dynamic_port_selection(
    config: EvalControllerConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    config.controller_port = None
    monkeypatch.setattr(
        controller_mod,
        "_find_free_port",
        lambda: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    driver = EvalController(config=config)

    assert driver.controller_port is None


@pytest.mark.asyncio
async def test_run_posts_rollout_init_contract_and_returns_success(
    config: EvalControllerConfig,
) -> None:
    driver = EvalController(config=config)
    posted: dict[str, Any] = {}

    async def fake_post_rollout(
        url: str, json: dict[str, Any], timeout: float
    ) -> httpx.Response:
        posted["url"] = url
        posted["payload"] = json
        posted["timeout"] = timeout
        _complete_state(driver, json["rollout_id"])
        return httpx.Response(200, json={})

    driver._new_rollout_id = lambda: "protocol-1"  # type: ignore[method-assign]
    driver._post_rollout = fake_post_rollout  # type: ignore[method-assign]

    outcome = await driver.run(
        [{"role": "user", "content": "hello"}],
        label="row-1",
        rollout_id="dataset-row-id",
    )

    assert posted == {
        "url": "http://127.0.0.1:8000/rollout",
        "timeout": 30.0,
        "payload": {
            "rollout_id": "protocol-1",
            "initial_messages": [{"role": "user", "content": "hello"}],
            "label": "row-1",
            "metadata": None,
            "chat_completions_url": "http://127.0.0.1:8899",
            "controller_api_key": driver.api_key,
            "completion_callback_url": "http://127.0.0.1:8899/v1/rollout/completed",
            "grader_callback_url": "http://127.0.0.1:8899/v1/grader/completed",
            "agent_timeout_sec": 0.5,
            "grader_timeout_sec": 0.5,
            "extra_fields": None,
        },
    }
    assert outcome.status == RolloutStatus.SUCCESS
    assert outcome.rollout_id == "protocol-1"
    assert outcome.tokens == 0
    assert driver.server.get_rollout_state("protocol-1") is None
    assert driver.bridge.collect_tokens("protocol-1") == 0


@pytest.mark.asyncio
async def test_run_callback_failure_status_returns_failure(
    config: EvalControllerConfig,
) -> None:
    driver = EvalController(config=config)

    async def fake_post_rollout(
        url: str, json: dict[str, Any], timeout: float
    ) -> httpx.Response:
        _complete_failed_state(driver, json["rollout_id"])
        return httpx.Response(200, json={})

    driver._new_rollout_id = lambda: "protocol-failed"  # type: ignore[method-assign]
    driver._post_rollout = fake_post_rollout  # type: ignore[method-assign]

    outcome = await driver.run([{"role": "user", "content": "hello"}])

    assert outcome.status == RolloutStatus.FAILURE
    assert outcome.error == "Rollout callback failed: rollout failed"


@pytest.mark.asyncio
async def test_run_collects_bridge_tokens_without_double_counting(
    config: EvalControllerConfig,
) -> None:
    driver = EvalController(config=config)
    driver.bridge.tokens["protocol-tokens"] = 7

    async def fake_post_rollout(
        url: str, json: dict[str, Any], timeout: float
    ) -> httpx.Response:
        _complete_state(driver, json["rollout_id"])
        state = driver.server.get_rollout_state(json["rollout_id"])
        assert state is not None
        state.total_tokens = 5
        return httpx.Response(200, json={})

    driver._new_rollout_id = lambda: "protocol-tokens"  # type: ignore[method-assign]
    driver._post_rollout = fake_post_rollout  # type: ignore[method-assign]

    outcome = await driver.run([{"role": "user", "content": "hello"}])

    assert outcome.status == RolloutStatus.SUCCESS
    assert outcome.tokens == 5
    assert driver.bridge.collect_tokens("protocol-tokens") == 0


@pytest.mark.asyncio
async def test_run_timeout_during_rollout_init_returns_failure_and_cleans_state(
    config: EvalControllerConfig,
) -> None:
    driver = EvalController(config=config)

    async def fake_post_rollout(
        url: str, json: dict[str, Any], timeout: float
    ) -> httpx.Response:
        raise TimeoutError("init timed out")

    driver._new_rollout_id = lambda: "protocol-timeout"  # type: ignore[method-assign]
    driver._post_rollout = fake_post_rollout  # type: ignore[method-assign]

    outcome = await driver.run([{"role": "user", "content": "hello"}])

    assert outcome.status == RolloutStatus.FAILURE
    assert outcome.error is not None
    assert "30s" in outcome.error
    assert driver.server.get_rollout_state("protocol-timeout") is None


@pytest.mark.asyncio
async def test_run_non_2xx_rollout_init_returns_failure_and_collects_systemic(
    config: EvalControllerConfig,
) -> None:
    driver = EvalController(config=config)
    driver.bridge.tokens["protocol-http"] = 3
    driver.bridge.systemic_errors["protocol-http"] = "provider down"

    async def fake_post_rollout(
        url: str, json: dict[str, Any], timeout: float
    ) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    driver._new_rollout_id = lambda: "protocol-http"  # type: ignore[method-assign]
    driver._post_rollout = fake_post_rollout  # type: ignore[method-assign]

    outcome = await driver.run([{"role": "user", "content": "hello"}])

    assert outcome.status == RolloutStatus.FAILURE
    assert outcome.error == "/rollout returned HTTP 503: unavailable"
    assert outcome.systemic_error == "provider down"
    assert outcome.tokens == 0
    assert driver.bridge.collect_tokens("protocol-http") == 0
    assert driver.server.get_rollout_state("protocol-http") is None


@pytest.mark.asyncio
async def test_run_invalid_rollout_init_json_returns_failure(
    config: EvalControllerConfig,
) -> None:
    driver = EvalController(config=config)

    async def fake_post_rollout(
        url: str, json: dict[str, Any], timeout: float
    ) -> httpx.Response:
        return httpx.Response(200, content=b"{")

    driver._new_rollout_id = lambda: "protocol-json"  # type: ignore[method-assign]
    driver._post_rollout = fake_post_rollout  # type: ignore[method-assign]

    outcome = await driver.run([{"role": "user", "content": "hello"}])

    assert outcome.status == RolloutStatus.FAILURE
    assert outcome.error is not None
    assert "Invalid RolloutInitResponse JSON" in outcome.error
    assert driver.server.get_rollout_state("protocol-json") is None


@pytest.mark.asyncio
async def test_run_agent_callback_timeout_returns_failure_and_cleans_state(
    config: EvalControllerConfig,
) -> None:
    driver = EvalController(config=config)
    config.agent_timeout_sec = 0.01

    async def fake_post_rollout(
        url: str, json: dict[str, Any], timeout: float
    ) -> httpx.Response:
        return httpx.Response(200, json={})

    driver._new_rollout_id = (  # type: ignore[method-assign]
        lambda: "protocol-agent-timeout"
    )
    driver._post_rollout = fake_post_rollout  # type: ignore[method-assign]

    outcome = await driver.run([{"role": "user", "content": "hello"}])

    assert outcome.status == RolloutStatus.FAILURE
    assert outcome.error is not None
    assert "Timed out waiting for rollout callback" in outcome.error
    assert driver.server.get_rollout_state("protocol-agent-timeout") is None


@pytest.mark.asyncio
async def test_run_grader_callback_timeout_returns_failure_and_cleans_state(
    config: EvalControllerConfig,
) -> None:
    driver = EvalController(config=config)
    config.grader_timeout_sec = 0.01

    async def fake_post_rollout(
        url: str, json: dict[str, Any], timeout: float
    ) -> httpx.Response:
        state = driver.server.get_rollout_state(json["rollout_id"])
        assert state is not None
        state.mark_rollout_completed(
            RolloutCompleteRequest(
                rollout_id=json["rollout_id"],
                status=RolloutStatus.SUCCESS,
            )
        )
        return httpx.Response(200, json={})

    driver._new_rollout_id = (  # type: ignore[method-assign]
        lambda: "protocol-grader-timeout"
    )
    driver._post_rollout = fake_post_rollout  # type: ignore[method-assign]

    outcome = await driver.run([{"role": "user", "content": "hello"}])

    assert outcome.status == RolloutStatus.FAILURE
    assert outcome.error is not None
    assert "Timed out waiting for grader callback" in outcome.error
    assert driver.server.get_rollout_state("protocol-grader-timeout") is None


@pytest.mark.asyncio
async def test_post_rollout_uses_httpx_async_client(
    config: EvalControllerConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["json"] = request.read().decode()
        return httpx.Response(200, json={})

    driver = EvalController(config=config)

    class FakeAsyncClient(httpx.AsyncClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(controller_mod.httpx, "AsyncClient", FakeAsyncClient)

    response = await driver._post_rollout(
        "http://127.0.0.1:8000/rollout",
        json={"rollout_id": "r1"},
        timeout=30.0,
    )

    assert response.status_code == 200
    assert seen["url"] == "http://127.0.0.1:8000/rollout"
    assert seen["json"] == '{"rollout_id":"r1"}'


@pytest.mark.asyncio
async def test_start_rejects_occupied_controller_port(
    config: EvalControllerConfig,
) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        _, port = server.getsockname()
        config.controller_port = port
        driver = EvalController(config=config)

        with pytest.raises(TimeoutError, match="already occupied"):
            await driver.start()

        assert driver._server_task is None
        assert driver._uvicorn_server is None


def test_controller_package_lazily_exports_driver_without_heavy_imports() -> None:
    script = "\n".join(
        [
            "import sys",
            "import osmosis_ai.eval.controller as controller",
            "assert 'osmosis_ai.eval.controller.controller' not in sys.modules",
            "assert 'osmosis_ai.eval.controller.server' not in sys.modules",
            "assert 'osmosis_ai.eval.controller.state' not in sys.modules",
            "assert 'osmosis_ai.eval.controller.litellm_bridge' not in sys.modules",
            "assert 'litellm' not in sys.modules",
            "assert controller.EvalController.__name__ == 'EvalController'",
            "assert controller.EvalControllerConfig.__name__ == 'EvalControllerConfig'",
            "assert 'osmosis_ai.eval.controller.controller' in sys.modules",
            "assert 'osmosis_ai.eval.controller.server' in sys.modules",
        ]
    )

    completed = asyncio.run(
        asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
    )

    assert completed.returncode == 0, completed.stderr
