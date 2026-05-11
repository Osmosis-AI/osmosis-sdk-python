import asyncio
import contextlib
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pytest
import uvicorn
from fastapi import BackgroundTasks, FastAPI

from osmosis_ai.eval.controller.controller import EvalController, EvalControllerConfig
from osmosis_ai.eval.controller.locks import (
    FixedPortLock,
    assert_user_server_port_free,
)
from osmosis_ai.rollout.types import RolloutSample, RolloutStatus

FAKE_ROLLOUT_SERVER_HOST = "127.0.0.1"
FAKE_ROLLOUT_SERVER_PORT = 8000
FAKE_ROLLOUT_SERVER_LOCK_PATH = (
    Path(tempfile.gettempdir()) / "osmosis-sdk-python-user-server-8000.lock"
)


class FakeBridge:
    async def complete(self, rollout_id, sample_id, body):
        from types import SimpleNamespace

        return SimpleNamespace(
            id="chatcmpl-1",
            created=123,
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason="stop",
                    message=SimpleNamespace(content="answer", tool_calls=None),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    def collect_tokens(self, rollout_id):
        return 0

    def collect_systemic_error(self, rollout_id):
        return None


async def _wait_for_fake_rollout_server(
    server: uvicorn.Server,
    server_task: asyncio.Task,
    *,
    timeout_sec: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout_sec
    last_error: Exception | None = None
    async with httpx.AsyncClient(timeout=0.2) as client:
        while time.monotonic() < deadline:
            if server.started:
                return
            if server_task.done():
                task_error: BaseException | None = None
                with contextlib.suppress(asyncio.CancelledError):
                    task_error = server_task.exception()
                raise RuntimeError(
                    "Fake rollout server exited before readiness"
                ) from task_error
            try:
                response = await client.get(
                    f"http://{FAKE_ROLLOUT_SERVER_HOST}:{FAKE_ROLLOUT_SERVER_PORT}/health"
                )
                if response.status_code == 200:
                    return
            except Exception as exc:
                last_error = exc
            await asyncio.sleep(0.05)

    raise TimeoutError(
        "Timed out waiting for fake rollout server readiness"
    ) from last_error


async def _stop_fake_rollout_server(
    server: uvicorn.Server,
    server_task: asyncio.Task,
    *,
    timeout_sec: float = 5.0,
) -> None:
    server.should_exit = True
    try:
        await asyncio.wait_for(server_task, timeout=timeout_sec)
    except TimeoutError:
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server_task


@contextlib.asynccontextmanager
async def _fake_rollout_server(app: FastAPI):
    fixed_port_lock = FixedPortLock(lock_path=FAKE_ROLLOUT_SERVER_LOCK_PATH)
    fixed_port_lock.acquire()
    server: uvicorn.Server | None = None
    server_task: asyncio.Task | None = None
    try:
        assert_user_server_port_free()
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=FAKE_ROLLOUT_SERVER_HOST,
                port=FAKE_ROLLOUT_SERVER_PORT,
                log_level="warning",
            )
        )
        server_task = asyncio.create_task(server.serve())
        await _wait_for_fake_rollout_server(server, server_task)
        yield
    finally:
        try:
            if server is not None and server_task is not None:
                await _stop_fake_rollout_server(server, server_task)
        finally:
            fixed_port_lock.release()


@pytest.mark.asyncio
async def test_eval_controller_fake_rollout_server_round_trip(tmp_path: Path) -> None:
    app = FastAPI()
    received = {}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/rollout")
    async def rollout(payload: dict, background_tasks: BackgroundTasks):
        received.update(payload)

        async def run_callbacks():
            headers = {"Authorization": f"Bearer {payload['controller_api_key']}"}
            async with httpx.AsyncClient() as client:
                stream_response = await client.post(
                    f"{payload['chat_completions_url']}/chat/completions",
                    headers={
                        **headers,
                        "x-rollout-id": payload["rollout_id"],
                        "x-sample-id": "s1",
                    },
                    json={
                        "model": "openai/osmosis-rollout",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                    timeout=5.0,
                )
                assert stream_response.status_code == 200
                assert "data: [DONE]" in stream_response.text
                await client.post(
                    payload["completion_callback_url"],
                    headers=headers,
                    json={"rollout_id": payload["rollout_id"], "status": "success"},
                    timeout=5.0,
                )
                await client.post(
                    payload["grader_callback_url"],
                    headers=headers,
                    json={
                        "rollout_id": payload["rollout_id"],
                        "status": "success",
                        "samples": {
                            "s1": RolloutSample(id="s1", reward=1.0).model_dump()
                        },
                    },
                    timeout=5.0,
                )

        background_tasks.add_task(run_callbacks)
        return {}

    async with _fake_rollout_server(app):
        controller = EvalController(
            config=EvalControllerConfig(
                project_root=tmp_path,
                rollout_name="demo",
                rollout_dir=tmp_path,
                entrypoint="main.py",
                llm_model="openai/gpt-5-mini",
                api_key=None,
                base_url=None,
                agent_timeout_sec=5.0,
                grader_timeout_sec=5.0,
            )
        )
        controller.bridge = FakeBridge()
        controller.server.bridge = controller.bridge
        await controller.start()
        try:
            outcome = await controller.run(
                messages=[{"role": "user", "content": "hi"}], label="answer"
            )
        finally:
            await controller.stop()

    assert outcome.status is RolloutStatus.SUCCESS
    assert outcome.rollout_id == received["rollout_id"]
    chat_completions_url = urlparse(received["chat_completions_url"])
    assert chat_completions_url.scheme == "http"
    assert chat_completions_url.hostname == "127.0.0.1"
    assert chat_completions_url.port is not None
    assert chat_completions_url.netloc == f"127.0.0.1:{chat_completions_url.port}"
    assert chat_completions_url.path == ""
    assert chat_completions_url.params == ""
    assert chat_completions_url.query == ""
    assert chat_completions_url.fragment == ""
    assert not received["chat_completions_url"].endswith("/v1")
    assert received["completion_callback_url"].endswith("/v1/rollout/completed")
    assert received["grader_callback_url"].endswith("/v1/grader/completed")


async def _run_fake_rollout_scenario(
    tmp_path: Path,
    *,
    samples_payload: dict | None,
    post_grader: bool = True,
    call_chat: bool = True,
    agent_timeout_sec: float = 1.0,
    grader_timeout_sec: float = 1.0,
):
    app = FastAPI()
    recorder = {
        "chat_sample_ids": [],
        "rollout_completion_posted": False,
        "grader_completion_posted": False,
    }

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/rollout")
    async def rollout(payload: dict, background_tasks: BackgroundTasks):
        async def run_callbacks():
            headers = {"Authorization": f"Bearer {payload['controller_api_key']}"}
            async with httpx.AsyncClient() as client:
                if call_chat:
                    stream_response = await client.post(
                        f"{payload['chat_completions_url']}/chat/completions",
                        headers={
                            **headers,
                            "x-rollout-id": payload["rollout_id"],
                            "x-sample-id": "s1",
                        },
                        json={
                            "model": "openai/osmosis-rollout",
                            "messages": [{"role": "user", "content": "hi"}],
                        },
                        timeout=5.0,
                    )
                    assert stream_response.status_code == 200
                    recorder["chat_sample_ids"].append("s1")
                rollout_response = await client.post(
                    payload["completion_callback_url"],
                    headers=headers,
                    json={"rollout_id": payload["rollout_id"], "status": "success"},
                    timeout=5.0,
                )
                assert rollout_response.status_code == 200
                recorder["rollout_completion_posted"] = True
                if post_grader:
                    grader_response = await client.post(
                        payload["grader_callback_url"],
                        headers=headers,
                        json={
                            "rollout_id": payload["rollout_id"],
                            "status": "success",
                            "samples": samples_payload or {},
                        },
                        timeout=5.0,
                    )
                    assert grader_response.status_code == 200
                    recorder["grader_completion_posted"] = True

        background_tasks.add_task(run_callbacks)
        return {}

    async with _fake_rollout_server(app):
        controller = EvalController(
            config=EvalControllerConfig(
                project_root=tmp_path,
                rollout_name="demo",
                rollout_dir=tmp_path,
                entrypoint="main.py",
                llm_model="openai/gpt-5-mini",
                api_key=None,
                base_url=None,
                agent_timeout_sec=agent_timeout_sec,
                grader_timeout_sec=grader_timeout_sec,
            )
        )
        controller.bridge = FakeBridge()
        controller.server.bridge = controller.bridge
        await controller.start()
        try:
            outcome = await controller.run(
                messages=[{"role": "user", "content": "hi"}],
                label="answer",
            )
        finally:
            await controller.stop()
    return outcome, recorder


@pytest.mark.asyncio
async def test_missing_grader_callback_times_out(tmp_path: Path) -> None:
    outcome, recorder = await _run_fake_rollout_scenario(
        tmp_path,
        samples_payload=None,
        post_grader=False,
        agent_timeout_sec=1.0,
        grader_timeout_sec=0.2,
    )

    assert outcome.status is RolloutStatus.FAILURE
    assert outcome.error and "Timed out waiting for grader callback" in outcome.error
    assert recorder["chat_sample_ids"] == ["s1"]
    assert recorder["rollout_completion_posted"] is True
    assert recorder["grader_completion_posted"] is False


@pytest.mark.asyncio
async def test_missing_grader_rewards_fail(tmp_path: Path) -> None:
    outcome, _ = await _run_fake_rollout_scenario(tmp_path, samples_payload={})

    assert outcome.status is RolloutStatus.FAILURE
    assert outcome.error and "Missing grader rewards" in outcome.error


@pytest.mark.asyncio
async def test_unknown_grader_rewards_fail(tmp_path: Path) -> None:
    outcome, _ = await _run_fake_rollout_scenario(
        tmp_path,
        samples_payload={"s2": RolloutSample(id="s2", reward=1.0).model_dump()},
    )

    assert outcome.status is RolloutStatus.FAILURE
    assert outcome.error and "not created by the controller" in outcome.error


@pytest.mark.asyncio
async def test_none_grader_reward_fails(tmp_path: Path) -> None:
    outcome, _ = await _run_fake_rollout_scenario(
        tmp_path,
        samples_payload={"s1": RolloutSample(id="s1", reward=None).model_dump()},
    )

    assert outcome.status is RolloutStatus.FAILURE
    assert outcome.error and "returned None as reward" in outcome.error
