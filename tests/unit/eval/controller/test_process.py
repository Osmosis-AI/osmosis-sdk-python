import inspect
import os
import socket
from pathlib import Path

import httpx
import pytest

import osmosis_ai.eval.controller.process as process_mod
from osmosis_ai.eval.controller.locks import FixedPortLock, assert_user_server_port_free
from osmosis_ai.eval.controller.process import (
    UserServerProcess,
    build_user_server_command,
    build_user_server_env,
    start_user_server_process,
    wait_for_user_server_health,
)


def test_assert_user_server_port_free_raises_for_occupied_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        host, port = server.getsockname()

        with pytest.raises(RuntimeError, match=f"{host}:{port}"):
            assert_user_server_port_free(host=host, port=port)


def test_build_user_server_command_uses_uv_python_entrypoint() -> None:
    assert build_user_server_command("rollout.py") == [
        "uv",
        "run",
        "python",
        "rollout.py",
    ]


def test_build_user_server_env_sets_contract_and_pythonpath(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project_root = tmp_path / "project"
    rollout_dir = tmp_path / "project" / "rollouts" / "demo"
    monkeypatch.setenv("PYTHONPATH", "/existing")
    monkeypatch.setenv("KEEP_ME", "yes")

    env = build_user_server_env(
        project_root=project_root,
        rollout_dir=rollout_dir,
        rollout_name="demo",
        entrypoint="rollout.py",
        invocation_id="inv-1",
    )

    assert env["ENTRYPOINT_SCRIPT"] == "rollout.py"
    assert env["REPOSITORY_PATH"] == str(rollout_dir)
    assert env["TRAINING_RUN_ID"] == "inv-1"
    assert env["ROLLOUT_NAME"] == "demo"
    assert env["ROLLOUT_PORT"] == "8000"
    assert env["KEEP_ME"] == "yes"
    assert env["PYTHONPATH"].split(os.pathsep) == [
        str(rollout_dir),
        str(project_root),
        "/existing",
    ]


def test_fixed_port_lock_accepts_timeout_and_custom_path(tmp_path: Path) -> None:
    lock = FixedPortLock(timeout=1, lock_path=tmp_path / "port.lock")

    lock.acquire()
    lock.release()


def test_process_lifecycle_helpers_are_async() -> None:
    assert inspect.iscoroutinefunction(start_user_server_process)
    assert inspect.iscoroutinefunction(wait_for_user_server_health)


@pytest.mark.asyncio
async def test_wait_for_user_server_health_requires_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, timeout):
            return httpx.Response(404)

    monkeypatch.setattr(process_mod.httpx, "AsyncClient", lambda: FakeClient())
    monkeypatch.setattr(process_mod.asyncio, "sleep", lambda _: _instant_sleep())

    with pytest.raises(TimeoutError, match=r"127\.0\.0\.1:8000"):
        await wait_for_user_server_health(timeout_sec=0.01)


async def _instant_sleep():
    return None


@pytest.mark.asyncio
async def test_user_server_process_terminate_suppresses_process_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        pid = 123
        returncode = None

        async def wait(self):
            self.returncode = 0

    process = UserServerProcess(
        process=FakeProcess(),  # type: ignore[arg-type]
        log_path=Path("server.log"),
    )
    monkeypatch.setattr(process_mod.sys, "platform", "darwin")
    monkeypatch.setattr(
        process_mod.os,
        "killpg",
        lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError()),
    )

    await process.terminate()
