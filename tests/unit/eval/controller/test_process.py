import inspect
import os
import socket
import subprocess
from pathlib import Path

import httpx
import pytest

import osmosis_ai.eval.controller.process as process_mod
from osmosis_ai.eval.controller.locks import (
    FixedPortLock,
    assert_user_server_port_free,
    fixed_port_lock_path,
)
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
    workspace_directory = tmp_path / "project"
    rollout_dir = tmp_path / "project" / "rollouts" / "demo"
    monkeypatch.setenv("PYTHONPATH", "/existing")
    monkeypatch.setenv("KEEP_ME", "yes")

    env = build_user_server_env(
        workspace_directory=workspace_directory,
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
        str(workspace_directory),
        "/existing",
    ]


def test_fixed_port_lock_accepts_timeout_and_custom_path(tmp_path: Path) -> None:
    lock = FixedPortLock(timeout=1, lock_path=tmp_path / "port.lock")

    lock.acquire()
    lock.release()


def test_fixed_port_lock_path_is_shared_across_projects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    for project in (project_a, project_b):
        subprocess.run(
            ["git", "init", "-b", "main", str(project)],
            check=True,
            capture_output=True,
        )
        for rel_path in (
            ".osmosis/research",
            "rollouts",
            "configs/eval",
            "configs/training",
            "data",
        ):
            (project / rel_path).mkdir(parents=True, exist_ok=True)

    monkeypatch.chdir(project_a)
    lock_a = fixed_port_lock_path()
    monkeypatch.chdir(project_b)
    lock_b = fixed_port_lock_path()

    assert lock_a == lock_b
    assert project_a.resolve() not in lock_a.parents
    assert project_b.resolve() not in lock_a.parents
    assert lock_a.name == "user-server-8000.lock"


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


@pytest.mark.asyncio
async def test_wait_for_user_server_health_rejects_healthy_response_after_process_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        returncode = None

    fake_process = FakeProcess()

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, timeout):
            fake_process.returncode = 2
            return httpx.Response(200)

    monkeypatch.setattr(process_mod.httpx, "AsyncClient", lambda: FakeClient())

    user_server = UserServerProcess(
        process=fake_process,  # type: ignore[arg-type]
        log_path=Path("server.log"),
    )
    with pytest.raises(RuntimeError, match="exited before becoming healthy"):
        await wait_for_user_server_health(timeout_sec=1.0, process=user_server)


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
