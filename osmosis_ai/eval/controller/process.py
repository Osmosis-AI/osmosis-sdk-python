from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

USER_SERVER_PORT = 8000


def build_user_server_command(entrypoint: str) -> list[str]:
    return ["uv", "run", "python", entrypoint]


def build_user_server_env(
    *,
    project_root: Path,
    rollout_dir: Path,
    rollout_name: str,
    entrypoint: str,
    invocation_id: str,
) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    path_parts = [str(rollout_dir), str(project_root)]
    if existing_pythonpath:
        path_parts.append(existing_pythonpath)
    env.update(
        {
            "ENTRYPOINT_SCRIPT": entrypoint,
            "REPOSITORY_PATH": str(rollout_dir),
            "TRAINING_RUN_ID": invocation_id,
            "ROLLOUT_NAME": rollout_name,
            "ROLLOUT_PORT": str(USER_SERVER_PORT),
            "PYTHONPATH": os.pathsep.join(path_parts),
        }
    )
    return env


@dataclass
class UserServerProcess:
    process: asyncio.subprocess.Process
    log_path: Path

    async def terminate(self, *, grace_sec: float = 5.0) -> None:
        if self.process.returncode is not None:
            return
        try:
            if sys.platform != "win32":
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGTERM)
            else:
                with contextlib.suppress(ProcessLookupError):
                    self.process.terminate()
            await asyncio.wait_for(self.process.wait(), timeout=grace_sec)
        except TimeoutError:
            if sys.platform != "win32":
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
            else:
                with contextlib.suppress(ProcessLookupError):
                    self.process.kill()
            await self.process.wait()


async def start_user_server_process(
    *,
    project_root: Path,
    rollout_dir: Path,
    rollout_name: str,
    entrypoint: str,
    invocation_id: str,
    log_dir: Path,
) -> UserServerProcess:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"user-server-{invocation_id}.log"
    kwargs = {}
    if sys.platform != "win32":
        kwargs["start_new_session"] = True

    log_file = log_path.open("ab")
    try:
        process = await asyncio.create_subprocess_exec(
            *build_user_server_command(entrypoint),
            cwd=rollout_dir,
            env=build_user_server_env(
                project_root=project_root,
                rollout_dir=rollout_dir,
                rollout_name=rollout_name,
                entrypoint=entrypoint,
                invocation_id=invocation_id,
            ),
            stdout=log_file,
            stderr=asyncio.subprocess.STDOUT,
            **kwargs,
        )
    except Exception:
        log_file.close()
        raise
    else:
        log_file.close()
    return UserServerProcess(process=process, log_path=log_path)


async def wait_for_user_server_health(*, timeout_sec: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_sec
    last_error: Exception | None = None
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(
                    f"http://127.0.0.1:{USER_SERVER_PORT}/health",
                    timeout=0.5,
                )
                if response.status_code == 200:
                    return
            except Exception as exc:
                last_error = exc
            await asyncio.sleep(0.1)
    raise TimeoutError(
        "User rollout server did not become healthy on "
        f"127.0.0.1:{USER_SERVER_PORT} within {timeout_sec:.1f}s: {last_error}"
    )


__all__ = [
    "USER_SERVER_PORT",
    "UserServerProcess",
    "build_user_server_command",
    "build_user_server_env",
    "start_user_server_process",
    "wait_for_user_server_health",
]
