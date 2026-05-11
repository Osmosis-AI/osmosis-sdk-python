from __future__ import annotations

import os
import socket
import tempfile
from pathlib import Path

from filelock import FileLock

USER_SERVER_HOST = "127.0.0.1"
USER_SERVER_PORT = 8000


def assert_user_server_port_free(
    host: str = USER_SERVER_HOST, port: int = USER_SERVER_PORT
) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        if sock.connect_ex((host, port)) == 0:
            raise RuntimeError(f"User server port is already occupied: {host}:{port}")


def fixed_port_lock_path() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir())
        return root / "OsmosisAI" / "locks" / f"user-server-{USER_SERVER_PORT}.lock"

    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        root = Path(runtime_dir) / "osmosis-ai"
    else:
        uid = os.getuid() if hasattr(os, "getuid") else os.environ.get("USER", "user")
        root = Path(tempfile.gettempdir()) / f"osmosis-ai-{uid}"
    return root / "locks" / f"user-server-{USER_SERVER_PORT}.lock"


class FixedPortLock:
    def __init__(self, *, timeout: int = 30, lock_path: Path | None = None) -> None:
        self.lock_path: Path = lock_path or fixed_port_lock_path()
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(self.lock_path), timeout=timeout)

    def acquire(self) -> None:
        self._lock.acquire()

    def release(self) -> None:
        self._lock.release()


__all__ = [
    "USER_SERVER_HOST",
    "USER_SERVER_PORT",
    "FixedPortLock",
    "assert_user_server_port_free",
    "fixed_port_lock_path",
]
