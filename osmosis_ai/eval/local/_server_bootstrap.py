"""Launch a rollout entrypoint with a supervisor-owned health marker.

The rollout project resolves its own SDK version, which may predate the
``instance_id`` field in ``create_rollout_server``. Patching FastAPI at the ASGI
boundary keeps the ownership handshake independent of that SDK version: only
the child process the supervisor launched can echo the random marker.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

from starlette.types import Message, Receive, Scope, Send

ROLLOUT_INSTANCE_HEADER = "x-osmosis-rollout-instance-id"
_INSTANCE_ENV = "_OSMOSIS_ROLLOUT_INSTANCE_ID"


def _install_health_marker(instance_id: str) -> None:
    from fastapi import FastAPI

    original_call = FastAPI.__call__
    header_name = ROLLOUT_INSTANCE_HEADER.encode("ascii")
    header_value = instance_id.encode("ascii")

    async def call_with_health_marker(
        self: FastAPI, scope: Scope, receive: Receive, send: Send
    ) -> None:
        if scope.get("type") != "http" or scope.get("path") != "/health":
            await original_call(self, scope, receive, send)
            return

        async def send_with_health_marker(message: Message) -> None:
            if message.get("type") == "http.response.start":
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() != header_name
                ]
                headers.append((header_name, header_value))
                message = {**message, "headers": headers}
            await send(message)

        await original_call(self, scope, receive, send_with_health_marker)

    FastAPI.__call__ = call_with_health_marker


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: _server_bootstrap.py <rollout-entrypoint>")
    entrypoint = Path(sys.argv[1]).resolve()
    if not entrypoint.is_file():
        raise SystemExit(f"rollout entrypoint does not exist: {entrypoint}")
    instance_id = os.environ.get(_INSTANCE_ENV)
    if not instance_id:
        raise SystemExit(f"{_INSTANCE_ENV} is required")

    _install_health_marker(instance_id)
    sys.argv = [str(entrypoint)]
    sys.path.insert(0, str(entrypoint.parent))
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
