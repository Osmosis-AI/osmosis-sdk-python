"""Launch a rollout entrypoint with supervisor-owned health metadata.

The rollout project resolves its own SDK version, which may predate the
``instance_id`` field in ``create_rollout_server``. Patching FastAPI at the ASGI
boundary keeps the handshake independent of that SDK version: only the child
process the supervisor launched can echo the random marker and its installed
SDK version.
"""

from __future__ import annotations

import os
import runpy
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from starlette.types import Message, Receive, Scope, Send

ROLLOUT_INSTANCE_HEADER = "x-osmosis-rollout-instance-id"
ROLLOUT_SDK_VERSION_HEADER = "x-osmosis-rollout-sdk-version"
_INSTANCE_ENV = "_OSMOSIS_ROLLOUT_INSTANCE_ID"


def _install_health_marker(instance_id: str, sdk_version: str) -> None:
    from fastapi import FastAPI

    original_call = FastAPI.__call__
    health_headers = (
        (ROLLOUT_INSTANCE_HEADER.encode("ascii"), instance_id.encode("ascii")),
        (
            ROLLOUT_SDK_VERSION_HEADER.encode("ascii"),
            sdk_version.encode("ascii"),
        ),
    )
    header_names = {name for name, _value in health_headers}

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
                    if name.lower() not in header_names
                ]
                headers.extend(health_headers)
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

    try:
        sdk_version = version("osmosis-ai")
    except PackageNotFoundError:
        sdk_version = "unknown"
    _install_health_marker(instance_id, sdk_version)
    sys.argv = [str(entrypoint)]
    sys.path.insert(0, str(entrypoint.parent))
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
