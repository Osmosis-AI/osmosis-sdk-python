from __future__ import annotations

import asyncio
import socket
from contextlib import suppress
from typing import Any
from urllib.parse import quote

from osmosis_ai._imports import raise_optional_dependency_error

try:
    import uvicorn
    from fastapi import FastAPI
except ModuleNotFoundError as _exc:
    raise_optional_dependency_error(
        _exc,
        extra="eval",
        feature="Local evaluation",
    )

from osmosis_ai.eval.local.llm_bridge import (
    LiteLLMBridge,
    create_bridge_router,
)

_START_POLL_ATTEMPTS = 500
_START_POLL_INTERVAL_SEC = 0.01


class LocalhostUvicornServer:
    """Bind a socket first, then serve. Never pick-free-close-bind."""

    def __init__(self, app: Any, *, port: int | None = None) -> None:
        self._app = app
        self._port = port
        self._socket: socket.socket | None = None
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None
        self._base_url: str | None = None

    @property
    def base_url(self) -> str:
        if self._base_url is None:
            raise RuntimeError("localhost server has not started")
        return self._base_url

    async def start(self) -> str:
        if self._task is not None:
            return self.base_url
        sock: socket.socket | None = None
        try:
            sock = self._reserve_socket(self._port)
            self._socket = sock
            host, port = sock.getsockname()[:2]
            self._base_url = f"http://{host}:{port}"
            config = uvicorn.Config(
                self._app,
                host=host,
                port=port,
                log_level="warning",
                lifespan="off",
                ws="none",
            )
            server = uvicorn.Server(config)
            self._server = server
            self._task = asyncio.create_task(server.serve(sockets=[sock]))
            for _ in range(_START_POLL_ATTEMPTS):
                if server.started:
                    return self._base_url
                if self._task.done():
                    await self._task
                    raise RuntimeError("localhost server exited before becoming ready")
                await asyncio.sleep(_START_POLL_INTERVAL_SEC)
            raise RuntimeError("localhost server failed to start")
        except BaseException:
            await self._reset(sock)
            raise

    async def stop(self) -> None:
        try:
            if self._server is not None:
                self._server.should_exit = True
            if self._task is not None:
                await self._task
        finally:
            await self._reset()

    async def __aenter__(self) -> LocalhostUvicornServer:
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.stop()

    async def _reset(self, sock: socket.socket | None = None) -> None:
        task = self._task
        server = self._server
        to_close = self._socket if self._socket is not None else sock
        self._task = None
        self._server = None
        self._socket = None
        self._base_url = None
        if server is not None:
            server.should_exit = True
        if task is not None:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
        if to_close is not None:
            with suppress(OSError):
                to_close.close()

    @staticmethod
    def _reserve_socket(port: int | None = None) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port or 0))
        except OSError:
            sock.close()
            raise
        return sock


def create_llm_bridge_app(
    bridge: LiteLLMBridge,
    *,
    auth_token: str,
    keepalive: bool = False,
) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.include_router(
        create_bridge_router(
            bridge,
            auth_token=auth_token,
            non_stream_keepalive=keepalive,
        )
    )
    return app


class LlmBridgeListener:
    def __init__(
        self,
        bridge: LiteLLMBridge,
        *,
        auth_token: str,
        port: int | None = None,
        advertised_base_url: str | None = None,
        bridge_keepalive: bool = False,
    ) -> None:
        # Writable after start: an auto-managed tunnel learns its public URL
        # only once the listener is already serving.
        self.advertised_base_url = advertised_base_url
        self._server = LocalhostUvicornServer(
            create_llm_bridge_app(
                bridge,
                auth_token=auth_token,
                keepalive=bridge_keepalive,
            ),
            port=port,
        )

    @property
    def base_url(self) -> str:
        return self._server.base_url

    def chat_completions_url(self, rollout_id: str) -> str:
        """Per-rollout OpenAI-compatible api_base served by the mounted bridge.

        Clients append ``/chat/completions``; the URL itself must not include
        that suffix. This is the only URL a sandbox ever dials, so it is the
        only one ``advertised_base_url`` replaces.
        """
        base = (self.advertised_base_url or self.base_url).rstrip("/")
        encoded = quote(rollout_id, safe="")
        return f"{base}/v1/rollouts/{encoded}"

    async def start(self) -> str:
        return await self._server.start()

    async def stop(self) -> None:
        await self._server.stop()

    async def __aenter__(self) -> LlmBridgeListener:
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.stop()


__all__ = ["LlmBridgeListener", "LocalhostUvicornServer", "create_llm_bridge_app"]
