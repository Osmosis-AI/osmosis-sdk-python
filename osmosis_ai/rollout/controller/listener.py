"""Localhost-only callback listener around :class:`CallbackStore`.

The listener authenticates bearer tokens, checks that the URL-scoped rollout
id matches the body, accepts only terminal callback statuses, and returns the
store's acknowledgment only after a terminal commit hook has finished. It
always binds ``127.0.0.1``; the port is ephemeral by default and may be fixed
so an externally-run tunnel can point at it.

When local eval provides a :class:`~osmosis_ai.rollout.controller.llm_bridge.LiteLLMBridge`,
its chat-completions routes mount on the same app under their own bearer —
one process, one port, two credentials, mirroring the hosted eval controller.
An ``advertised_base_url`` (a tunnel's public URL) replaces the loopback base
in ``chat_completions_url()`` only: completion and grader callbacks come from
the rollout server, a host process, and stay on loopback.
"""

from __future__ import annotations

import asyncio
import secrets
import socket
from contextlib import suppress
from typing import Any
from urllib.parse import quote

from osmosis_ai._imports import raise_optional_dependency_error

try:
    import uvicorn
    from fastapi import Depends, FastAPI, HTTPException, Request
except ModuleNotFoundError as _exc:
    raise_optional_dependency_error(
        _exc,
        extra="eval",
        feature="Local evaluation",
    )

from osmosis_ai.rollout.controller.llm_bridge import (
    LiteLLMBridge,
    create_bridge_router,
)
from osmosis_ai.rollout.controller.store import CallbackStore, UnknownRolloutIdError
from osmosis_ai.rollout.types import (
    GraderCompleteRequest,
    GraderStatus,
    RolloutCompleteRequest,
    RolloutStatus,
)
from osmosis_ai.rollout.utils.identifiers import is_single_path_segment

_START_POLL_ATTEMPTS = 500
_START_POLL_INTERVAL_SEC = 0.01

# Only terminal statuses may cross the callback boundary; in-flight states
# must never be able to win a rollout's terminal result.
_TERMINAL_COMPLETION_STATUSES = frozenset(
    {RolloutStatus.SUCCESS, RolloutStatus.FAILURE, RolloutStatus.CANCELLED}
)
_TERMINAL_GRADER_STATUSES = frozenset({GraderStatus.SUCCESS, GraderStatus.FAILURE})


def _assert_non_empty_auth_token(auth_token: str) -> None:
    if not auth_token or not auth_token.strip():
        raise ValueError("callback auth_token must be a non-empty string")


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


def create_callback_app(store: CallbackStore, *, auth_token: str) -> FastAPI:
    """Build the FastAPI app that receives completion and grader callbacks."""
    _assert_non_empty_auth_token(auth_token)
    # No docs/openapi surface: a tunnel can expose this app to the internet,
    # and FastAPI's defaults would hand out the full route list and callback
    # body schemas unauthenticated.
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    async def require_auth(request: Request) -> None:
        header = request.headers.get("Authorization")
        scheme, _, credentials = (header or "").partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(
            credentials.encode(), auth_token.encode()
        ):
            raise HTTPException(status_code=401, detail="Unauthorized")

    def _rollout_id(path_id: str, body_id: str | None) -> str:
        if not is_single_path_segment(path_id):
            raise HTTPException(status_code=422, detail="invalid rollout_id")
        if body_id is not None and body_id != path_id:
            raise HTTPException(status_code=422, detail="rollout_id mismatch")
        return path_id

    @app.post(
        "/v1/rollouts/{rollout_id}/completion",
        dependencies=[Depends(require_auth)],
    )
    async def completion(
        rollout_id: str, request: RolloutCompleteRequest
    ) -> dict[str, Any]:
        _rollout_id(rollout_id, request.rollout_id)
        if request.status not in _TERMINAL_COMPLETION_STATUSES:
            raise HTTPException(
                status_code=422,
                detail="completion callback status must be terminal",
            )
        try:
            return await store.handle_completion(rollout_id, request)
        except UnknownRolloutIdError as exc:
            raise HTTPException(status_code=404, detail="unknown rollout_id") from exc

    @app.post(
        "/v1/rollouts/{rollout_id}/grader",
        dependencies=[Depends(require_auth)],
    )
    async def grader(rollout_id: str, request: GraderCompleteRequest) -> dict[str, Any]:
        _rollout_id(rollout_id, request.rollout_id)
        if request.status not in _TERMINAL_GRADER_STATUSES:
            raise HTTPException(
                status_code=422,
                detail="grader callback status must be terminal",
            )
        try:
            return await store.handle_grader(rollout_id, request)
        except UnknownRolloutIdError as exc:
            raise HTTPException(status_code=404, detail="unknown rollout_id") from exc

    return app


class CallbackListener:
    """Owns a reserved localhost socket and serves :func:`create_callback_app`."""

    def __init__(
        self,
        store: CallbackStore,
        *,
        auth_token: str,
        bridge: LiteLLMBridge | None = None,
        bridge_token: str | None = None,
        port: int | None = None,
        advertised_base_url: str | None = None,
        bridge_keepalive: bool = False,
    ) -> None:
        _assert_non_empty_auth_token(auth_token)
        app = create_callback_app(store, auth_token=auth_token)
        if bridge is not None:
            if bridge_token is None:
                raise ValueError("bridge_token is required when a bridge is provided")
            app.include_router(
                create_bridge_router(
                    bridge,
                    auth_token=bridge_token,
                    non_stream_keepalive=bridge_keepalive,
                )
            )
        # Writable after start: an auto-managed tunnel learns its public URL
        # only once the listener is already serving.
        self.advertised_base_url = advertised_base_url
        self._server = LocalhostUvicornServer(app, port=port)

    @property
    def base_url(self) -> str:
        return self._server.base_url

    def completion_url(self, rollout_id: str) -> str:
        encoded = quote(rollout_id, safe="")
        return f"{self.base_url}/v1/rollouts/{encoded}/completion"

    def grader_url(self, rollout_id: str) -> str:
        encoded = quote(rollout_id, safe="")
        return f"{self.base_url}/v1/rollouts/{encoded}/grader"

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

    async def __aenter__(self) -> CallbackListener:
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.stop()
