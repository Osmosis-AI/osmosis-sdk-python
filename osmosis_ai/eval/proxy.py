"""EvalProxy — local OpenAI-compatible HTTP proxy with per-request metrics.

Sits between the agent workflow and the ExternalLLMClient, forwarding
``/v1/chat/completions`` requests while tracking token usage, latency,
and call counts keyed by ``rollout_id``.  Optionally writes per-request
JSONL traces for debugging.
"""

import asyncio
import contextlib
import json
import logging
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Per-request metrics collected at the proxy level."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    num_calls: int = 0
    total_latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class EvalProxy:
    """Local OpenAI-compatible HTTP proxy with eval instrumentation.

    Responsibilities:
    - Forward ``/v1/chat/completions`` to an :class:`ExternalLLMClient`
    - Track per-request metrics keyed by ``rollout_id`` (concurrent-safe)
    - Write per-request JSONL traces (optional, when *trace_dir* is set)
    - Translate provider errors to eval error types

    Parameters
    ----------
    client:
        An ``ExternalLLMClient`` (or any object with an async
        ``chat_completions(messages, **kwargs)`` returning a dict).
    trace_dir:
        If set, JSONL trace files are written under this directory,
        one file per *rollout_id*.
    """

    def __init__(self, client: Any, trace_dir: str | None = None) -> None:
        self.client = client
        self.trace_dir: Path | None = Path(trace_dir) if trace_dir else None

        # Per-rollout metric accumulators.  Concurrent-safe because asyncio
        # is single-threaded and there are no await points between reading
        # and mutating a metrics entry in _handle_chat_completions.
        self._metrics: dict[str, RequestMetrics] = {}

        self._server_task: asyncio.Task[None] | None = None
        self._server: Any = None
        self._port: int | None = None
        self.systemic_error: str | None = None

    @property
    def url(self) -> str:
        """Base URL of the running proxy (e.g. ``http://127.0.0.1:8321``)."""
        if self._port is None:
            raise RuntimeError("EvalProxy has not been started yet")
        return f"http://127.0.0.1:{self._port}/v1"

    async def start(self) -> None:
        """Start the proxy server in a background asyncio task."""
        if self._server_task is not None:
            raise RuntimeError("EvalProxy is already running")

        import uvicorn

        app = self._build_app()
        self._port = _find_free_port()

        config = uvicorn.Config(
            app=app,
            host="127.0.0.1",
            port=self._port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        self._server = server

        self._server_task = asyncio.create_task(server.serve())

        # Wait for the server socket to be ready.
        try:
            await _wait_for_port("127.0.0.1", self._port, timeout=10.0)
        except Exception:
            self._server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._server_task
            self._server_task = None
            self._port = None
            raise
        logger.info("EvalProxy listening on %s", self.url)

    async def stop(self) -> None:
        """Shutdown the proxy server gracefully."""
        if self._server is not None:
            self._server.should_exit = True
        if self._server_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._server_task
            self._server_task = None
        self._server = None
        self._port = None
        self._metrics.clear()

    async def _handle_chat_completions(
        self,
        rollout_id: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """Forward a chat-completions request and track metrics."""
        metrics = self._metrics.setdefault(rollout_id, RequestMetrics())

        messages = request.get("messages", [])
        # Only forward semantic keys that affect response content.
        # Sampling params (temperature, top_p, …) come from the Strands SDK
        # defaults and may be unsupported by the real model.  The
        # ExternalLLMClient owns the model and its own defaults.
        _FORWARD_KEYS = {"tools", "tool_choice"}
        params = {k: v for k, v in request.items() if k in _FORWARD_KEYS}

        from osmosis_ai.eval.common.errors import SystemicProviderError

        start = time.monotonic()
        try:
            response = await self.client.chat_completions(messages=messages, **params)
        except SystemicProviderError as e:
            self.systemic_error = str(e)
            raise
        elapsed_ms = (time.monotonic() - start) * 1000

        usage = response.get("usage", {})
        metrics.prompt_tokens += usage.get("prompt_tokens", 0)
        metrics.completion_tokens += usage.get("completion_tokens", 0)
        metrics.num_calls += 1
        metrics.total_latency_ms += elapsed_ms

        if self.trace_dir is not None:
            await asyncio.to_thread(
                self._write_trace, rollout_id, request, response, elapsed_ms
            )

        return response

    def collect_metrics(self, rollout_id: str) -> RequestMetrics:
        """Pop and return accumulated metrics for *rollout_id*.

        If no metrics have been recorded for the given id an empty
        :class:`RequestMetrics` instance (all zeros) is returned.
        """
        return self._metrics.pop(rollout_id, RequestMetrics())

    def _write_trace(
        self,
        rollout_id: str,
        request: dict[str, Any],
        response: dict[str, Any],
        latency_ms: float,
    ) -> None:
        """Append a JSONL line to ``trace_dir/{rollout_id}.jsonl``."""
        assert self.trace_dir is not None
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.trace_dir / f"{rollout_id}.jsonl"
        line = json.dumps(
            {
                "ts": time.time(),
                "type": "llm_call",
                "request": request,
                "response": response,
                "latency_ms": round(latency_ms, 2),
            },
            ensure_ascii=False,
        )
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def _build_app(self) -> Any:
        """Construct the FastAPI application with a single endpoint."""
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse

        app = FastAPI(title="EvalProxy", docs_url=None, redoc_url=None)

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request) -> JSONResponse:
            rollout_id = request.headers.get("x-rollout-id", "unknown")
            body = await request.json()

            try:
                result = await self._handle_chat_completions(rollout_id, body)
            except Exception as exc:
                logger.exception("EvalProxy error for rollout_id=%s", rollout_id)
                return JSONResponse(
                    status_code=502,
                    content={"error": str(exc)},
                )
            return JSONResponse(content=result)

        return app


def _find_free_port() -> int:
    """Bind to port 0 and return the OS-assigned ephemeral port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _wait_for_port(
    host: str,
    port: int,
    timeout: float = 10.0,
    interval: float = 0.05,
) -> None:
    """Poll until a TCP connection to *host*:*port* succeeds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=0.2
            )
            writer.close()
            await writer.wait_closed()
            return
        except (OSError, asyncio.TimeoutError):
            await asyncio.sleep(interval)
    raise TimeoutError(f"EvalProxy did not start within {timeout}s")


__all__ = ["EvalProxy", "RequestMetrics"]
