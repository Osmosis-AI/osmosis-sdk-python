"""LiteLLMProxy — lightweight OpenAI-compatible proxy with token counting.

Responsibilities:
1. Forward /v1/chat/completions to the real LLM via litellm
2. Count tokens per rollout_id from response usage
3. Detect systemic LLM errors (auth, quota, model-not-found)
4. (Optional) Write per-request JSONL traces when trace_dir is set (--debug)
"""

import asyncio
import contextlib
import json
import logging
import socket
import time
from pathlib import Path
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)


class LiteLLMProxy:
    """Local LLM proxy with per-rollout token counting and systemic error detection."""

    SYSTEMIC_EXCEPTIONS = (
        "AuthenticationError",
        "BudgetExceededError",
        "NotFoundError",
    )

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        trace_dir: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.trace_dir: Path | None = Path(trace_dir) if trace_dir else None
        self._tokens: dict[str, int] = {}
        self._systemic_errors: dict[str, str] = {}
        self._port: int | None = None
        self._server_task: asyncio.Task[None] | None = None
        self._server: Any = None

    @property
    def url(self) -> str:
        if self._port is None:
            raise RuntimeError("LiteLLMProxy has not been started yet")
        return f"http://127.0.0.1:{self._port}/v1"

    def collect_tokens(self, rollout_id: str) -> int:
        return self._tokens.pop(rollout_id, 0)

    def collect_systemic_error(self, rollout_id: str) -> str | None:
        return self._systemic_errors.pop(rollout_id, None)

    async def start(self) -> str:
        """Start proxy and return the base URL."""
        if self._server_task is not None:
            raise RuntimeError("LiteLLMProxy is already running")

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

        try:
            await _wait_for_port("127.0.0.1", self._port, timeout=10.0)
        except Exception:
            self._server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._server_task
            self._server_task = None
            self._port = None
            raise

        logger.info("LiteLLMProxy listening on %s", self.url)
        return self.url

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._server_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._server_task
            self._server_task = None
        self._server = None
        self._port = None
        self._tokens.clear()
        self._systemic_errors.clear()

    async def preflight_check(self) -> None:
        """Send a minimal 1-token request to verify LLM connectivity."""
        from osmosis_ai.errors import CLIError

        litellm = _get_litellm()

        try:
            litellm.get_llm_provider(model=self.model, api_base=self.base_url)
        except Exception as exc:
            msg = getattr(exc, "message", str(exc))
            raise CLIError(
                f"Invalid LiteLLM model format. Use 'provider/model' "
                f"(e.g. openai/gpt-5-mini). Received: '{self.model}'. Details: {msg}"
            ) from exc

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url

        try:
            await litellm.acompletion(**kwargs)
        except Exception as e:
            ename = type(e).__name__
            if ename == "RateLimitError":
                return
            msg = getattr(e, "message", str(e))
            if ename in self.SYSTEMIC_EXCEPTIONS or ename == "APIConnectionError":
                raise CLIError(f"LLM preflight check failed: {msg}") from e
            logger.debug("Preflight non-fatal error: %s", e)

    # Transport & credential keys are always proxy-controlled, never from request body.
    _PROXY_CONTROLLED_KEYS = frozenset(
        {
            "model",
            "messages",
            "stream",
            "stream_options",
            "api_key",
            "api_base",
            "base_url",
            "organization",
            "api_version",
            "custom_llm_provider",
        }
    )

    def _build_litellm_kwargs(self, body: dict) -> dict[str, Any]:
        """Build kwargs for litellm.acompletion from the incoming request body."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": body.get("messages", []),
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url

        for key, value in body.items():
            if key not in self._PROXY_CONTROLLED_KEYS and key not in kwargs:
                kwargs[key] = value
        return kwargs

    async def _handle_request(self, rollout_id: str, body: dict) -> dict:
        """Forward non-streaming request to LLM, accumulate token usage."""
        litellm = _get_litellm()
        kwargs = self._build_litellm_kwargs(body)

        start = time.monotonic()
        try:
            response: Any = await litellm.acompletion(**kwargs)
        except Exception as e:
            if type(e).__name__ in self.SYSTEMIC_EXCEPTIONS:
                self._systemic_errors[rollout_id] = str(e)
            raise
        elapsed_ms = (time.monotonic() - start) * 1000

        usage = response.usage
        total = (usage.total_tokens if usage else 0) or 0
        self._tokens[rollout_id] = self._tokens.get(rollout_id, 0) + total

        if self.trace_dir is not None:
            await asyncio.to_thread(
                self._write_trace, rollout_id, body, response, elapsed_ms
            )

        data = response.model_dump(exclude_none=True)
        if usage:
            data["usage"] = {
                "prompt_tokens": usage.prompt_tokens or 0,
                "completion_tokens": usage.completion_tokens or 0,
                "total_tokens": total,
            }
        return data

    async def _stream_response(self, rollout_id: str, body: dict):
        """Forward streaming request to LLM, yield SSE chunks, count tokens."""
        litellm = _get_litellm()
        kwargs = self._build_litellm_kwargs(body)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        try:
            response: Any = await litellm.acompletion(**kwargs)
        except Exception as e:
            if type(e).__name__ in self.SYSTEMIC_EXCEPTIONS:
                self._systemic_errors[rollout_id] = str(e)
            error_data = {"error": {"message": str(e), "type": type(e).__name__}}
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"
            return

        try:
            async for chunk in response:
                if hasattr(chunk, "usage") and chunk.usage:
                    total = chunk.usage.total_tokens or 0
                    if total > 0:
                        self._tokens[rollout_id] = (
                            self._tokens.get(rollout_id, 0) + total
                        )

                chunk_data = chunk.model_dump(exclude_none=True)
                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.warning("Streaming error for rollout_id=%s: %s", rollout_id, e)
            if type(e).__name__ in self.SYSTEMIC_EXCEPTIONS:
                self._systemic_errors[rollout_id] = str(e)
            error_data = {"error": {"message": str(e), "type": type(e).__name__}}
            yield f"data: {json.dumps(error_data)}\n\n"

        yield "data: [DONE]\n\n"

    def _write_trace(
        self, rollout_id: str, request: dict, response: Any, latency_ms: float
    ) -> None:
        assert self.trace_dir is not None
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.trace_dir / f"{rollout_id}.jsonl"
        resp_data = (
            response.model_dump() if hasattr(response, "model_dump") else response
        )
        line = json.dumps(
            {
                "ts": time.time(),
                "type": "llm_call",
                "request": request,
                "response": resp_data,
                "latency_ms": round(latency_ms, 2),
            },
            ensure_ascii=False,
        )
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def _build_app(self) -> Any:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse

        app = FastAPI(title="LiteLLMProxy", docs_url=None, redoc_url=None)

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            rollout_id = request.headers.get("x-rollout-id", "unknown")
            body = await request.json()

            if body.get("stream"):
                return StreamingResponse(
                    self._stream_response(rollout_id, body),
                    media_type="text/event-stream",
                )

            try:
                result = await self._handle_request(rollout_id, body)
            except Exception as exc:
                logger.exception("LiteLLMProxy error for rollout_id=%s", rollout_id)
                return JSONResponse(status_code=502, content={"error": str(exc)})
            return JSONResponse(content=result)

        return app


def _get_litellm():
    """Lazy import LiteLLM."""
    try:
        import litellm

        litellm.suppress_debug_info = True
        if hasattr(litellm, "_async_client_cleanup_registered"):
            litellm._async_client_cleanup_registered = True
        return litellm
    except ImportError as exc:
        from osmosis_ai.errors import CLIError

        raise CLIError(
            "LiteLLM is required. Install with: pip install litellm"
        ) from exc


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _wait_for_port(host: str, port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=0.2
            )
            writer.close()
            await writer.wait_closed()
            return
        except (TimeoutError, OSError):
            await asyncio.sleep(0.05)
    raise TimeoutError(f"LiteLLMProxy did not start within {timeout}s")


__all__ = ["LiteLLMProxy"]
