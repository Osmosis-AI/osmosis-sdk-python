"""Real stdio and HTTP lifecycle contracts for the optional MCP prototype."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

pytest.importorskip("mcp.server.lowlevel", reason="optional MCP prototype dependencies")

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Bridge:
    process: asyncio.subprocess.Process
    started: asyncio.Queue[str]
    closed: asyncio.Queue[str]
    responses: dict[int, dict[str, Any]] = field(default_factory=dict)

    async def send(self, method: str, params: dict[str, Any], request_id=None):
        message = {"jsonrpc": "2.0", "method": method, "params": params}
        if request_id is not None:
            message["id"] = request_id
        assert self.process.stdin is not None
        self.process.stdin.write((json.dumps(message) + "\n").encode())
        await self.process.stdin.drain()

    async def receive(self, request_id: int, timeout: float = 3) -> dict[str, Any]:
        assert self.process.stdout is not None
        async with asyncio.timeout(timeout):
            while request_id not in self.responses:
                line = await self.process.stdout.readline()
                assert line, "Bridge exited before returning a JSON-RPC response"
                response = json.loads(line)
                if "id" in response:
                    self.responses[response["id"]] = response
            return self.responses.pop(request_id)

    async def call(self, request_id: int, name: str, **arguments):
        await self.send(
            "tools/call", {"name": name, "arguments": arguments}, request_id
        )


@pytest_asyncio.fixture
async def bridge(tmp_path: Path) -> AsyncIterator[Bridge]:
    started: asyncio.Queue[str] = asyncio.Queue()
    closed: asyncio.Queue[str] = asyncio.Queue()
    handlers: set[asyncio.Task] = set()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        task = asyncio.current_task()
        assert task is not None
        handlers.add(task)
        disconnected = None
        try:
            request = await reader.readuntil(b"\r\n\r\n")
            target = request.split(b" ", 2)[1].decode()
            assert b"Authorization: Bearer synthetic-lifecycle-token\r\n" in request
            assert b"X-Osmosis-Workspace: workspace-a\r\n" in request
            if target.startswith("/api/cli/training-runs?"):
                prefix = (
                    b'{"training_runs":[],"total_count":0,"has_more":false,"padding":"'
                )
                body = prefix + b"x" * 600 + b'"}'
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    + f"Content-Length: {len(body)}\r\n\r\n".encode()
                    + prefix
                )
                await writer.drain()
                started.put_nowait(target)
                disconnected = asyncio.create_task(reader.read())
                for byte in body[len(prefix) :]:
                    done, _ = await asyncio.wait({disconnected}, timeout=0.1)
                    if done:
                        assert disconnected.result() == b""
                        closed.put_nowait(target)
                        return
                    writer.write(bytes([byte]))
                    await writer.drain()
            else:
                body = b'{"datasets":[],"total_count":0,"has_more":false}'
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    + f"Content-Length: {len(body)}\r\n\r\n".encode()
                    + body
                )
                await writer.drain()
        finally:
            if disconnected is not None:
                disconnected.cancel()
                await asyncio.gather(disconnected, return_exceptions=True)
            writer.close()
            await writer.wait_closed()
            handlers.discard(task)

    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    (repository / "configs/training").mkdir(parents=True)
    (repository / "configs/training/tiny.toml").write_text("private config contents")
    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(ROOT / "prototypes/stdio_mcp/bridge.py"),
        "--workspace-directory",
        str(repository),
        "--workspace",
        "workspace-a",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=repository,
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(tmp_path / "home"),
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
            "PYTHONPATH": str(ROOT),
            "OSMOSIS_PLATFORM_URL": url,
            "OSMOSIS_TOKEN": "synthetic-lifecycle-token",
            "OSMOSIS_TOKEN_PLATFORM_URL": url,
        },
    )
    instance = Bridge(process, started, closed)
    try:
        await instance.send(
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "lifecycle-test", "version": "1"},
            },
            1,
        )
        assert "result" in await instance.receive(1, timeout=10)
        await instance.send("notifications/initialized", {})
        yield instance
    finally:
        if process.returncode is None:
            process.kill()
        await process.wait()
        server.close()
        await server.wait_closed()
        for task in handlers:
            task.cancel()
        await asyncio.gather(*handlers, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["deadline", "cancel", "eof"])
async def test_active_http_body_is_closed_on_deadline_cancellation_and_eof(
    bridge: Bridge, finish: str
):
    started_at = asyncio.get_running_loop().time()
    await bridge.call(2, "list_training_runs", limit=1)
    target = await asyncio.wait_for(bridge.started.get(), timeout=3)

    await bridge.call(3, "list_local_configs")
    local = await bridge.receive(3)
    assert local["result"]["structuredContent"]["data"]["config_paths"] == [
        "configs/training/tiny.toml"
    ]

    if finish == "deadline":
        response = await bridge.receive(2, timeout=12)
        elapsed = asyncio.get_running_loop().time() - started_at
        assert 9 <= elapsed <= 12
        assert response["result"]["isError"] is True
        assert response["result"]["structuredContent"] == {
            "data": None,
            "error": "PLATFORM_UNAVAILABLE",
        }
    elif finish == "cancel":
        await bridge.send("notifications/cancelled", {"requestId": 2})
    else:
        assert bridge.process.stdin is not None
        bridge.process.stdin.close()
        await asyncio.wait_for(bridge.process.wait(), timeout=3)
        assert bridge.process.returncode == 0

    assert await asyncio.wait_for(bridge.closed.get(), timeout=3) == target

    if finish != "eof":
        for request_id, tool, expected in (
            (4, "list_local_configs", {"config_paths": ["configs/training/tiny.toml"]}),
            (5, "list_datasets", {"items": [], "total_count": 0, "has_more": False}),
        ):
            await bridge.call(request_id, tool)
            response = (await bridge.receive(request_id))["result"]
            assert response["isError"] is False
            assert response["structuredContent"]["error"] is None
            assert expected.items() <= response["structuredContent"]["data"].items()


@pytest.mark.asyncio
async def test_remote_admission_is_bounded_and_queue_wait_uses_request_deadline(
    bridge: Bridge,
):
    started_at = asyncio.get_running_loop().time()
    for request_id in range(2, 8):
        await bridge.call(request_id, "list_training_runs", offset=request_id)
    active = {await asyncio.wait_for(bridge.started.get(), timeout=3) for _ in range(4)}
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(bridge.started.get(), timeout=0.3)

    await bridge.call(8, "list_local_configs")
    assert (await bridge.receive(8))["result"]["isError"] is False
    await bridge.send("notifications/cancelled", {"requestId": 2})
    cancelled = await asyncio.wait_for(bridge.closed.get(), timeout=3)
    assert cancelled.endswith("&offset=2")
    active.remove(cancelled)
    replacement = await asyncio.wait_for(bridge.started.get(), timeout=3)
    assert replacement not in active
    active.add(replacement)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(bridge.started.get(), timeout=0.3)

    async with asyncio.timeout(12):
        for request_id in range(3, 8):
            result = (await bridge.receive(request_id, timeout=12))["result"]
            assert result["isError"] is True
            assert result["structuredContent"] == {
                "data": None,
                "error": "PLATFORM_UNAVAILABLE",
            }
    assert 9 <= asyncio.get_running_loop().time() - started_at <= 12
    async with asyncio.timeout(3):
        while active:
            active.discard(await bridge.closed.get())
