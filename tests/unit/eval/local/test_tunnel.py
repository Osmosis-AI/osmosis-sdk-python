"""Tests for the cloudflared quick-tunnel manager.

A shell script stands in for the cloudflared binary, so the tests cover the
real subprocess contract — log parsing on stderr, exit codes, process-group
teardown — without any network or the binary itself.
"""

from __future__ import annotations

import asyncio
import shutil
import socket
import time
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from osmosis_ai.eval.local import tunnel as tunnel_module
from osmosis_ai.eval.local.tunnel import CloudflaredTunnel, TunnelError
from osmosis_ai.rollout.controller.listener import LocalhostUvicornServer

FAKE_URL = "https://fake-name.trycloudflare.com"


def _fake_cloudflared(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "cloudflared"
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(0o755)
    return script


async def _no_probe(
    self: CloudflaredTunnel, url: str, deadline: float
) -> bool:  # pragma: no cover - trivial
    return True


async def test_missing_binary_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(TunnelError, match="brew install cloudflared"):
        await CloudflaredTunnel(local_url="http://127.0.0.1:1").start()


async def test_start_parses_url_and_stop_kills_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _fake_cloudflared(
        tmp_path,
        f"echo 'INF |  {FAKE_URL}  |' >&2\n"
        "echo 'INF Registered tunnel connection' >&2\n"
        "exec sleep 60",
    )
    monkeypatch.setattr(shutil, "which", lambda name: str(script))
    monkeypatch.setattr(CloudflaredTunnel, "_probe_ready", _no_probe)
    tunnel = CloudflaredTunnel(local_url="http://127.0.0.1:1")
    url = await tunnel.start()
    assert url == FAKE_URL
    assert tunnel.public_url == FAKE_URL
    process = tunnel._process
    assert process is not None and process.returncode is None
    await tunnel.stop()
    assert process.returncode is not None
    # Stopped: there is no child left to wait on.
    assert await tunnel.wait() is None


async def test_registered_connection_skips_host_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _fake_cloudflared(
        tmp_path,
        f"echo 'INF |  {FAKE_URL}  |' >&2\n"
        "sleep 0.05\n"
        "echo 'INF Registered tunnel connection' >&2\n"
        "exec sleep 60",
    )
    monkeypatch.setattr(shutil, "which", lambda name: str(script))

    async def unexpected_host_probe(
        self: CloudflaredTunnel, url: str, deadline: float
    ) -> bool:
        raise AssertionError("registered tunnels must not wait on the host probe")

    monkeypatch.setattr(CloudflaredTunnel, "_probe_ready", unexpected_host_probe)
    tunnel = CloudflaredTunnel(local_url="http://127.0.0.1:1")
    assert await asyncio.wait_for(tunnel.start(), timeout=1.0) == FAKE_URL
    assert tunnel.verified is True
    assert tunnel.unverified_reason is None
    await tunnel.stop()


async def test_registered_child_that_exits_immediately_is_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _fake_cloudflared(
        tmp_path,
        f"echo 'INF |  {FAKE_URL}  |' >&2\n"
        "echo 'INF Registered tunnel connection' >&2\n"
        "exit 7",
    )
    monkeypatch.setattr(shutil, "which", lambda name: str(script))
    spawned: list[asyncio.subprocess.Process] = []
    tunnel = CloudflaredTunnel(local_url="http://127.0.0.1:1", on_spawn=spawned.append)
    with pytest.raises(TunnelError, match="exited with code 7"):
        await asyncio.wait_for(tunnel.start(), timeout=1.0)
    assert spawned[0].returncode == 7
    assert tunnel._process is None


async def test_registration_cancels_failed_host_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _fake_cloudflared(
        tmp_path,
        f"echo 'INF |  {FAKE_URL}  |' >&2\n"
        "sleep 0.05\n"
        "echo 'INF Registered tunnel connection' >&2\n"
        "exec sleep 60",
    )
    monkeypatch.setattr(shutil, "which", lambda name: str(script))
    monkeypatch.setattr(tunnel_module, "_CONNECTION_REGISTRATION_GRACE_SEC", 0.01)
    probe_started = asyncio.Event()
    probe_cancelled = asyncio.Event()

    async def failed_host_probe(
        self: CloudflaredTunnel, url: str, deadline: float
    ) -> bool:
        probe_started.set()
        self.unverified_reason = "HTTP 530"
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            probe_cancelled.set()
            raise

    monkeypatch.setattr(CloudflaredTunnel, "_probe_ready", failed_host_probe)
    tunnel = CloudflaredTunnel(local_url="http://127.0.0.1:1")
    assert await asyncio.wait_for(tunnel.start(), timeout=1.0) == FAKE_URL
    assert probe_started.is_set()
    assert probe_cancelled.is_set()
    assert tunnel.verified is True
    assert tunnel.unverified_reason is None
    await tunnel.stop()


async def test_probe_failure_wins_registration_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection_registered = asyncio.Event()

    async def child_exit(self: CloudflaredTunnel, url: str, deadline: float) -> bool:
        connection_registered.set()
        raise TunnelError("cloudflared exited with code 7")

    monkeypatch.setattr(CloudflaredTunnel, "_probe_ready", child_exit)
    tunnel = CloudflaredTunnel(local_url="http://127.0.0.1:1")
    with pytest.raises(TunnelError, match="exited with code 7"):
        await tunnel._probe_ready_until_registered(
            FAKE_URL, time.monotonic() + 1.0, connection_registered
        )


@pytest.mark.parametrize("probe_reason", ["DNS lookup failed", "HTTP 530"])
async def test_unregistered_unreachable_tunnel_fails_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probe_reason: str
) -> None:
    script = _fake_cloudflared(
        tmp_path,
        f"echo 'INF |  {FAKE_URL}  |' >&2\nexec sleep 60",
    )
    monkeypatch.setattr(shutil, "which", lambda name: str(script))
    monkeypatch.setattr(tunnel_module, "_CONNECTION_REGISTRATION_GRACE_SEC", 0.01)
    spawned: list[asyncio.subprocess.Process] = []

    async def unreachable_host_probe(
        self: CloudflaredTunnel, url: str, deadline: float
    ) -> bool:
        self.unverified_reason = probe_reason
        return False

    monkeypatch.setattr(CloudflaredTunnel, "_probe_ready", unreachable_host_probe)
    tunnel = CloudflaredTunnel(local_url="http://127.0.0.1:1", on_spawn=spawned.append)
    with pytest.raises(
        TunnelError, match="neither registered a connection nor"
    ) as excinfo:
        await tunnel.start()
    message = str(excinfo.value)
    assert probe_reason in message
    assert "--advertise-url" in message
    assert tunnel_module._QUICK_TUNNEL_DOCS_URL in message
    assert spawned[0].returncode is not None
    assert tunnel._process is None


async def test_host_http_success_does_not_require_registration_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _fake_cloudflared(
        tmp_path,
        f"echo 'INF |  {FAKE_URL}  |' >&2\nexec sleep 60",
    )
    monkeypatch.setattr(shutil, "which", lambda name: str(script))
    monkeypatch.setattr(tunnel_module, "_CONNECTION_REGISTRATION_GRACE_SEC", 0.01)

    async def reachable_host_probe(
        self: CloudflaredTunnel, url: str, deadline: float
    ) -> bool:
        return True

    monkeypatch.setattr(CloudflaredTunnel, "_probe_ready", reachable_host_probe)
    tunnel = CloudflaredTunnel(local_url="http://127.0.0.1:1")
    assert await tunnel.start() == FAKE_URL
    assert tunnel.verified is True
    await tunnel.stop()


async def test_exit_before_url_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _fake_cloudflared(tmp_path, "echo 'no tunnel for you' >&2\nexit 3")
    monkeypatch.setattr(shutil, "which", lambda name: str(script))
    with pytest.raises(TunnelError, match="exited with code 3"):
        await CloudflaredTunnel(local_url="http://127.0.0.1:1").start()


async def test_url_publish_timeout_raises_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _fake_cloudflared(tmp_path, "exec sleep 60")
    monkeypatch.setattr(shutil, "which", lambda name: str(script))
    monkeypatch.setattr(tunnel_module, "_START_TIMEOUT_SEC", 0.2)
    tunnel = CloudflaredTunnel(local_url="http://127.0.0.1:1")
    with pytest.raises(TunnelError, match="did not publish"):
        await tunnel.start()
    assert tunnel._process is None


async def test_probe_ready_accepts_app_404() -> None:
    # The listener serves no route at "/": its 404 is the passthrough proof.
    async with LocalhostUvicornServer(FastAPI()) as server:
        tunnel = CloudflaredTunnel(local_url=server.base_url)
        assert await tunnel._probe_ready(server.base_url, time.monotonic() + 5.0)


async def test_probe_ready_marks_edge_errors_unverified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()

    @app.get("/")
    async def edge_error() -> JSONResponse:
        return JSONResponse({"detail": "tunnel not ready"}, status_code=530)

    monkeypatch.setattr(tunnel_module, "_PROBE_INTERVAL_SEC", 0.01)
    async with LocalhostUvicornServer(app) as server:
        tunnel = CloudflaredTunnel(local_url=server.base_url)
        assert (
            await tunnel._probe_ready(server.base_url, time.monotonic() + 0.3) is False
        )
        assert tunnel.unverified_reason == "HTTP 530"


async def test_probe_ready_clears_transient_error_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_async_client = httpx.AsyncClient
    statuses = iter((530, 404))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(next(statuses), request=request)

    def client_factory(**kwargs: object) -> httpx.AsyncClient:
        return real_async_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(tunnel_module.httpx, "AsyncClient", client_factory)
    monkeypatch.setattr(tunnel_module, "_PROBE_INTERVAL_SEC", 0)
    tunnel = CloudflaredTunnel(local_url="http://127.0.0.1:1")
    assert await tunnel._probe_ready(FAKE_URL, time.monotonic() + 1.0) is True
    assert tunnel.unverified_reason is None


async def test_probe_with_no_http_response_is_unverified_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The host's DNS/egress may not see *.trycloudflare.com at all (Tailscale
    # MagicDNS, corporate filters); the sandbox resolves it independently, so
    # an unreachable probe downgrades to unverified instead of failing.
    monkeypatch.setattr(tunnel_module, "_PROBE_INTERVAL_SEC", 0.01)
    tunnel = CloudflaredTunnel(local_url="http://127.0.0.1:1")
    assert (
        await tunnel._probe_ready("http://127.0.0.1:1", time.monotonic() + 0.3) is False
    )
    assert tunnel.unverified_reason == "connection failed"


def test_probe_failure_reason_identifies_dns() -> None:
    try:
        try:
            raise OSError("resolver failed") from socket.gaierror(8, "not known")
        except OSError as cause:
            raise httpx.ConnectError("connect failed") from cause
    except httpx.ConnectError as exc:
        assert tunnel_module._probe_failure_reason(exc) == "DNS lookup failed"


async def test_drain_forwards_cloudflared_log_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _fake_cloudflared(
        tmp_path,
        f"echo 'INF |  {FAKE_URL}  |' >&2\n"
        "echo 'INF Registered tunnel connection' >&2\n"
        "exec sleep 60",
    )
    monkeypatch.setattr(shutil, "which", lambda name: str(script))
    monkeypatch.setattr(CloudflaredTunnel, "_probe_ready", _no_probe)
    lines: list[str] = []
    tunnel = CloudflaredTunnel(local_url="http://127.0.0.1:1", on_log=lines.append)
    await tunnel.start()
    await tunnel.stop()
    assert any(FAKE_URL in line for line in lines)


async def test_on_spawn_receives_the_child_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _fake_cloudflared(
        tmp_path,
        f"echo 'INF |  {FAKE_URL}  |' >&2\n"
        "echo 'INF Registered tunnel connection' >&2\n"
        "exec sleep 60",
    )
    monkeypatch.setattr(shutil, "which", lambda name: str(script))
    monkeypatch.setattr(CloudflaredTunnel, "_probe_ready", _no_probe)
    seen: list[int] = []
    tunnel = CloudflaredTunnel(
        local_url="http://127.0.0.1:1",
        on_spawn=lambda process: seen.append(process.pid),
    )
    await tunnel.start()
    process = tunnel._process
    assert process is not None
    assert seen == [process.pid]
    await tunnel.stop()


async def test_on_spawn_failure_stops_child_and_fails_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _fake_cloudflared(
        tmp_path,
        f"echo 'INF |  {FAKE_URL}  |' >&2\n"
        "echo 'INF Registered tunnel connection' >&2\n"
        "exec sleep 60",
    )
    monkeypatch.setattr(shutil, "which", lambda name: str(script))
    monkeypatch.setattr(CloudflaredTunnel, "_probe_ready", _no_probe)
    spawned: list[asyncio.subprocess.Process] = []

    def explode(process: asyncio.subprocess.Process) -> None:
        spawned.append(process)
        raise RuntimeError("disk full")

    tunnel = CloudflaredTunnel(local_url="http://127.0.0.1:1", on_spawn=explode)
    with pytest.raises(TunnelError, match="orphan cleanup"):
        await tunnel.start()
    assert spawned[0].returncode is not None


async def test_interrupted_stop_never_claims_the_child_exited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _fake_cloudflared(
        tmp_path,
        f"echo 'INF |  {FAKE_URL}  |' >&2\n"
        "echo 'INF Registered tunnel connection' >&2\n"
        "exec sleep 60",
    )
    monkeypatch.setattr(shutil, "which", lambda name: str(script))
    monkeypatch.setattr(CloudflaredTunnel, "_probe_ready", _no_probe)
    tunnel = CloudflaredTunnel(local_url="http://127.0.0.1:1")
    await tunnel.start()
    process = tunnel._process
    assert process is not None

    async def interrupted_to_thread(*args: object, **kwargs: object) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(tunnel_module.asyncio, "to_thread", interrupted_to_thread)
    with pytest.raises(asyncio.CancelledError):
        await tunnel.stop()
    assert await tunnel.stop() is False

    process.kill()
    await process.wait()


async def test_spawn_failure_raises_tunnel_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `which` succeeding does not make the spawn safe: the path can be gone
    # (or not executable) by exec time, and the CLI owes the same guidance.
    missing = tmp_path / "cloudflared"
    monkeypatch.setattr(shutil, "which", lambda name: str(missing))
    with pytest.raises(TunnelError, match="could not start cloudflared"):
        await CloudflaredTunnel(local_url="http://127.0.0.1:1").start()


async def test_spawn_pins_an_empty_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Quick tunnels are unsupported when a default ~/.cloudflared config
    # exists, so the argv must pin an explicitly empty one.
    import os

    script = _fake_cloudflared(
        tmp_path,
        f"echo \"ARGS:$@\" >&2\necho 'INF |  {FAKE_URL}  |' >&2\n"
        "echo 'INF Registered tunnel connection' >&2\n"
        "exec sleep 60",
    )
    monkeypatch.setattr(shutil, "which", lambda name: str(script))
    monkeypatch.setattr(CloudflaredTunnel, "_probe_ready", _no_probe)
    lines: list[str] = []
    tunnel = CloudflaredTunnel(local_url="http://127.0.0.1:1", on_log=lines.append)
    await tunnel.start()
    await tunnel.stop()
    args_line = next(line for line in lines if line.startswith("ARGS:"))
    assert f"--config {os.devnull}" in args_line


async def test_rate_limited_creation_names_the_cause_and_the_docs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The real failure shape measured against Cloudflare: cloudflared logs the
    # 429 and exits non-zero without ever publishing a URL. The generic
    # exit-code message would hide the known cause.
    script = _fake_cloudflared(
        tmp_path,
        "echo 'ERR Error unmarshaling QuickTunnel response: error code: 1015"
        ' error="invalid character" status_code="429 Too Many Requests"\' >&2\n'
        "exit 1",
    )
    monkeypatch.setattr(shutil, "which", lambda name: str(script))
    with pytest.raises(TunnelError, match="rate-limiting") as excinfo:
        await CloudflaredTunnel(local_url="http://127.0.0.1:1").start()
    assert "--advertise-url" in str(excinfo.value)
    assert "developers.cloudflare.com" in str(excinfo.value)
