"""Tests for the cloudflared quick-tunnel manager.

A shell script stands in for the cloudflared binary, so the tests cover the
real subprocess contract — log parsing on stderr, exit codes, process-group
teardown — without any network or the binary itself.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

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
        f"echo 'INF |  {FAKE_URL}  |' >&2\nexec sleep 60",
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


async def test_probe_ready_times_out_on_edge_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()

    @app.get("/")
    async def edge_error() -> JSONResponse:
        return JSONResponse({"detail": "tunnel not ready"}, status_code=530)

    monkeypatch.setattr(tunnel_module, "_PROBE_INTERVAL_SEC", 0.01)
    async with LocalhostUvicornServer(app) as server:
        tunnel = CloudflaredTunnel(local_url=server.base_url)
        with pytest.raises(TunnelError, match="did not become reachable"):
            await tunnel._probe_ready(server.base_url, time.monotonic() + 0.3)


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


async def test_drain_forwards_cloudflared_log_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _fake_cloudflared(
        tmp_path,
        f"echo 'INF |  {FAKE_URL}  |' >&2\nexec sleep 60",
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
        f"echo 'INF |  {FAKE_URL}  |' >&2\nexec sleep 60",
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


async def test_on_spawn_failure_does_not_break_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _fake_cloudflared(
        tmp_path,
        f"echo 'INF |  {FAKE_URL}  |' >&2\nexec sleep 60",
    )
    monkeypatch.setattr(shutil, "which", lambda name: str(script))
    monkeypatch.setattr(CloudflaredTunnel, "_probe_ready", _no_probe)

    def explode(process: object) -> None:
        raise RuntimeError("disk full")

    tunnel = CloudflaredTunnel(local_url="http://127.0.0.1:1", on_spawn=explode)
    assert await tunnel.start() == FAKE_URL
    await tunnel.stop()
