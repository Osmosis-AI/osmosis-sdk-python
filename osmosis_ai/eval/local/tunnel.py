"""cloudflared Quick Tunnel manager for local eval runs.

Spawns ``cloudflared tunnel --url <listener>`` as a child process (same
process-group discipline as the rollout server), parses the public
``https://*.trycloudflare.com`` URL from its log output, probes readiness
through the edge, and terminates with the run. The ``cloudflared`` binary on
PATH is the whole contract: no Python dependency, no account, no config.

Quick tunnels run a single edge connection, so the child can die mid-run;
the supervisor watches :meth:`CloudflaredTunnel.wait` and halts dispatch with
a clear error instead of letting pending rollouts hang.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import time
from collections.abc import Callable

import httpx

from osmosis_ai.eval.local.state import process_group_of, terminate_process_group

_URL_PATTERN = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
_START_TIMEOUT_SEC = 30.0
_PROBE_INTERVAL_SEC = 1.0
_TERM_GRACE_SEC = 5.0

_INSTALL_HINT = (
    "--tunnel cloudflared requires the `cloudflared` binary on PATH. Install "
    "it (macOS: `brew install cloudflared`; other platforms: "
    "https://developers.cloudflare.com/cloudflare-one/connections/"
    "connect-networks/downloads/), or run your own tunnel and pass "
    "--advertise-url instead."
)

_QUICK_TUNNEL_DOCS_URL = (
    "https://developers.cloudflare.com/cloudflare-one/networks/connectors/"
    "cloudflare-tunnel/do-more-with-tunnels/trycloudflare/"
)

# Cloudflare rate-limits quick-tunnel *creation* per source IP (measured:
# ~20 tunnels inside 30 minutes tripped it for 25+ minutes; already-running
# tunnels keep serving). cloudflared surfaces the 429 only as log lines and a
# cryptic unmarshal error before exiting, so the lines are the signal.
_RATE_LIMIT_MARKERS = (
    "429 Too Many Requests",
    "error code: 1015",
    "failed to unmarshal quick Tunnel",
)
_RATE_LIMIT_HINT = (
    "Cloudflare is rate-limiting quick-tunnel creation for this network "
    "(HTTP 429). Wait a few minutes and retry, or run your own tunnel and "
    f"pass --advertise-url. Quick-tunnel limits: {_QUICK_TUNNEL_DOCS_URL}"
)


class TunnelError(RuntimeError):
    """The tunnel could not be started."""


class CloudflaredTunnel:
    """One quick-tunnel child process; :meth:`start` returns the public URL."""

    def __init__(
        self,
        *,
        local_url: str,
        on_log: Callable[[str], None] | None = None,
        on_spawn: Callable[[asyncio.subprocess.Process], None] | None = None,
    ) -> None:
        self._local_url = local_url
        self._on_log = on_log
        # Called the moment the child exists, before the URL parse and the
        # readiness probe: the supervisor records the process for orphan
        # reaping, and a kill landing inside the (up to 30s) startup window
        # must still leave that record behind.
        self._on_spawn = on_spawn
        self._process: asyncio.subprocess.Process | None = None
        self._drain_task: asyncio.Task[None] | None = None
        # True until a stop() attempt observes the child NOT exiting; a
        # never-started tunnel has nothing running, so it starts confirmed.
        self._stop_confirmed = True
        self.public_url: str | None = None
        # False when the readiness probe got no HTTP response at all — the
        # host's DNS/egress cannot see the edge, which says nothing about
        # whether the sandbox's can.
        self.verified: bool = False

    async def start(self) -> str:
        executable = shutil.which("cloudflared")
        if executable is None:
            raise TunnelError(_INSTALL_HINT)
        try:
            process = await asyncio.create_subprocess_exec(
                executable,
                "tunnel",
                # Quick tunnels are unsupported when a default config file
                # (~/.cloudflared/config.yml, typically from a named-tunnel
                # setup) exists; pinning an explicitly empty config keeps a
                # user's named-tunnel installation from hijacking the run.
                "--config",
                os.devnull,
                "--url",
                self._local_url,
                "--no-autoupdate",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            # `which` succeeding does not make the spawn safe (binary removed
            # or not executable); keep every startup failure a TunnelError so
            # the CLI reports the same guidance.
            raise TunnelError(f"could not start cloudflared: {exc}") from exc
        self._process = process
        self._stop_confirmed = False
        try:
            if self._on_spawn is not None:
                try:
                    self._on_spawn(process)
                except Exception as exc:
                    raise TunnelError(
                        f"could not record cloudflared for orphan cleanup: {exc}"
                    ) from exc
            deadline = time.monotonic() + _START_TIMEOUT_SEC
            url_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
            self._drain_task = asyncio.create_task(self._drain(process, url_future))
            try:
                url = await asyncio.wait_for(
                    url_future, timeout=deadline - time.monotonic()
                )
            except TimeoutError:
                raise TunnelError(
                    "cloudflared did not publish a trycloudflare.com URL within "
                    f"{_START_TIMEOUT_SEC:.0f}s; see the run log for its output"
                ) from None
            self.verified = await self._probe_ready(url, deadline)
        except BaseException:
            # Never let cleanup mask the original error: a failing stop()
            # would otherwise replace the TunnelError mid-flight.
            with contextlib.suppress(Exception):
                await self.stop()
            raise
        self.public_url = url
        return url

    async def _drain(
        self, process: asyncio.subprocess.Process, url_future: asyncio.Future[str]
    ) -> None:
        """Read cloudflared's log stream for the tunnel URL, then keep draining.

        The pipe must be drained for the child's lifetime — a full pipe would
        block cloudflared — so this task outlives the URL parse.
        """
        stream = process.stderr
        assert stream is not None
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode(errors="replace").rstrip()
            if self._on_log is not None:
                with contextlib.suppress(Exception):
                    self._on_log(text)
            if not url_future.done():
                match = _URL_PATTERN.search(text)
                if match is not None:
                    url_future.set_result(match.group(0))
                elif any(marker in text for marker in _RATE_LIMIT_MARKERS):
                    # Known failure with a known cause: name it instead of
                    # the generic exit-code message the child produces.
                    url_future.set_exception(TunnelError(_RATE_LIMIT_HINT))
        if not url_future.done():
            returncode = await process.wait()
            url_future.set_exception(
                TunnelError(
                    f"cloudflared exited with code {returncode} before "
                    "publishing a tunnel URL"
                )
            )

    async def _probe_ready(self, url: str, deadline: float) -> bool:
        """GET through the edge until our app answers (any non-5xx status).

        The listener serves no route at ``/``, so its 404 is the readiness
        proof; 5xx comes from the edge while the tunnel is still connecting.

        Every transport-level failure — connection refusals, DNS errors, and
        connect timeouts alike — is retried until the deadline: a fresh
        ``*.trycloudflare.com`` name can take longer than the startup budget
        to propagate through some resolvers, so giving up early would misread
        a routine propagation delay. Returns False when no HTTP response
        arrived by the deadline: the host's own DNS or egress may never see
        the edge at all (Tailscale MagicDNS and corporate filters both do
        this) even though the sandbox resolves it independently —
        unverifiable is a warning, not an error. An edge that answers but
        keeps answering 5xx is a real failure.
        """
        saw_response = False
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
            while True:
                process = self._process
                if process is not None and process.returncode is not None:
                    raise TunnelError(
                        f"cloudflared exited with code {process.returncode} "
                        "before the tunnel became reachable"
                    )
                try:
                    response = await client.get(url)
                except httpx.HTTPError:
                    pass
                else:
                    saw_response = True
                    if response.status_code < 500:
                        return True
                if time.monotonic() >= deadline:
                    if saw_response:
                        raise TunnelError(
                            f"tunnel {url} did not become reachable within "
                            f"{_START_TIMEOUT_SEC:.0f}s. Some networks block "
                            "*.trycloudflare.com; if yours does, run your own "
                            "tunnel and pass --advertise-url. Quick-tunnel "
                            f"docs: {_QUICK_TUNNEL_DOCS_URL}"
                        )
                    return False
                await asyncio.sleep(_PROBE_INTERVAL_SEC)

    async def wait(self) -> int | None:
        """Block until the child exits; ``None`` if it was already stopped."""
        process = self._process
        if process is None:
            return None
        return await process.wait()

    async def stop(self) -> bool:
        """Terminate the child; True iff it is confirmed exited.

        False means the process may still be running (a termination or wait
        failure), so the caller must keep its ownership record for a later
        reap rather than assume the tunnel is gone. Repeated calls keep
        returning the first call's verdict.
        """
        process, self._process = self._process, None
        if process is not None and process.returncode is None:
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                await asyncio.to_thread(
                    terminate_process_group,
                    process_group_of(process.pid),
                    grace_sec=_TERM_GRACE_SEC,
                )
        if process is not None:
            with contextlib.suppress(Exception):
                await process.wait()
            self._stop_confirmed = process.returncode is not None
        if self._drain_task is not None:
            self._drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._drain_task
            self._drain_task = None
        return self._stop_confirmed


__all__ = ["CloudflaredTunnel", "TunnelError"]
