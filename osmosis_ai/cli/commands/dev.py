"""Internal developer commands (hidden).

`osmosis dev serve` runs the rollout in the current directory as a local server
and exposes it to the GPU cluster over an iroh (`dumbpipe`) tunnel. It prints the
tunnel ticket; paste that into your training config under [advanced] as
`local_rollout_address` and submit — the per-run ECS rollout slot then forwards
its rollout calls to this machine instead of cloning the rollout repo.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError

app: typer.Typer = typer.Typer(
    help="Internal developer tooling.",
    no_args_is_help=True,
)

DEFAULT_ROLLOUT_PORT = 8000

_SERVE_LOG_DIR = Path.home() / ".cache" / "osmosis" / "dev-serve"
_SERVE_LOG_KEEP = 5

# Keep DUMBPIPE_VERSION and the checksums in lockstep with the rollout-server
# Dockerfile (iac/.../rollout-server/Dockerfile) so both ends of the tunnel
# speak the same ticket format.
DUMBPIPE_VERSION = "v0.39.0"
_DUMBPIPE_SHA256 = {
    (
        "darwin",
        "x86_64",
    ): "4480d175d7888f8f705418cd81b1512aa3c29bb84bc0f8566424c32cdc636042",
    (
        "darwin",
        "aarch64",
    ): "13f284b36f2df429487975878ac8890f50ddd7916d040d988f467a919b0fba3e",
    (
        "linux",
        "x86_64",
    ): "9ac5e71983eba4cf4f47e92e4694dad0f0133abba4f14dd2a2a8428cece88fef",
    (
        "linux",
        "aarch64",
    ): "8cd099afe80c69ac58bfbc975b28e29d41caea141f3dd396457345a739836a0e",
}

_ARCH_ALIASES = {
    "arm64": "aarch64",
    "aarch64": "aarch64",
    "x86_64": "x86_64",
    "amd64": "x86_64",
}


def _progress(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def _ensure_dumbpipe() -> str:
    """Return a path to the pinned `dumbpipe` binary, downloading it if needed.

    dumbpipe has no version flag, so a PATH binary's ticket-protocol version can't
    be verified. We therefore prefer the pinned, checksum-verified build and only
    fall back to PATH on platforms without one (where compatibility isn't guaranteed).
    """
    system = platform.system().lower()
    arch = _ARCH_ALIASES.get(platform.machine().lower())
    sha = _DUMBPIPE_SHA256.get((system, arch or ""))
    if system not in ("darwin", "linux") or arch is None or sha is None:
        existing = shutil.which("dumbpipe")
        if existing:
            _progress(
                f"No pinned dumbpipe build for {system}/{platform.machine()}; using "
                f"{existing} (version unverified, ticket compatibility not guaranteed)."
            )
            return existing
        raise CLIError(
            f"No pinned dumbpipe build for {system}/{platform.machine()}. "
            "Install dumbpipe manually and put it on PATH.",
            code="INTERNAL",
        )

    bin_dir = Path.home() / ".cache" / "osmosis" / "bin"
    dest = bin_dir / f"dumbpipe-{DUMBPIPE_VERSION}"
    if dest.exists() and os.access(dest, os.X_OK):
        return str(dest)

    asset = f"dumbpipe-{DUMBPIPE_VERSION}-{system}-{arch}.tar.gz"
    url = (
        "https://github.com/n0-computer/dumbpipe/releases/download/"
        f"{DUMBPIPE_VERSION}/{asset}"
    )
    bin_dir.mkdir(parents=True, exist_ok=True)
    tgz = bin_dir / asset
    _progress(f"Downloading dumbpipe {DUMBPIPE_VERSION} ...")
    urllib.request.urlretrieve(url, tgz)

    actual = hashlib.sha256(tgz.read_bytes()).hexdigest()
    if actual != sha:
        tgz.unlink(missing_ok=True)
        raise CLIError(
            f"dumbpipe checksum mismatch (expected {sha}, got {actual}).",
            code="INTERNAL",
        )

    with tarfile.open(tgz) as tar:
        inner = next(
            (n for n in tar.getnames() if n.rstrip("/").endswith("dumbpipe")),
            None,
        )
        if inner is None:
            tgz.unlink(missing_ok=True)
            raise CLIError(
                "dumbpipe binary not found in release archive.", code="INTERNAL"
            )
        tar.extract(inner, bin_dir)
    (bin_dir / inner).rename(dest)
    dest.chmod(0o755)
    tgz.unlink(missing_ok=True)
    return str(dest)


def _wait_until_healthy(
    port: int, server: subprocess.Popen[bytes], log_path: Path, timeout: float = 60.0
) -> None:
    url = f"http://localhost:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise CLIError(
                "Rollout server exited before becoming healthy:\n"
                + log_path.read_text(errors="replace"),
                code="INTERNAL",
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except OSError:
            pass
        time.sleep(1)
    raise CLIError("Rollout server did not become healthy in time.", code="INTERNAL")


def _capture_ticket(
    tunnel: subprocess.Popen[bytes], log_path: Path, timeout: float = 30.0
) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if tunnel.poll() is not None:
            raise CLIError(
                "dumbpipe exited before printing a ticket:\n"
                + log_path.read_text(errors="replace"),
                code="INTERNAL",
            )
        for line in log_path.read_text(errors="replace").splitlines():
            if "connect-tcp" in line:
                parts = line.split()
                if parts:
                    return parts[-1]
        time.sleep(0.5)
    raise CLIError("Could not capture an iroh ticket from dumbpipe.", code="INTERNAL")


def _new_run_dir() -> Path:
    """Create a per-run log dir under the stable cache, pruning all but the newest few."""
    _SERVE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        (p for p in _SERVE_LOG_DIR.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    for stale in (
        existing[: -(_SERVE_LOG_KEEP - 1)] if _SERVE_LOG_KEEP > 1 else existing
    ):
        shutil.rmtree(stale, ignore_errors=True)

    run = _SERVE_LOG_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    run.mkdir(parents=True, exist_ok=True)
    return run


def _stream_log(log_path: Path, stop: threading.Event) -> None:
    """Tail ``log_path`` (from the start) to stderr until ``stop`` is set and drained."""
    with log_path.open("r", errors="replace") as fh:
        while True:
            line = fh.readline()
            if line:
                sys.stderr.write(line)
                sys.stderr.flush()
                continue
            if stop.is_set():
                rest = fh.read()
                if rest:
                    sys.stderr.write(rest)
                    sys.stderr.flush()
                return
            time.sleep(0.2)


def _terminate(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _emit_ticket(port: int, ticket: str) -> None:
    """Surface the tunnel ticket: machine-readable on stdout, styled on stderr."""
    import json

    from osmosis_ai.cli.output.context import OutputFormat, get_output_context

    ctx = get_output_context()

    # serve() never returns a CommandResult (it blocks until interrupted), so we
    # emit this command's machine output here and mark it so the result callback
    # doesn't flag the None return as "no structured output".
    if ctx.format is OutputFormat.json:
        sys.stdout.write(
            json.dumps(
                {
                    "schema_version": ctx.schema_version,
                    "local_rollout_address": ticket,
                    "port": port,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        sys.stdout.flush()
        ctx.output_emitted = True
        return

    # The ticket is this command's machine output. Emit it raw on stdout whenever
    # output is being consumed (piped/redirected, or the low-noise plain format);
    # in an interactive rich TTY the styled block below is the interface, so skip
    # the duplicate raw dump.
    if ctx.format is OutputFormat.plain or not sys.stdout.isatty():
        sys.stdout.write(ticket + "\n")
        sys.stdout.flush()
    ctx.output_emitted = True

    if ctx.format is OutputFormat.plain or not ctx.interactive:
        _progress("")
        _progress("Add this to your training config under [advanced], then submit:")
        _progress(f'    local_rollout_address = "{ticket}"')
        _progress("")
        _progress(f"Rollout server + tunnel running on :{port} (Ctrl-C to stop).")
        return

    from rich.console import Console as RichConsole

    err = RichConsole(stderr=True, highlight=False)
    err.print()
    err.rule("[bold green]Local rollout ready[/]", style="green")
    err.print()
    err.print(
        "Add this to your training config under [bold]\\[advanced][/], then submit:"
    )
    err.print()
    # soft_wrap keeps the long ticket on one logical line so it copies cleanly.
    err.print(
        f'    [cyan]local_rollout_address[/] = [green]"{ticket}"[/]', soft_wrap=True
    )
    err.print()
    err.print(
        f"Rollout server + tunnel running on :{port} (Ctrl-C to stop).", style="dim"
    )


@app.command("serve")
def serve(
    port: int = typer.Option(
        DEFAULT_ROLLOUT_PORT,
        "--port",
        help="Local port for the rollout server.",
    ),
) -> None:
    """Serve the rollout in the current directory over an iroh tunnel.

    Run from a rollout folder (containing ``main.py``). Prints the tunnel
    ticket to stdout, then holds the server + tunnel open until interrupted.
    """
    repo = Path.cwd()
    main_file = repo / "main.py"
    if not main_file.is_file():
        raise CLIError(
            f"No main.py in {repo}. Run `osmosis dev serve` from a rollout folder.",
            code="INVALID_USAGE",
        )

    dumbpipe = _ensure_dumbpipe()
    work = _new_run_dir()
    server_log = work / "rollout-server.log"
    tunnel_log = work / "dumbpipe.log"
    server: subprocess.Popen[bytes] | None = None
    tunnel: subprocess.Popen[bytes] | None = None
    stop_stream = threading.Event()
    log_thread: threading.Thread | None = None

    env = os.environ.copy()
    env["ROLLOUT_PORT"] = str(port)
    env["_OSMOSIS_ROLLOUT_PORT"] = str(port)

    # Route SIGTERM (plain `kill`, most OOM kills) into the same path as Ctrl-C so
    # the finally below tears down the children instead of orphaning them — a
    # leaked server holds the port and a leaked tunnel keeps forwarding stale code.
    def _on_sigterm(*_: object) -> None:
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, _on_sigterm)

    try:
        with console.status(f"Starting rollout server on :{port}"):
            with server_log.open("wb") as fh:
                server = subprocess.Popen(
                    [sys.executable, str(main_file)],
                    cwd=str(repo),
                    env=env,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                )
            _wait_until_healthy(port, server, server_log)

        with console.status("Opening iroh tunnel"):
            with tunnel_log.open("wb") as fh:
                tunnel = subprocess.Popen(
                    [dumbpipe, "listen-tcp", "--host", f"localhost:{port}"],
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                )
            ticket = _capture_ticket(tunnel, tunnel_log)

        _emit_ticket(port, ticket)

        _progress("")
        _progress(f"── rollout server logs ({server_log}) ──")
        log_thread = threading.Thread(
            target=_stream_log, args=(server_log, stop_stream), daemon=True
        )
        log_thread.start()

        while True:
            if server.poll() is not None:
                raise CLIError(
                    "Rollout server exited:\n" + server_log.read_text(errors="replace"),
                    code="INTERNAL",
                )
            if tunnel.poll() is not None:
                raise CLIError(
                    "iroh tunnel exited:\n" + tunnel_log.read_text(errors="replace"),
                    code="INTERNAL",
                )
            time.sleep(1)
    except KeyboardInterrupt:
        _progress("\nShutting down rollout server + tunnel ...")
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        stop_stream.set()
        if log_thread is not None:
            log_thread.join(timeout=2)
        _terminate(tunnel)
        _terminate(server)
        _progress(f"Logs preserved in {work} (rollout-server.log, dumbpipe.log).")
