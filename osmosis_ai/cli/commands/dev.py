"""Internal developer commands (hidden).

`osmosis dev serve` runs the rollout in the current directory as a local server
and exposes it to the GPU cluster over an iroh (`dumbpipe`) tunnel. It prints the
tunnel ticket; paste that into your training config under [advanced] as
`local_rollout_ticket` and submit — the per-run ECS rollout slot then forwards
its rollout calls to this machine instead of cloning the rollout repo.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

import typer

from osmosis_ai.cli.errors import CLIError

app: typer.Typer = typer.Typer(
    help="Internal developer tooling.",
    no_args_is_help=True,
)

DEFAULT_ROLLOUT_PORT = 8000

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
    """Return a path to a `dumbpipe` binary, downloading a pinned release if needed."""
    existing = shutil.which("dumbpipe")
    if existing:
        return existing

    system = platform.system().lower()
    arch = _ARCH_ALIASES.get(platform.machine().lower())
    sha = _DUMBPIPE_SHA256.get((system, arch or ""))
    if system not in ("darwin", "linux") or arch is None or sha is None:
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


def _terminate(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@app.command("serve")
def serve(
    port: int = typer.Option(
        DEFAULT_ROLLOUT_PORT,
        "--port",
        help="Local port for the rollout server.",
    ),
) -> None:
    """Serve the rollout in the current directory over an iroh tunnel.

    Run from a rollout repo root (containing ``main.py``). Prints the tunnel
    ticket to stdout, then holds the server + tunnel open until interrupted.
    """
    repo = Path.cwd()
    main_file = repo / "main.py"
    if not main_file.is_file():
        raise CLIError(
            f"No main.py in {repo}. Run `osmosis dev serve` from a rollout repo root.",
            code="INVALID_USAGE",
        )

    dumbpipe = _ensure_dumbpipe()
    work = Path(tempfile.mkdtemp(prefix="osmosis-dev-serve-"))
    server_log = work / "rollout-server.log"
    tunnel_log = work / "dumbpipe.log"
    server: subprocess.Popen[bytes] | None = None
    tunnel: subprocess.Popen[bytes] | None = None

    env = os.environ.copy()
    env["ROLLOUT_PORT"] = str(port)
    env["_OSMOSIS_ROLLOUT_PORT"] = str(port)

    try:
        _progress(f"Starting rollout server on :{port} ...")
        with server_log.open("wb") as fh:
            server = subprocess.Popen(
                [sys.executable, str(main_file)],
                cwd=str(repo),
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
            )
        _wait_until_healthy(port, server, server_log)

        _progress("Rollout server healthy. Opening iroh tunnel ...")
        with tunnel_log.open("wb") as fh:
            tunnel = subprocess.Popen(
                [dumbpipe, "listen-tcp", "--host", f"localhost:{port}"],
                stdout=fh,
                stderr=subprocess.STDOUT,
            )
        ticket = _capture_ticket(tunnel, tunnel_log)

        # The ticket is the command's output and must reach stdout reliably even
        # when piped — console.print() no-ops off a TTY, so write it directly.
        sys.stdout.write(ticket + "\n")
        sys.stdout.flush()

        _progress("")
        _progress("Add this to your training config under [advanced], then submit:")
        _progress(f'    local_rollout_ticket = "{ticket}"')
        _progress("")
        _progress(f"Rollout server + tunnel running on :{port} (Ctrl-C to stop).")

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
        _terminate(tunnel)
        _terminate(server)
        shutil.rmtree(work, ignore_errors=True)
