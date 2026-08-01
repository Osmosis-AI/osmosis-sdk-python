"""Benchmark the Harbor rollout backends end to end on local Docker.

Starts a stub OpenAI endpoint, a controller that receives the protocol
callbacks, and a real rollout server, then drives concurrent rollouts through
the full container pipeline for several consecutive runs.

Backends:
    new — HarborBackendV2: pure task image, bundle wheel pip-installed
          per trial, Harbor's content-addressed image cache.
    old — HarborBackend: harness code and SDK baked into the image at
          backend construction, no per-trial install.

Usage:
    uv run benchmarks/container_lifecycle/container_lifecycle_bench.py \\
        --backend both --runs 5 --concurrency 20

Point it at any Harbor task folder instead of the generated one:

    uv run benchmarks/container_lifecycle/container_lifecycle_bench.py \\
        --tasks-dir path/to/task --backend new

The bench harness declares its SDK source in bench_harness/pyproject.toml,
so any image with python3 and pip works; the bundle install pulls the SDK.
Arbitrary tasks run on the new backend only: the old backend requires a
Dockerfile that COPYs /workspace, which normal Harbor tasks don't have.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from harbor.trial.queue import TrialQueue

from osmosis_ai.packaging import build_bundle
from osmosis_ai.rollout.backend.harbor import HarborBackendV2, HarborBackend
from osmosis_ai.rollout.server import create_rollout_server

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_DIR = Path(__file__).resolve().parent / "bench_harness"
sys.path.insert(0, str(HARNESS_DIR))  # a real server runs inside its own project
WORKFLOW = "bench_harness.solver:BenchWorkflow"
GRADER = "bench_harness.grade:BenchGrader"

NEW_DOCKERFILE = """\
FROM python:3.12-slim
"""

OLD_DOCKERFILE = """\
FROM python:3.12-slim
COPY {wheel} /tmp/{wheel}
RUN pip install --no-cache-dir /tmp/{wheel}
COPY workspace /workspace
CMD ["sleep", "infinity"]
"""


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def stub_llm(latency: float) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def completions(request: Request) -> dict:
        await asyncio.sleep(latency)
        return {
            "id": "bench",
            "object": "chat.completion",
            "model": "bench",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "bench response"},
                    "finish_reason": "stop",
                }
            ],
        }

    return app


class Controller:
    """The protocol's controller half: submits rollouts, receives callbacks."""

    def __init__(self, port: int, rollout_url: str, stub_url: str):
        self.port = port
        self.rollout_url = rollout_url
        self.stub_url = stub_url
        self.outcomes: dict[str, asyncio.Future[str]] = {}
        self.app = FastAPI()

        @self.app.post("/rollout/{rollout_id}/completed")
        async def completed(rollout_id: str, request: Request) -> dict:
            return {"status": "ok"}

        @self.app.post("/grader/{rollout_id}/completed")
        async def graded(rollout_id: str, request: Request) -> dict:
            body = await request.json()
            future = self.outcomes.get(rollout_id)
            if future and not future.done():
                future.set_result(body.get("status", "unknown"))
            return {"status": "ok"}

    async def submit(
        self, client: httpx.AsyncClient, rollout_id: str, timeout: float
    ) -> tuple[str, float]:
        """Run one rollout; return its outcome status and wall time."""
        self.outcomes[rollout_id] = asyncio.get_running_loop().create_future()
        base = f"http://127.0.0.1:{self.port}"
        start = time.monotonic()
        response = await client.post(
            f"{self.rollout_url}/rollout",
            json={
                "rollout_id": rollout_id,
                "initial_messages": [{"role": "user", "content": "bench"}],
                "label": "bench",
                "chat_completions_url": f"{self.stub_url}/v1",
                "completion_callback_url": f"{base}/rollout/{rollout_id}/completed",
                "grader_callback_url": f"{base}/grader/{rollout_id}/completed",
                "controller_api_key": "bench",
                "agent_timeout_sec": timeout,
                "grader_timeout_sec": timeout,
            },
        )
        response.raise_for_status()
        try:
            status = await asyncio.wait_for(self.outcomes[rollout_id], timeout)
        except asyncio.TimeoutError:
            status = "timeout"
        return status, time.monotonic() - start


async def serve(app: FastAPI, port: int) -> tuple[uvicorn.Server, asyncio.Task]:
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    return server, task


def build_sdk_wheel(work_dir: Path) -> Path:
    print("building osmosis-ai wheel...")
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(work_dir), str(REPO_ROOT)],
        check=True,
        capture_output=True,
    )
    return next(work_dir.glob("osmosis_ai-*.whl"))


def prepare_task(work_dir: Path, wheel: Path, kind: str) -> Path:
    task_dir = work_dir / f"bench-task-{kind}"
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True)
    if kind == "new":
        (env_dir / "Dockerfile").write_text(NEW_DOCKERFILE)
    else:
        shutil.copy2(wheel, env_dir / wheel.name)
        (env_dir / "Dockerfile").write_text(OLD_DOCKERFILE.format(wheel=wheel.name))
    (task_dir / "task.toml").write_text(f'[task]\nname = "osmosis/bench-{kind}"\n')
    return task_dir


def make_backend(
    kind: str,
    task_dir: Path,
    concurrency: int,
    keep_trials: bool,
    custom_sdk_pip_package: str | None = None,
):
    orchestrator = TrialQueue(n_concurrent=concurrency)
    if kind == "new":
        bundle = (
            build_bundle(HARNESS_DIR, workflow=WORKFLOW, grader=GRADER, deps=[custom_sdk_pip_package])
            if custom_sdk_pip_package
            else None
        )
        return HarborBackendV2(
            orchestrator=orchestrator,
            tasks_dir=task_dir,
            agent=WORKFLOW,
            grader=GRADER,
            bundle=bundle,
            cleanup_successful_trials=not keep_trials,
        )
    return HarborBackend(
        orchestrator=orchestrator,
        task_dir=task_dir,
        user_code_dir=HARNESS_DIR / "bench_harness",
        workflow=WORKFLOW,
        grader=GRADER,
        cleanup_successful_trials=not keep_trials,
    )


async def run_series(
    kind: str,
    task_dir: Path,
    stub_url: str,
    runs: int,
    concurrency: int,
    timeout: float,
    keep_trials: bool,
    custom_sdk_pip_package: str | None = None,
) -> dict:
    setup_start = time.monotonic()
    backend = make_backend(kind, task_dir, concurrency, keep_trials, custom_sdk_pip_package)
    setup = time.monotonic() - setup_start

    controller_port, rollout_port = free_port(), free_port()
    controller = Controller(
        controller_port,
        rollout_url=f"http://127.0.0.1:{rollout_port}",
        stub_url=stub_url,
    )
    servers = [
        await serve(controller.app, controller_port),
        await serve(create_rollout_server(backend=backend), rollout_port),
    ]

    walls: list[float] = []
    warm_latencies: list[float] = []
    failures = 0
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            for run in range(1, runs + 1):
                start = time.monotonic()
                outcomes = await asyncio.gather(
                    *(
                        controller.submit(client, f"{kind}-run{run}-{i}", timeout)
                        for i in range(concurrency)
                    )
                )
                wall = time.monotonic() - start
                walls.append(wall)
                if run > 1:
                    warm_latencies.extend(latency for status, latency in outcomes)
                succeeded = sum(1 for status, latency in outcomes if status == "success")
                failures += concurrency - succeeded
                print(
                    f"[{kind}] run {run}: {wall:.1f}s, {succeeded}/{concurrency} ok, "
                    f"{concurrency / wall:.2f} rollouts/s"
                )
    finally:
        for server, task in servers:
            server.should_exit = True
        await asyncio.gather(*(task for server, task in servers))

    return {
        "kind": kind,
        "setup": setup,
        "walls": walls,
        "warm_latencies": warm_latencies,
        "failures": failures,
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def report(results: list[dict], concurrency: int) -> None:
    header = (
        f"{'backend':<8} {'setup':>7} {'cold':>7} "
        f"{'warm mean':>10} {'warm max':>9} {'warm rps':>9} "
        f"{'lat p50':>8} {'lat p95':>8} {'lat max':>8}"
    )
    print(f"\nwarm = runs 2+; lat = per-rollout submit->graded seconds\n{header}")
    for r in results:
        warm = r["walls"][1:]
        latencies = r["warm_latencies"]
        if not warm:
            print(f"{r['kind']:<8} {r['setup']:>6.1f}s {r['walls'][0]:>6.1f}s  (single run)")
            continue
        mean = statistics.mean(warm)
        print(
            f"{r['kind']:<8} {r['setup']:>6.1f}s {r['walls'][0]:>6.1f}s "
            f"{mean:>9.1f}s {max(warm):>8.1f}s {concurrency / mean:>9.2f} "
            f"{percentile(latencies, 0.5):>7.1f}s "
            f"{percentile(latencies, 0.95):>7.1f}s "
            f"{max(latencies):>7.1f}s"
        )
    if any(r["failures"] for r in results):
        raise SystemExit(
            f"failures: {({r['kind']: r['failures'] for r in results})}"
        )


async def bench(args: argparse.Namespace) -> None:
    kinds = ["old", "new"] if args.backend == "both" else [args.backend]
    if args.tasks_dir and kinds != ["new"]:
        raise SystemExit(
            "--tasks-dir requires --backend new: the old backend needs a "
            "Dockerfile that COPYs /workspace, which normal Harbor tasks lack"
        )

    work_dir = Path(tempfile.mkdtemp(prefix="harbor-bench-"))
    print(f"work dir: {work_dir}")
    wheel = build_sdk_wheel(work_dir) if "old" in kinds else None

    stub_port = free_port()
    stub_server, stub_task = await serve(stub_llm(args.latency), stub_port)
    stub_url = f"http://127.0.0.1:{stub_port}"

    results = []
    try:
        for kind in kinds:
            task_dir = args.tasks_dir or prepare_task(work_dir, wheel, kind)
            results.append(
                await run_series(
                    kind,
                    task_dir,
                    stub_url,
                    args.runs,
                    args.concurrency,
                    args.timeout,
                    args.keep_trials,
                    args.custom_sdk_pip_package,
                )
            )
    finally:
        stub_server.should_exit = True
        await stub_task

    report(results, args.concurrency)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=None,
        help="benchmark an arbitrary Harbor task folder instead of the "
        "generated default (see module docstring for image requirements)",
    )
    parser.add_argument(
        "--custom-sdk-pip-package",
        default=None,
        help="extra pip requirement added to the bundle, overriding the SDK "
        "source declared in bench_harness/pyproject.toml (edit that file "
        "instead for a persistent change). Example: 'osmosis-ai @ https://"
        "github.com/Osmosis-AI/osmosis-sdk-python/archive/refs/heads/"
        "<branch>.tar.gz'",
    )
    parser.add_argument("--backend", choices=["new", "old", "both"], default="both")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--latency", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--keep-trials", action="store_true")
    args = parser.parse_args()
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        raise SystemExit("docker daemon is required")
    asyncio.run(bench(args))


if __name__ == "__main__":
    main()
