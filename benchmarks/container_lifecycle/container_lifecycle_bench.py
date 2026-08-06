"""Benchmark the Harbor rollout backend end to end on local Docker.

Starts a stub OpenAI endpoint, a controller that receives the protocol
callbacks, and a real rollout server, then drives concurrent rollouts through
the full container pipeline for several consecutive runs.

Usage:
    uv run benchmarks/container_lifecycle/container_lifecycle_bench.py \\
        --runs 5 --concurrency 20

Point it at any Harbor task folder instead of the generated one:

    uv run benchmarks/container_lifecycle/container_lifecycle_bench.py \\
        --tasks-dir path/to/task

The bench harness declares its SDK source in bench_harness/pyproject.toml,
so any image with python3 and pip works; the bundle install pulls the SDK.
"""

from __future__ import annotations

import argparse
import asyncio
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from harbor.trial.queue import TrialQueue

from osmosis_ai.rollout.backend.harbor import HarborBackend
from osmosis_ai.rollout.server import create_rollout_server

HARNESS_DIR = Path(__file__).resolve().parent / "bench_harness"
sys.path.insert(0, str(HARNESS_DIR))  # a real server runs inside its own project
WORKFLOW = "bench_harness.solver:BenchWorkflow"
GRADER = "bench_harness.grade:BenchGrader"

DOCKERFILE = """\
FROM python:3.12-slim
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
        except TimeoutError:
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


def prepare_task(work_dir: Path) -> Path:
    task_dir = work_dir / "bench-task"
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True)
    (env_dir / "Dockerfile").write_text(DOCKERFILE)
    (task_dir / "task.toml").write_text('[task]\nname = "osmosis/bench"\n')
    return task_dir


def make_backend(
    task_dir: Path,
    concurrency: int,
    keep_trials: bool,
    patch_dockerfile_with_sdk: bool = True,
):
    return HarborBackend(
        orchestrator=TrialQueue(n_concurrent=concurrency),
        tasks_dir=task_dir,
        agent=WORKFLOW,
        grader=GRADER,
        cleanup_successful_trials=not keep_trials,
        patch_dockerfile_with_sdk=patch_dockerfile_with_sdk,
    )


async def run_series(
    task_dir: Path,
    stub_url: str,
    runs: int,
    concurrency: int,
    timeout: float,
    keep_trials: bool,
    patch_dockerfile_with_sdk: bool = True,
) -> dict:
    setup_start = time.monotonic()
    backend = make_backend(
        task_dir, concurrency, keep_trials, patch_dockerfile_with_sdk
    )
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
    # Unique per invocation: rollout ids become docker compose project names
    # and trial dirs, so two overlapping bench runs must never share them.
    nonce = uuid.uuid4().hex[:6]
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            for run in range(1, runs + 1):
                start = time.monotonic()
                outcomes = await asyncio.gather(
                    *(
                        controller.submit(
                            client, f"bench-{nonce}-run{run}-{i}", timeout
                        )
                        for i in range(concurrency)
                    )
                )
                wall = time.monotonic() - start
                walls.append(wall)
                if run > 1:
                    warm_latencies.extend(latency for status, latency in outcomes)
                succeeded = sum(
                    1 for status, latency in outcomes if status == "success"
                )
                failures += concurrency - succeeded
                print(
                    f"run {run}: {wall:.1f}s, {succeeded}/{concurrency} ok, "
                    f"{concurrency / wall:.2f} rollouts/s"
                )
    finally:
        for server, _task in servers:
            server.should_exit = True
        await asyncio.gather(*(task for server, task in servers))

    return {
        "setup": setup,
        "walls": walls,
        "warm_latencies": warm_latencies,
        "failures": failures,
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def report(result: dict, concurrency: int) -> None:
    header = (
        f"{'setup':>7} {'cold':>7} "
        f"{'warm mean':>10} {'warm max':>9} {'warm rps':>9} "
        f"{'lat p50':>8} {'lat p95':>8} {'lat max':>8}"
    )
    print(f"\nwarm = runs 2+; lat = per-rollout submit->graded seconds\n{header}")
    warm = result["walls"][1:]
    latencies = result["warm_latencies"]
    if not warm:
        print(f"{result['setup']:>6.1f}s {result['walls'][0]:>6.1f}s  (single run)")
    else:
        mean = statistics.mean(warm)
        print(
            f"{result['setup']:>6.1f}s {result['walls'][0]:>6.1f}s "
            f"{mean:>9.1f}s {max(warm):>8.1f}s {concurrency / mean:>9.2f} "
            f"{percentile(latencies, 0.5):>7.1f}s "
            f"{percentile(latencies, 0.95):>7.1f}s "
            f"{max(latencies):>7.1f}s"
        )
    if result["failures"]:
        raise SystemExit(f"failures: {result['failures']}")


async def bench(args: argparse.Namespace) -> None:
    work_dir = Path(tempfile.mkdtemp(prefix="harbor-bench-"))
    print(f"work dir: {work_dir}")

    stub_port = free_port()
    stub_server, stub_task = await serve(stub_llm(args.latency), stub_port)
    stub_url = f"http://127.0.0.1:{stub_port}"

    try:
        result = await run_series(
            args.tasks_dir or prepare_task(work_dir),
            stub_url,
            args.runs,
            args.concurrency,
            args.timeout,
            args.keep_trials,
            args.patch_dockerfile_with_sdk,
        )
    finally:
        stub_server.should_exit = True
        await stub_task

    report(result, args.concurrency)


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
        "--no-patch-dockerfile-with-sdk",
        dest="patch_dockerfile_with_sdk",
        action="store_false",
        help="disable the default Dockerfile patch that pre-installs the "
        "harness's dependencies into the task image; dependencies then "
        "download inside every trial container",
    )
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
