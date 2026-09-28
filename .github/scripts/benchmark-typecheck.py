"""Compare warm type-check wall times in one checkout and Python environment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "typecheck-benchmark"
REPETITIONS = 5
ENVIRONMENT_OVERRIDES = {"PYRIGHT_PYTHON_IGNORE_WARNINGS": "1"}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def input_hashes() -> dict[str, str]:
    source = hashlib.sha256()
    for path in sorted((ROOT / "osmosis_ai").rglob("*")):
        if path.is_file() and (
            path.suffix in {".py", ".pyi"} or path.name == "py.typed"
        ):
            source.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
            source.update(hashlib.sha256(path.read_bytes()).digest())
    return {
        "source_sha256": source.hexdigest(),
        **{
            f"{name}_sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("uv.lock", "pyproject.toml")
        },
    }


def run_check(command: list[str], log: Path) -> tuple[int, float]:
    with log.open("w") as output:
        start = time.perf_counter()
        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=os.environ | ENVIRONMENT_OVERRIDES,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
            returncode = result.returncode
        except OSError as error:
            output.write(f"Unable to start checker: {error}\n")
            returncode = 127
        return returncode, time.perf_counter() - start


def main() -> int:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    python_version = config["tool"]["pyright"]["pythonVersion"]
    if python_version != config["tool"]["ty"]["environment"]["python-version"]:
        raise SystemExit("ty and Pyright must target the same Python version")

    scripts = Path(sys.executable).parent
    commands = {
        "ty": [
            str(scripts / "ty"),
            "check",
            "--python",
            sys.executable,
            "--python-version",
            python_version,
            "--python-platform",
            sys.platform,
            "--error-on-warning",
            "osmosis_ai/",
        ],
        "pyright": [
            str(scripts / "pyright"),
            "--pythonpath",
            sys.executable,
            "--pythonversion",
            python_version,
            "--pythonplatform",
            platform.system(),
            "osmosis_ai/",
        ],
    }
    # Benchmark artifacts do not make a previously clean checkout dirty.
    status = git("status", "--porcelain", "--", ".", ":(exclude)typecheck-benchmark")
    hashes = input_hashes()
    result = {
        "started_at": datetime.now(UTC).isoformat(),
        "execution_environment": (
            "github-actions" if os.getenv("GITHUB_ACTIONS") == "true" else "local"
        ),
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_status": status,
        "input_hashes": hashes,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "target_python_version": python_version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "github": {
            name: os.environ[name]
            for name in (
                "GITHUB_RUN_ID",
                "GITHUB_RUN_ATTEMPT",
                "GITHUB_SHA",
                "RUNNER_OS",
                "RUNNER_ARCH",
                "ImageOS",
                "ImageVersion",
            )
            if name in os.environ
        },
        "installed_distributions": dict(
            sorted(
                (dist.metadata["Name"], dist.version)
                for dist in importlib.metadata.distributions()
            )
        ),
        "scope": "osmosis_ai/",
        "commands": commands,
        "environment_overrides": ENVIRONMENT_OVERRIDES,
        "warmups_per_checker": 1,
        "measured_runs_per_checker": REPETITIONS,
    }
    OUTPUT.mkdir(exist_ok=True)
    runs = []
    for iteration in range(REPETITIONS + 1):
        phase = "warmup" if iteration == 0 else "measured"
        # Reverse order each round to reduce systematic first-run bias.
        order = list(commands) if iteration % 2 == 0 else list(reversed(commands))
        for checker in order:
            log = OUTPUT / f"{checker}-{phase}-{iteration}.log"
            returncode, seconds = run_check(commands[checker], log)
            runs.append(
                {
                    "checker": checker,
                    "phase": phase,
                    "iteration": iteration,
                    "returncode": returncode,
                    "seconds": seconds,
                    "log": log.name,
                }
            )
            print(
                f"{checker} {phase} {iteration}: {seconds:.3f}s (exit {returncode})",
                flush=True,
            )

    unchanged = hashes == input_hashes() and result["git_head"] == git(
        "rev-parse", "HEAD"
    )
    valid = unchanged and all(run["returncode"] == 0 for run in runs)
    summary = {}
    for checker in commands:
        times = [
            run["seconds"]
            for run in runs
            if run["checker"] == checker and run["phase"] == "measured"
        ]
        summary[checker] = {
            "median_seconds": statistics.median(times),
            "min_seconds": min(times),
            "max_seconds": max(times),
        }
    result.update(
        runs=runs,
        summary=summary,
        inputs_unchanged=unchanged,
        valid_comparison=valid,
        completed_at=datetime.now(UTC).isoformat(),
    )
    (OUTPUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"valid_comparison": valid, "summary": summary}, indent=2))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
