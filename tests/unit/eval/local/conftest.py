"""Shared scaffolding for local-eval runner tests.

The E2E fixtures build a real rollout project on disk and let the supervisor
spawn it as a subprocess, so the tests exercise the same path a user does:
subprocess lifecycle, artifact-root override, HTTP dispatch, callbacks,
journalling, and materialization.
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from osmosis_ai.eval.local.dataset import ResolvedDataset, RowSelection, select_rows
from osmosis_ai.eval.local.runner import (
    EvalRunSpec,
    LocalEvalOptions,
    LocalEvalRunner,
    ProgressSnapshot,
)

# A self-contained rollout project. The workflow makes one real chat call
# through the controller's LiteLLM bridge so the whole LLM path is exercised
# (bridge -> litellm -> OpenAI-compatible upstream stub), and the grader
# rewards an exact match against the row's label.
ROLLOUT_MAIN = """\
import json
import os
import sys

import httpx
import uvicorn

from osmosis_ai.rollout import (
    AgentWorkflow,
    AgentWorkflowContext,
    AgentWorkflowOutput,
    Grader,
    GraderContext,
    LocalBackend,
    get_rollout_context,
)
from osmosis_ai.rollout.server import create_rollout_server


class EchoWorkflow(AgentWorkflow):
    async def run(self, ctx: AgentWorkflowContext) -> AgentWorkflowOutput:
        import asyncio

        if os.environ.get("OSMOSIS_TEST_ECHO_SECRET"):
            print("workflow saw MY_TOKEN=" + os.environ.get("MY_TOKEN", ""), flush=True)
        delay = float(os.environ.get("OSMOSIS_TEST_WORKFLOW_SLEEP", "0") or 0)
        if delay:
            await asyncio.sleep(delay)
        rollout = get_rollout_context()
        prompt = ctx.prompt[-1]["content"] if ctx.prompt else ""
        reply = "no-llm"
        if rollout is not None and rollout.chat_completions_url:
            headers = {"Authorization": f"Bearer {rollout.api_key}"}
            body = {
                "model": "local-eval",
                "messages": list(ctx.prompt),
                "stream": True,
            }
            content = []
            async with httpx.AsyncClient(timeout=30.0) as client:
                url = rollout.chat_completions_url.rstrip("/") + "/chat/completions"
                async with client.stream("POST", url, json=body, headers=headers) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:].strip()
                        if not payload or payload == "[DONE]":
                            continue
                        chunk = json.loads(payload)
                        for choice in chunk.get("choices") or []:
                            piece = (choice.get("delta") or {}).get("content")
                            if piece:
                                content.append(piece)
            reply = "".join(content) or "empty"
        if os.environ.get("OSMOSIS_TEST_WRITE_ARTIFACT") and ctx.artifacts_dir:
            (ctx.artifacts_dir / "note.txt").write_text(reply)
        return AgentWorkflowOutput(
            messages=[*ctx.prompt, {"role": "assistant", "content": reply}]
        )


class MatchGrader(Grader):
    async def grade(self, ctx: GraderContext) -> None:
        if os.environ.get("OSMOSIS_TEST_GRADER_CRASH"):
            raise RuntimeError("grader exploded on purpose")
        if os.environ.get("OSMOSIS_TEST_REMOVE_SAMPLE") and ctx.sample is not None:
            ctx.sample.remove_sample = True
            return
        messages = list(ctx.sample.messages) if ctx.sample is not None else []
        final = messages[-1]["content"] if messages else ""
        ctx.set_reward(1.0 if ctx.label is not None and ctx.label in final else 0.0)


backend = LocalBackend(workflow=EchoWorkflow, grader=MatchGrader)
app = create_rollout_server(backend=backend)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.environ["_OSMOSIS_ROLLOUT_PORT"]),
        log_level="warning",
    )
"""

DATASET_ROWS = [
    {"user_prompt": "say ok", "ground_truth": "ok"},
    {"user_prompt": "say ok twice", "ground_truth": "ok"},
    {"user_prompt": "say ok thrice", "ground_truth": "ok"},
    {"user_prompt": "say ok again", "ground_truth": "ok"},
]


@dataclass
class RecordingHooks:
    """Captures everything the supervisor reports back to the CLI layer."""

    secrets: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    stages: list[str] = field(default_factory=list)
    confirmations: list[tuple[int, str]] = field(default_factory=list)
    progress_snapshots: list[ProgressSnapshot] = field(default_factory=list)
    secret_requests: list[list[str]] = field(default_factory=list)
    refuse_confirmation: bool = False
    accept_new_run: bool = False
    new_run_prompts: list[tuple[str, int]] = field(default_factory=list)

    def note(self, message: str) -> None:
        self.notes.append(message)

    def stage(self, message: str) -> None:
        self.stages.append(message)

    async def confirm_dispatch(self, *, pending: int, model_path: str) -> None:
        self.confirmations.append((pending, model_path))
        if self.refuse_confirmation:
            raise RuntimeError("dispatch declined")

    async def confirm_new_run(self, *, run_name: str, total: int) -> bool:
        self.new_run_prompts.append((run_name, total))
        return self.accept_new_run

    def resolve_secrets(self, names: Sequence[str]) -> dict[str, str]:
        self.secret_requests.append(list(names))
        return dict(self.secrets)

    def progress(self, snapshot: ProgressSnapshot) -> None:
        self.progress_snapshots.append(snapshot)


@dataclass
class RunnerHarness:
    """A ready-to-run supervisor plus the paths its assertions need.

    The stub's base URL and key travel through the litellm provider env vars
    (set by the ``harness`` fixture), exactly as a user's credentials would.
    """

    rollout_dir: Path
    output_root: Path
    dataset: ResolvedDataset
    selection: RowSelection
    hooks: RecordingHooks

    def spec(self, **overrides: Any) -> EvalRunSpec:
        payload: dict[str, Any] = {
            "rollout_name": "echo-rollout",
            "entrypoint": "main.py",
            "model_path": "openai/gpt-5-mini",
            "dataset_name": "echo",
            "n": 1,
            "pass_threshold": 1.0,
            "agent_timeout_sec": 60.0,
            "grader_timeout_sec": 30.0,
        }
        payload.update(overrides)
        return EvalRunSpec(**payload)

    def runner(
        self,
        *,
        spec: EvalRunSpec | None = None,
        options: LocalEvalOptions | None = None,
        selection: RowSelection | None = None,
        hooks: RecordingHooks | None = None,
    ) -> LocalEvalRunner:
        # The fake rollout project below has no pyproject.toml and needs none:
        # pinning the interpreter keeps every E2E spawn direct, offline, and out
        # of uv's resolver.
        options = replace(
            options or LocalEvalOptions(name="run-1"),
            server_interpreter=sys.executable,
        )
        return LocalEvalRunner(
            spec=spec or self.spec(),
            options=options,
            dataset=self.dataset,
            selection=selection or self.selection,
            rollout_dir=self.rollout_dir,
            output_root=self.output_root,
            hooks=hooks or self.hooks,
        )

    def run_dir(self, name: str = "run-1") -> Path:
        return self.output_root / name

    def index_rows(self, name: str = "run-1") -> list[dict[str, Any]]:
        path = self.run_dir(name) / "index.jsonl"
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def journal_lines(self, name: str = "run-1") -> list[dict[str, Any]]:
        path = self.run_dir(name) / "events.jsonl"
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]


@pytest.fixture
def rollout_project(tmp_path: Path) -> Path:
    project = tmp_path / "rollouts" / "echo-rollout"
    project.mkdir(parents=True)
    (project / "main.py").write_text(ROLLOUT_MAIN, encoding="utf-8")
    return project


@pytest.fixture
def dataset_file(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "echo.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in DATASET_ROWS), encoding="utf-8"
    )
    return path


@pytest.fixture
async def openai_stub() -> AsyncIterator[str]:
    """A running OpenAI-compatible upstream. Yields its base URL."""
    from osmosis_ai.rollout.controller.listener import LocalhostUvicornServer
    from tests.unit.rollout.openai_stub import create_openai_stub_app

    async with LocalhostUvicornServer(create_openai_stub_app()) as server:
        yield server.base_url


@pytest.fixture
async def harness(
    tmp_path: Path,
    rollout_project: Path,
    dataset_file: Path,
    openai_stub: str,
    monkeypatch: pytest.MonkeyPatch,
) -> RunnerHarness:
    from osmosis_ai.eval.local.dataset import resolve_explicit_dataset_file

    # The supervisor passes no credentials of its own: litellm resolves the
    # openai/* provider from the environment, in-process and in the spawned
    # crash-resume supervisor alike (subprocesses inherit os.environ).
    monkeypatch.setenv("OPENAI_API_BASE", openai_stub)
    monkeypatch.setenv("OPENAI_BASE_URL", openai_stub)
    monkeypatch.setenv("OPENAI_API_KEY", "stub-llm-key")
    return RunnerHarness(
        rollout_dir=rollout_project,
        output_root=tmp_path / "evals",
        dataset=resolve_explicit_dataset_file(dataset_file),
        selection=select_rows(dataset_file),
        hooks=RecordingHooks(),
    )
