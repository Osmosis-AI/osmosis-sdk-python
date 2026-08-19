"""The local evaluation supervisor.

Owns run state and resume, the rollout-server subprocess, bounded row x run
scheduling, cancellation, and result materialization (design
``local-eval-run-plan.md`` §3.1). Model calls stay on this machine: the
callback listener also mounts an in-process LiteLLM bridge, mirroring the
hosted eval controller, and litellm resolves provider credentials from the
environment (a ``--secrets-file`` entry reaches it the same way).

The one ordering invariant everything else serves: a terminal callback is
acknowledged **only after** its journal record is fully written and ``fsync``-ed,
so a durably acknowledged work item never runs again after ``kill -9``.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import logging
import os
import signal
import socket
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

from osmosis_ai._uv import _uv_executable
from osmosis_ai.consts import PACKAGE_VERSION
from osmosis_ai.eval.local._server_bootstrap import ROLLOUT_INSTANCE_HEADER
from osmosis_ai.eval.local.dataset import (
    EvalDatasetRow,
    ResolvedDataset,
    RowSelection,
    format_row_selector,
)
from osmosis_ai.eval.local.results import (
    CANONICAL_TRAJECTORY_FILENAME,
    TRIALS_DIRNAME,
    Materializer,
    RunIdentity,
    aggregate_metrics,
    read_valid_trajectory,
    select_attempts,
)
from osmosis_ai.eval.local.state import (
    DATASET_NORMALIZATION_VERSION,
    JOURNAL_FILENAME,
    LOCAL_STATE_SCHEMA_VERSION,
    LOCKS_DIRNAME,
    MANIFEST_FILENAME,
    ROLLOUT_PROTOCOL_VERSION,
    RunLock,
    RunManifest,
    TerminalJournal,
    TerminalRecord,
    TerminalStatus,
    WorkKey,
    archive_run_directory,
    digest_of,
    terminate_process_group,
    utc_now,
    validate_run_name,
)
from osmosis_ai.rollout.backend.harbor.diagnostics import REDACTED
from osmosis_ai.rollout.controller import (
    CallbackListener,
    CallbackStore,
    LiteLLMBridge,
    TerminalCallbackResult,
)
from osmosis_ai.rollout.driver import RolloutOutcome, RolloutRunRequest
from osmosis_ai.rollout.http_driver import HttpRolloutDriver, RolloutProtocolError
from osmosis_ai.rollout.types.protocol import GraderStatus
from osmosis_ai.rollout.types.sample import RolloutStatus
from osmosis_ai.source_scan import reject_directory_symlinks, source_digest

logger: logging.Logger = logging.getLogger(__name__)

LOGS_FILENAME = "logs.txt"

_HEALTH_TIMEOUT_SEC = 90.0
_HEALTH_POLL_INTERVAL_SEC = 0.2
# One retry covers a lost bind race on an ephemeral port; more would only make a
# hung server cost another full health timeout before the run gives up.
_PORT_ATTEMPTS = 2
_SERVER_TERM_GRACE_SEC = 5.0
_TRAJECTORY_GRACE_SEC = 30.0
_TRAJECTORY_POLL_INTERVAL_SEC = 0.2
_CALLBACK_NETWORK_GRACE_SEC = 60.0
# Floor for the per-item supervisor deadline when the config sets no timeouts.
_DEFAULT_ITEM_DEADLINE_SEC = 3600.0
# How long a cancelled run waits for the backend to unwind before the server is
# terminated. Sandboxed backends need real time here to destroy a container.
_CANCEL_SETTLE_SEC = 120.0
_CANCEL_POLL_INTERVAL_SEC = 0.5

#: Supervisor-owned subprocess variables a config must never set (§8).
RESERVED_ENV_NAMES: frozenset[str] = frozenset(
    {
        "_OSMOSIS_ROLLOUT_PORT",
        "_OSMOSIS_ROLLOUT_ARTIFACT_ROOT",
        "_OSMOSIS_ROLLOUT_INSTANCE_ID",
    }
)


class LocalEvalError(RuntimeError):
    """The local run cannot proceed."""


class ResumeRefusedError(LocalEvalError):
    """A named run's resolved inputs changed, so resuming would mix versions."""


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvalRunSpec:
    """Execution-semantic values read from the shared eval TOML (§5).

    Extracted by the CLI layer so the supervisor never imports config or CLI
    machinery, and so every field here is one the fingerprint may legitimately
    depend on.
    """

    rollout_name: str
    entrypoint: str
    model_path: str
    dataset_name: str
    n: int = 1
    batch_size: int | None = None
    pass_threshold: float = 1.0
    agent_timeout_sec: float | None = None
    grader_timeout_sec: float | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    secret_names: tuple[str, ...] = ()
    branch: str | None = None
    commit_sha: str | None = None


@dataclass(frozen=True)
class LocalEvalOptions:
    """Runtime-only knobs. All CLI flags -- never a config section (§5)."""

    name: str | None = None
    fresh: bool = False
    retry_failed: bool = False
    max_in_flight: int | None = None
    rollout_port: int | None = None
    verbose: bool = False
    # Test seam, not a CLI flag: the CLI always leaves this ``None``, which runs
    # the rollout server through uv so its dependencies resolve from
    # ``rollouts/<name>/pyproject.toml``. Tests set it to ``sys.executable`` so
    # their fake rollout servers spawn directly -- offline and fast.
    server_interpreter: str | None = None


class RunnerHooks(Protocol):
    """CLI-facing callbacks, so the supervisor holds no CLI dependencies."""

    def note(self, message: str) -> None:
        """Surface a human-readable status line."""

    def stage(self, message: str) -> None:
        """Surface a run milestone.

        Separate from ``note`` because the two feed different displays: ``note``
        carries the log echo a ``--verbose`` run streams, while stages are the
        handful of lines a plain run prints so the long gaps -- model preflight,
        rollout-server startup -- are not silent (§4.3).
        """

    async def confirm_dispatch(self, *, pending: int, model_path: str) -> None:
        """Cost boundary before any rollout is dispatched. Raise to abort.

        Awaited, not called: the CLI answers this by prompting the user, and a
        terminal prompt library needs the supervisor's own event loop to render
        on. A synchronous implementation that starts a second loop would fail.
        """

    def resolve_secrets(self, names: Sequence[str]) -> dict[str, str]:
        """Resolve workflow-secret values. Called only when work is pending."""
        ...

    def progress(self, snapshot: ProgressSnapshot) -> None:
        """Report live counts for the terminal progress bar."""


@dataclass(frozen=True)
class ProgressSnapshot:
    completed: int
    total: int
    passed: int
    failed: int

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.completed) if self.completed else 0.0


@dataclass(frozen=True)
class WorkItem:
    """One ``(row_index, run_index)`` work item awaiting a terminal result."""

    row: EvalDatasetRow
    run_index: int

    @property
    def key(self) -> WorkKey:
        return (self.row.row_index, self.run_index)


@dataclass
class FailedWorkItem:
    """A failed or skipped work item, for the end-of-run report (§4.3)."""

    row_index: int
    source_row_index: int
    run_index: int
    rollout_id: str
    error_type: str | None
    rollout_dir: Path


@dataclass
class RunSummary:
    """What the CLI reports when the supervisor returns."""

    run_dir: Path
    local_run_id: str
    run_name: str
    total_work_items: int
    dispatched: int
    succeeded: int
    failed: int
    skipped: int
    resumed: int
    cancelled: bool
    duration_ms: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    failures: list[FailedWorkItem] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #


def compute_source_digest(project_dir: Path, *, exclude: Path | None = None) -> str:
    """Full SHA-256 of the rollout project's source bytes (§5).

    Shares the packager's traversal, so the bundle cache and the resume lock can
    never disagree about which files describe the project. *exclude* keeps a run
    output directory nested inside the project from changing its own digest.
    """
    reject_directory_symlinks(project_dir, exclude=exclude, label="rollout source")
    return source_digest(project_dir, exclude=exclude)


def build_run_inputs(
    spec: EvalRunSpec,
    *,
    dataset: ResolvedDataset,
    selection: RowSelection,
    rollout_source_digest: str,
) -> dict[str, Any]:
    """The resolved-input lock: only execution-semantic inputs (§9.5).

    Deliberately excluded so an unrelated change never refuses a resume: run
    name, output path, throughput knobs (``--max-in-flight``,
    ``evaluation.batch_size``), UI flags, SDK version, branch display name, and
    timestamps. Secret *names* are included; values never are.
    """
    return {
        "model_path": spec.model_path,
        "dataset": {
            "sha256": dataset.sha256,
            "selected_source_rows": format_row_selector(selection.source_row_indices),
        },
        "n": spec.n,
        "rollout": {
            "name": spec.rollout_name,
            "entrypoint": spec.entrypoint,
            "source_digest": rollout_source_digest,
        },
        "env": dict(sorted(spec.env.items())),
        "secret_names": sorted(spec.secret_names),
        "timeouts": {
            "agent_timeout_sec": spec.agent_timeout_sec,
            "grader_timeout_sec": spec.grader_timeout_sec,
        },
        "versions": {
            "rollout_protocol": ROLLOUT_PROTOCOL_VERSION,
            "dataset_normalization": DATASET_NORMALIZATION_VERSION,
            "state_schema": LOCAL_STATE_SCHEMA_VERSION,
        },
    }


def generated_run_name(
    config_stem: str, inputs_digest: str, *, now: str | None = None
) -> str:
    """``<config-stem>-<timestamp>-<short-fingerprint>`` for an unnamed run (§4.4)."""
    stamp = now or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    safe_stem = (
        "".join(
            char if char.isalnum() or char in "._-" else "-" for char in config_stem
        ).strip("-._")
        or "eval"
    )
    return validate_run_name(f"{safe_stem}-{stamp}-{inputs_digest[:8]}")


def changed_input_keys(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> list[str]:
    """Top-level ``inputs`` keys that differ, added or removed included (§9.5).

    Naming the key is what makes a refusal actionable; the values themselves are
    digests and selectors that nobody reads out of an error message.
    """
    return sorted(
        key for key in {*previous, *current} if previous.get(key) != current.get(key)
    )


# --------------------------------------------------------------------------- #
# Run log
# --------------------------------------------------------------------------- #


class SecretRedactor:
    """Replaces known secret values in text bound for the run log (§7.4).

    Substring replacement rather than dropping the whole line: a redacted line
    still carries its stack frame or status code, which is the reason the line
    exists. Short values are ignored -- ``dummy`` and friends are placeholders,
    and redacting them would blank unrelated text.
    """

    _MIN_LENGTH = 8

    def __init__(self, values: Sequence[str] = ()) -> None:
        self._values: list[str] = []
        self.extend(values)

    def extend(self, values: Sequence[str]) -> None:
        for value in values:
            if value and len(value) >= self._MIN_LENGTH and value not in self._values:
                self._values.append(value)
        # Longest first, so a value containing another is redacted whole.
        self._values.sort(key=len, reverse=True)

    def scrub(self, text: str) -> str:
        for value in self._values:
            text = text.replace(value, REDACTED)
        return text


class RunLog:
    """Appends download-format lines to the run's combined ``logs.txt`` (§2.4).

    Format: ``<ISO> <LEVEL> [<step>] <message> <json details>`` -- shape
    compatible with what ``eval download`` synthesizes from ``eval_run_log``
    rows, so a local run and a downloaded run read the same way.
    """

    def __init__(
        self,
        path: Path,
        *,
        echo: Callable[[str], None] | None = None,
        redact: Callable[[str], str] | None = None,
    ) -> None:
        self._path = path
        self._echo = echo
        self._redact = redact
        path.parent.mkdir(parents=True, exist_ok=True)
        # The log carries rollout-server output, so it gets the same owner-only
        # mode as the journal even though redaction should keep it clean.
        path.touch(mode=0o600, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")

    def write(self, level: str, step: str, message: str, **details: Any) -> None:
        line = f"{utc_now()} {level.upper()} [{step}] {message}"
        if details:
            from osmosis_ai.eval.local.state import canonical_json

            line = f"{line} {canonical_json(details)}"
        if self._redact is not None:
            line = self._redact(line)
        self._handle.write(line + "\n")
        self._handle.flush()
        if self._echo is not None:
            self._echo(line)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._handle.close()


# --------------------------------------------------------------------------- #
# Rollout-server subprocess
# --------------------------------------------------------------------------- #


def reserve_free_port() -> int:
    """Pick a free localhost port.

    Inherently racy -- the rollout server binds it in another process -- so the
    caller must retry startup on a bind failure rather than trust this (§20.3).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def inherited_env(base: Mapping[str, str]) -> dict[str, str]:
    """Copy an inherited environment, minus ``VIRTUAL_ENV``.

    A parent shell's activated venv makes uv warn about a mismatched
    environment, and the direct-interpreter path does not need it either.
    """
    return {name: value for name, value in base.items() if name != "VIRTUAL_ENV"}


def uv_sync_command(uv_executable: str, rollout_dir: Path) -> list[str]:
    """Argv that resolves the rollout's own environment from its pyproject."""
    return [uv_executable, "sync", "--project", str(rollout_dir)]


def uv_run_command(
    uv_executable: str, rollout_dir: Path, entrypoint: Path
) -> list[str]:
    """Argv that runs the rollout server inside the rollout's own environment.

    ``--no-sync`` because dependencies are synced once beforehand, as their own
    attributable stage.
    """
    return [
        uv_executable,
        "run",
        "--no-sync",
        "--project",
        str(rollout_dir),
        str(Path(__file__).with_name("_server_bootstrap.py")),
        str(entrypoint),
    ]


def build_subprocess_env(
    *,
    base: Mapping[str, str],
    config_env: Mapping[str, str],
    secrets: Mapping[str, str],
    port: int,
    artifact_root: Path,
    instance_id: str,
) -> dict[str, str]:
    """Compose the rollout-server child environment, internals applied last (§8).

    A config must not redirect artifacts, spoof the instance id, or change the
    selected port, so colliding ``[env]``/``[secrets]`` names are refused rather
    than silently overridden.
    """
    collisions = sorted((set(config_env) | set(secrets)) & RESERVED_ENV_NAMES)
    if collisions:
        raise LocalEvalError(
            "[env]/[secrets] must not set supervisor-owned variables: "
            + ", ".join(collisions)
        )
    env = inherited_env(base)
    env.update(config_env)
    env.update(secrets)
    env.update(
        {
            "_OSMOSIS_ROLLOUT_PORT": str(port),
            "_OSMOSIS_ROLLOUT_ARTIFACT_ROOT": str(artifact_root),
            "_OSMOSIS_ROLLOUT_INSTANCE_ID": instance_id,
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


@dataclass(frozen=True)
class _HealthProbe:
    payload: dict[str, Any]
    instance_id: str | None


async def _probe_health(
    client: httpx.AsyncClient, base_url: str
) -> _HealthProbe | None:
    try:
        response = await client.get(f"{base_url}/health", timeout=5.0)
    except httpx.HTTPError:
        return None
    if response.status_code >= 400:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    return _HealthProbe(
        payload=payload,
        instance_id=response.headers.get(ROLLOUT_INSTANCE_HEADER),
    )


async def probe_health(
    client: httpx.AsyncClient, base_url: str
) -> dict[str, Any] | None:
    """Return ``/health`` when reachable, else ``None``."""
    probe = await _probe_health(client, base_url)
    return probe.payload if probe is not None else None


def health_capacity(health: Mapping[str, Any]) -> int | None:
    """Backend-advertised in-flight capacity from ``/health``, if any.

    Two shapes, because both backends answer this endpoint: Harbor reports
    ``max_queue_depth`` at top level, LocalBackend reports its limiter snapshot
    under ``concurrency``. Anything absent, non-integer, boolean, or
    non-positive means "unbounded" and yields no cap.
    """
    nested = health.get("concurrency")
    for value in (
        health.get("max_queue_depth"),
        nested.get("max_concurrent") if isinstance(nested, Mapping) else None,
    ):
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


# --------------------------------------------------------------------------- #
# Supervisor
# --------------------------------------------------------------------------- #


class LocalEvalRunner:
    """One local evaluation run, from lock acquisition to final materialization."""

    def __init__(
        self,
        *,
        spec: EvalRunSpec,
        options: LocalEvalOptions,
        dataset: ResolvedDataset,
        selection: RowSelection,
        rollout_dir: Path,
        output_root: Path,
        hooks: RunnerHooks,
        provenance: Mapping[str, Any] | None = None,
        config_stem: str = "eval",
    ) -> None:
        self._spec = spec
        self._options = options
        self._dataset = dataset
        self._selection = selection
        self._rollout_dir = rollout_dir
        self._output_root = output_root
        self._hooks = hooks
        self._provenance = dict(provenance or {})
        self._config_stem = config_stem

        self._run_dir: Path | None = None
        self._log: RunLog | None = None
        self._journal: TerminalJournal | None = None
        self._latest: dict[WorkKey, TerminalRecord] = {}
        self._resumed_keys: set[WorkKey] = set()
        self._dispatch_context: dict[str, WorkItem] = {}
        self._dispatch_started: dict[str, float] = {}
        self._bridge: LiteLLMBridge | None = None
        self._materializer: Materializer | None = None
        self._identity: RunIdentity | None = None
        self._child: asyncio.subprocess.Process | None = None
        self._child_reader: asyncio.Task[None] | None = None
        self._cancelled = asyncio.Event()
        self._halt_reason: str | None = None
        self._redactor = SecretRedactor()
        self._started_at = utc_now()
        self._started_monotonic = time.monotonic()
        self._dispatched = 0

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    async def run(self) -> RunSummary:
        """Execute the full startup order from §8, then schedule and finalize."""
        run_name = self._options.name or generated_run_name(
            self._config_stem, self._pending_inputs_digest()
        )
        validate_run_name(run_name)
        lock_path = self._output_root / LOCKS_DIRNAME / f"{run_name}.lock"
        with RunLock(lock_path):
            return await self._run_locked(run_name=run_name)

    # ------------------------------------------------------------------ #
    # Startup
    # ------------------------------------------------------------------ #

    def _pending_inputs_digest(self) -> str:
        return digest_of(self._build_inputs())

    def _build_inputs(self) -> dict[str, Any]:
        return build_run_inputs(
            self._spec,
            dataset=self._dataset,
            selection=self._selection,
            rollout_source_digest=self._source_digest(),
        )

    def _source_digest(self) -> str:
        cached = getattr(self, "_source_digest_value", None)
        if cached is None:
            if self._output_root == self._rollout_dir:
                raise LocalEvalError(
                    f"--output must not be the rollout source directory "
                    f"{self._rollout_dir}: excluding the output tree would then "
                    "exclude the whole project, freezing the resume fingerprint."
                )
            exclude = (
                self._output_root
                if self._output_root.is_relative_to(self._rollout_dir)
                else None
            )
            cached = compute_source_digest(self._rollout_dir, exclude=exclude)
            self._source_digest_value = cached
        return cached

    async def _run_locked(self, *, run_name: str) -> RunSummary:
        run_dir = self._output_root / run_name
        self._run_dir = run_dir
        inputs = self._build_inputs()
        manifest = self._open_or_create_run(run_dir, inputs=inputs, run_name=run_name)
        self._log = RunLog(
            run_dir / LOGS_FILENAME,
            echo=self._hooks.note if self._options.verbose else None,
            redact=self._redactor.scrub,
        )
        self._identity = RunIdentity(
            local_run_id=manifest.local_run_id,
            run_name=run_name,
            dataset_name=self._spec.dataset_name,
            model_name=self._spec.model_path,
            rollout_name=self._spec.rollout_name,
            started_at=self._started_at,
        )
        self._materializer = Materializer(run_dir)

        journal = TerminalJournal(run_dir / JOURNAL_FILENAME)
        try:
            replay = journal.replay()
            if replay.truncated_bytes:
                self._write_log(
                    "warning",
                    "resume",
                    "discarded a partial trailing journal record",
                    bytes=replay.truncated_bytes,
                )
            journal.open_for_append(replay)
            self._journal = journal
            self._latest = replay.latest
            self._resumed_keys = set(self._latest)

            pending = self._pending_work_items()
            self._refresh_snapshots()
            total = len(self._selection.rows) * max(1, self._spec.n)
            if not pending:
                self._stage(
                    "resume",
                    f"{run_name}: all {total} work items already have results",
                )
                return self._finalize(cancelled=False)
            recorded = len(self._latest)
            self._stage(
                "run",
                f"{run_name}: {len(pending)} of {total} work items pending"
                + (f", {recorded} already recorded" if recorded else ""),
            )

            await self._hooks.confirm_dispatch(
                pending=len(pending), model_path=self._spec.model_path
            )
            secrets = self._hooks.resolve_secrets(list(self._spec.secret_names))
            self._redactor.extend(list(secrets.values()))
            return await self._execute(pending, secrets=secrets)
        finally:
            journal.close()
            if self._log is not None:
                self._log.close()

    def _open_or_create_run(
        self, run_dir: Path, *, inputs: Mapping[str, Any], run_name: str
    ) -> RunManifest:
        """Compare, archive, or create -- the resolved-input lock gate (§4.4)."""
        manifest_path = run_dir / MANIFEST_FILENAME
        if manifest_path.is_file() and self._options.fresh:
            archived = archive_run_directory(run_dir)
            self._hooks.note(f"archived previous results to {archived}")
        elif manifest_path.is_file():
            existing = RunManifest.read(manifest_path)
            changed = changed_input_keys(existing.inputs, inputs)
            if changed:
                raise ResumeRefusedError(
                    f"run {run_name!r} was created with different resolved inputs, "
                    "so resuming it would mix versions inside one set of metrics. "
                    f"Changed: {', '.join(changed)}. Restart under the same name "
                    "with --fresh (the previous results are archived, never "
                    "deleted)."
                )
            return existing

        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / TRIALS_DIRNAME).mkdir(exist_ok=True)
        manifest = RunManifest.create(
            local_run_id=uuid.uuid4().hex,
            run_name=run_name,
            inputs=inputs,
            provenance={"sdk_version": PACKAGE_VERSION, **self._provenance},
        )
        manifest.write(manifest_path)
        return manifest

    def _pending_work_items(self) -> list[WorkItem]:
        """Work items with no terminal result, plus retries when asked (§9.4)."""
        pending: list[WorkItem] = []
        for row in self._selection.rows:
            for run_index in range(max(1, self._spec.n)):
                key = (row.row_index, run_index)
                record = self._latest.get(key)
                if record is None:
                    pending.append(WorkItem(row=row, run_index=run_index))
                    continue
                if self._options.retry_failed and record.status in (
                    "failed",
                    "skipped",
                ):
                    pending.append(WorkItem(row=row, run_index=run_index))
        return pending

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    async def _execute(
        self, pending: Sequence[WorkItem], *, secrets: Mapping[str, str]
    ) -> RunSummary:
        assert self._run_dir is not None
        controller_token = uuid.uuid4().hex
        bridge_token = uuid.uuid4().hex
        self._redactor.extend([controller_token, bridge_token])
        store = CallbackStore(on_terminal_commit=self._commit_terminal)

        # litellm resolves provider credentials from the environment; a secret
        # supplied through --secrets-file must reach it the same way.
        for name, value in secrets.items():
            os.environ.setdefault(name, value)

        bridge = LiteLLMBridge(model=self._spec.model_path)
        self._bridge = bridge
        listener = CallbackListener(
            store,
            auth_token=controller_token,
            bridge=bridge,
            bridge_token=bridge_token,
        )
        cancelled = False
        try:
            await listener.start()
            self._write_log("info", "listener", "callback listener started")
            self._stage("preflight", f"checking model {self._spec.model_path}")
            await self._model_preflight(bridge)
            async with httpx.AsyncClient() as client:
                base_url = await self._start_rollout_server(
                    secrets=secrets, client=client
                )
                driver = HttpRolloutDriver(
                    rollout_base_url=base_url,
                    callback_store=store,
                    completion_url_for=listener.completion_url,
                    grader_url_for=listener.grader_url,
                    chat_completions_url_for=listener.chat_completions_url,
                    chat_api_key=bridge_token,
                    controller_api_key=controller_token,
                    http_client=client,
                    callback_timeout_sec=self._callback_deadline(),
                )
                concurrency = await self._resolve_concurrency(client, base_url)
                self._stage(
                    "schedule",
                    f"running {len(pending)} work items, up to {concurrency} in flight",
                )
                # Open the progress display before the first result, not after
                # it: a rollout can take minutes, and that wait is exactly when
                # the user needs to see that something is running.
                self._report_progress()
                cancelled = await self._schedule(pending, driver, concurrency)
                if cancelled:
                    await self._settle_cancellations(client, base_url)
        finally:
            await self._stop_rollout_server()
            with contextlib.suppress(Exception):
                await listener.stop()
        return self._finalize(cancelled=cancelled)

    async def _settle_cancellations(
        self, client: httpx.AsyncClient, base_url: str
    ) -> None:
        """Give the rollout server a bounded grace to unwind cancelled work (§8).

        Without this the supervisor terminates the server milliseconds after the
        cancel is acknowledged, and a backend that owns real resources never
        reaches its teardown -- a Harbor trial leaves its sandbox container
        running with nothing left to stop it. Polling per-rollout status is the
        signal that the backend's own cancellation handler has finished.
        """
        rollout_ids = list(self._dispatch_context)
        if not rollout_ids:
            return
        self._stage(
            "cancel",
            f"waiting for {len(rollout_ids)} cancelled rollouts to unwind",
        )
        try:
            observed = await asyncio.wait_for(
                asyncio.gather(
                    *(
                        self._wait_settled(client, base_url, rollout_id)
                        for rollout_id in rollout_ids
                    )
                ),
                timeout=_CANCEL_SETTLE_SEC,
            )
        except TimeoutError:
            observed = None
        if observed is not None and all(observed):
            self._write_log("info", "cancel", "cancelled work unwound")
            return
        self._write_log(
            "warning",
            "cancel",
            "cancelled work did not unwind within the grace period; the backend "
            "may have left resources behind",
            grace_sec=_CANCEL_SETTLE_SEC,
        )

    async def _wait_settled(
        self, client: httpx.AsyncClient, base_url: str, rollout_id: str
    ) -> bool:
        """Poll one rollout until it stops running; False when unobservable.

        Nothing observed is no evidence the backend unwound -- the caller must
        not report it as such.
        """
        while True:
            running = await self._rollout_is_running(client, base_url, rollout_id)
            if running is None:
                return False
            if not running:
                return True
            await asyncio.sleep(_CANCEL_POLL_INTERVAL_SEC)

    async def _rollout_is_running(
        self, client: httpx.AsyncClient, base_url: str, rollout_id: str
    ) -> bool | None:
        """Whether the rollout is still working; ``None`` when unobservable.

        A status the supervisor cannot read leaves nothing to wait on, but it is
        also no evidence that the backend unwound -- callers must not report it
        as one. UNKNOWN is different: terminal records are retained for far
        longer than a teardown takes, so an id the server no longer knows is
        not in flight.
        """
        try:
            response = await client.get(
                f"{base_url}/rollout/{rollout_id}/status", timeout=5.0
            )
        except httpx.HTTPError:
            # An unreachable server cannot be waited on any further.
            return None
        if response.status_code >= 400:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        try:
            status = RolloutStatus(payload.get("status"))
        except ValueError:
            # An unrecognized status is no evidence the backend unwound.
            return None
        return status in (
            RolloutStatus.QUEUED,
            RolloutStatus.RUNNING,
            RolloutStatus.GRADING,
        )

    async def _model_preflight(self, bridge: LiteLLMBridge) -> None:
        """One-shot completion so a bad model or key fails once, loudly,
        before any dispatch instead of failing every work item."""
        try:
            await bridge.preflight_check()
        except Exception as exc:
            raise LocalEvalError(
                f"model preflight failed for {self._spec.model_path!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self._write_log("info", "preflight", "model preflight passed")

    def _halt_dispatch(self, reason: str, **details: Any) -> None:
        """Stop new dispatch without stamping the remaining queue failed (§9.3).

        A process-wide fault -- proxy auth, a dead rollout server, a network
        outage -- says nothing about the work items that have not run. Leaving
        them with no terminal record keeps them pending, so a later invocation
        resumes them once the cause is fixed. Stamping them ``failed`` would
        instead make a plain resume skip work that never executed.
        """
        if self._halt_reason is not None:
            return
        self._halt_reason = reason
        self._write_log(
            "error", "dispatch", "halting dispatch", reason=reason, **details
        )
        self._hooks.note(f"stopping dispatch: {reason}")

    def _callback_deadline(self) -> float:
        """Supervisor deadline = server timeout + callback/network grace (§10).

        Timeouts are optional in the config, but an unbounded wait is not an
        option: a lost callback would hang every worker forever with no output.
        A ``None`` phase means "run unbounded", so summing only the phases that
        are bounded would stamp still-running work ``callback_timeout``. With
        anything unbounded, fall back to a generous floor.
        """
        agent = self._spec.agent_timeout_sec
        grader = self._spec.grader_timeout_sec
        if agent is None or grader is None:
            return _DEFAULT_ITEM_DEADLINE_SEC
        server_budget = agent + grader
        if server_budget <= 0:
            return _DEFAULT_ITEM_DEADLINE_SEC
        return server_budget + _CALLBACK_NETWORK_GRACE_SEC

    async def _resolve_concurrency(
        self, client: httpx.AsyncClient, base_url: str
    ) -> int:
        """``--max-in-flight`` -> ``batch_size`` -> ``/health`` capacity -> 1 (§10)."""
        health = await probe_health(client, base_url) or {}
        hard_cap = health_capacity(health)
        for candidate in (self._options.max_in_flight, self._spec.batch_size):
            if isinstance(candidate, int) and candidate > 0:
                return min(candidate, hard_cap) if hard_cap else candidate
        return hard_cap or 1

    async def _schedule(
        self, pending: Sequence[WorkItem], driver: HttpRolloutDriver, concurrency: int
    ) -> bool:
        """Feed work through a bounded worker pool. Returns True when cancelled.

        A pool rather than a task per row: over-queueing lets LocalBackend queue
        time consume workflow deadlines and, on Harbor, creates sandboxes the
        backend cannot service (§10).
        """
        queue: asyncio.Queue[WorkItem] = asyncio.Queue()
        for item in pending:
            queue.put_nowait(item)
        workers = [
            asyncio.create_task(self._worker(queue, driver))
            for _ in range(max(1, min(concurrency, len(pending))))
        ]
        loop = asyncio.get_running_loop()
        installed: list[signal.Signals] = []
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, ValueError):
                loop.add_signal_handler(sig, self._request_cancel, workers)
                installed.append(sig)
        watchdog = asyncio.create_task(self._watch_child(workers))
        try:
            results = await asyncio.gather(*workers, return_exceptions=True)
        finally:
            watchdog.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watchdog
            for sig in installed:
                with contextlib.suppress(NotImplementedError, ValueError):
                    loop.remove_signal_handler(sig)
        for result in results:
            # Cancellation is how the supervisor stops in-flight work, so only an
            # unexpected exception means a worker died with items still queued.
            # Halting records it and marks the run incomplete, instead of
            # returning a short run that reads as a clean one.
            if isinstance(result, BaseException) and not isinstance(
                result, asyncio.CancelledError
            ):
                self._halt_dispatch(
                    f"a worker failed: {type(result).__name__}: {result}"
                )
        return self._cancelled.is_set() or self._halt_reason is not None

    async def _watch_child(self, workers: Sequence[asyncio.Task[None]]) -> None:
        """Abort in-flight waits when the rollout server dies (§10 deadline).

        Without this, every worker sits on a callback that can no longer arrive
        until its deadline expires -- minutes of silence per item, and each one
        then looks like a per-item timeout rather than one dead server.
        """
        child = self._child
        if child is None:
            return
        returncode = await child.wait()
        if self._cancelled.is_set() or self._halt_reason is not None:
            return
        self._halt_dispatch(
            f"the rollout server exited with code {returncode}; see {LOGS_FILENAME}",
            returncode=returncode,
        )
        for worker in workers:
            worker.cancel()

    def _request_cancel(self, workers: Sequence[asyncio.Task[None]]) -> None:
        """First interrupt: stop dispatch and cancel in-flight work (§8).

        A cancelled attempt writes no terminal record, so it stays pending and
        the next invocation runs it again -- unlike Harbor, which records a
        ``CancelledError`` result and then skips the trial as complete.
        """
        if self._cancelled.is_set():
            self._hooks.note("second interrupt: exiting now")
            raise KeyboardInterrupt
        self._cancelled.set()
        self._hooks.note(
            "interrupted: cancelling in-flight rollouts, press Ctrl-C again to exit now"
        )
        self._write_log("warning", "cancel", "supervisor cancellation requested")
        for worker in workers:
            worker.cancel()

    async def _worker(
        self, queue: asyncio.Queue[WorkItem], driver: HttpRolloutDriver
    ) -> None:
        while not self._cancelled.is_set() and self._halt_reason is None:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await self._run_work_item(item, driver)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if (
                    isinstance(exc, RolloutProtocolError)
                    and 400 <= exc.status_code < 500
                ):
                    # The rollout server actively refused *this* request, so the
                    # failure is attributable to the work item and gets a
                    # terminal record.
                    await self._journal_supervisor_failure(item, exc)
                    continue
                # Everything else -- a 5xx from a restarting server, proxy
                # errors, transport failures, an indeterminate admission, a
                # supervisor bug -- is process-wide as far as this item is
                # concerned. Halt dispatch and leave it pending rather than
                # durably recording a failure that says nothing about the row
                # (§9.3).
                self._halt_dispatch(
                    f"{type(exc).__name__}: {exc}",
                    row_index=item.row.row_index,
                    run_index=item.run_index,
                )
                return

    async def _run_work_item(self, item: WorkItem, driver: HttpRolloutDriver) -> None:
        rollout_id = uuid.uuid4().hex
        self._dispatch_context[rollout_id] = item
        self._dispatch_started[rollout_id] = time.monotonic()
        request = RolloutRunRequest(
            messages=list(item.row.initial_messages),
            label=item.row.label,
            metadata=dict(item.row.metadata) if item.row.metadata else None,
            rollout_id=rollout_id,
            agent_timeout_sec=self._spec.agent_timeout_sec,
            grader_timeout_sec=self._spec.grader_timeout_sec,
            extra_fields={
                "row_index": item.row.row_index,
                "run_index": item.run_index,
            },
        )
        self._write_log(
            "info",
            "dispatch",
            "dispatching rollout",
            rollout_id=rollout_id,
            row_index=item.row.row_index,
            run_index=item.run_index,
        )
        try:
            outcome = await driver.run(request)
        finally:
            self._dispatched += 1
        if outcome.status is RolloutStatus.CANCELLED:
            # Supervisor-requested cancellation writes no terminal event, so the
            # work item stays pending for the next invocation.
            self._write_log(
                "warning", "cancel", "rollout cancelled", rollout_id=rollout_id
            )
            return
        await self._await_trajectory(rollout_id)
        with contextlib.suppress(Exception):
            self._refresh_snapshots(project_keys={item.key})
        self._report_progress()
        self._forget(rollout_id)
        self._log_outcome(item, rollout_id, outcome)

    def _log_outcome(
        self, item: WorkItem, rollout_id: str, outcome: RolloutOutcome
    ) -> None:
        record = self._latest.get(item.key)
        self._write_log(
            "info",
            "result",
            "work item finished",
            rollout_id=rollout_id,
            row_index=item.row.row_index,
            run_index=item.run_index,
            status=record.status if record else str(outcome.status),
            reward=record.reward if record else None,
        )

    # ------------------------------------------------------------------ #
    # Terminal commit -- the durability boundary
    # ------------------------------------------------------------------ #

    async def _commit_terminal(
        self, result: TerminalCallbackResult
    ) -> Mapping[str, Any] | None:
        """Journal the terminal result, then let the callback be acknowledged.

        Everything the record needs is gathered *before* the append, and the
        append ``fsync``s before returning, so an HTTP 200 on the grader callback
        means the result is durable (§9.3).
        """
        item = self._dispatch_context.get(result.rollout_id)
        if item is None:
            # Returning here would let the store ack a callback with nothing
            # journaled, breaking "HTTP 200 on the grader callback means the
            # result is durable". Raising makes the listener 500 and leaves the
            # terminal slot open, so the work item simply stays pending.
            raise LocalEvalError(
                f"terminal callback for rollout {result.rollout_id} has no "
                "dispatch context; refusing to acknowledge it"
            )
        status, reward, error_type = _classify_terminal(result)
        record = TerminalRecord(
            row_index=item.row.row_index,
            run_index=item.run_index,
            rollout_id=result.rollout_id,
            status=status,
            source_row_index=item.row.source_row_index,
            reward=reward,
            tokens=self._collect_tokens(result.rollout_id),
            duration_ms=self._elapsed_ms(result.rollout_id),
            error_type=error_type,
        )
        await self._append_record(record)
        return None

    async def _append_record(self, record: TerminalRecord) -> None:
        assert self._journal is not None
        await self._journal.append(record)
        self._latest[record.key] = record
        # This result was produced now, so it is no longer a carried-forward one:
        # `resumed` is the platform's carry-forward flag, not "the run resumed".
        self._resumed_keys.discard(record.key)

    async def _journal_supervisor_failure(
        self, item: WorkItem, exc: BaseException
    ) -> None:
        """Journal an unambiguous per-item failure that produced no callback (§9.3)."""
        rollout_id = next(
            (
                candidate
                for candidate, context in self._dispatch_context.items()
                if context is item
            ),
            uuid.uuid4().hex,
        )
        existing = self._latest.get(item.key)
        # Only *this* attempt's own terminal record makes the failure redundant.
        # A record from an earlier attempt is what --retry-failed is replacing.
        if existing is not None and existing.rollout_id == rollout_id:
            self._forget(rollout_id)
            return
        self._write_log(
            "error",
            "result",
            "work item failed before a terminal callback",
            rollout_id=rollout_id,
            row_index=item.row.row_index,
            run_index=item.run_index,
            error=f"{type(exc).__name__}: {exc}",
        )
        await self._append_record(
            TerminalRecord(
                row_index=item.row.row_index,
                run_index=item.run_index,
                rollout_id=rollout_id,
                status="failed",
                source_row_index=item.row.source_row_index,
                tokens=self._collect_tokens(rollout_id),
                duration_ms=self._elapsed_ms(rollout_id),
                error_type=_error_type_for(exc),
            )
        )
        self._forget(rollout_id)
        with contextlib.suppress(Exception):
            self._refresh_snapshots(project_keys={item.key})
        self._report_progress()

    def _elapsed_ms(self, rollout_id: str) -> float:
        started = self._dispatch_started.get(rollout_id)
        if started is None:
            return 0.0
        return (time.monotonic() - started) * 1000.0

    def _collect_tokens(self, rollout_id: str) -> int | None:
        if self._bridge is None:
            return None
        return self._bridge.collect_tokens(rollout_id)

    def _forget(self, rollout_id: str) -> None:
        self._dispatch_context.pop(rollout_id, None)
        self._dispatch_started.pop(rollout_id, None)
        if self._bridge is not None:
            self._bridge.discard(rollout_id)

    async def _await_trajectory(self, rollout_id: str) -> None:
        """Poll for a parseable ``trajectory.json`` after the durable result (§11.3).

        The server writes the trajectory in ``finally``, *after* the grader
        callback is acknowledged, and ``save_trajectory`` never raises -- so the
        file can arrive late or never. Waiting here, after the journal append,
        is the only ordering in which the server can reach its own write.
        """
        assert self._run_dir is not None
        path = (
            self._run_dir / TRIALS_DIRNAME / rollout_id / CANONICAL_TRAJECTORY_FILENAME
        )
        deadline = time.monotonic() + _TRAJECTORY_GRACE_SEC
        while time.monotonic() < deadline:
            if read_valid_trajectory(path, rollout_id=rollout_id) is not None:
                return
            await asyncio.sleep(_TRAJECTORY_POLL_INTERVAL_SEC)
        self._write_log(
            "warning",
            "trajectory",
            "no parseable trajectory within the archive grace period",
            rollout_id=rollout_id,
            rollout_dir=str(path.parent),
        )

    # ------------------------------------------------------------------ #
    # Rollout-server lifecycle
    # ------------------------------------------------------------------ #

    def _resolve_uv(self) -> str:
        try:
            return _uv_executable()
        except RuntimeError as exc:
            raise LocalEvalError(
                "uv is required to launch the rollout server from "
                f"rollouts/{self._spec.rollout_name}/pyproject.toml; install uv "
                "and retry"
            ) from exc

    async def _sync_rollout_dependencies(self, uv_executable: str) -> None:
        """Resolve the rollout's own environment before the server is spawned.

        Its own stage, so a dependency-resolution failure is attributed to deps
        instead of surfacing later as a server-health timeout.
        """
        self._stage("deps", f"syncing rollout dependencies ({self._spec.rollout_name})")
        child = await asyncio.create_subprocess_exec(
            *uv_sync_command(uv_executable, self._rollout_dir),
            cwd=str(self._rollout_dir),
            env=inherited_env(os.environ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            await self._tee_child_output(child, step="deps")
            returncode = await child.wait()
        except asyncio.CancelledError:
            # The sync child is outside _stop_rollout_server's lifecycle, so
            # cancellation must stop its whole build tree and reap it here.
            if child.returncode is None:
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                    await asyncio.to_thread(
                        terminate_process_group,
                        _process_group_of(child.pid),
                        grace_sec=_SERVER_TERM_GRACE_SEC,
                    )
            with contextlib.suppress(Exception):
                await child.wait()
            raise
        if returncode != 0:
            raise LocalEvalError(
                f"uv sync failed for rollouts/{self._spec.rollout_name} with exit "
                f"code {returncode}; see {LOGS_FILENAME}"
            )

    async def _start_rollout_server(
        self, *, secrets: Mapping[str, str], client: httpx.AsyncClient
    ) -> str:
        assert self._run_dir is not None
        entrypoint = self._rollout_dir / self._spec.entrypoint
        if not entrypoint.is_file():
            raise LocalEvalError(
                f"rollout entrypoint {entrypoint} does not exist; check "
                "experiment.rollout and experiment.entrypoint"
            )
        artifact_root = self._run_dir / TRIALS_DIRNAME
        if self._options.server_interpreter is None:
            uv_executable = self._resolve_uv()
            # Once, outside the retry loop: dependencies do not change when the
            # port does.
            await self._sync_rollout_dependencies(uv_executable)
            argv = uv_run_command(uv_executable, self._rollout_dir, entrypoint)
        else:
            argv = [self._options.server_interpreter, str(entrypoint)]
        last_error: BaseException | None = None
        for attempt in range(1, _PORT_ATTEMPTS + 1):
            port = self._options.rollout_port or reserve_free_port()
            instance_id = uuid.uuid4().hex
            env = build_subprocess_env(
                base=os.environ,
                config_env=self._spec.env,
                secrets=secrets,
                port=port,
                artifact_root=artifact_root,
                instance_id=instance_id,
            )
            self._stage(
                "server",
                f"starting rollout server ({self._spec.entrypoint})",
                port=port,
                attempt=attempt,
            )
            child = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(self._rollout_dir),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            self._child = child
            self._child_reader = asyncio.create_task(self._tee_child_output(child))
            base_url = f"http://127.0.0.1:{port}"
            try:
                await self._wait_for_health(
                    client, base_url, instance_id=instance_id, child=child
                )
            except LocalEvalError as exc:
                last_error = exc
                await self._stop_rollout_server()
                if self._options.rollout_port is not None or attempt == _PORT_ATTEMPTS:
                    raise
                self._write_log(
                    "warning",
                    "server",
                    "retrying startup on a new port",
                    error=str(exc),
                )
                continue
            self._stage("server", f"rollout server healthy on port {port}", port=port)
            return base_url
        raise LocalEvalError(f"rollout server did not start: {last_error}")

    async def _wait_for_health(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        *,
        instance_id: str,
        child: asyncio.subprocess.Process,
    ) -> None:
        """Poll ``/health`` until the child is ready, it dies, or time runs out.

        ``child.returncode`` is checked every poll tick, so a rollout server
        that fails its startup (a Harbor prewarm that cannot build its image,
        say) surfaces within one tick instead of waiting out the full health
        timeout.
        """
        deadline = time.monotonic() + _HEALTH_TIMEOUT_SEC
        while True:
            if child.returncode is not None:
                raise LocalEvalError(
                    f"rollout server exited with code {child.returncode} "
                    f"before becoming healthy; see {LOGS_FILENAME}"
                )
            health = await _probe_health(client, base_url)
            if health is not None:
                # The compatibility bootstrap writes the ownership id to a
                # response header even when the rollout's independently-resolved
                # SDK predates the JSON field. Direct test spawns and current SDK
                # servers can still prove ownership through the body fallback.
                reported = health.instance_id or health.payload.get("instance_id")
                if reported != instance_id:
                    raise LocalEvalError(
                        f"another process is already listening on {base_url} "
                        f"and its /health reports instance_id={reported!r} "
                        f"instead of this run's; stop it, or choose a "
                        f"different --rollout-port"
                    )
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LocalEvalError(
                    f"rollout server did not become healthy within "
                    f"{_HEALTH_TIMEOUT_SEC:.0f}s; see {LOGS_FILENAME}"
                )
            await asyncio.sleep(min(_HEALTH_POLL_INTERVAL_SEC, remaining))

    async def _tee_child_output(
        self, child: asyncio.subprocess.Process, *, step: str = "rollout-server"
    ) -> None:
        """Tee a child's combined output into the run log under *step* (§4.3)."""
        stream = child.stdout
        if stream is None:
            return
        while True:
            try:
                raw = await stream.readline()
            except (asyncio.CancelledError, ValueError):
                raise
            if not raw:
                return
            # RunLog redacts every line it writes, so a secret echoed by the
            # workflow or an HTTP debug log never lands in logs.txt (§7.4).
            self._write_log("info", step, raw.decode(errors="replace").rstrip())

    async def _stop_rollout_server(self) -> None:
        child = self._child
        if child is None:
            return
        self._child = None
        if child.returncode is None:
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                await asyncio.to_thread(
                    terminate_process_group,
                    _process_group_of(child.pid),
                    grace_sec=_SERVER_TERM_GRACE_SEC,
                )
        with contextlib.suppress(Exception):
            await child.wait()
        if self._child_reader is not None:
            self._child_reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._child_reader
            self._child_reader = None

    # ------------------------------------------------------------------ #
    # Snapshots and finalization
    # ------------------------------------------------------------------ #

    def _refresh_snapshots(
        self, *, project_keys: set[WorkKey] | None = None
    ) -> list[dict[str, Any]]:
        """Rewrite snapshots; copy file projections only for changed work items.

        Snapshots are small and always rewritten whole so partial output stays
        readable. File copies are not: projecting every attempt on every refresh
        is quadratic in the row count, so *project_keys* narrows them to the work
        item that just finished. ``None`` projects everything.
        """
        assert self._materializer is not None and self._identity is not None
        attempts = select_attempts(
            self._latest,
            trials_dir=self._materializer.trials_dir,
            resumed_keys=self._resumed_keys,
        )
        return self._materializer.refresh(
            attempts,
            identity=self._identity,
            pass_threshold=self._spec.pass_threshold,
            sampled_rows=len(self._selection.rows),
            total_dataset_rows=self._selection.total_dataset_rows,
            total_runs=len(self._selection.rows) * max(1, self._spec.n),
            project_keys=project_keys,
        )

    def _report_progress(self) -> None:
        statuses = [record.status for record in self._latest.values()]
        rewards = [
            record.reward
            for record in self._latest.values()
            if record.status != "skipped" and record.reward is not None
        ]
        passed = sum(1 for reward in rewards if reward >= self._spec.pass_threshold)
        with contextlib.suppress(Exception):
            self._hooks.progress(
                ProgressSnapshot(
                    completed=len(statuses),
                    total=len(self._selection.rows) * max(1, self._spec.n),
                    passed=passed,
                    failed=sum(1 for status in statuses if status == "failed"),
                )
            )

    def _finalize(self, *, cancelled: bool) -> RunSummary:
        run_dir = self._run_dir
        started = self._identity
        assert run_dir is not None and started is not None
        duration_ms = (time.monotonic() - self._started_monotonic) * 1000.0
        total = len(self._selection.rows) * max(1, self._spec.n)
        complete = len(self._latest) >= total
        identity = RunIdentity(
            local_run_id=started.local_run_id,
            run_name=started.run_name,
            dataset_name=started.dataset_name,
            model_name=started.model_name,
            rollout_name=started.rollout_name,
            started_at=started.started_at,
            status="finished" if complete and not cancelled else "incomplete",
            completed_at=utc_now() if complete and not cancelled else None,
            duration_ms=duration_ms,
        )
        self._identity = identity
        rows = self._refresh_snapshots()
        statuses = [record.status for record in self._latest.values()]
        return RunSummary(
            run_dir=run_dir,
            local_run_id=identity.local_run_id,
            run_name=identity.run_name,
            total_work_items=total,
            dispatched=self._dispatched,
            succeeded=sum(1 for status in statuses if status == "success"),
            failed=sum(1 for status in statuses if status == "failed"),
            skipped=sum(1 for status in statuses if status == "skipped"),
            resumed=len(self._resumed_keys),
            cancelled=cancelled,
            duration_ms=duration_ms,
            metrics=aggregate_metrics(rows, pass_threshold=self._spec.pass_threshold),
            failures=self._collect_failures(),
        )

    def _collect_failures(self) -> list[FailedWorkItem]:
        assert self._run_dir is not None
        failures: list[FailedWorkItem] = []
        for key in sorted(self._latest):
            record = self._latest[key]
            if record.status == "success":
                continue
            failures.append(
                FailedWorkItem(
                    row_index=record.row_index,
                    source_row_index=(
                        record.source_row_index
                        if record.source_row_index is not None
                        else record.row_index
                    ),
                    run_index=record.run_index,
                    rollout_id=record.rollout_id,
                    error_type=record.error_type,
                    rollout_dir=self._run_dir / TRIALS_DIRNAME / record.rollout_id,
                )
            )
        return failures

    # ------------------------------------------------------------------ #
    # Logging helpers
    # ------------------------------------------------------------------ #

    def _stage(self, step: str, message: str, **details: Any) -> None:
        """Log a milestone and surface it to the CLI.

        One call site keeps the log and the terminal from drifting apart: a
        stage a plain run prints is always a line the run log also carries.
        """
        self._write_log("info", step, message, **details)
        with contextlib.suppress(Exception):
            self._hooks.stage(message)

    def _write_log(self, level: str, step: str, message: str, **details: Any) -> None:
        if self._log is not None:
            self._log.write(level, step, message, **details)
        elif level in ("warning", "error"):
            self._hooks.note(message)


def _process_group_of(pid: int) -> int:
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, PermissionError, OSError) as exc:
        if isinstance(exc, OSError) and exc.errno not in (
            errno.ESRCH,
            errno.EPERM,
            errno.EINVAL,
        ):
            raise
        # ``start_new_session=True`` makes the child its own group leader, so its
        # pid is its pgid whenever the lookup itself is unavailable.
        return pid


def _classify_terminal(
    result: TerminalCallbackResult,
) -> tuple[TerminalStatus, float | None, str | None]:
    """Map a callback-store terminal result onto the index status vocabulary."""
    if result.source == "timeout":
        return "failed", None, "callback_timeout"
    grader = result.grader
    if grader is None:
        return "failed", None, "missing_grader_callback"
    sample = grader.sample
    reward = sample.reward if sample is not None else None
    # A crashed grader is a failure even when it asked to drop the sample:
    # "skipped" would leave the row out of the scored denominator entirely.
    if grader.status is not GraderStatus.SUCCESS:
        return "failed", reward, grader.err_category or "grader_failed"
    if sample is not None and sample.remove_sample:
        # ``remove_sample`` is the workflow saying "do not score this row".
        return "skipped", None, None
    completion = result.completion
    if completion is not None and completion.status is not RolloutStatus.SUCCESS:
        return "failed", reward, completion.err_category or "workflow_failed"
    return "success", reward, None


def _error_type_for(exc: BaseException) -> str:
    if isinstance(exc, RolloutProtocolError):
        return "rollout_protocol_error"
    return "supervisor_error"
