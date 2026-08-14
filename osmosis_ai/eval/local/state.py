"""Durable state for a local evaluation run: journal, manifest, and run lock.

Three concepts, and deliberately no more (design ``local-eval-run-plan.md`` §9):

* ``manifest.json`` -- immutable resolved-input lock plus provenance. Written
  once at run creation and compared structurally on resume, so a semantic
  change refuses with a field-level diff instead of silently mixing versions.
* ``events.jsonl`` -- the resume authority. One newline-terminated JSON record
  per terminal attempt. The record is written in full, ``fsync``-ed, and only
  then may the terminal callback be acknowledged, so a durably acknowledged
  work item never runs again after ``kill -9``.
* everything else -- projections, rebuilt from the journal and the artifact
  tree. They are never consulted to decide whether work reruns.

The process flock lives outside the run directory because ``--fresh`` renames
that directory; holding an inode inside it would leave the replacement path
unlocked and admit a second supervisor.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import fcntl
import hashlib
import json
import os
import signal
import tempfile
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from osmosis_ai.rollout.utils.identifiers import is_single_path_segment

# Bumped when the on-disk meaning of events.jsonl or manifest.json changes.
LOCAL_STATE_SCHEMA_VERSION = 1
# Bumped when the layout under rollout_trials/ or the projections change.
LOCAL_ARTIFACT_SCHEMA_VERSION = 1
# Bumped when dataset row normalization changes what a row_index means.
DATASET_NORMALIZATION_VERSION = 1
# The rollout HTTP/callback protocol this runner speaks.
ROLLOUT_PROTOCOL_VERSION = "0.3"

MANIFEST_FILENAME = "manifest.json"
JOURNAL_FILENAME = "events.jsonl"
LOCKS_DIRNAME = ".locks"

#: ``index.jsonl`` status values the platform schema accepts (§2.2).
TerminalStatus = Literal["success", "failed", "skipped"]
_TERMINAL_STATUSES: frozenset[str] = frozenset({"success", "failed", "skipped"})

#: Stable work-item identity within a manifest-locked run: ``(row, run)``.
WorkKey = tuple[int, int]

_RUN_NAME_MAX_LEN = 96


class LocalEvalStateError(RuntimeError):
    """A local run's on-disk state cannot be used as-is."""


class JournalCorruptionError(LocalEvalStateError):
    """A committed journal record is unparseable, so resume is unsafe."""


class RunLockedError(LocalEvalStateError):
    """Another supervisor holds this run's lock."""


class OrphanChildError(LocalEvalStateError):
    """A recorded rollout-server child is alive but could not be verified."""


def utc_now() -> str:
    """Timestamp for provenance fields. Never used to arbitrate attempts."""
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# Durable writes
# --------------------------------------------------------------------------- #


def fsync_dir(directory: Path) -> None:
    """Persist a directory entry so a rename or create survives a crash."""
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError as exc:  # pragma: no cover - platform dependent
        # Some filesystems refuse fsync on a directory fd; the replace itself
        # is still atomic, so this is a durability downgrade, not a failure.
        if exc.errno not in (errno.EINVAL, errno.EACCES):
            raise
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write *data* to *path* atomically and durably.

    Temp file in the destination directory -> fsync the file -> atomic replace
    -> fsync the parent directory (§11.1). Readers therefore only ever see a
    complete previous or complete new file, including across a crash.
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    fsync_dir(parent)


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write *payload* in the repo's 2-space + trailing-newline form."""
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def canonical_json(payload: Any) -> str:
    """Stable serialization used for digests: sorted keys, no incidental space."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest_of(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def drop_none_values(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Omit ``None``-valued keys rather than writing JSON ``null`` (§2.2)."""
    return {key: value for key, value in payload.items() if value is not None}


# --------------------------------------------------------------------------- #
# Run naming and directories
# --------------------------------------------------------------------------- #


def validate_run_name(name: str) -> str:
    """Return *name* if it is one safe path segment, else raise (§4.4)."""
    if not name:
        raise LocalEvalStateError("run name must not be empty")
    if len(name) > _RUN_NAME_MAX_LEN:
        raise LocalEvalStateError(
            f"run name must be at most {_RUN_NAME_MAX_LEN} characters: {name!r}"
        )
    if not is_single_path_segment(name):
        raise LocalEvalStateError(
            f"run name must be a single path segment (no separators, and not "
            f"'.' or '..'): {name!r}"
        )
    if name == LOCKS_DIRNAME:
        raise LocalEvalStateError(f"run name {name!r} is reserved")
    if name[0] in "-." or any(
        char not in "._-" and not char.isalnum() for char in name
    ):
        raise LocalEvalStateError(
            "run name must start with a letter or digit and use only letters, "
            f"digits, '.', '_', or '-': {name!r}"
        )
    return name


def archive_run_directory(run_dir: Path, *, now: str | None = None) -> Path:
    """Rename *run_dir* aside for ``--fresh``. Archive, never silent-delete."""
    stamp = now or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = run_dir.parent / f"{run_dir.name}.archive-{stamp}"
    suffix = 1
    while destination.exists():
        destination = run_dir.parent / f"{run_dir.name}.archive-{stamp}-{suffix}"
        suffix += 1
    os.replace(run_dir, destination)
    fsync_dir(run_dir.parent)
    return destination


# --------------------------------------------------------------------------- #
# Terminal journal
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TerminalRecord:
    """One terminal result for one attempt of one work item.

    ``recorded_at`` is provenance only: replay arbitrates by append order, so a
    skewed clock can never change which attempt wins (§9.4).
    """

    row_index: int
    run_index: int
    rollout_id: str
    status: TerminalStatus
    recorded_at: str
    source_row_index: int | None = None
    reward: float | None = None
    tokens: int | None = None
    duration_ms: float = 0.0
    error_type: str | None = None

    @property
    def key(self) -> WorkKey:
        return (self.row_index, self.run_index)

    def to_payload(self) -> dict[str, Any]:
        return drop_none_values(
            {
                "row_index": self.row_index,
                "run_index": self.run_index,
                "rollout_id": self.rollout_id,
                "status": self.status,
                "recorded_at": self.recorded_at,
                "source_row_index": self.source_row_index,
                "reward": self.reward,
                "tokens": self.tokens,
                "duration_ms": self.duration_ms,
                "error_type": self.error_type,
            }
        )

    def to_journal_line(self) -> bytes:
        return (canonical_json(self.to_payload()) + "\n").encode("utf-8")

    @classmethod
    def from_payload(cls, payload: Any, *, where: str) -> TerminalRecord:
        if not isinstance(payload, dict):
            raise JournalCorruptionError(f"{where}: record is not a JSON object")

        def _int(name: str) -> int:
            value = payload.get(name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise JournalCorruptionError(f"{where}: {name} must be an integer")
            return value

        def _optional_int(name: str) -> int | None:
            value = payload.get(name)
            if value is None:
                return None
            if not isinstance(value, int) or isinstance(value, bool):
                raise JournalCorruptionError(
                    f"{where}: {name} must be an integer when present"
                )
            return value

        def _optional_float(name: str) -> float | None:
            value = payload.get(name)
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise JournalCorruptionError(
                    f"{where}: {name} must be a number when present"
                )
            return float(value)

        def _optional_str(name: str) -> str | None:
            value = payload.get(name)
            if value is None:
                return None
            if not isinstance(value, str):
                raise JournalCorruptionError(
                    f"{where}: {name} must be a string when present"
                )
            return value

        rollout_id = payload.get("rollout_id")
        if not isinstance(rollout_id, str) or not rollout_id:
            raise JournalCorruptionError(f"{where}: rollout_id must be a string")
        status = payload.get("status")
        if status not in _TERMINAL_STATUSES:
            raise JournalCorruptionError(
                f"{where}: status must be one of {sorted(_TERMINAL_STATUSES)}"
            )
        recorded_at = _optional_str("recorded_at") or ""
        return cls(
            row_index=_int("row_index"),
            run_index=_int("run_index"),
            rollout_id=rollout_id,
            status=status,  # type: ignore[arg-type]  # membership checked above
            recorded_at=recorded_at,
            source_row_index=_optional_int("source_row_index"),
            reward=_optional_float("reward"),
            tokens=_optional_int("tokens"),
            duration_ms=_optional_float("duration_ms") or 0.0,
            error_type=_optional_str("error_type"),
        )


@dataclass(frozen=True)
class JournalReplay:
    """Result of replaying the journal at startup."""

    records: tuple[TerminalRecord, ...] = ()
    #: Byte offset just past the last complete record; an append must start here.
    committed_size: int = 0
    #: Size of a discarded non-newline EOF fragment, if any.
    truncated_bytes: int = 0

    @property
    def latest(self) -> dict[WorkKey, TerminalRecord]:
        """Last valid terminal record per work key, in append order (§9.4)."""
        selected: dict[WorkKey, TerminalRecord] = {}
        for record in self.records:
            selected[record.key] = record
        return selected


class TerminalJournal:
    """Append-only journal of terminal results; the resume authority.

    The commit unit is a complete newline-terminated record. ``append`` writes
    every byte, then ``fsync``s, before returning -- callers may only
    acknowledge a terminal callback after it returns.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None
        self._lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def replay(self) -> JournalReplay:
        """Read committed records, reporting any partial trailing record.

        A non-newline EOF fragment is a crash mid-append and is discarded even
        when its JSON happens to parse. A malformed *committed* record is
        corruption and stops the run.
        """
        if not self._path.exists():
            return JournalReplay()
        records: list[TerminalRecord] = []
        committed = 0
        truncated = 0
        with self._path.open("rb") as handle:
            for lineno, raw in enumerate(handle, start=1):
                if not raw.endswith(b"\n"):
                    truncated = len(raw)
                    break
                text = raw[:-1].strip()
                if not text:
                    raise JournalCorruptionError(
                        f"{self._path}:{lineno}: blank journal record"
                    )
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise JournalCorruptionError(
                        f"{self._path}:{lineno}: invalid JSON ({exc.msg})"
                    ) from exc
                records.append(
                    TerminalRecord.from_payload(payload, where=f"{self._path}:{lineno}")
                )
                committed += len(raw)
        return JournalReplay(
            records=tuple(records),
            committed_size=committed,
            truncated_bytes=truncated,
        )

    def open_for_append(self, replay: JournalReplay) -> None:
        """Open the journal for appending, discarding a partial trailing record.

        *replay* must come from :meth:`replay` on this same journal: its
        ``committed_size`` is where the next record starts.
        """
        if self._fd is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        created = not self._path.exists()
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            if replay.truncated_bytes:
                os.ftruncate(fd, replay.committed_size)
                os.fsync(fd)
            elif os.lseek(fd, 0, os.SEEK_END) != replay.committed_size:
                raise LocalEvalStateError(
                    f"{self._path} changed between replay and open; "
                    "another supervisor may be writing to this run"
                )
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        if created:
            fsync_dir(self._path.parent)

    async def append(self, record: TerminalRecord) -> None:
        """Durably append *record*. Returns only once ``fsync`` has completed."""
        if self._fd is None:
            raise LocalEvalStateError("journal is not open for appending")
        payload = record.to_journal_line()
        async with self._lock:
            await asyncio.to_thread(self._append_sync, payload)

    def _append_sync(self, payload: bytes) -> None:
        fd = self._fd
        if fd is None:  # pragma: no cover - guarded by append()
            raise LocalEvalStateError("journal is not open for appending")
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


# --------------------------------------------------------------------------- #
# Manifest and resolved-input lock
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class InputDiff:
    """One field-level difference between two resolved-input locks."""

    field: str
    previous: Any
    current: Any


def diff_inputs(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> list[InputDiff]:
    """Field-level diff of two ``inputs`` objects, with dotted nested keys."""
    diffs: list[InputDiff] = []
    _collect_diffs(previous, current, prefix="", diffs=diffs)
    return diffs


_MISSING = object()


def _collect_diffs(
    previous: Any, current: Any, *, prefix: str, diffs: list[InputDiff]
) -> None:
    if isinstance(previous, Mapping) and isinstance(current, Mapping):
        for key in sorted({*previous.keys(), *current.keys()}):
            child = f"{prefix}.{key}" if prefix else str(key)
            _collect_diffs(
                previous.get(key, _MISSING),
                current.get(key, _MISSING),
                prefix=child,
                diffs=diffs,
            )
        return
    if previous == current:
        return
    diffs.append(
        InputDiff(
            field=prefix or "inputs",
            previous=None if previous is _MISSING else previous,
            current=None if current is _MISSING else current,
        )
    )


@dataclass(frozen=True)
class RunManifest:
    """Immutable local provenance plus the resolved-input lock. Never uploaded."""

    local_run_id: str
    run_name: str
    created_at: str
    inputs: dict[str, Any]
    inputs_digest: str
    provenance: dict[str, Any] = field(default_factory=dict)
    schema_version: int = LOCAL_STATE_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        local_run_id: str,
        run_name: str,
        inputs: Mapping[str, Any],
        provenance: Mapping[str, Any],
    ) -> RunManifest:
        resolved = dict(inputs)
        return cls(
            local_run_id=local_run_id,
            run_name=run_name,
            created_at=utc_now(),
            inputs=resolved,
            inputs_digest=digest_of(resolved),
            provenance=dict(provenance),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "local_run_id": self.local_run_id,
            "run_name": self.run_name,
            "created_at": self.created_at,
            "inputs": self.inputs,
            "inputs_digest": self.inputs_digest,
            "provenance": self.provenance,
        }

    def write(self, path: Path) -> None:
        atomic_write_json(path, self.to_payload())

    @classmethod
    def read(cls, path: Path) -> RunManifest:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalEvalStateError(f"{path} is unreadable: {exc}") from exc
        if not isinstance(payload, dict):
            raise LocalEvalStateError(f"{path} is not a JSON object")
        schema_version = payload.get("schema_version")
        if schema_version != LOCAL_STATE_SCHEMA_VERSION:
            raise LocalEvalStateError(
                f"{path} was written by state schema version {schema_version!r}; "
                f"this SDK writes version {LOCAL_STATE_SCHEMA_VERSION}. Start a "
                "new run with --fresh."
            )
        inputs = payload.get("inputs")
        if not isinstance(inputs, dict):
            raise LocalEvalStateError(f"{path} has no inputs object")
        digest = payload.get("inputs_digest")
        if not isinstance(digest, str):
            raise LocalEvalStateError(f"{path} has no inputs_digest")
        return cls(
            local_run_id=str(payload.get("local_run_id", "")),
            run_name=str(payload.get("run_name", "")),
            created_at=str(payload.get("created_at", "")),
            inputs=inputs,
            inputs_digest=digest,
            provenance=payload.get("provenance") or {},
            schema_version=schema_version,
        )


# --------------------------------------------------------------------------- #
# Process lock and orphan-child metadata
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChildProcessRecord:
    """Runtime metadata for verified orphan cleanup. Never resume authority."""

    supervisor_pid: int
    child_pid: int
    child_pgid: int
    port: int
    instance_id: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "supervisor_pid": self.supervisor_pid,
            "child_pid": self.child_pid,
            "child_pgid": self.child_pgid,
            "port": self.port,
            "instance_id": self.instance_id,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ChildProcessRecord | None:
        try:
            return cls(
                supervisor_pid=int(payload["supervisor_pid"]),
                child_pid=int(payload["child_pid"]),
                child_pgid=int(payload["child_pgid"]),
                port=int(payload["port"]),
                instance_id=str(payload["instance_id"]),
            )
        except (KeyError, TypeError, ValueError):
            # Unreadable metadata is a stale record, not a reason to refuse:
            # it can only cost us an orphan we fail to reap, and the health
            # instance-id check is what actually authorizes any kill.
            return None


class RunLock:
    """Exclusive flock for one named run, held outside the run directory.

    The same inode also carries rollout-server child metadata so the next
    startup can reap a verified orphan. The path is never replaced while held.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None

    @property
    def path(self) -> Path:
        return self._path

    def acquire(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise RunLockedError(
                    f"another `osmosis eval run` already holds {self._path}. "
                    "Wait for it to finish, or use a different --name."
                ) from exc
            raise
        self._fd = fd

    def read_child(self) -> ChildProcessRecord | None:
        if self._fd is None:
            raise LocalEvalStateError("run lock is not held")
        raw = os.pread(self._fd, 64 * 1024, 0)
        if not raw.strip():
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        child = payload.get("child")
        if not isinstance(child, dict):
            return None
        return ChildProcessRecord.from_payload(child)

    def write_child(self, record: ChildProcessRecord | None) -> None:
        """Update the held inode in place, then fsync. Never replaces the path."""
        if self._fd is None:
            raise LocalEvalStateError("run lock is not held")
        payload = {"child": record.to_payload()} if record is not None else {}
        data = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
        os.ftruncate(self._fd, 0)
        os.pwrite(self._fd, data, 0)
        os.fsync(self._fd)

    def clear_child(self) -> None:
        self.write_child(None)

    def release(self) -> None:
        if self._fd is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> RunLock:
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but is owned by someone else, so it is definitely not our child.
        return True
    return True


def process_group_is_alive(pgid: int) -> bool:
    if pgid <= 0:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_process_group(
    pgid: int, *, grace_sec: float, poll_sec: float = 0.05
) -> None:
    """SIGTERM a process group, escalating to SIGKILL after *grace_sec*.

    Blocking: call from startup reaping, or via ``asyncio.to_thread`` during an
    async shutdown.
    """
    if pgid <= 0:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + max(0.0, grace_sec)
    while time.monotonic() < deadline:
        if not process_group_is_alive(pgid):
            return
        time.sleep(poll_sec)
    if process_group_is_alive(pgid):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)


def iter_run_directories(root: Path) -> Iterator[Path]:
    """Yield run directories under ``<output>/``, skipping ``.locks``."""
    if not root.is_dir():
        return
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name != LOCKS_DIRNAME:
            yield child
