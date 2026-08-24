"""Durable state for a local evaluation run: journal, manifest, and run lock.

Four concepts, and deliberately no more (design ``local-eval-run-plan.md`` §9):

* ``manifest.json`` -- immutable resolved-input lock plus provenance. Written
  once at run creation and compared on resume, so a semantic change refuses
  by name instead of silently mixing versions.
* ``events.jsonl`` -- the resume authority. One newline-terminated JSON record
  per terminal attempt. The record is written in full, ``fsync``-ed, and only
  then may the terminal callback be acknowledged, so a durably acknowledged
  work item never runs again after ``kill -9``.
* ``server.json`` -- the rollout-server ownership record. Present only while
  a spawned server may be running; a record that survives a supervisor death
  is how the next invocation finds -- and proves ownership of -- the orphan.
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
import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from osmosis_ai.rollout.utils.identifiers import is_single_path_segment

# Bumped when the on-disk meaning of events.jsonl or manifest.json changes.
LOCAL_STATE_SCHEMA_VERSION = 1
# Bumped when dataset row normalization changes what a row_index means.
DATASET_NORMALIZATION_VERSION = 1
# The rollout HTTP/callback protocol this runner speaks.
ROLLOUT_PROTOCOL_VERSION = "0.3"

MANIFEST_FILENAME = "manifest.json"
JOURNAL_FILENAME = "events.jsonl"
SERVER_STATE_FILENAME = "server.json"
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


#: ``(field, accepted JSON types, human label, required)`` -- the whole record
#: schema, checked in one pass. Unlisted keys are ignored, so a journal written
#: by an older build of the same state schema still replays.
_RECORD_FIELDS: tuple[tuple[str, type | tuple[type, ...], str, bool], ...] = (
    ("row_index", int, "an integer", True),
    ("run_index", int, "an integer", True),
    ("rollout_id", str, "a string", True),
    ("status", str, "a string", True),
    ("source_row_index", int, "an integer", False),
    ("reward", (int, float), "a number", False),
    ("tokens", int, "an integer", False),
    ("duration_ms", (int, float), "a number", False),
    ("error_type", str, "a string", False),
)


@dataclass(frozen=True)
class TerminalRecord:
    """One terminal result for one attempt of one work item.

    Carries no timestamp on purpose: replay arbitrates by append order, so a
    skewed clock can never change which attempt wins (§9.4).
    """

    row_index: int
    run_index: int
    rollout_id: str
    status: TerminalStatus
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
        values: dict[str, Any] = {}
        for name, kinds, label, required in _RECORD_FIELDS:
            value = payload.get(name)
            if value is None:
                if required:
                    raise JournalCorruptionError(f"{where}: {name} is missing")
                continue
            # ``bool`` is an ``int`` subclass, and no field accepts a JSON bool.
            if isinstance(value, bool) or not isinstance(value, kinds):
                raise JournalCorruptionError(f"{where}: {name} must be {label}")
            values[name] = value
        # The id names a directory under rollout_trials/ and a projection file,
        # so a traversal or separator in a replayed record would resolve reads
        # and copies outside the run directory. Also covers the empty id.
        if not is_single_path_segment(values["rollout_id"]):
            raise JournalCorruptionError(
                f"{where}: rollout_id must be a single path segment (no "
                "separators, and not '.' or '..')"
            )
        if values["status"] not in _TERMINAL_STATUSES:
            raise JournalCorruptionError(
                f"{where}: status must be one of {sorted(_TERMINAL_STATUSES)}"
            )
        for name in ("reward", "duration_ms"):
            if name in values:
                values[name] = float(values[name])
        return cls(**values)


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
class RunManifest:
    """Immutable local provenance plus the resolved-input lock. Never uploaded."""

    local_run_id: str
    run_name: str
    created_at: str
    inputs: dict[str, Any]
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
            provenance=dict(provenance),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "local_run_id": self.local_run_id,
            "run_name": self.run_name,
            "created_at": self.created_at,
            "inputs": self.inputs,
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
        return cls(
            local_run_id=str(payload.get("local_run_id", "")),
            run_name=str(payload.get("run_name", "")),
            created_at=str(payload.get("created_at", "")),
            inputs=inputs,
            provenance=payload.get("provenance") or {},
            schema_version=schema_version,
        )


# --------------------------------------------------------------------------- #
# Process lock
# --------------------------------------------------------------------------- #


class RunLock:
    """Exclusive flock for one named run, held outside the run directory.

    The path is never replaced while the lock is held, so ``--fresh`` renaming
    the run directory cannot leave the replacement path unlocked.
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


def terminate_process_group(
    pgid: int, *, grace_sec: float, poll_sec: float = 0.05
) -> None:
    """SIGTERM a process group, escalating to SIGKILL after *grace_sec*.

    Blocking: call via ``asyncio.to_thread`` during an async shutdown.
    """
    if pgid <= 0:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + max(0.0, grace_sec)
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            # Alive, but no longer ours to signal; the SIGKILL below is a no-op.
            break
        time.sleep(poll_sec)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)


# --------------------------------------------------------------------------- #
# Rollout-server ownership record
# --------------------------------------------------------------------------- #


def process_start_token(pid: int) -> str | None:
    """Opaque token for when *pid* started; ``None`` when it cannot be read.

    Equal tokens for the same pid mean the pid was not recycled in between,
    which is what makes signalling a recorded process group safe. Linux reads
    the kernel's exact starttime tick from ``/proc``; macOS has no ``/proc``,
    so ``ps``'s second-resolution ``lstart`` stands in -- a recycled pid that
    also lands on the same start second is not a realistic collision.
    """
    if pid <= 0:
        return None
    if os.path.isdir("/proc"):
        try:
            stat = Path(f"/proc/{pid}/stat").read_bytes()
            # comm (field 2) may contain spaces and parentheses, so fields are
            # only positional after the last ')'. starttime is field 22.
            return stat[stat.rindex(b")") + 2 :].split()[19].decode("ascii")
        except (OSError, ValueError, IndexError, UnicodeDecodeError):
            return None
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    token = result.stdout.strip()
    return token if result.returncode == 0 and token else None


@dataclass(frozen=True)
class ServerProcessState:
    """Identity of the spawned rollout-server process group, for orphan reaping.

    Written right after the spawn and removed on clean shutdown, so a record
    that survives names a server whose supervisor died without cleanup
    (``kill -9``, OOM, a crashed terminal). ``pid`` plus ``start_token`` is
    the ownership proof: a recycled pid or pgid cannot reproduce the original
    start time, so a failed :meth:`is_owner_alive` means the group must not
    be signalled.
    """

    pid: int
    pgid: int
    start_token: str
    instance_id: str
    port: int
    created_at: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "pgid": self.pgid,
            "start_token": self.start_token,
            "instance_id": self.instance_id,
            "port": self.port,
            "created_at": self.created_at,
        }

    def write(self, path: Path) -> None:
        atomic_write_json(path, self.to_payload())

    @classmethod
    def read(cls, path: Path) -> ServerProcessState | None:
        """Read a record, or ``None``. Tolerant: this file only ever enables
        extra cleanup, so nothing about it may stop a run."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        pid = payload.get("pid")
        pgid = payload.get("pgid")
        start_token = payload.get("start_token")
        if (
            not isinstance(pid, int)
            or isinstance(pid, bool)
            or pid <= 0
            or not isinstance(pgid, int)
            or isinstance(pgid, bool)
            or pgid <= 0
            or not isinstance(start_token, str)
            or not start_token
        ):
            return None
        port = payload.get("port")
        return cls(
            pid=pid,
            pgid=pgid,
            start_token=start_token,
            instance_id=str(payload.get("instance_id", "")),
            port=port if isinstance(port, int) and not isinstance(port, bool) else 0,
            created_at=str(payload.get("created_at", "")),
        )

    def is_owner_alive(self) -> bool:
        """Whether *pid* is provably still the recorded process, in its group."""
        if process_start_token(self.pid) != self.start_token:
            return False
        try:
            return os.getpgid(self.pid) == self.pgid
        except OSError:
            return False


def reap_orphan_server(path: Path, *, grace_sec: float) -> ServerProcessState | None:
    """Terminate the server group recorded at *path* iff it is provably ours.

    Returns the state that was terminated, else ``None``. The record is
    removed in every case: after a kill its group is dead, and a record that
    fails verification names a pid that is gone or recycled and will never
    verify again. Blocking for up to *grace_sec*: call via
    ``asyncio.to_thread`` from async code.
    """
    state = ServerProcessState.read(path)
    reaped: ServerProcessState | None = None
    if state is not None and state.is_owner_alive():
        terminate_process_group(state.pgid, grace_sec=grace_sec)
        reaped = state
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)
    return reaped
