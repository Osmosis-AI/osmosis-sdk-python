"""Portable integrity checks for finalized native trial evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from osmosis_ai.rollout.utils.identifiers import ensure_single_path_segment

EVIDENCE_SCHEMA = "harbor-evidence-v1"


def _open_directory(path: Path, *, dir_fd: int | None = None) -> int:
    """Walk beneath an open directory without following any component's links."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    current = os.open(path.anchor or ".", flags, dir_fd=dir_fd)
    try:
        for part in path.parts[1:] if path.anchor else path.parts:
            child = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = child
    except BaseException:
        os.close(current)
        raise
    return current


def _open_regular(directory_fd: int, name: str) -> BinaryIO:
    path = Path(evidence_path(name))
    parent = _open_directory(path.parent, dir_fd=directory_fd)
    try:
        descriptor = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
    finally:
        os.close(parent)
    stream = os.fdopen(descriptor, "rb")
    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
        stream.close()
        raise ValueError("Evidence must contain regular files")
    return stream


def _regular_files(directory_fd: int, prefix: str = "") -> set[str]:
    names = set()
    for name in os.listdir(directory_fd):
        relative = evidence_path(prefix + name)
        mode = os.stat(name, dir_fd=directory_fd, follow_symlinks=False).st_mode
        if stat.S_ISDIR(mode):
            child = _open_directory(Path(name), dir_fd=directory_fd)
            try:
                names.update(_regular_files(child, relative + "/"))
            finally:
                os.close(child)
        elif stat.S_ISREG(mode):
            names.add(relative)
        else:
            raise ValueError("Evidence contains a link or special file")
    return names


def evidence_path(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or ":" in value
        or any(unicodedata.category(char) == "Cc" for char in value)
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or PurePosixPath(value).is_absolute()
    ):
        raise ValueError("Evidence path must be a safe relative path")
    return value


def verify_trial_evidence(
    directory: Path, rollout_id: str, *, process_id: str | None = None
) -> dict[str, Any]:
    """Verify an exact inventory; a partial copy is never finalized evidence."""
    ensure_single_path_segment(rollout_id, label="rollout_id")
    evidence_path(rollout_id)
    try:
        directory_fd = _open_directory(directory)
        try:
            return _verify_trial_evidence(directory_fd, rollout_id, process_id)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise ValueError("Evidence directory or file is missing or unsafe") from exc


def _verify_trial_evidence(
    directory_fd: int, rollout_id: str, process_id: str | None
) -> dict[str, Any]:
    with _open_regular(directory_fd, "manifest.json") as stream:
        manifest = json.load(stream)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != EVIDENCE_SCHEMA
        or manifest.get("rollout_id") != rollout_id
        or (process_id is not None and manifest.get("process_id") != process_id)
        or manifest.get("complete") is not True
        or manifest.get("errors") != []
        or not isinstance(manifest.get("files"), list)
    ):
        raise ValueError("Evidence is incomplete or belongs to another rollout")
    declared = set()
    for entry in manifest["files"]:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("path"), str)
            or type(entry.get("size_bytes")) is not int
            or entry["size_bytes"] < 0
            or not isinstance(entry.get("sha256"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"])
        ):
            raise ValueError("Evidence file record is invalid")
        name = evidence_path(entry["path"])
        if name in declared or name == "manifest.json":
            raise ValueError("Evidence inventory has duplicate or reserved paths")
        declared.add(name)
        with _open_regular(directory_fd, name) as stream:
            digest = hashlib.sha256()
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
            size = os.fstat(stream.fileno()).st_size
        if size != entry["size_bytes"] or digest.hexdigest() != entry["sha256"]:
            raise ValueError("Evidence checksum mismatch")
    actual = _regular_files(directory_fd) - {"manifest.json"}
    if declared != actual or "result.json" not in declared:
        raise ValueError("Evidence inventory does not exactly cover its files")
    return manifest


def trial_evidence_inventory(
    root: Path, rollout_ids: list[str], *, process_id: str | None = None
) -> dict[str, Any]:
    """Report missing/partial rollouts explicitly using a caller's expected IDs."""
    records, missing = {}, []
    for rollout_id in sorted(set(rollout_ids)):
        ensure_single_path_segment(rollout_id, label="rollout_id")
        evidence_path(rollout_id)
        try:
            if (root / rollout_id).is_symlink():
                raise ValueError("Linked rollout directory")
            records[rollout_id] = verify_trial_evidence(
                root / rollout_id / "harbor", rollout_id, process_id=process_id
            )
        except (OSError, ValueError, KeyError, TypeError):
            missing.append(rollout_id)
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "process_id": process_id,
        "complete": not missing,
        "rollout_ids": sorted(set(rollout_ids)),
        "missing_rollout_ids": missing,
        "rollouts": records,
    }
