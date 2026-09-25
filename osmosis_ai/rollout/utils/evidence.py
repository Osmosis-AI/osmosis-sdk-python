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


def _open_regular(path: Path) -> BinaryIO:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    stream = os.fdopen(descriptor, "rb")
    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
        stream.close()
        raise ValueError("Evidence must contain regular files")
    return stream


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
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Evidence directory is missing or linked")
    manifest_path = directory / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Evidence manifest must be a regular file")
    with _open_regular(manifest_path) as stream:
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
        current = directory
        for part in PurePosixPath(name).parts:
            current /= part
            if current.is_symlink():
                raise ValueError("Evidence contains a link")
        if not current.is_file():
            raise ValueError("Evidence file is missing")
        with _open_regular(current) as stream:
            digest = hashlib.sha256()
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
            size = os.fstat(stream.fileno()).st_size
        if size != entry["size_bytes"] or digest.hexdigest() != entry["sha256"]:
            raise ValueError("Evidence checksum mismatch")
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("Evidence contains a link or special file")
        if path.is_file() and path != manifest_path:
            actual.add(path.relative_to(directory).as_posix())
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
