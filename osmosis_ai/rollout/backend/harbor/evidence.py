"""Sanitize and finalize native trial evidence before its working copy is removed."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any

from osmosis_ai.rollout.backend.harbor.diagnostics import REDACTED, redact_secrets
from osmosis_ai.rollout.utils.evidence import EVIDENCE_SCHEMA, evidence_path
from osmosis_ai.rollout.utils.identifiers import ensure_single_path_segment

# Credentials in structured records, shell assignments, HTTP headers and URLs.
_ASSIGNMENT = re.compile(
    r"""(?ix)([\w-]*(?:api[_-]?key|authorization|password|credential|secret|token)[\w-]*\s*[=:]\s*)("[^"\n]*"|'[^'\n]*'|[^\s,;\]}]+)"""
)
_BEARER = re.compile(r"(?i)\bBearer\s+[^\s\"',;]+")
_URL_USERINFO = re.compile(r"(https?://)[^\s/@]+:[^\s/@]+@", re.IGNORECASE)
MAX_NATIVE_FILE_BYTES = 64 * 1024 * 1024


def sanitized_text(data: bytes, api_key: str | None) -> bytes:
    text = data.decode("utf-8")  # Binary files cannot be safely inspected as logs.

    def clean(value: str) -> str:
        if api_key:
            value = value.replace(api_key, REDACTED)
        value = _BEARER.sub("Bearer " + REDACTED, value)
        value = _ASSIGNMENT.sub(lambda match: match[1] + REDACTED, value)
        return _URL_USERINFO.sub(r"\1[REDACTED]@", value)

    def strings(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: strings(child) for key, child in value.items()}
        if isinstance(value, list):
            return [strings(child) for child in value]
        return clean(value) if isinstance(value, str) else value

    try:
        value = json.loads(text)
    except ValueError:
        text = clean(text)
    else:
        text = (
            json.dumps(
                strings(redact_secrets(value, api_key)), ensure_ascii=True, indent=2
            )
            + "\n"
        )
    return text.encode()


def retain_trial_evidence(
    trial_dir: Path,
    artifact_root: Path,
    rollout_id: str,
    *,
    api_key: str | None = None,
    trial_result: Any = None,
    process_id: str | None = None,
) -> bool:
    """Publish a manifest last; skipped or unreadable evidence stays incomplete.

    Only native result/log records are selected. Configuration and task sources
    never enter the export. Every byte is sanitized again even after upstream
    credential scrubbing. Existing diagnostic logs remain a separate surface.
    """
    ensure_single_path_segment(rollout_id, label="rollout_id")
    evidence_path(rollout_id)
    artifact_root.mkdir(parents=True, exist_ok=True)
    rollout_root = artifact_root / rollout_id
    if rollout_root.is_symlink():
        raise ValueError("Linked rollout evidence destination")
    rollout_root.mkdir(exist_ok=True)
    destination = rollout_root / "harbor"
    if destination.is_symlink():
        raise ValueError("Linked native evidence destination")
    errors: list[str] = []
    files: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=".harbor-", dir=rollout_root) as temporary:
        staged = Path(temporary) / "evidence"
        staged.mkdir(mode=0o700)

        def write(name: str, data: bytes) -> None:
            evidence_path(name)
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            target.write_bytes(data)
            target.chmod(0o600)
            files.append(
                {
                    "path": name,
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )

        def copy(path: Path, name: str) -> None:
            try:
                evidence_path(name)
                if api_key and api_key in name:
                    raise ValueError("credential in filename")
                if path.is_symlink():
                    raise ValueError("linked source")
                if path.is_dir():
                    for child in sorted(path.iterdir()):
                        copy(child, name + "/" + child.name)
                else:
                    descriptor = os.open(
                        path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
                    )
                    with os.fdopen(descriptor, "rb") as stream:
                        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                            raise ValueError("special source")
                        data = stream.read(MAX_NATIVE_FILE_BYTES + 1)
                        if len(data) > MAX_NATIVE_FILE_BYTES:
                            raise ValueError("native file exceeds sanitization limit")
                        if name == "result.json" and not isinstance(
                            json.loads(data), dict
                        ):
                            raise ValueError("invalid native result")
                        write(name, sanitized_text(data, api_key))
            except (OSError, ValueError):
                # Do not expose an untrusted filename or exception containing a credential.
                errors.append("unreadable_or_unsafe_native_file")

        if trial_dir.is_symlink() or not trial_dir.is_dir():
            errors.append("native_trial_missing")
        else:
            result = trial_dir / "result.json"
            if result.exists() or result.is_symlink():
                copy(result, "result.json")
            elif trial_result is not None and hasattr(trial_result, "model_dump"):
                try:
                    write(
                        "result.json",
                        sanitized_text(
                            json.dumps(trial_result.model_dump(mode="json")).encode(),
                            api_key,
                        ),
                    )
                except (TypeError, ValueError):
                    errors.append("native_result_invalid")
            else:
                errors.append("native_result_missing")
            for name in (
                "trial.log",
                "exception.txt",
                "agent",
                "user-agent",
                "verifier",
            ):
                source = trial_dir / name
                if source.exists() or source.is_symlink():
                    copy(source, "logs/" + name)
            steps = trial_dir / "steps"
            if steps.is_symlink():
                errors.append("unsafe_native_steps")
            elif steps.is_dir():
                for step in sorted(steps.iterdir()):
                    if step.is_symlink() or not step.is_dir():
                        errors.append("unsafe_native_step")
                        continue
                    for name in ("agent", "user-agent", "verifier"):
                        source = step / name
                        if source.exists() or source.is_symlink():
                            copy(source, f"logs/steps/{step.name}/{name}")
        manifest = {
            "schema_version": EVIDENCE_SCHEMA,
            "rollout_id": rollout_id,
            "process_id": process_id,
            "complete": not errors,
            "files": sorted(files, key=lambda item: item["path"]),
            "errors": errors,
        }
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (staged / "manifest.json").chmod(0o600)
        previous = Path(temporary) / "previous"
        if destination.exists():
            destination.rename(previous)
        staged.rename(destination)
        # TemporaryDirectory removes any replaced partial attempt only after publication.
    return not errors
