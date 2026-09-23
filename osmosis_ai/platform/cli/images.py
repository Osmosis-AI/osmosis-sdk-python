"""Repository image builds with resumable requests and verified downloads."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import DetailResult, OperationResult, detail_fields
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.auth.config import get_platform_url
from osmosis_ai.platform.auth.platform_client import PlatformAPIError
from osmosis_ai.platform.cli.workspace_repo import normalize_git_identity

_IMAGE = re.compile(r"[A-Za-z0-9._:/-]+@sha256:[0-9a-f]{64}")
_MAX_BYTES = 2 * 1024**3


def write_json(path: Path, data: dict[str, Any]) -> None:
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_request(
    output: Path,
    repository: str,
    ref: str,
    tasks_dir: str,
    request_id: str | None = None,
) -> dict[str, Any]:
    identity = normalize_git_identity(repository).identity
    if (
        not ref
        or len(ref) > 255
        or any(ord(char) < 32 or ord(char) == 127 for char in ref)
    ):
        raise CLIError("ref must be a branch, tag, or commit SHA", code="VALIDATION")
    if (
        "\\" in tasks_dir
        or "\0" in tasks_dir
        or any(part in ("", ".", "..") for part in tasks_dir.split("/"))
    ):
        raise CLIError(
            "tasks_dir must be a safe directory relative to the repository root",
            code="VALIDATION",
        )
    if request_id:
        try:
            request_id = str(UUID(request_id))
        except ValueError:
            raise CLIError("request_id must be a UUID", code="VALIDATION") from None
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    inputs = {"platform": get_platform_url(), "git_identity": identity}
    request = {
        "repository": f"https://github.com/{identity}",
        "ref": ref,
        "tasks_dir": tasks_dir,
    }
    path = output / "request.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if (
            saved["inputs"] != inputs
            or {key: saved["request"].get(key) for key in request} != request
            or (request_id and saved["request_id"] != request_id)
        ):
            raise CLIError(
                "Build inputs changed; use a new --output-dir", code="CONFLICT"
            )
        return saved
    request_id = request_id or str(uuid4())
    saved = {
        "inputs": inputs,
        "request_id": request_id,
        "request": {"request_id": request_id, **request},
    }
    write_json(path, saved)
    return saved


def wait_for_build(
    client: OsmosisClient,
    state: dict[str, Any],
    output: Path,
    timeout: float,
    wait: bool,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    submitted = False
    previous = None
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            status = (
                client.get_image_build(
                    state["request_id"],
                    git_identity=state["inputs"]["git_identity"],
                    timeout=min(30, remaining),
                )
                if submitted
                else client.submit_image_build(
                    **state["request"], timeout=min(30, remaining)
                )
            )
            submitted = True
            write_json(output / "status.json", status)
            progress = (
                status["phase"],
                status.get("completed_image_count", 0),
                status.get("image_count", 0),
            )
            if progress != previous:
                console.print(
                    f"Images: {progress[0]}, {progress[1]}/{progress[2]} complete"
                )
                previous = progress
            if status["phase"] in ("failed", "cancelled"):
                raise CLIError(
                    f"Image build {status['phase']}; inspect {output / 'status.json'}. After fixing the cause, use a new --output-dir.",
                    code="PLATFORM_ERROR",
                )
            if not wait or status["phase"] == "completed":
                return status
        except PlatformAPIError as error:
            if error.status_code not in (None, 408, 429, 500, 502, 503, 504):
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(min(10, max(0, deadline - time.monotonic())))
    raise CLIError(
        f"Local wait limit reached; cloud builds continue. Rerun the same command with the same --output-dir to resume: {output}",
        code="PLATFORM_ERROR",
    )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


def download(reference: dict[str, Any], access: dict[str, Any], path: Path) -> None:
    if access["reference"] != reference:
        raise CLIError(
            "Download reference differs from the published artifact",
            code="PLATFORM_ERROR",
        )
    try:
        parsed = urlsplit(access["url"])
        host = parsed.hostname or ""
        valid_origin = (
            parsed.scheme == "https"
            and (
                host == "storage.googleapis.com"
                or host.endswith(".storage.googleapis.com")
            )
            and not parsed.username
            and not parsed.password
            and parsed.port in (None, 443)
        )
    except ValueError:
        valid_origin = False
    if not valid_origin:
        raise CLIError(
            "Unexpected image artifact download origin", code="PLATFORM_ERROR"
        )
    digest = hashlib.sha256()
    size = 0
    temporary = None
    try:
        descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        temporary = Path(name)
        with (
            os.fdopen(descriptor, "wb") as output,
            urllib.request.build_opener(_NoRedirect).open(
                access["url"], timeout=300
            ) as source,
        ):
            while chunk := source.read(1024 * 1024):
                size += len(chunk)
                if size > _MAX_BYTES:
                    raise CLIError(
                        "Image artifact exceeds 2 GiB", code="PLATFORM_ERROR"
                    )
                output.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != reference["sha256"]:
            raise CLIError("Image artifact checksum mismatch", code="PLATFORM_ERROR")
        temporary.replace(path)
    except (urllib.error.URLError, OSError):
        raise CLIError(
            "Image artifact transfer failed; rerun to obtain fresh download access",
            code="PLATFORM_ERROR",
        ) from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _tree_identity(root: Path) -> dict[str, Any]:
    result = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise CLIError("Task symlinks are unsupported", code="VALIDATION")
        if path.is_file():
            with path.open("rb") as handle:
                result[path.relative_to(root).as_posix()] = (
                    hashlib.file_digest(handle, "sha256").hexdigest(),
                    path.stat().st_mode & 0o777,
                )
    return result


def collect(
    client: OsmosisClient, state: dict[str, Any], status: dict[str, Any], output: Path
) -> dict[str, Any]:
    result = status["result"]
    access = client.get_image_build_artifacts(
        state["request_id"], git_identity=state["inputs"]["git_identity"]
    )
    download(result["manifest"], access["manifest"], output / "manifest.json")
    manifest = json.loads((output / "manifest.json").read_text())
    if (
        manifest["source_revision"] != status["source_revision"]
        or manifest["bundle"] != result["bundle"]
    ):
        raise CLIError(
            "Published source or bundle does not match build status",
            code="PLATFORM_ERROR",
        )
    images = {entry["key"]: entry["image"] for entry in manifest["images"]}
    tasks = {entry["task"]: entry["environments"] for entry in manifest["tasks"]}
    if (
        len(images) != len(manifest["images"])
        or len(tasks) != len(manifest["tasks"])
        or len(tasks) != result["task_count"]
    ):
        raise CLIError(
            "Published image/task inventory is inconsistent", code="PLATFORM_ERROR"
        )
    if any(not _IMAGE.fullmatch(image) for image in images.values()):
        raise CLIError("Published image is not pinned by digest", code="PLATFORM_ERROR")
    named = manifest.get("named_images", {})
    if any(image not in images.values() for image in named.values()):
        raise CLIError(
            "Named image is missing from the manifest", code="PLATFORM_ERROR"
        )
    download(result["bundle"], access["bundle"], output / "bundle.tar.gz")
    prefix = state["request"]["tasks_dir"] + "/"
    bindings = {}
    with tempfile.TemporaryDirectory(prefix="image-bundle-", dir=output) as directory:
        root = Path(directory)
        size = 0
        try:
            with tarfile.open(output / "bundle.tar.gz") as archive:
                for index, member in enumerate(archive):
                    size += member.size
                    if (
                        index >= 100000
                        or size > _MAX_BYTES
                        or not (member.isfile() or member.isdir())
                    ):
                        raise CLIError(
                            "Task bundle exceeds extraction limits or contains unsupported entries",
                            code="PLATFORM_ERROR",
                        )
                    archive.extract(member, root, filter="data")
        except tarfile.TarError:
            raise CLIError(
                "Task bundle is invalid or contains unsafe archive entries",
                code="PLATFORM_ERROR",
            ) from None
        source = root / state["request"]["tasks_dir"]
        selected_root = source.resolve()
        for name, roles in tasks.items():
            if not name.startswith(prefix) or not (
                root / name
            ).resolve().is_relative_to(selected_root):
                raise CLIError(
                    "Published task is outside the selected directory",
                    code="PLATFORM_ERROR",
                )
            config = tomllib.loads((root / name / "task.toml").read_text())
            expected = {"environment"}
            owners = [
                ("", config),
                *(
                    (f"steps.{i}.", step)
                    for i, step in enumerate(config.get("steps", []))
                ),
            ]
            for role_prefix, owner in owners:
                if owner.get("verifier", {}).get("environment") is not None:
                    expected.add(role_prefix + "verifier.environment")
            if set(roles) != expected:
                raise CLIError(
                    "Published verifier/agent roles differ from the task",
                    code="PLATFORM_ERROR",
                )
            mapped = {}
            for role, key in roles.items():
                if not isinstance(key, str) or key not in images:
                    raise CLIError(
                        "Published task references an undefined image",
                        code="PLATFORM_ERROR",
                    )
                table: Any = config
                for component in role.split("."):
                    table = (
                        table[int(component)]
                        if isinstance(table, list)
                        else table[component]
                    )
                if table.get("docker_image") != images[key]:
                    raise CLIError(
                        "Published task image binding is inconsistent",
                        code="PLATFORM_ERROR",
                    )
                mapped[role] = images[key]
            bindings[name.removeprefix(prefix)] = mapped
        discovered = set()
        for current, directories, files in os.walk(source):
            if "task.toml" in files:
                discovered.add(Path(current).relative_to(root).as_posix())
                directories.clear()
        if discovered != set(tasks):
            raise CLIError(
                "Task bundle inventory differs from the manifest", code="PLATFORM_ERROR"
            )
        destination = output / "tasks"
        if destination.is_symlink():
            raise CLIError(
                "Downloaded tasks path must not be a symlink", code="CONFLICT"
            )
        if destination.exists():
            if _tree_identity(destination) != _tree_identity(source):
                raise CLIError(
                    "Downloaded tasks were changed locally; use a new --output-dir",
                    code="CONFLICT",
                )
        else:
            shutil.move(source, destination)
    summary = {
        "named_images": named,
        "tasks": bindings,
        "source_revision": manifest["source_revision"],
    }
    write_json(output / "images.json", summary)
    return summary


def build(
    *,
    repository: str,
    ref: str,
    tasks_dir: str,
    output_dir: Path,
    request_id: str | None,
    wait: bool,
    timeout: float,
) -> OperationResult:
    state = prepare_request(output_dir, repository, ref, tasks_dir, request_id)
    client = OsmosisClient()
    status = wait_for_build(client, state, output_dir, timeout, wait)
    if status["phase"] == "completed":
        collect(client, state, status, output_dir)
    return OperationResult(
        operation="images.build",
        status=status["phase"],
        resource={
            "request_id": state["request_id"],
            **status,
            "output_dir": str(output_dir),
        },
        message=f"Image build {status['phase']}. State saved in {output_dir}",
    )


def info(*, repository: str, request_id: str) -> DetailResult:
    identity = normalize_git_identity(repository).identity
    status = OsmosisClient().get_image_build(request_id, git_identity=identity)
    return DetailResult(
        title="Image build",
        data={"request_id": request_id, **status},
        fields=detail_fields(
            [
                ("Request", request_id),
                ("Phase", status["phase"]),
                ("Commit", status.get("source_revision") or "Resolving"),
                ("Tasks", str(status.get("task_count", 0))),
                (
                    "Images",
                    f"{status.get('completed_image_count', 0)}/{status.get('image_count', 0)}",
                ),
            ]
        ),
    )
