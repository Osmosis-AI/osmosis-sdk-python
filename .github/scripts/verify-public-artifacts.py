"""Reject internal tooling and private dependencies in public release artifacts."""

from __future__ import annotations

import re
import sys
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import NoReturn

FORBIDDEN_PACKAGE_PATHS = (
    "osmosis_ai/cli/commands/dev/",
    "osmosis_ai/cli/commands/images.py",
    "osmosis_ai/platform/cli/dev_server.py",
    "osmosis_ai/platform/cli/images.py",
)
FORBIDDEN_SYMBOLS = (
    b"DevServerBackend",
    b"DevServerSandboxEnvironment",
    b"DevRolloutServerInfo",
    b"PaginatedDevRolloutServers",
    b"serialize_dev_rollout_server",
    b"provision_dev_rollout_server",
    b"teardown_dev_rollout_server",
    b"get_dev_rollout_server_logs",
    b"stream_dev_rollout_server_logs",
    b"list_dev_rollout_servers",
    b"/api/cli/dev-rollout-server",
    b"submit_image_build",
    b"get_image_build",
    b"get_image_build_artifacts",
    b"get_image_pull_credentials",
    b"/api/cli/image-builds",
)


def fail(message: str) -> NoReturn:
    # Explicit exits, not asserts, so ``python -O`` cannot skip the checks.
    raise SystemExit(message)


def check_member(name: str, content: bytes) -> None:
    parts = PurePosixPath(name).parts
    if "osmosis_ai" in parts:
        package_path = "/".join(parts[parts.index("osmosis_ai") :])
        if package_path.startswith(FORBIDDEN_PACKAGE_PATHS):
            fail(f"Internal module shipped: {name}")
        if name.endswith(".py"):
            for symbol in FORBIDDEN_SYMBOLS:
                if symbol in content:
                    fail(f"Internal API {symbol.decode()} shipped in {name}")
    if parts[-1] in {"METADATA", "PKG-INFO"}:
        metadata = BytesParser().parsebytes(content, headersonly=True)
        for requirement in metadata.get_all("Requires-Dist", []):
            if re.match(r"osmo(?:\W|$)", requirement, re.IGNORECASE):
                fail(f"Public artifact has a private dependency: {requirement}")


def check_artifact(path: Path) -> None:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.endswith("/"):
                    check_member(name, archive.read(name))
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile():
                    source = archive.extractfile(member)
                    if source is None:
                        fail(f"Unreadable sdist member: {member.name}")
                    check_member(member.name, source.read())
    else:
        raise ValueError(f"Unsupported release artifact: {path}")
    print(f"Public artifact boundary verified: {path.name}")


if __name__ == "__main__":
    artifacts = [Path(argument) for argument in sys.argv[1:]]
    if not any(path.suffix == ".whl" for path in artifacts):
        fail("Missing wheel")
    if not any(path.name.endswith(".tar.gz") for path in artifacts):
        fail("Missing sdist")
    for artifact in artifacts:
        check_artifact(artifact)
