"""Keep internal image-build tooling out of both public package formats."""

from __future__ import annotations

import io
import runpy
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
check_artifact = cast(
    Callable[[Path], None],
    runpy.run_path(str(REPO_ROOT / ".github/scripts/verify-public-artifacts.py"))[
        "check_artifact"
    ],
)


@pytest.fixture(params=["wheel", "sdist"])
def archive_format(request: pytest.FixtureRequest) -> str:
    return request.param


def _write_artifact(
    tmp_path: Path, archive_format: str, members: dict[str, bytes]
) -> Path:
    if archive_format == "wheel":
        path = tmp_path / "osmosis_ai-0.0.0-py3-none-any.whl"
        with zipfile.ZipFile(path, "w") as archive:
            for name, content in members.items():
                archive.writestr(name, content)
    else:
        path = tmp_path / "osmosis_ai-0.0.0.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            for name, content in members.items():
                member = tarfile.TarInfo(f"osmosis_ai-0.0.0/{name}")
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
    return path


@pytest.mark.parametrize(
    "module",
    [
        "osmosis_ai/cli/commands/images.py",
        "osmosis_ai/platform/cli/images.py",
    ],
)
def test_rejects_internal_image_build_modules(
    tmp_path: Path, archive_format: str, module: str
) -> None:
    artifact = _write_artifact(tmp_path, archive_format, {module: b""})

    with pytest.raises(SystemExit, match="Internal module shipped") as error:
        check_artifact(artifact)

    assert module in str(error.value)


@pytest.mark.parametrize(
    "content",
    [
        b"def submit_image_build(): pass",
        b"def get_image_build(): pass",
        b"def get_image_build_artifacts(): pass",
        b"def get_image_pull_credentials(): pass",
        b'ENDPOINT = "/api/cli/image-builds"',
    ],
)
def test_rejects_image_build_api_moved_to_another_module(
    tmp_path: Path, archive_format: str, content: bytes
) -> None:
    module = "osmosis_ai/platform/relocated.py"
    artifact = _write_artifact(tmp_path, archive_format, {module: content})

    with pytest.raises(SystemExit, match="Internal API") as error:
        check_artifact(artifact)

    assert module in str(error.value)


def test_permits_retained_harbor_runtime_and_historical_mentions(
    tmp_path: Path, archive_format: str
) -> None:
    retained_modules = [
        "osmosis_ai/harbor_images.py",
        "osmosis_ai/rollout/backend/harbor/source.py",
    ]
    members = {name: (REPO_ROOT / name).read_bytes() for name in retained_modules}
    members["CHANGELOG.md"] = (
        b"Moved osmosis_ai/cli/commands/images.py and submit_image_build to Osmo."
    )
    members["tests/unit/test_removed_image_builds.py"] = (
        b'assert not hasattr(client, "get_image_build")\n'
        b'OLD_ROUTE = "/api/cli/image-builds"\n'
    )

    check_artifact(_write_artifact(tmp_path, archive_format, members))
