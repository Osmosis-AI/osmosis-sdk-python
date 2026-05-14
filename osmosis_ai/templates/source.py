"""Template source resolution.

The configured upstream repository provides starter file contents. Control
metadata stays in the SDK catalog so user-editable repos cannot change CLI
behavior. Tests and local development can point at a checkout with
``OSMOSIS_WORKSPACE_TEMPLATE_PATH``.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import requests

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.auth.config import CACHE_DIR

DEFAULT_WORKSPACE_TEMPLATE_REPO = "Osmosis-AI/workspace-template"
DEFAULT_WORKSPACE_TEMPLATE_REF = "main"

_PATH_ENV = "OSMOSIS_WORKSPACE_TEMPLATE_PATH"
_REPO_ENV = "OSMOSIS_WORKSPACE_TEMPLATE_REPO"
_REF_ENV = "OSMOSIS_WORKSPACE_TEMPLATE_REF"


def _template_repo() -> str:
    return os.environ.get(_REPO_ENV) or DEFAULT_WORKSPACE_TEMPLATE_REPO


def _template_ref() -> str:
    return os.environ.get(_REF_ENV) or DEFAULT_WORKSPACE_TEMPLATE_REF


def _cache_key(repo: str, ref: str) -> str:
    return "__".join(part.replace("/", "_") for part in (repo, ref))


def _safe_extract(archive: tarfile.TarFile, target: Path) -> None:
    """Extract a tar archive while rejecting path traversal."""
    target_resolved = target.resolve()
    for member in archive.getmembers():
        if member.issym() or member.islnk():
            raise CLIError(
                f"Template archive contains an unsafe link: {member.name}",
                code="VALIDATION",
            )
        member_path = target / member.name
        try:
            member_path.resolve().relative_to(target_resolved)
        except ValueError as exc:
            raise CLIError(
                f"Template archive contains an unsafe path: {member.name}",
                code="VALIDATION",
            ) from exc
    archive.extractall(target, filter="data")


def _download_workspace_template(repo: str, ref: str, destination: Path) -> None:
    url = f"https://github.com/{repo}/archive/{ref}.tar.gz"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise CLIError(
            f"Unable to fetch starter templates from {url}: {exc}",
            code="NETWORK",
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="osmosis-template-") as tmp_name:
        tmp_dir = Path(tmp_name)
        archive_path = tmp_dir / "workspace-template.tar.gz"
        archive_path.write_bytes(response.content)
        extract_dir = tmp_dir / "extract"
        extract_dir.mkdir()
        try:
            with tarfile.open(archive_path, "r:gz") as archive:
                _safe_extract(archive, extract_dir)
        except tarfile.TarError as exc:
            raise CLIError(
                f"Template archive from {url} is invalid: {exc}",
                code="VALIDATION",
            ) from exc

        roots = [path for path in extract_dir.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise CLIError(
                "Template archive must contain exactly one root directory.",
                code="VALIDATION",
            )
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(roots[0]), destination)


def workspace_template_root(*, refresh: bool = False) -> Path:
    """Return a local directory containing the template source checkout.

    ``refresh=True`` re-fetches the configured repo/ref so scaffold repair can
    use the latest template contents instead of a stale cache.
    """
    override = os.environ.get(_PATH_ENV)
    if override:
        root = Path(override).expanduser().resolve()
        if not root.is_dir():
            raise CLIError(
                f"Configured template path does not exist: {root}",
                code="NOT_FOUND",
            )
        return root

    repo = _template_repo()
    ref = _template_ref()
    cache_root = Path(CACHE_DIR) / "workspace-template" / _cache_key(repo, ref)
    if refresh or not cache_root.is_dir():
        _download_workspace_template(repo, ref, cache_root)
    return cache_root


__all__ = [
    "DEFAULT_WORKSPACE_TEMPLATE_REF",
    "DEFAULT_WORKSPACE_TEMPLATE_REPO",
    "workspace_template_root",
]
