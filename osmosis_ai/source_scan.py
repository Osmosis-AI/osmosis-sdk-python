"""Shared traversal for digesting a rollout project's source tree.

Two callers need the same answer to "which files describe this project's
code?": the Harbor bundle packager keys its wheel cache on it, and local
evaluation puts it in a run's resolved-input lock so a named run refuses to
resume across a code change. Keeping one traversal here means those two
digests can never disagree about which files count -- only the width differs,
since the packager truncates its cache key while eval stores a full SHA-256.

This module must stay free of optional dependencies: ``osmosis_ai.packaging``
requires the ``harbor`` extra and local evaluation must not.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from fnmatch import fnmatch
from pathlib import Path

EXCLUDE_DIRS: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "dist",
        "build",
        "node_modules",
        ".pytest_cache",
        ".ruff_cache",
    }
)

# Glob-style names, matched against every path component so a matching
# directory takes its contents with it. Resolving a project's environment
# writes these into its source tree -- ``uv sync`` drops ``uv.lock``, and a
# setuptools build backend drops ``<name>.egg-info/`` -- so counting them would
# let running a project change the digest that describes that project.
EXCLUDE_PATTERNS: tuple[str, ...] = ("uv.lock", "*.egg-info")


def _is_excluded(relative: Path, *, path: Path, exclude: Path | None) -> bool:
    parts = relative.parts
    if set(parts) & EXCLUDE_DIRS:
        return True
    if any(fnmatch(part, pattern) for pattern in EXCLUDE_PATTERNS for part in parts):
        return True
    return exclude is not None and path.is_relative_to(exclude)


def iter_source_files(root: Path, *, exclude: Path | None = None) -> Iterator[Path]:
    """Yield every source file under *root*, in unspecified order.

    Skips anything matching :data:`EXCLUDE_DIRS` or :data:`EXCLUDE_PATTERNS`
    and, when given, anything under *exclude* -- used to keep a cache or
    run-output directory nested in the project from changing the project's own
    digest.
    """
    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root)
        if _is_excluded(relative, path=candidate, exclude=exclude):
            continue
        yield candidate


def reject_directory_symlinks(
    root: Path, *, exclude: Path | None = None, label: str = "source"
) -> None:
    """Refuse to digest through a directory (or broken) symlink.

    ``rglob`` does not recurse into symlinked directories, so the digest cannot
    see their contents, while consumers that copy the tree dereference them --
    the digest would not describe what actually ships, and a mutated link
    target would keep serving stale cached output. A link resolving back into
    the output tree would even recurse. File symlinks stay allowed: hashing and
    copying both read the target's bytes, so the two agree.
    """
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root)
        if _is_excluded(relative, path=candidate, exclude=exclude):
            continue
        if candidate.is_symlink() and not candidate.is_file():
            raise ValueError(
                f"{label} contains a directory or broken symlink: "
                f"{candidate} -> {os.readlink(candidate)}; replace it with a real "
                "directory or file so the digest can describe what it covers"
            )


def source_digest(root: Path, *, extra: str = "", exclude: Path | None = None) -> str:
    """Return the full SHA-256 hex digest of *root*'s source bytes.

    Relative paths are hashed alongside file contents so a rename changes the
    digest. *extra* is mixed in first, letting a caller bind the digest to
    out-of-tree inputs of its own.
    """
    digest = hashlib.sha256(extra.encode())
    for file in sorted(iter_source_files(root, exclude=exclude)):
        digest.update(str(file.relative_to(root)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()
