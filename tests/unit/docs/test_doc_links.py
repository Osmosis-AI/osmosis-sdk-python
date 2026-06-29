"""Guard the "code-anchored" docs contract.

The developer docs under ``docs/`` link to package source with relative paths
(e.g. ``../osmosis_ai/rollout/server/app.py``) and to each other (``./cli.md``).
Nothing else enforces that those targets still exist, so a renamed or moved file
silently rots the docs. This test fails when any relative link in repo markdown
points at a path that no longer exists, forcing the doc to be fixed in the same
PR as the code change.

Scope: file/directory existence only. External URLs (``https://``, ``mailto:``)
and in-file heading anchors (``#section``) are intentionally out of scope —
heading-anchor slugging is too fragile to check reliably, and external links
can't be verified offline.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Directory names that never contain documentation we author/control.
_EXCLUDED_DIR_PARTS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "build",
        "dist",
        "__pycache__",
        ".ruff_cache",
        ".pytest_cache",
        ".mypy_cache",
        ".tox",
        "htmlcov",
        # Local-only working artifacts (gitignored), not published docs.
        "superpowers",
    }
)

# [label](target ...) — capture target up to whitespace or the closing paren,
# tolerating an optional <...> wrapper. A trailing `"title"` is dropped because
# the capture stops at the first whitespace.
_LINK_RE = re.compile(r"\[[^\]]*\]\(\s*<?([^)>\s]+)")

# Inline code spans can hold bracket/paren noise that mimics a link; drop them.
_INLINE_CODE_RE = re.compile(r"`[^`]*`")

# Scheme-prefixed (http:, mailto:, …) or protocol-relative (//host) targets.
_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*:")


def _is_external(target: str) -> bool:
    return bool(_SCHEME_RE.match(target)) or target.startswith("//")


def _iter_local_link_targets(text: str) -> list[tuple[int, str]]:
    """Return (lineno, target) for local markdown links, skipping code blocks."""
    results: list[tuple[int, str]] = []
    in_fence = False
    fence_marker = ""
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.lstrip()
        if in_fence:
            if stripped.startswith(fence_marker):
                in_fence = False
            continue
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = True
            fence_marker = stripped[:3]
            continue
        line = _INLINE_CODE_RE.sub("", raw_line)
        for match in _LINK_RE.finditer(line):
            target = match.group(1)
            if _is_external(target) or target.startswith("#"):
                continue
            results.append((lineno, target))
    return results


def _discover_markdown_files() -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(_REPO_ROOT):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in _EXCLUDED_DIR_PARTS and not d.endswith(".egg-info")
        ]
        for name in filenames:
            if name.endswith(".md"):
                files.append(Path(dirpath) / name)
    return sorted(files)


_MARKDOWN_FILES = _discover_markdown_files()


def test_scanner_is_not_vacuous() -> None:
    """Fail loudly if discovery/parsing silently matches nothing (regex rot)."""
    rels = {p.relative_to(_REPO_ROOT).as_posix() for p in _MARKDOWN_FILES}
    assert "docs/architecture.md" in rels
    assert "README.md" in rels

    arch = _REPO_ROOT / "docs" / "architecture.md"
    targets = [t for _, t in _iter_local_link_targets(arch.read_text("utf-8"))]
    # architecture.md anchors to package source via ../osmosis_ai/... paths.
    assert any(t.startswith("../osmosis_ai/") for t in targets)


def test_local_markdown_links_resolve() -> None:
    broken: list[str] = []
    for md_file in _MARKDOWN_FILES:
        rel = md_file.relative_to(_REPO_ROOT).as_posix()
        for lineno, target in _iter_local_link_targets(md_file.read_text("utf-8")):
            path_part = target.split("#", 1)[0].split("?", 1)[0]
            if not path_part:  # was a pure anchor/query after stripping
                continue
            resolved = (md_file.parent / path_part).resolve()
            if not resolved.exists():
                broken.append(f"{rel}:{lineno} -> {target}")

    assert not broken, "Broken relative links in markdown:\n" + "\n".join(broken)
