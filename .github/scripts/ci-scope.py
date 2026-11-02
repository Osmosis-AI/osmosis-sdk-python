"""Skip expensive PR checks only for a verified development-documentation diff."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def documentation_only() -> bool:
    if os.environ.get("GITHUB_EVENT_NAME") != "pull_request":
        return False
    try:
        parents = subprocess.check_output(
            ["git", "show", "-s", "--format=%P", "HEAD"], text=True
        ).split()
        if parents != [os.environ.get("PR_BASE_SHA"), os.environ.get("PR_HEAD_SHA")]:
            return False
        # Include both sides of renames so moving source into docs cannot hide it.
        paths = subprocess.check_output(
            ["git", "diff", "--name-only", "--no-renames", "-z", "HEAD^1", "HEAD"]
        ).split(b"\0")[:-1]
    except (OSError, subprocess.CalledProcessError):
        return False
    return bool(paths) and all(
        path in {b"AGENTS.md", b"CONTRIBUTING.md"}
        or (path.startswith(b"docs/") and path.endswith(b".md"))
        for path in paths
    )


if __name__ == "__main__":
    output = f"docs-only={str(documentation_only()).lower()}"
    print(output)
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as handle:
        handle.write(output + "\n")
