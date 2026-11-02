"""The documentation fast path must never hide code, packaging, or CI changes."""

from __future__ import annotations

import runpy
import subprocess
from pathlib import Path

import pytest

documentation_only = runpy.run_path(
    str(Path(__file__).parents[2] / ".github/scripts/ci-scope.py")
)["documentation_only"]


def git(*args: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            "user.name=CI Test",
            "-c",
            "user.email=ci@example.test",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            *args,
        ],
        text=True,
        stderr=subprocess.PIPE,
    ).strip()


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    git("init", "-b", "main")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/guide.md").write_text("Original guide\n")
    (tmp_path / "source.py").write_text("print('hello')\n")
    git("add", ".")
    git("commit", "-m", "base")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("PR_BASE_SHA", git("rev-parse", "HEAD"))
    git("checkout", "-b", "feature")
    return tmp_path


def merge(monkeypatch: pytest.MonkeyPatch) -> None:
    git("add", "-A")
    git("commit", "--allow-empty", "-m", "change")
    monkeypatch.setenv("PR_HEAD_SHA", git("rev-parse", "HEAD"))
    git("checkout", "main")
    git("merge", "--no-ff", "feature", "-m", "merge")


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("docs/nested/new guide.md", True),
        ("CONTRIBUTING.md", True),
        ("AGENTS.md", True),
        ("README.md", False),
        ("pyproject.toml", False),
        ("uv.lock", False),
        ("docs/example.py", False),
        (".github/workflows/tests.yml", False),
        ("source.py", False),
    ],
)
def test_merge_diff_is_conservative(
    checkout: Path, monkeypatch: pytest.MonkeyPatch, path: str, expected: bool
) -> None:
    changed = checkout / path
    changed.parent.mkdir(parents=True, exist_ok=True)
    changed.write_text("Changed\n")
    merge(monkeypatch)
    assert documentation_only() is expected


def test_renaming_code_into_docs_cannot_hide_its_removal(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (checkout / "source.py").rename(checkout / "docs/source.md")
    merge(monkeypatch)
    assert not documentation_only()


def test_docs_deletion_is_checked_but_unknown_diffs_run_everything(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (checkout / "docs/guide.md").unlink()
    merge(monkeypatch)
    assert documentation_only()
    for event in ("push", "release", "workflow_dispatch"):
        monkeypatch.setenv("GITHUB_EVENT_NAME", event)
        assert not documentation_only()
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("PR_HEAD_SHA", "0" * 40)
    assert not documentation_only()


def test_empty_diff_and_missing_history_run_everything(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    merge(monkeypatch)
    assert not documentation_only()
    git("checkout", "HEAD^1")
    assert not documentation_only()


def test_action_output_with_a_shallow_merge_checkout(
    checkout: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (checkout / "docs/guide.md").write_text("Updated guide\n")
    merge(monkeypatch)
    clone = tmp_path / "shallow"
    git("clone", "--depth=2", "--branch=main", checkout.as_uri(), str(clone))
    monkeypatch.chdir(clone)
    assert documentation_only()
    output = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    runpy.run_path(
        str(Path(__file__).parents[2] / ".github/scripts/ci-scope.py"),
        run_name="__main__",
    )
    assert output.read_text() == "docs-only=true\n"
    git("fetch", "--depth=1", "origin", "main")
    assert not documentation_only()
