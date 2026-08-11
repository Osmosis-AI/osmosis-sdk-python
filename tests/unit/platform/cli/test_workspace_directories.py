"""Tests for the remembered workspace directory locations."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from osmosis_ai.platform.cli.workspace_directories import (
    forget_workspace_directory,
    recall_workspace_directory,
    remember_workspace_directory,
)

PLATFORM = "https://platform.osmosis.ai"
OTHER_PLATFORM = "https://staging.osmosis.ai"


@pytest.fixture(autouse=True)
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "config" / "workspace-directories.json"
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace_directories.WORKSPACE_DIRECTORIES_FILE", path
    )
    monkeypatch.setenv("OSMOSIS_PLATFORM_URL", PLATFORM)
    return path


def test_a_remembered_clone_is_recalled(tmp_path: Path) -> None:
    remember_workspace_directory("ws-1", tmp_path / "acme-workspace")

    assert recall_workspace_directory("ws-1") == (tmp_path / "acme-workspace").resolve()


def test_an_unknown_workspace_recalls_nothing() -> None:
    assert recall_workspace_directory("ws-unknown") is None


def test_remembering_again_replaces_the_previous_path(tmp_path: Path) -> None:
    remember_workspace_directory("ws-1", tmp_path / "first")
    remember_workspace_directory("ws-1", tmp_path / "second")

    assert recall_workspace_directory("ws-1") == (tmp_path / "second").resolve()


def test_forgetting_leaves_other_workspaces_alone(tmp_path: Path) -> None:
    remember_workspace_directory("ws-1", tmp_path / "acme")
    remember_workspace_directory("ws-2", tmp_path / "globex")

    forget_workspace_directory("ws-1")

    assert recall_workspace_directory("ws-1") is None
    assert recall_workspace_directory("ws-2") == (tmp_path / "globex").resolve()


def test_forgetting_an_unknown_workspace_is_harmless() -> None:
    forget_workspace_directory("ws-unknown")

    assert recall_workspace_directory("ws-unknown") is None


def test_clones_are_scoped_to_the_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remember_workspace_directory("ws-1", tmp_path / "prod-clone")

    monkeypatch.setenv("OSMOSIS_PLATFORM_URL", OTHER_PLATFORM)
    assert recall_workspace_directory("ws-1") is None

    remember_workspace_directory("ws-1", tmp_path / "staging-clone")
    assert recall_workspace_directory("ws-1") == (tmp_path / "staging-clone").resolve()

    monkeypatch.setenv("OSMOSIS_PLATFORM_URL", PLATFORM)
    assert recall_workspace_directory("ws-1") == (tmp_path / "prod-clone").resolve()


def test_a_relative_path_is_stored_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    remember_workspace_directory("ws-1", Path("acme-workspace"))

    assert recall_workspace_directory("ws-1") == (tmp_path / "acme-workspace").resolve()


def test_a_malformed_file_reads_as_empty(store: Path, tmp_path: Path) -> None:
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("{not json")

    assert recall_workspace_directory("ws-1") is None

    remember_workspace_directory("ws-1", tmp_path / "acme")
    assert recall_workspace_directory("ws-1") == (tmp_path / "acme").resolve()


def test_an_unexpected_version_reads_as_empty(store: Path) -> None:
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(
        json.dumps({"version": 99, "platforms": {PLATFORM: {"ws-1": "/somewhere"}}})
    )

    assert recall_workspace_directory("ws-1") is None


def test_an_unwritable_store_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace_directories.WORKSPACE_DIRECTORIES_FILE",
        tmp_path / "missing-file" / "nested" / "directories.json",
    )

    def deny(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("read-only file system")

    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace_directories.atomic_write_json", deny
    )

    remember_workspace_directory("ws-1", tmp_path / "acme")

    assert recall_workspace_directory("ws-1") is None


def test_the_store_is_not_world_readable(store: Path, tmp_path: Path) -> None:
    remember_workspace_directory("ws-1", tmp_path / "acme")

    assert stat.S_IMODE(store.stat().st_mode) == 0o600
