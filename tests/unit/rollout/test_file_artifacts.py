"""Tests for osmosis_ai.rollout.utils.file_artifacts."""

import os
from pathlib import Path

import pytest

from osmosis_ai.rollout.agent_workflow import AgentWorkflow
from osmosis_ai.rollout.backend.local.backend import LocalBackend
from osmosis_ai.rollout.types import AgentWorkflowConfig
from osmosis_ai.rollout.utils import file_artifacts
from osmosis_ai.rollout.utils.file_artifacts import (
    artifact_tree_state,
    copy_artifact_tree,
    create_rollout_artifacts_dir,
    default_artifact_root,
)


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(file_artifacts, "CREATE_BACKOFF_SECONDS", 0)


class TestCreateRolloutArtifactsDir:
    async def test_creates_and_returns_harbor_shaped_dir(self, tmp_path):
        result = await create_rollout_artifacts_dir(tmp_path, "r1")

        assert result == tmp_path / "r1" / "artifacts"
        assert result.is_dir()
        # Marker lives outside artifacts/, so the collected dir starts empty.
        assert list(result.iterdir()) == []
        assert (tmp_path / "r1" / ".osmosis-health-check").is_file()

    async def test_retries_past_transient_mkdir_failure(self, tmp_path, monkeypatch):
        original_mkdir = Path.mkdir
        calls = {"n": 0}

        def flaky_mkdir(self, *args, **kwargs):
            calls["n"] += 1
            # Only the first top-level call fails; pathlib re-enters mkdir
            # recursively for parents, so raw counts don't map to attempts.
            if calls["n"] == 1:
                raise OSError("mount not ready")
            return original_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", flaky_mkdir)

        result = await create_rollout_artifacts_dir(tmp_path, "r1")

        assert result is not None and result.is_dir()

    async def test_second_call_skips_write_when_marker_exists(
        self, tmp_path, monkeypatch
    ):
        # Workflow + grader probe the same rollout_id; on a no-overwrite mount a
        # second write to the same marker key would fail, so it must be skipped.
        first = await create_rollout_artifacts_dir(tmp_path, "r1")
        assert first is not None

        def no_overwrite(self, *args, **kwargs):
            raise OSError("overwrite not permitted")

        monkeypatch.setattr(Path, "write_bytes", no_overwrite)

        second = await create_rollout_artifacts_dir(tmp_path, "r1")

        assert second == first

    async def test_returns_none_when_dir_is_not_writable(self, tmp_path, monkeypatch):
        def unwritable(self, *args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(Path, "write_bytes", unwritable)

        result = await create_rollout_artifacts_dir(tmp_path, "r1")

        assert result is None

    async def test_returns_none_when_mkdir_always_fails(self, tmp_path, monkeypatch):
        def boom(self, *args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(Path, "mkdir", boom)

        result = await create_rollout_artifacts_dir(tmp_path, "r1")

        assert result is None


class TestCopyArtifactTree:
    def test_merges_nested_files_and_replaces_conflicting_types(self, tmp_path):
        source = tmp_path / "source"
        destination = tmp_path / "destination"
        (source / "nested").mkdir(parents=True)
        (source / "nested" / "grader.json").write_text("{}")
        (source / "now-a-file").write_text("final")
        (destination / "nested").mkdir(parents=True)
        (destination / "nested" / "workflow.txt").write_text("kept")
        (destination / "now-a-file").mkdir(parents=True)

        copied = copy_artifact_tree(
            source,
            destination,
            destination_root=tmp_path,
        )

        assert copied == 2
        assert (destination / "nested" / "workflow.txt").read_text() == "kept"
        assert (destination / "nested" / "grader.json").read_text() == "{}"
        assert (destination / "now-a-file").read_text() == "final"

    def test_skips_symlinks(self, tmp_path):
        source = tmp_path / "source"
        destination = tmp_path / "destination"
        source.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("do not copy")
        (source / "linked.txt").symlink_to(secret)

        copy_artifact_tree(
            source,
            destination,
            destination_root=tmp_path,
        )

        assert not (destination / "linked.txt").exists()

    def test_skips_special_files(self, tmp_path):
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO files are unavailable on this platform")

        source = tmp_path / "source"
        destination = tmp_path / "destination"
        source.mkdir()
        os.mkfifo(source / "pipe")

        copied = copy_artifact_tree(
            source,
            destination,
            destination_root=tmp_path,
        )

        assert copied == 0
        assert not (destination / "pipe").exists()

    def test_rejects_intermediate_destination_symlink(self, tmp_path):
        source = tmp_path / "source"
        trusted_root = tmp_path / "trusted"
        outside = tmp_path / "outside"
        source.mkdir()
        trusted_root.mkdir()
        outside.mkdir()
        (source / "grader.txt").write_text("grader")
        (trusted_root / "logs").symlink_to(outside, target_is_directory=True)

        with pytest.raises(OSError, match="symlink in artifact destination path"):
            copy_artifact_tree(
                source,
                trusted_root / "logs" / "artifacts",
                destination_root=trusted_root,
            )

        assert not (outside / "artifacts" / "grader.txt").exists()

    def test_rejects_destination_outside_trusted_root(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()

        with pytest.raises(ValueError, match="outside trusted root"):
            copy_artifact_tree(
                source,
                tmp_path / "destination",
                destination_root=tmp_path / "trusted",
            )

    def test_replace_destination_removes_stale_entries(self, tmp_path):
        source = tmp_path / "source"
        destination = tmp_path / "destination"
        source.mkdir()
        destination.mkdir()
        (source / "fresh.txt").write_text("fresh")
        (destination / "stale.txt").write_text("stale")

        copy_artifact_tree(
            source,
            destination,
            destination_root=tmp_path,
            replace_destination=True,
        )

        assert (destination / "fresh.txt").read_text() == "fresh"
        assert not (destination / "stale.txt").exists()

    def test_baseline_skips_unchanged_files(self, tmp_path):
        source = tmp_path / "source"
        destination = tmp_path / "destination"
        source.mkdir()
        unchanged = source / "workflow.txt"
        unchanged.write_text("workflow")
        baseline = artifact_tree_state(source)
        changed = source / "grader.txt"
        changed.write_text("grader")

        copied = copy_artifact_tree(
            source,
            destination,
            destination_root=tmp_path,
            baseline=baseline,
        )

        assert copied == 1
        assert not (destination / "workflow.txt").exists()
        assert (destination / "grader.txt").read_text() == "grader"


class _StubWorkflow(AgentWorkflow):
    async def run(self, ctx):
        return None


class TestDefaultArtifactRoot:
    def test_falls_back_to_home_osmosis_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("_OSMOSIS_ROLLOUT_ARTIFACT_ROOT", raising=False)

        assert default_artifact_root() == Path.home() / ".osmosis"

    def test_returns_env_override_without_changing_home(self, tmp_path, monkeypatch):
        home_before = Path.home()
        override = tmp_path / "rollout_trials"
        monkeypatch.setenv("_OSMOSIS_ROLLOUT_ARTIFACT_ROOT", str(override))

        assert default_artifact_root() == override
        assert Path.home() == home_before

    def test_local_backend_observes_override_at_construction(
        self, tmp_path, monkeypatch
    ):
        home_before = Path.home()
        override = tmp_path / "local_trials"
        monkeypatch.setenv("_OSMOSIS_ROLLOUT_ARTIFACT_ROOT", str(override))

        backend = LocalBackend(
            workflow=_StubWorkflow,
            workflow_config=AgentWorkflowConfig(name="test"),
        )

        assert backend.artifact_root == override
        assert Path.home() == home_before

    def test_harbor_backend_observes_override_at_construction(
        self, tmp_path, monkeypatch
    ):
        from harbor.trial.queue import TrialQueue

        from osmosis_ai.rollout.backend.harbor.backend import HarborBackend

        home_before = Path.home()
        override = tmp_path / "harbor_trials"
        monkeypatch.setenv("_OSMOSIS_ROLLOUT_ARTIFACT_ROOT", str(override))

        task = tmp_path / "template-task"
        (task / "environment").mkdir(parents=True)
        (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
        (task / "task.toml").write_text('[task]\nname = "template-task"\n')

        backend = HarborBackend(
            orchestrator=TrialQueue(n_concurrent=1),
            tasks_dir=task,
            agent="terminus-2",
        )

        assert backend.artifact_root == override
        assert Path.home() == home_before
