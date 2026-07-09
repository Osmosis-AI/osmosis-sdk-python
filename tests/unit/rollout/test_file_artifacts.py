"""Tests for osmosis_ai.rollout.utils.file_artifacts."""

from pathlib import Path

import pytest

from osmosis_ai.rollout.utils import file_artifacts
from osmosis_ai.rollout.utils.file_artifacts import create_rollout_artifacts_dir


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

    @pytest.mark.parametrize(
        "rollout_id",
        ["../other", "a/b", "/tmp/other", "..", ".", ""],
    )
    async def test_rejects_ids_that_escape_root(
        self, tmp_path, monkeypatch, rollout_id
    ):
        # An untrusted rollout id must never place artifacts outside the root.
        result = await create_rollout_artifacts_dir(tmp_path, rollout_id)

        assert result is None
        assert list(tmp_path.iterdir()) == []
