"""Tests for osmosis_ai.platform.auth.local_config."""

from __future__ import annotations

from pathlib import Path

import pytest

from osmosis_ai.platform.auth import local_config


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect CONFIG_DIR / CACHE_DIR / CONFIG_FILE to a temp directory."""
    config_dir = tmp_path / "osmosis"
    cache_dir = config_dir / "cache"
    config_file = config_dir / "config.json"

    monkeypatch.setattr(local_config, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(local_config, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(local_config, "CONFIG_FILE", config_file)


# ── _safe_ws_name ────────────────────────────────────────────────


class TestSafeWsName:
    def test_simple_name(self):
        assert local_config._safe_ws_name("my-workspace") == "my-workspace"

    def test_uppercase(self):
        assert local_config._safe_ws_name("MyWorkspace") == "myworkspace"

    def test_spaces(self):
        assert local_config._safe_ws_name("my workspace") == "my_workspace"

    def test_special_chars(self):
        assert local_config._safe_ws_name("ws@#$.test!") == "ws____test_"

    def test_already_safe(self):
        assert local_config._safe_ws_name("foo_bar-123") == "foo_bar-123"


# ── Subscription cache (cache/ directory) ────────────────────────


class TestSubscriptionCache:
    def test_load_empty(self):
        assert local_config.load_subscription_status("ws") is None

    def test_save_and_load_true(self):
        local_config.save_subscription_status("ws", True)
        assert local_config.load_subscription_status("ws") is True

    def test_save_and_load_false(self):
        local_config.save_subscription_status("ws", False)
        assert local_config.load_subscription_status("ws") is False

    def test_written_to_cache_dir(self, tmp_path: Path):
        local_config.save_subscription_status("My WS", True)
        cache_dir = tmp_path / "osmosis" / "cache"
        files = list(cache_dir.glob("subscription_*.json"))
        assert len(files) == 1
        assert "my_ws" in files[0].name

    def test_does_not_touch_config_json(self, tmp_path: Path):
        local_config.save_subscription_status("ws", True)
        config_file = tmp_path / "osmosis" / "config.json"
        assert not config_file.exists()

    def test_overwrite(self):
        local_config.save_subscription_status("ws", True)
        local_config.save_subscription_status("ws", False)
        assert local_config.load_subscription_status("ws") is False


# ── clear_workspace_data ─────────────────────────────────────────


class TestClearWorkspaceData:
    def test_clears_cache_files(self, tmp_path: Path):
        local_config.save_subscription_status("ws", True)
        cache_dir = tmp_path / "osmosis" / "cache"
        legacy_cache_file = cache_dir / "projects_ws.json"
        legacy_cache_file.write_text("{}", encoding="utf-8")
        assert len(list(cache_dir.glob("*.json"))) == 2

        local_config.clear_workspace_data("ws")
        assert list(cache_dir.glob("subscription_*.json")) == []
        assert legacy_cache_file.exists()

    def test_does_not_affect_other_workspaces(self):
        local_config.save_subscription_status("ws-a", True)
        local_config.save_subscription_status("ws-b", True)

        local_config.clear_workspace_data("ws-a")

        assert local_config.load_subscription_status("ws-b") is True

    def test_noop_when_nothing_exists(self):
        local_config.clear_workspace_data("nonexistent")
