"""Tests for osmosis_ai.platform.auth.local_config."""

from __future__ import annotations

from pathlib import Path

import pytest

from osmosis_ai.platform.auth import local_config


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect CONFIG_DIR / CONFIG_FILE to a temp directory."""
    config_dir = tmp_path / "osmosis"
    config_file = config_dir / "config.json"

    monkeypatch.setattr(local_config, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(local_config, "CONFIG_FILE", config_file)


def test_local_config_does_not_expose_subscription_cache_helpers() -> None:
    assert not hasattr(local_config, "clear_workspace_data")
    assert not hasattr(local_config, "load_subscription_status")
    assert not hasattr(local_config, "save_subscription_status")


# ── session reset ────────────────────────────────────────────────


def test_reset_session_deletes_credentials_and_legacy_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    config_file = tmp_path / "osmosis" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(local_config, "CONFIG_FILE", config_file)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.delete_credentials",
        lambda: calls.append("delete_credentials"),
    )

    local_config.reset_session()

    assert calls == ["delete_credentials"]
    assert not config_file.exists()


def test_clear_all_local_data_delegates_to_session_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(local_config, "reset_session", lambda: calls.append("reset"))

    local_config.clear_all_local_data()

    assert calls == ["reset"]
