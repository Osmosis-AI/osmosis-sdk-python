"""Tests for Osmosis auth configuration."""

from __future__ import annotations

import pytest

from osmosis_ai.platform.auth.config import get_platform_url, normalize_platform_url


@pytest.mark.parametrize(
    ("raw_url", "expected"),
    [
        ("https://platform.osmosis.ai/", "https://platform.osmosis.ai"),
        ("https://platform.osmosis.ai///", "https://platform.osmosis.ai"),
        (" https://platform.osmosis.ai/ ", "https://platform.osmosis.ai"),
        ("https://staging.osmosis.ai/", "https://staging.osmosis.ai"),
    ],
)
def test_normalize_platform_url_strips_trailing_slashes(
    raw_url: str,
    expected: str,
) -> None:
    assert normalize_platform_url(raw_url) == expected


def test_get_platform_url_normalizes_env_trailing_slash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OSMOSIS_PLATFORM_URL", "https://platform.osmosis.ai/")

    assert get_platform_url() == "https://platform.osmosis.ai"
