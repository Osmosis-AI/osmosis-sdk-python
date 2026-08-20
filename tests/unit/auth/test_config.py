"""Tests for Osmosis auth configuration."""

from __future__ import annotations

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.auth.config import (
    get_platform_url,
    normalize_platform_url,
    validate_env_token_platform,
)


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


@pytest.mark.parametrize(
    ("raw_url", "expected"),
    [
        ("localhost:3000", "http://localhost:3000"),
        ("127.0.0.1:8000", "http://127.0.0.1:8000"),
        ("[::1]:3000", "http://[::1]:3000"),
        ("platform.osmosis.ai", "https://platform.osmosis.ai"),
        ("staging.osmosis.ai:8443", "https://staging.osmosis.ai:8443"),
        # "://" in the query must not be mistaken for a scheme.
        ("localhost:3000/cb?next=https://x", "http://localhost:3000/cb"),
    ],
)
def test_normalize_platform_url_adds_scheme_when_missing(
    raw_url: str,
    expected: str,
) -> None:
    """Scheme-less dev URLs get http:// for loopback and https:// otherwise."""
    assert normalize_platform_url(raw_url) == expected


def test_get_platform_url_normalizes_env_trailing_slash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OSMOSIS_PLATFORM_URL", "https://platform.osmosis.ai/")

    assert get_platform_url() == "https://platform.osmosis.ai"


@pytest.mark.parametrize(
    "raw_url",
    ["http://host:notaport", "http://[::1"],
)
def test_normalize_platform_url_invalid_port_or_ipv6_raises_validation(
    raw_url: str,
) -> None:
    with pytest.raises(CLIError) as exc_info:
        normalize_platform_url(raw_url)

    assert exc_info.value.code == "VALIDATION"
    assert raw_url in exc_info.value.message


def test_env_token_without_binding_remains_compatible_with_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OSMOSIS_TOKEN", "prod-token")
    monkeypatch.delenv("OSMOSIS_PLATFORM_URL", raising=False)
    monkeypatch.delenv("OSMOSIS_TOKEN_PLATFORM_URL", raising=False)

    validate_env_token_platform()


def test_env_token_requires_binding_for_non_production_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OSMOSIS_TOKEN", "staging-token")
    monkeypatch.setenv("OSMOSIS_PLATFORM_URL", "https://platform-staging.osmosis.ai")
    monkeypatch.delenv("OSMOSIS_TOKEN_PLATFORM_URL", raising=False)

    with pytest.raises(CLIError) as exc_info:
        validate_env_token_platform()

    assert exc_info.value.code == "ENV_TOKEN_PLATFORM_REQUIRED"


def test_env_token_rejects_mismatched_platform_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OSMOSIS_TOKEN", "staging-token")
    monkeypatch.setenv("OSMOSIS_PLATFORM_URL", "https://platform-staging.osmosis.ai")
    monkeypatch.setenv("OSMOSIS_TOKEN_PLATFORM_URL", "https://platform.osmosis.ai")

    with pytest.raises(CLIError) as exc_info:
        validate_env_token_platform()

    assert exc_info.value.code == "ENV_TOKEN_PLATFORM_MISMATCH"


def test_env_token_accepts_matching_normalized_platform_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OSMOSIS_TOKEN", "local-token")
    monkeypatch.setenv("OSMOSIS_PLATFORM_URL", "localhost:3000")
    monkeypatch.setenv("OSMOSIS_TOKEN_PLATFORM_URL", "http://localhost:3000/")

    validate_env_token_platform()
