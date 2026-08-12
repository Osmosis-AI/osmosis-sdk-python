"""Fail-closed handling for non-HTTPS non-loopback OSMOSIS_PLATFORM_URL."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from osmosis_ai.cli import main as cli
from osmosis_ai.platform.auth.config import (
    InsecurePlatformURLWarning,
    get_platform_url,
    is_insecure_platform_url,
)


@pytest.fixture
def _dotenv_platform_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Write a cwd ``.env`` URL and ensure it cannot leak into later tests."""
    monkeypatch.delenv("OSMOSIS_PLATFORM_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    os.environ.pop("OSMOSIS_PLATFORM_URL", None)


def test_is_insecure_platform_url_distinguishes_loopback() -> None:
    assert is_insecure_platform_url("http://example.invalid") is True
    assert is_insecure_platform_url("http://127.0.0.1:8000") is False
    assert is_insecure_platform_url("http://localhost:8000") is False
    assert is_insecure_platform_url("https://platform.osmosis.ai") is False


def test_get_platform_url_emits_named_insecure_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OSMOSIS_PLATFORM_URL", "http://example.invalid")
    with pytest.warns(InsecurePlatformURLWarning, match="not HTTPS"):
        assert get_platform_url() == "http://example.invalid"


def test_cwd_dotenv_http_platform_url_fails_closed_without_connecting(
    _dotenv_platform_url: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tmp_path = _dotenv_platform_url
    monkeypatch.setenv("OSMOSIS_TOKEN", "secret-token")
    monkeypatch.delenv("OSMOSIS_ALLOW_INSECURE_PLATFORM_URL", raising=False)
    (tmp_path / ".env").write_text(
        "OSMOSIS_PLATFORM_URL=http://example.invalid\n",
        encoding="utf-8",
    )

    def _must_not_connect(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("must not open a connection")

    monkeypatch.setattr(
        "osmosis_ai.platform.auth.platform_client.platform_request",
        _must_not_connect,
    )
    monkeypatch.setattr("urllib.request.urlopen", _must_not_connect)

    rc = cli.main(["--json", "auth", "whoami"])

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    stderr_envelopes = [json.loads(line) for line in captured.err.splitlines()]
    assert stderr_envelopes[0]["warning"]["code"] == "INSECURE_PLATFORM_URL"
    assert "http://example.invalid" in stderr_envelopes[0]["warning"]["message"]
    error = stderr_envelopes[1]["error"]
    assert error["code"] == "VALIDATION"
    assert "http://example.invalid" in error["message"]
    assert "OSMOSIS_ALLOW_INSECURE_PLATFORM_URL=1" in error["message"]


def test_insecure_platform_url_allowed_with_opt_in(
    _dotenv_platform_url: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tmp_path = _dotenv_platform_url
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.setenv("OSMOSIS_ALLOW_INSECURE_PLATFORM_URL", "1")
    (tmp_path / ".env").write_text(
        "OSMOSIS_PLATFORM_URL=http://example.invalid\n",
        encoding="utf-8",
    )

    rc = cli.main(["--json", "auth", "login"])

    captured = capsys.readouterr()
    assert rc == 1
    stderr_envelopes = [json.loads(line) for line in captured.err.splitlines()]
    assert stderr_envelopes[0]["warning"]["code"] == "INSECURE_PLATFORM_URL"
    error = stderr_envelopes[1]["error"]
    assert error["code"] == "INTERACTIVE_REQUIRED"


def test_loopback_http_platform_url_is_allowed(
    _dotenv_platform_url: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tmp_path = _dotenv_platform_url
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.delenv("OSMOSIS_ALLOW_INSECURE_PLATFORM_URL", raising=False)
    (tmp_path / ".env").write_text(
        "OSMOSIS_PLATFORM_URL=http://127.0.0.1:8000\n",
        encoding="utf-8",
    )

    rc = cli.main(["--json", "auth", "login"])

    captured = capsys.readouterr()
    assert rc == 1
    error = json.loads(captured.err)["error"]
    assert error["code"] == "INTERACTIVE_REQUIRED"
