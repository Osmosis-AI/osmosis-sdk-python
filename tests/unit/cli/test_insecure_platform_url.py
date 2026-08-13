"""Fail-closed handling for non-HTTPS non-loopback OSMOSIS_PLATFORM_URL."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import pytest

from osmosis_ai.cli import main as cli
from osmosis_ai.platform.auth.config import (
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


def test_get_platform_url_is_silent_for_insecure_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Surfacing insecure URLs is the CLI gate's job; the resolver stays quiet."""
    monkeypatch.setenv("OSMOSIS_PLATFORM_URL", "http://example.invalid")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
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


def test_invalid_platform_url_port_is_validation_not_internal(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OSMOSIS_PLATFORM_URL", "http://host:notaport")
    monkeypatch.delenv("OSMOSIS_ALLOW_INSECURE_PLATFORM_URL", raising=False)

    rc = cli.main(["--json", "auth", "whoami"])

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    envelope = json.loads(captured.err)
    assert envelope["error"]["code"] == "VALIDATION"
    assert "http://host:notaport" in envelope["error"]["message"]


def test_schemeless_loopback_platform_url_is_allowed(
    _dotenv_platform_url: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A scheme-less loopback URL like ``localhost:3000`` normalizes to http."""
    tmp_path = _dotenv_platform_url
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.delenv("OSMOSIS_ALLOW_INSECURE_PLATFORM_URL", raising=False)
    (tmp_path / ".env").write_text(
        "OSMOSIS_PLATFORM_URL=localhost:3000\n",
        encoding="utf-8",
    )

    rc = cli.main(["--json", "auth", "login"])

    captured = capsys.readouterr()
    assert rc == 1
    error = json.loads(captured.err)["error"]
    assert error["code"] == "INTERACTIVE_REQUIRED"


def test_invalid_platform_url_emits_envelope_in_fresh_process(tmp_path: Path) -> None:
    """A bad URL must not escape as an import-time traceback while handling errors."""
    import subprocess
    import sys as _sys

    result = subprocess.run(
        [
            _sys.executable,
            "-c",
            "from osmosis_ai.cli.main import main; raise SystemExit(main(['--json', 'dataset', 'list']))",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={
            **os.environ,
            "OSMOSIS_PLATFORM_URL": "https://host:notaport",
            "OSMOSIS_TOKEN": "tok",
        },
    )

    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    error = json.loads(result.stderr.splitlines()[0])["error"]
    assert error["code"] == "VALIDATION"
    assert "Invalid platform URL" in error["message"]


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
