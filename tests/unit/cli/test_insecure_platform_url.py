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
def _platform_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Isolate explicit platform and dotenv configuration."""
    for name in (
        "OSMOSIS_PLATFORM_URL",
        "OSMOSIS_TOKEN",
        "OSMOSIS_TOKEN_PLATFORM_URL",
        "OSMOSIS_ENV_FILE",
        "OSMOSIS_ALLOW_INSECURE_PLATFORM_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    for name in (
        "OSMOSIS_PLATFORM_URL",
        "OSMOSIS_TOKEN",
        "OSMOSIS_TOKEN_PLATFORM_URL",
    ):
        os.environ.pop(name, None)


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


def test_parent_dotenv_platform_is_loaded_implicitly(
    _platform_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tmp_path = _platform_env
    (tmp_path / ".env").write_text(
        "OSMOSIS_PLATFORM_URL=https://platform-staging.osmosis.ai\n",
        encoding="utf-8",
    )
    nested = tmp_path / "rollouts" / "demo"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    rc = cli.main(["--json", "auth", "login"])

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    error = json.loads(captured.err)["error"]
    assert error["code"] == "INTERACTIVE_REQUIRED"
    assert get_platform_url() == "https://platform-staging.osmosis.ai"
    assert os.environ.get("OSMOSIS_TOKEN") is None


def test_implicit_dotenv_http_platform_url_fails_closed(
    _platform_env: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (_platform_env / ".env").write_text(
        "OSMOSIS_PLATFORM_URL=http://example.invalid\n",
        encoding="utf-8",
    )

    rc = cli.main(["--json", "auth", "login"])

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    stderr_envelopes = [json.loads(line) for line in captured.err.splitlines()]
    assert stderr_envelopes[0]["warning"]["code"] == "INSECURE_PLATFORM_URL"
    assert stderr_envelopes[1]["error"]["code"] == "VALIDATION"


def test_implicit_env_file_cannot_mix_with_an_unrelated_ambient_token(
    _platform_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OSMOSIS_TOKEN", "ambient-production-token")
    (_platform_env / ".env").write_text(
        "OSMOSIS_PLATFORM_URL=https://platform-staging.osmosis.ai\n",
        encoding="utf-8",
    )

    rc = cli.main(["--json", "auth", "whoami"])

    error = json.loads(capsys.readouterr().err)["error"]
    assert rc == 1
    assert error["code"] == "CONFLICT"
    assert "different auth profile" in error["message"]
    assert os.environ.get("OSMOSIS_PLATFORM_URL") is None


@pytest.mark.parametrize(
    ("ambient_name", "ambient_value"),
    [
        ("OSMOSIS_PLATFORM_URL", "https://platform.osmosis.ai"),
        ("OSMOSIS_TOKEN_PLATFORM_URL", "https://platform.osmosis.ai"),
    ],
)
def test_dotenv_token_cannot_merge_with_an_inherited_auth_profile_value(
    _platform_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    ambient_name: str,
    ambient_value: str,
) -> None:
    monkeypatch.setenv(ambient_name, ambient_value)
    (_platform_env / ".env").write_text(
        "OSMOSIS_TOKEN=dotenv-token\n",
        encoding="utf-8",
    )

    rc = cli.main(["--json", "auth", "whoami"])

    error = json.loads(capsys.readouterr().err)["error"]
    assert rc == 1
    assert error["code"] == "CONFLICT"
    assert "complete auth profile" in error["message"]
    assert os.environ.get("OSMOSIS_TOKEN") is None


def test_uv_preloaded_url_only_env_cannot_mix_with_an_ambient_token(
    _platform_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    platform_url = "https://platform-staging.osmosis.ai"
    monkeypatch.setenv("OSMOSIS_PLATFORM_URL", platform_url)
    monkeypatch.setenv("OSMOSIS_TOKEN", "ambient-production-token")
    (_platform_env / ".env").write_text(
        f"OSMOSIS_PLATFORM_URL={platform_url}\n",
        encoding="utf-8",
    )

    rc = cli.main(["--json", "auth", "whoami"])

    error = json.loads(capsys.readouterr().err)["error"]
    assert rc == 1
    assert error["code"] == "CONFLICT"
    assert "complete auth profile" in error["message"]
    assert os.environ["OSMOSIS_PLATFORM_URL"] == platform_url


def test_implicit_env_file_accepts_profile_already_loaded_by_uv(
    _platform_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile = {
        "OSMOSIS_PLATFORM_URL": "https://platform-staging.osmosis.ai",
        "OSMOSIS_TOKEN": "staging-token",
        "OSMOSIS_TOKEN_PLATFORM_URL": "https://platform-staging.osmosis.ai",
    }
    for name, value in profile.items():
        monkeypatch.setenv(name, value)
    (_platform_env / ".env").write_text(
        "".join(f"{name}={value}\n" for name, value in profile.items()),
        encoding="utf-8",
    )

    rc = cli.main(["--json", "auth", "login"])

    error = json.loads(capsys.readouterr().err)["error"]
    assert rc == 1
    assert error["code"] == "CONFLICT"
    assert "already active for this process" in error["message"]


def test_implicit_env_file_accepts_equivalent_uv_loaded_profile_urls(
    _platform_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(
        "OSMOSIS_PLATFORM_URL",
        "https://platform-staging.osmosis.ai/",
    )
    monkeypatch.setenv("OSMOSIS_TOKEN", "staging-token")
    monkeypatch.setenv(
        "OSMOSIS_TOKEN_PLATFORM_URL",
        "https://platform-staging.osmosis.ai/",
    )
    (_platform_env / ".env").write_text(
        "OSMOSIS_PLATFORM_URL=https://platform-staging.osmosis.ai\n"
        "OSMOSIS_TOKEN=staging-token\n"
        "OSMOSIS_TOKEN_PLATFORM_URL=https://platform-staging.osmosis.ai\n",
        encoding="utf-8",
    )

    rc = cli.main(["--json", "auth", "login"])

    error = json.loads(capsys.readouterr().err)["error"]
    assert rc == 1
    assert error["code"] == "CONFLICT"
    assert "already active for this process" in error["message"]


def test_explicit_env_file_http_platform_url_fails_closed(
    _platform_env: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env_file = _platform_env / "dev.env"
    env_file.write_text(
        "OSMOSIS_PLATFORM_URL=http://example.invalid\nOSMOSIS_TOKEN=secret-token\n",
        encoding="utf-8",
    )

    rc = cli.main(["--json", "--env-file", str(env_file), "auth", "whoami"])

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    stderr_envelopes = [json.loads(line) for line in captured.err.splitlines()]
    assert stderr_envelopes[0]["warning"]["code"] == "INSECURE_PLATFORM_URL"
    error = stderr_envelopes[1]["error"]
    assert error["code"] == "VALIDATION"
    assert "http://example.invalid" in error["message"]


def test_env_file_auth_profile_cannot_merge_with_ambient_token(
    _platform_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OSMOSIS_TOKEN", "ambient-production-token")
    env_file = _platform_env / "staging.env"
    env_file.write_text(
        "OSMOSIS_PLATFORM_URL=https://platform-staging.osmosis.ai\n"
        "OSMOSIS_TOKEN=staging-token\n"
        "OSMOSIS_TOKEN_PLATFORM_URL=https://platform-staging.osmosis.ai\n",
        encoding="utf-8",
    )

    rc = cli.main(["--json", "--env-file", str(env_file), "auth", "whoami"])

    error = json.loads(capsys.readouterr().err)["error"]
    assert rc == 1
    assert error["code"] == "CONFLICT"
    assert os.environ["OSMOSIS_TOKEN"] == "ambient-production-token"
    assert os.environ.get("OSMOSIS_PLATFORM_URL") is None
    assert os.environ.get("OSMOSIS_TOKEN_PLATFORM_URL") is None


def test_insecure_platform_url_allowed_with_opt_in(
    _platform_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tmp_path = _platform_env
    monkeypatch.setenv("OSMOSIS_ALLOW_INSECURE_PLATFORM_URL", "1")
    env_file = tmp_path / "dev.env"
    env_file.write_text(
        "OSMOSIS_PLATFORM_URL=http://example.invalid\n",
        encoding="utf-8",
    )

    rc = cli.main(["--json", "--env-file", str(env_file), "auth", "login"])

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
    _platform_env: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A scheme-less loopback URL like ``localhost:3000`` normalizes to http."""
    tmp_path = _platform_env
    env_file = tmp_path / "dev.env"
    env_file.write_text(
        "OSMOSIS_PLATFORM_URL=localhost:3000\n",
        encoding="utf-8",
    )

    rc = cli.main(["--json", "--env-file", str(env_file), "auth", "login"])

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
    _platform_env: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tmp_path = _platform_env
    env_file = tmp_path / "dev.env"
    env_file.write_text(
        "OSMOSIS_PLATFORM_URL=http://127.0.0.1:8000\n",
        encoding="utf-8",
    )

    rc = cli.main(["--json", "--env-file", str(env_file), "auth", "login"])

    captured = capsys.readouterr()
    assert rc == 1
    error = json.loads(captured.err)["error"]
    assert error["code"] == "INTERACTIVE_REQUIRED"


def test_platform_option_overrides_env_file(
    _platform_env: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env_file = _platform_env / "staging.env"
    env_file.write_text(
        "OSMOSIS_PLATFORM_URL=https://platform-staging.osmosis.ai\n",
        encoding="utf-8",
    )

    rc = cli.main(
        [
            "--json",
            "--env-file",
            str(env_file),
            "--platform",
            "localhost:3000",
            "auth",
            "login",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert json.loads(captured.err)["error"]["code"] == "INTERACTIVE_REQUIRED"
    assert get_platform_url() == "http://localhost:3000"


def test_platform_option_overrides_url_only_env_with_ambient_token(
    _platform_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OSMOSIS_TOKEN", "ambient-production-token")
    env_file = _platform_env / "staging.env"
    env_file.write_text(
        "OSMOSIS_PLATFORM_URL=https://platform-staging.osmosis.ai\n",
        encoding="utf-8",
    )

    rc = cli.main(
        [
            "--json",
            "--env-file",
            str(env_file),
            "--platform",
            "https://platform.osmosis.ai",
            "auth",
            "login",
        ]
    )

    error = json.loads(capsys.readouterr().err)["error"]
    assert rc == 1
    assert error["code"] == "CONFLICT"
    assert "already active for this process" in error["message"]
    assert get_platform_url() == "https://platform.osmosis.ai"
