from __future__ import annotations

import json
import subprocess

import pytest

from osmosis_ai.cli import main as cli


def test_upgrade_json_no_update(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr("osmosis_ai.cli.upgrade._fetch_latest_version", lambda: "0.0.0")

    exit_code = cli.main(["--json", "upgrade"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["schema_version"] == 1
    assert payload["operation"] == "upgrade"
    assert payload["status"] == "no_update"
    assert payload["resource"]["latest_version"] == "0.0.0"


def test_upgrade_json_captures_subprocess_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        "osmosis_ai.cli.upgrade._fetch_latest_version", lambda: "99.0.0"
    )
    monkeypatch.setattr("osmosis_ai.cli.upgrade._detect_install_method", lambda: "pipx")
    monkeypatch.setattr("shutil.which", lambda command: f"/usr/bin/{command}")

    def fake_run(cmd, **kwargs):
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="installed ok\n",
            stderr="warning only\n",
        )

    monkeypatch.setattr("subprocess.run", fake_run)

    exit_code = cli.main(["--json", "upgrade"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["status"] == "success"
    assert payload["resource"]["method"] == "pipx"
    assert payload["resource"]["command"] == ["pipx", "upgrade", "osmosis-ai"]
    assert payload["resource"]["stdout"] == "installed ok\n"
    assert payload["resource"]["stderr"] == "warning only\n"


def test_upgrade_json_failed_subprocess_is_parseable_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        "osmosis_ai.cli.upgrade._fetch_latest_version", lambda: "99.0.0"
    )
    monkeypatch.setattr("osmosis_ai.cli.upgrade._detect_install_method", lambda: "pipx")
    monkeypatch.setattr("shutil.which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd,
            1,
            stdout="",
            stderr="failed\n",
        ),
    )

    exit_code = cli.main(["--json", "upgrade"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["status"] == "failed"
    assert payload["resource"]["stderr"] == "failed\n"


def test_upgrade_json_tries_next_fallback_after_failed_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        "osmosis_ai.cli.upgrade._fetch_latest_version", lambda: "99.0.0"
    )
    monkeypatch.setattr("osmosis_ai.cli.upgrade._detect_install_method", lambda: "pip")
    monkeypatch.setattr("shutil.which", lambda command: f"/usr/bin/{command}")

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd,
            1 if len(calls) == 1 else 0,
            stdout="",
            stderr="first failed\n" if len(calls) == 1 else "second ok\n",
        )

    monkeypatch.setattr("subprocess.run", fake_run)

    exit_code = cli.main(["--json", "upgrade"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["status"] == "success"
    assert len(calls) == 2
    assert payload["resource"]["command"] == calls[1]
