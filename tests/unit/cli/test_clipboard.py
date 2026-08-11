"""Tests for clipboard copying across local, remote, and multiplexed sessions."""

from __future__ import annotations

import base64
from io import StringIO
from typing import Any

import pytest

from osmosis_ai.cli.clipboard import copy_to_clipboard

TEXT = "train a model for triage"
PAYLOAD = base64.b64encode(TEXT.encode("utf-8")).decode("ascii")
ESC = "\x1b"
BEL = "\x07"

_SESSION_VARS = (
    "SSH_CONNECTION",
    "SSH_TTY",
    "SSH_CLIENT",
    "TMUX",
    "WAYLAND_DISPLAY",
    "WSL_DISTRO_NAME",
)


class _Tty(StringIO):
    def __init__(self, *, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


@pytest.fixture(autouse=True)
def session(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean local Linux session with no clipboard tool installed."""
    for name in _SESSION_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("osmosis_ai.cli.clipboard.platform.system", lambda: "Linux")
    monkeypatch.setattr("osmosis_ai.cli.clipboard.shutil.which", lambda _name: None)


def _stdout(monkeypatch: pytest.MonkeyPatch, *, tty: bool = True) -> _Tty:
    """Patch stdout inside the test body; pytest replaces it between phases."""
    stream = _Tty(tty=tty)
    monkeypatch.setattr("osmosis_ai.cli.clipboard.sys.stdout", stream)
    return stream


@pytest.fixture
def runs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record local clipboard tool invocations instead of running them."""
    calls: list[dict[str, Any]] = []

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        calls.append({"command": command, **kwargs})
        return None

    monkeypatch.setattr("osmosis_ai.cli.clipboard.subprocess.run", fake_run)
    return calls


def _installed(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    monkeypatch.setattr(
        "osmosis_ai.cli.clipboard.shutil.which",
        lambda name: f"/usr/bin/{name}" if name in names else None,
    )


# ── OSC 52 sequence ──────────────────────────────────────────────────


def test_osc52_carries_the_text_as_base64(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = _stdout(monkeypatch)

    assert copy_to_clipboard(TEXT) is True
    assert stdout.getvalue() == f"{ESC}]52;c;{PAYLOAD}{BEL}"


def test_osc52_is_wrapped_in_tmux_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = _stdout(monkeypatch)
    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,123,0")

    assert copy_to_clipboard(TEXT) is True

    written = stdout.getvalue()
    assert written.startswith(f"{ESC}Ptmux;")
    assert written.endswith(f"{ESC}\\")
    # Doubled ESCs are required for tmux DCS passthrough.
    assert f"{ESC}{ESC}]52;c;{PAYLOAD}{BEL}" in written


def test_osc52_writes_nothing_when_stdout_is_not_a_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = _stdout(monkeypatch, tty=False)

    assert copy_to_clipboard(TEXT) is False
    assert stdout.getvalue() == ""


# ── Routing between OSC 52 and the local tool ────────────────────────


def test_remote_session_skips_the_local_tool(
    runs: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Over SSH the local tool would write the remote machine's clipboard."""
    stdout = _stdout(monkeypatch)
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.1 22 10.0.0.2 51234")
    _installed(monkeypatch, "xclip")

    assert copy_to_clipboard(TEXT) is True

    assert runs == []
    assert stdout.getvalue() == f"{ESC}]52;c;{PAYLOAD}{BEL}"


def test_local_session_uses_the_local_tool(
    runs: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    stdout = _stdout(monkeypatch)
    monkeypatch.setattr("osmosis_ai.cli.clipboard.platform.system", lambda: "Darwin")
    _installed(monkeypatch, "pbcopy")

    assert copy_to_clipboard(TEXT) is True

    assert [call["command"] for call in runs] == [["pbcopy"]]
    assert runs[0]["input"] == TEXT.encode()
    assert stdout.getvalue() == ""


def test_wayland_session_uses_wl_copy(
    runs: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    _installed(monkeypatch, "wl-copy", "xclip")

    assert copy_to_clipboard(TEXT) is True

    assert [call["command"] for call in runs] == [["wl-copy"]]


def test_wsl_session_uses_windows_clip(
    runs: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    _installed(monkeypatch, "clip.exe")

    assert copy_to_clipboard(TEXT) is True

    assert [call["command"] for call in runs] == [["clip.exe"]]


def test_plain_linux_session_uses_xclip(
    runs: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _stdout(monkeypatch)
    _installed(monkeypatch, "xclip")

    assert copy_to_clipboard(TEXT) is True

    assert [call["command"] for call in runs] == [["xclip", "-selection", "clipboard"]]


def test_falls_back_to_osc52_when_no_tool_is_installed(
    runs: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    stdout = _stdout(monkeypatch)

    assert copy_to_clipboard(TEXT) is True

    assert runs == []
    assert stdout.getvalue() == f"{ESC}]52;c;{PAYLOAD}{BEL}"


def test_falls_back_to_osc52_when_the_local_tool_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """xclip is installed but has no display to talk to."""
    stdout = _stdout(monkeypatch)
    _installed(monkeypatch, "xclip")

    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("cannot open display")

    monkeypatch.setattr("osmosis_ai.cli.clipboard.subprocess.run", explode)

    assert copy_to_clipboard(TEXT) is True
    assert stdout.getvalue() == f"{ESC}]52;c;{PAYLOAD}{BEL}"


def test_reports_failure_when_nothing_can_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stdout(monkeypatch, tty=False)

    assert copy_to_clipboard(TEXT) is False
