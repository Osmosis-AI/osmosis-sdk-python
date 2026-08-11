"""Clipboard support for interactive CLI flows.

Uses a local tool when one can serve the session; otherwise OSC 52 so the
terminal sets the clipboard even over SSH and tmux.
"""

from __future__ import annotations

import base64
import os
import platform
import shutil
import subprocess
import sys

_LOCAL_TOOL_TIMEOUT = 3.0
_REMOTE_SESSION_VARS = ("SSH_CONNECTION", "SSH_TTY", "SSH_CLIENT")


def _is_remote_session() -> bool:
    return any(os.environ.get(name) for name in _REMOTE_SESSION_VARS)


def _local_tool() -> list[str] | None:
    system = platform.system()
    if system == "Darwin":
        candidates = [["pbcopy"]]
    elif system == "Windows":
        candidates = [["clip"]]
    elif system == "Linux":
        candidates = []
        if os.environ.get("WAYLAND_DISPLAY"):
            candidates.append(["wl-copy"])
        if os.environ.get("WSL_DISTRO_NAME"):
            candidates.append(["clip.exe"])
        candidates.append(["xclip", "-selection", "clipboard"])
    else:
        candidates = []

    for command in candidates:
        if shutil.which(command[0]) is not None:
            return command
    return None


def _copy_via_local_tool(command: list[str], text: str) -> bool:
    try:
        subprocess.run(
            command,
            input=text.encode(),
            check=True,
            timeout=_LOCAL_TOOL_TIMEOUT,
        )
    except Exception:
        return False
    return True


def _copy_via_osc52(text: str) -> bool:
    """Emit OSC 52. ``True`` means written, not that the clipboard changed."""
    stream = sys.stdout
    try:
        if not stream.isatty():
            return False
    except Exception:
        return False

    payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
    sequence = f"\x1b]52;c;{payload}\x07"
    if os.environ.get("TMUX"):
        # tmux only forwards inside DCS passthrough with every ESC doubled.
        escaped = sequence.replace("\x1b", "\x1b\x1b")
        sequence = f"\x1bPtmux;{escaped}\x1b\\"

    try:
        stream.write(sequence)
        stream.flush()
    except Exception:
        return False
    return True


def copy_to_clipboard(text: str) -> bool:
    """Copy *text* to the clipboard. Returns False when no route was available."""
    if not _is_remote_session():
        command = _local_tool()
        if command is not None and _copy_via_local_tool(command, text):
            return True
    return _copy_via_osc52(text)


__all__ = ["copy_to_clipboard"]
