"""Project-wide pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _neutralize_color_forcing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep Rich's terminal detection independent of the developer's shell.

    Rich forces terminal mode when FORCE_COLOR is merely present, whatever its
    value, so a shell exporting FORCE_COLOR=0 makes Console emit ANSI styles
    into assertions that expect plain text.
    """
    for key in ("FORCE_COLOR", "CLICOLOR_FORCE"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _reset_output_context_var() -> None:
    """Keep CLI output context state isolated across tests."""
    from osmosis_ai.cli.output.context import _output_context_var

    token = _output_context_var.set(None)
    try:
        yield
    finally:
        _output_context_var.reset(token)
