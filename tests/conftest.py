"""Project-wide pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_output_context_var() -> None:
    """Keep CLI output context state isolated across tests."""
    from osmosis_ai.cli.output.context import _output_context_var

    token = _output_context_var.set(None)
    try:
        yield
    finally:
        _output_context_var.reset(token)
