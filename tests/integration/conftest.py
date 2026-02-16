"""Fixtures specific to integration tests.

Integration tests exercise real component interactions with minimal mocking.
They may use httpx.AsyncClient against real FastAPI apps, real file I/O, etc.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def anyio_backend():
    """Use asyncio as the async backend for integration tests."""
    return "asyncio"
