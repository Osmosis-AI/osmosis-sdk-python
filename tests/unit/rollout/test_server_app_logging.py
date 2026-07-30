"""Tests that create_rollout_server only configures logging it is asked to own."""

from __future__ import annotations

import logging

from osmosis_ai.rollout.backend.base import ExecutionBackend, ResultCallback
from osmosis_ai.rollout.server.app import (
    _get_rollout_server_backend,
    create_rollout_server,
)
from osmosis_ai.rollout.types import ExecutionRequest


class _Backend(ExecutionBackend):
    async def execute(
        self,
        request: ExecutionRequest,
        on_workflow_complete: ResultCallback,
        on_grader_complete: ResultCallback | None = None,
    ) -> None:  # pragma: no cover - never invoked
        raise AssertionError("execute should not be called")


def _clear_root_handlers(monkeypatch) -> logging.Logger:
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [], raising=False)
    return root


class TestConfigureLogging:
    def test_installs_an_info_handler_when_the_process_has_none(self, monkeypatch):
        root = _clear_root_handlers(monkeypatch)
        monkeypatch.setattr(root, "level", logging.WARNING, raising=False)

        create_rollout_server(backend=_Backend())

        assert root.handlers
        assert root.level == logging.INFO

    def test_leaves_an_already_configured_process_alone(self, monkeypatch):
        root = _clear_root_handlers(monkeypatch)
        existing = logging.NullHandler()
        root.handlers.append(existing)
        monkeypatch.setattr(root, "level", logging.WARNING, raising=False)

        create_rollout_server(backend=_Backend())

        assert root.handlers == [existing]
        assert root.level == logging.WARNING

    def test_opting_out_installs_nothing(self, monkeypatch):
        root = _clear_root_handlers(monkeypatch)
        monkeypatch.setattr(root, "level", logging.WARNING, raising=False)

        create_rollout_server(backend=_Backend(), configure_logging=False)

        assert root.handlers == []
        assert root.level == logging.WARNING


def test_create_rollout_server_records_backend_marker() -> None:
    backend = _Backend()

    app = create_rollout_server(backend=backend, configure_logging=False)

    assert _get_rollout_server_backend(app) is backend
