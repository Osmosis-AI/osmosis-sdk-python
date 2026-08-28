"""In-process coverage for the supervisor-owned /health bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osmosis_ai.eval.local._server_bootstrap import (
    _INSTANCE_ENV,
    ROLLOUT_INSTANCE_HEADER,
    ROLLOUT_SDK_VERSION_HEADER,
    _install_health_marker,
    main,
)

_HEADER_NAME = ROLLOUT_INSTANCE_HEADER.encode("ascii")
_SDK_HEADER_NAME = ROLLOUT_SDK_VERSION_HEADER.encode("ascii")


@pytest.fixture
def restore_fastapi_call() -> Any:
    original = FastAPI.__call__
    yield
    FastAPI.__call__ = original


@pytest.fixture
def restore_sys_argv() -> Any:
    original = list(sys.argv)
    yield
    sys.argv = original


@pytest.fixture
def restore_sys_path() -> Any:
    original = list(sys.path)
    yield
    sys.path[:] = original


def test_health_marker_is_injected_only_on_health(restore_fastapi_call: None) -> None:
    _install_health_marker("owned", "0.3.0")
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> dict[str, bool]:
        return {"ok": True}

    client = TestClient(app)
    health_response = client.get("/health")
    ready_response = client.get("/ready")

    assert health_response.json() == {"status": "ok"}
    assert health_response.headers[ROLLOUT_INSTANCE_HEADER] == "owned"
    assert health_response.headers[ROLLOUT_SDK_VERSION_HEADER] == "0.3.0"
    assert ROLLOUT_INSTANCE_HEADER not in ready_response.headers
    assert ROLLOUT_SDK_VERSION_HEADER not in ready_response.headers


async def test_health_marker_replaces_a_stale_header(
    restore_fastapi_call: None,
) -> None:
    async def original_call(self: FastAPI, scope: Any, receive: Any, send: Any) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (_HEADER_NAME, b"stale"),
                    (_SDK_HEADER_NAME, b"stale"),
                    (b"content-type", b"application/json"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": b"{}"})

    FastAPI.__call__ = original_call  # type: ignore[method-assign]
    _install_health_marker("fresh", "0.3.1")

    messages: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    async def receive() -> dict[str, Any]:
        return {"type": "http.disconnect"}

    app = FastAPI()
    await app(
        {"type": "http", "asgi": {"version": "3.0"}, "path": "/health"}, receive, send
    )

    start = next(
        message for message in messages if message["type"] == "http.response.start"
    )
    assert [
        value for name, value in start["headers"] if name.lower() == _HEADER_NAME
    ] == [b"fresh"]
    assert [
        value for name, value in start["headers"] if name.lower() == _SDK_HEADER_NAME
    ] == [b"0.3.1"]
    assert messages[-1]["type"] == "http.response.body"


async def test_non_http_scopes_are_left_untouched(restore_fastapi_call: None) -> None:
    seen: list[str] = []

    async def original_call(self: FastAPI, scope: Any, receive: Any, send: Any) -> None:
        seen.append(str(scope.get("type")))

    FastAPI.__call__ = original_call  # type: ignore[method-assign]
    _install_health_marker("owned", "0.3.0")

    async def receive() -> dict[str, Any]:
        return {"type": "lifespan.shutdown"}

    async def send(_message: dict[str, Any]) -> None:
        return None

    app = FastAPI()
    await app({"type": "lifespan"}, receive, send)
    assert seen == ["lifespan"]


def test_main_rejects_wrong_argv(restore_sys_argv: None) -> None:
    sys.argv = ["_server_bootstrap.py"]
    with pytest.raises(SystemExit, match="usage:"):
        main()


def test_main_rejects_a_missing_entrypoint(
    tmp_path: Path, restore_sys_argv: None
) -> None:
    sys.argv = ["_server_bootstrap.py", str(tmp_path / "missing.py")]
    with pytest.raises(SystemExit, match="does not exist"):
        main()


def test_main_requires_the_instance_env(
    tmp_path: Path, restore_sys_argv: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    entrypoint = tmp_path / "main.py"
    entrypoint.write_text("pass\n", encoding="utf-8")
    sys.argv = ["_server_bootstrap.py", str(entrypoint)]
    monkeypatch.delenv(_INSTANCE_ENV, raising=False)
    with pytest.raises(SystemExit, match=f"{_INSTANCE_ENV} is required"):
        main()


def test_main_runs_the_entrypoint(
    tmp_path: Path,
    restore_sys_argv: None,
    restore_sys_path: None,
    restore_fastapi_call: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "ran.txt"
    entrypoint = tmp_path / "server.py"
    entrypoint.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ok')\n",
        encoding="utf-8",
    )
    sys.argv = ["_server_bootstrap.py", str(entrypoint)]
    monkeypatch.setenv(_INSTANCE_ENV, "owned")

    main()

    assert marker.read_text(encoding="utf-8") == "ok"
    assert sys.argv == [str(entrypoint.resolve())]
    assert str(entrypoint.parent) in sys.path
