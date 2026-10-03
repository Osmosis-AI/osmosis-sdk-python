"""Contract tests for the opt-in source-only stdio prototype."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp.server.lowlevel", reason="optional MCP prototype dependencies")
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from osmosis_ai.platform.auth import credentials as credential_store
from osmosis_ai.platform.auth.credentials import Credentials, UserInfo
from prototypes.stdio_mcp.bridge import (
    BridgeError,
    LocalInput,
    PageInput,
    Settings,
    list_datasets,
    local_context,
    read_platform,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
BRIDGE = REPO_ROOT / "prototypes/stdio_mcp/bridge.py"


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    (root / ".git").mkdir(parents=True)
    (root / "configs/training").mkdir(parents=True)
    (root / "configs/training/tiny.toml").write_text("do not disclose file contents")
    return root


@pytest.fixture
def platform(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, dict(self.headers)))
            if self.path.startswith("/api/cli/training-runs"):
                entered.set()
                release.wait(timeout=5)
            self.send_response(response["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Location", "/redirect-target")
            self.end_headers()
            self.wfile.write(json.dumps(response["body"]).encode())

        def log_message(self, *_args):
            pass

    requests: list[tuple[str, dict[str, str]]] = []
    response: dict[str, Any] = {"status": 200, "body": {}}
    entered = threading.Event()
    release = threading.Event()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    url = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OSMOSIS_PLATFORM_URL", url)
    monkeypatch.delenv("OSMOSIS_TOKEN", raising=False)
    monkeypatch.delenv("OSMOSIS_TOKEN_PLATFORM_URL", raising=False)
    path = tmp_path / ".config/osmosis/credentials.json"
    path.parent.mkdir(parents=True)
    creds = Credentials(
        access_token="synthetic-test-token",
        token_type="Bearer",
        expires_at=datetime.now(UTC) + timedelta(days=1),
        created_at=datetime.now(UTC),
        user=UserInfo(id="user", email="test@example.invalid"),
    )
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "platforms": {
                    url: {**creds.to_dict(), "platform_url": url, "token_store": "file"}
                },
            }
        )
    )
    path.chmod(0o600)
    monkeypatch.setattr(credential_store, "CREDENTIALS_FILE", path)
    try:
        yield url, response, requests, path, entered, release
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_explicit_repository_containment(repository: Path, tmp_path: Path):
    settings = Settings.create(repository, "workspace-a")
    assert local_context(settings, LocalInput()).config_paths == [
        "configs/training/tiny.toml"
    ]
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.toml").write_text("private")
    (repository / "configs/training/leak.toml").symlink_to(outside / "private.toml")
    assert local_context(settings, LocalInput()).config_paths == [
        "configs/training/tiny.toml"
    ]
    (repository / "configs/eval").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        local_context(settings, LocalInput(kind="eval"))
    with pytest.raises(ValueError, match="absolute"):
        Settings.create(Path("."), "workspace-a")
    with pytest.raises(ValueError, match="repository root"):
        Settings.create(outside, "workspace-a")
    with pytest.raises(ValueError, match="workspace"):
        Settings.create(repository, "workspace-a\r\nAuthorization: injected")


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (401, "AUTH_REQUIRED"),
        (403, "FORBIDDEN"),
        (426, "UPGRADE_REQUIRED"),
        (302, "PLATFORM_UNAVAILABLE"),
    ],
)
def test_failed_reads_preserve_auth_and_never_follow_redirects(
    platform, repository: Path, status: int, error: str
):
    _url, response, requests, path, _entered, _release = platform
    response.update(status=status, body={"message": "synthetic-secret-in-error"})
    before = path.read_bytes()
    settings = Settings.create(repository, "workspace-a")
    with pytest.raises(BridgeError, match=error):
        read_platform(settings, "datasets", PageInput())
    assert path.read_bytes() == before
    assert len(requests) == 1
    assert requests[0][1]["X-Osmosis-Workspace"] == "workspace-a"
    assert "X-Osmosis-Git" not in requests[0][1]
    assert requests[0][1]["Authorization"] == "Bearer synthetic-test-token"


def test_response_allowlist_and_explicit_workspace_header(platform, repository: Path):
    _url, response, requests, _path, _entered, _release = platform
    response["body"] = {
        "datasets": [
            {
                "id": "dataset-a",
                "file_name": "data.jsonl",
                "status": "ready",
                "data_preview": {"secret": "hidden"},
                "creator_email": "private@example.invalid",
                "platform_url": "https://evil.invalid/private",
            }
        ],
        "total_count": 1,
        "has_more": False,
    }
    for workspace in ("workspace-a", "workspace-b"):
        page = list_datasets(Settings.create(repository, workspace), PageInput(limit=1))
        assert page.model_dump() == {
            "items": [
                {
                    "id": "dataset-a",
                    "file_name": "data.jsonl",
                    "status": "ready",
                    "row_count": None,
                }
            ],
            "total_count": 1,
            "has_more": False,
            "next_offset": None,
        }
    assert [headers["X-Osmosis-Workspace"] for _, headers in requests] == [
        "workspace-a",
        "workspace-b",
    ]
    response["body"] = {"padding": "x" * 1_048_576}
    with pytest.raises(BridgeError, match="INVALID_RESPONSE"):
        read_platform(
            Settings.create(repository, "workspace-a"), "datasets", PageInput()
        )


def test_successful_continuation_preserves_limit_offset_and_workspace(
    platform, repository: Path
):
    _url, response, requests, _path, _entered, _release = platform
    response["body"] = {
        "datasets": [
            {"id": "dataset-two", "file_name": "second.jsonl", "status": "ready"}
        ],
        "total_count": 3,
        "has_more": True,
        "next_offset": 2,
    }
    page = list_datasets(
        Settings.create(repository, "workspace-a"), PageInput(limit=1, offset=1)
    )
    assert requests[0][0] == "/api/cli/datasets?limit=1&offset=1"
    assert requests[0][1]["X-Osmosis-Workspace"] == "workspace-a"
    assert page.items[0].id == "dataset-two"
    assert page.next_offset == 2


def test_platform_binding_and_https_fail_closed(
    monkeypatch: pytest.MonkeyPatch, platform, repository: Path
):
    url, _response, requests, _path, _entered, _release = platform
    settings = Settings.create(repository, "workspace-a")
    monkeypatch.setenv("OSMOSIS_TOKEN", "synthetic-env-token")
    monkeypatch.setenv("OSMOSIS_TOKEN_PLATFORM_URL", "https://wrong.invalid")
    with pytest.raises(BridgeError, match="AUTH_REQUIRED"):
        read_platform(settings, "datasets", PageInput())
    assert requests == []
    monkeypatch.setenv("OSMOSIS_PLATFORM_URL", "http://remote.invalid")
    with pytest.raises(ValueError, match="HTTPS"):
        Settings.create(repository, "workspace-a")
    monkeypatch.setenv("OSMOSIS_PLATFORM_URL", url + "/changed")
    with pytest.raises(BridgeError, match="AUTH_REQUIRED"):
        read_platform(settings, "datasets", PageInput())
    assert requests == []


@pytest.mark.asyncio
async def test_official_stdio_client_is_noninteractive_and_nonblocking(
    platform, repository: Path, tmp_path: Path
):
    url, response, requests, path, entered, release = platform
    response["body"] = {
        "training_runs": [
            {
                "id": "run-a",
                "name": "demo",
                "status": "finished",
                "created_at": "2026-10-03T00:00:00Z",
                "env_config": {"secret": "hidden"},
            }
        ],
        "total_count": 1,
        "has_more": False,
    }
    # Fail if the entry point loads UI dependencies, even when they are installed.
    (tmp_path / "sitecustomize.py").write_text(
        "import builtins\n"
        "_original_import = builtins.__import__\n"
        "def _guard(name, *args, **kwargs):\n"
        "    if name.split('.')[0] in {'rich', 'questionary', 'prompt_toolkit'}:\n"
        "        raise ImportError('UI imports disabled by protocol test')\n"
        "    return _original_import(name, *args, **kwargs)\n"
        "builtins.__import__ = _guard\n"
        "import os\n"
        "from pathlib import Path\n"
        "from osmosis_ai.platform.auth import credentials\n"
        "credentials.CREDENTIALS_FILE = Path(os.environ['MCP_TEST_CREDENTIALS_FILE'])\n"
    )
    env = {
        "MCP_TEST_CREDENTIALS_FILE": str(path),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.pathsep.join((str(tmp_path), str(REPO_ROOT))),
        "OSMOSIS_PLATFORM_URL": url,
        "NO_COLOR": "1",
    }
    before = path.read_bytes()
    with (tmp_path / "stderr.log").open("w+") as stderr:
        async with stdio_client(
            StdioServerParameters(
                command=sys.executable,
                args=[
                    str(BRIDGE),
                    "--workspace-directory",
                    str(repository),
                    "--workspace",
                    "workspace-a",
                ],
                cwd=str(tmp_path),
                env=env,
            ),
            errlog=stderr,
        ) as (read, write):
            async with ClientSession(read, write) as client:
                initialized = await client.initialize()
                assert initialized.serverInfo.name == "osmosis-local-prototype"
                tools = (await client.list_tools()).tools
                assert {tool.name for tool in tools} == {
                    "list_local_configs",
                    "list_datasets",
                    "list_training_runs",
                }
                assert all(
                    tool.outputSchema and tool.annotations.readOnlyHint
                    for tool in tools
                )
                local = await client.call_tool("list_local_configs", {})
                assert local.structuredContent["data"]["config_paths"] == [
                    "configs/training/tiny.toml"
                ]
                for arguments in (
                    {"kind": "../../private"},
                    {"token": "synthetic-input-secret"},
                ):
                    invalid = await client.call_tool("list_local_configs", arguments)
                    assert invalid.isError
                    assert invalid.structuredContent == {
                        "data": None,
                        "error": "INVALID_ARGUMENTS",
                    }
                unknown = await client.call_tool("synthetic-unknown-tool-canary", {})
                assert unknown.isError
                assert unknown.structuredContent == {
                    "data": None,
                    "error": "INVALID_ARGUMENTS",
                }
                pending = asyncio.create_task(
                    client.call_tool("list_training_runs", {"limit": 1})
                )
                assert await asyncio.to_thread(entered.wait, 3)
                await asyncio.wait_for(client.send_ping(), timeout=1)
                release.set()
                result = await pending
                assert result.structuredContent["data"]["items"] == [
                    {
                        "id": "run-a",
                        "name": "demo",
                        "status": "finished",
                        "created_at": "2026-10-03T00:00:00Z",
                    }
                ]
                response.update(status=401, body={"error": "synthetic-upstream-secret"})
                denied = await client.call_tool("list_datasets", {})
                assert denied.isError
                assert denied.structuredContent == {
                    "data": None,
                    "error": "AUTH_REQUIRED",
                }
        stderr.seek(0)
        log = stderr.read()
        assert "synthetic-" not in log
        assert "Traceback" not in log
        assert "Local MCP protocol event; request data withheld." in log
    assert path.read_bytes() == before
    assert len(requests) == 2


def test_keyring_credentials_are_read_only_on_401(
    platform, repository: Path, monkeypatch: pytest.MonkeyPatch
):
    url, response, _requests, path, _entered, _release = platform
    metadata = json.loads(path.read_text())
    entry = metadata["platforms"][url]
    entry.pop("access_token")
    entry.update(token_store="keyring", keyring_account="test-account")
    path.write_text(json.dumps(metadata))
    monkeypatch.setattr(credential_store.keyring, "get_keyring", lambda: object())
    monkeypatch.setattr(
        credential_store.keyring, "get_password", lambda *_: "synthetic-test-token"
    )

    def reject_mutation(*_args):
        pytest.fail("Read-only MCP requests must not modify keyring credentials")

    monkeypatch.setattr(credential_store.keyring, "set_password", reject_mutation)
    monkeypatch.setattr(credential_store.keyring, "delete_password", reject_mutation)
    response["status"] = 401
    before = path.read_bytes()
    with pytest.raises(BridgeError, match="AUTH_REQUIRED"):
        read_platform(
            Settings.create(repository, "workspace-a"), "datasets", PageInput()
        )
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"datasets": "unexpected"},
        {"datasets": [{"id": "a"}, {"id": "b"}], "total_count": 2, "has_more": False},
    ],
)
def test_invalid_or_unbounded_pages_fail_closed(platform, repository: Path, body):
    _url, response, _requests, _path, _entered, _release = platform
    response["body"] = body
    with pytest.raises(BridgeError, match="INVALID_RESPONSE"):
        read_platform(
            Settings.create(repository, "workspace-a"), "datasets", PageInput(limit=1)
        )


@pytest.mark.asyncio
async def test_malformed_wire_envelopes_never_reach_stderr(repository: Path):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(BRIDGE),
        "--workspace-directory",
        str(repository),
        "--workspace",
        "workspace-a",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(REPO_ROOT)},
        cwd=repository.parent,
    )
    assert process.stdin is not None and process.stdout is not None
    try:
        initialize = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "wire-test", "version": "1"},
            },
        }
        process.stdin.write((json.dumps(initialize) + "\n").encode())
        await process.stdin.drain()
        initialized = json.loads(await asyncio.wait_for(process.stdout.readline(), 5))
        assert initialized["id"] == 1 and "result" in initialized
        messages = [
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": {"credential": "synthetic-request-envelope-canary"}},
            },
            {
                "jsonrpc": "2.0",
                "method": "notifications/progress",
                "params": {
                    "progressToken": "test",
                    "progress": "synthetic-notification-envelope-canary",
                },
            },
        ]
        process.stdin.write(
            "".join(json.dumps(message) + "\n" for message in messages).encode()
        )
        await process.stdin.drain()
        denied = json.loads(await asyncio.wait_for(process.stdout.readline(), 5))
        assert denied["id"] == 2 and denied["error"]["code"] == -32602
        stdout, stderr = await asyncio.wait_for(process.communicate(), 5)
        assert b"synthetic-" not in stdout + stderr + json.dumps(denied).encode()
        assert b"Local MCP protocol event; request data withheld." in stderr
        assert process.returncode == 0
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
