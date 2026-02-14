"""Tests for MCP agent loop, loader, and CLI integration."""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osmosis_ai.rollout.core.schemas import (
    OpenAIFunctionToolSchema,
)
from osmosis_ai.rollout.mcp.agent_loop import (
    MCPAgentLoop,
    _extract_tool_value,
    _mcp_tool_to_openai,
)
from osmosis_ai.rollout.mcp.loader import MCPLoadError, load_mcp_server


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_mcp_server(tools: Dict[str, Dict[str, Any]]) -> MagicMock:
    """Create a mock FastMCP server with given tools.

    Args:
        tools: mapping of tool_name -> {"description": ..., "parameters": ...}
    """
    mock_tools = {}
    for name, info in tools.items():
        tool = MagicMock()
        tool.name = name
        tool.description = info.get("description", "")
        tool.parameters = info.get("parameters", {"type": "object", "properties": {}, "required": []})
        mock_tools[name] = tool

    tool_manager = MagicMock()
    tool_manager._tools = mock_tools
    server = MagicMock()
    server._tool_manager = tool_manager
    return server


def _install_fake_fastmcp(monkeypatch: pytest.MonkeyPatch) -> type:
    """Install a minimal in-memory fastmcp module for loader tests."""

    class FastMCP:
        def __init__(self, name: str) -> None:
            self.name = name
            self._tool_manager = types.SimpleNamespace(_tools={})

        def tool(self):
            def _decorator(fn):
                tool = types.SimpleNamespace(
                    name=fn.__name__,
                    description=(fn.__doc__ or "").strip(),
                    parameters={"type": "object", "properties": {}, "required": []},
                )
                self._tool_manager._tools[tool.name] = tool
                return fn

            return _decorator

    fastmcp_module = types.ModuleType("fastmcp")
    fastmcp_module.FastMCP = FastMCP
    monkeypatch.setitem(sys.modules, "fastmcp", fastmcp_module)
    return FastMCP


# ---------------------------------------------------------------------------
# _mcp_tool_to_openai conversion tests
# ---------------------------------------------------------------------------


class TestMCPToolToOpenAI:
    def test_simple_types(self):
        schema = _mcp_tool_to_openai(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        )

        assert schema.type == "function"
        assert schema.function.name == "add"
        assert schema.function.description == "Add two numbers"
        assert set(schema.function.parameters.required) == {"a", "b"}
        # integer should be normalised to number
        assert schema.function.parameters.properties["a"].type == "number"
        assert schema.function.parameters.properties["b"].type == "number"

    def test_description_on_property(self):
        schema = _mcp_tool_to_openai(
            name="greet",
            description="Greet someone",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The person's name"},
                },
                "required": ["name"],
            },
        )

        prop = schema.function.parameters.properties["name"]
        assert prop.type == "string"
        assert prop.description == "The person's name"

    def test_enum_via_ref(self):
        schema = _mcp_tool_to_openai(
            name="paint",
            description="Paint something",
            parameters={
                "$defs": {"Color": {"enum": ["red", "blue"], "type": "string"}},
                "type": "object",
                "properties": {
                    "color": {"$ref": "#/$defs/Color"},
                },
                "required": ["color"],
            },
        )

        prop = schema.function.parameters.properties["color"]
        assert prop.enum == ["red", "blue"]
        assert prop.type == "string"

    def test_optional_via_anyof(self):
        schema = _mcp_tool_to_openai(
            name="search",
            description="Search",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "default": None,
                    },
                },
                "required": ["query"],
            },
        )

        prop = schema.function.parameters.properties["limit"]
        assert prop.type == "number"  # integer → number

    def test_empty_description_fallback(self):
        schema = _mcp_tool_to_openai(
            name="noop",
            description="",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        assert schema.function.description == "Tool: noop"

    def test_no_properties(self):
        schema = _mcp_tool_to_openai(
            name="ping",
            description="Ping",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        assert schema.function.parameters.properties == {}
        assert schema.function.parameters.required == []


# ---------------------------------------------------------------------------
# MCPAgentLoop.get_tools tests
# ---------------------------------------------------------------------------


class TestMCPAgentLoopGetTools:
    def test_returns_converted_schemas(self):
        server = _make_mock_mcp_server(
            {
                "add": {
                    "description": "Add numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                    },
                },
                "greet": {
                    "description": "Say hello",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                },
            }
        )

        agent = MCPAgentLoop(server)
        tools = agent.get_tools(None)  # type: ignore[arg-type]

        assert len(tools) == 2
        names = {t.function.name for t in tools}
        assert names == {"add", "greet"}
        for t in tools:
            assert isinstance(t, OpenAIFunctionToolSchema)

    def test_custom_agent_name(self):
        server = _make_mock_mcp_server({})
        agent = MCPAgentLoop(server, agent_name="my_custom")
        assert agent.name == "my_custom"


# ---------------------------------------------------------------------------
# MCPAgentLoop.run tests
# ---------------------------------------------------------------------------


class TestMCPAgentLoopRun:
    async def test_no_tool_calls(self):
        """LLM responds without tool calls — loop should complete immediately."""
        server = _make_mock_mcp_server({
            "add": {
                "description": "Add",
                "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
            },
        })
        agent = MCPAgentLoop(server)

        # Build context
        from osmosis_ai.rollout.core.base import RolloutContext
        from osmosis_ai.rollout.core.schemas import RolloutRequest

        request = RolloutRequest(
            rollout_id="test-1",
            server_url="http://localhost:8080",
            messages=[{"role": "user", "content": "Hi"}],
            completion_params={},
        )

        mock_llm = MagicMock()
        mock_llm.get_metrics.return_value = MagicMock(
            llm_latency_ms=0, num_llm_calls=0, prompt_tokens=0, response_tokens=0
        )

        chat_result = MagicMock()
        chat_result.message = {"role": "assistant", "content": "Hello!"}
        chat_result.has_tool_calls = False
        chat_result.tool_calls = None

        mock_llm.chat_completions = AsyncMock(return_value=chat_result)

        ctx = RolloutContext(
            request=request,
            tools=agent.get_tools(request),
            llm=mock_llm,
        )

        result = await agent.run(ctx)

        assert result.status == "COMPLETED"
        assert len(result.final_messages) == 2  # user + assistant
        assert result.final_messages[-1]["content"] == "Hello!"

    async def test_tool_call_then_response(self):
        """LLM makes a tool call, then responds with final answer."""
        # Set up MCP server with a tool
        server = _make_mock_mcp_server({
            "add": {
                "description": "Add",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            },
        })

        # Mock call_tool to return a ToolResult-like object
        tool_result = MagicMock()
        tool_result.structured_content = {"result": 8}
        tool_result.content = []
        server._tool_manager.call_tool = AsyncMock(return_value=tool_result)

        agent = MCPAgentLoop(server)

        from osmosis_ai.rollout.core.base import RolloutContext
        from osmosis_ai.rollout.core.schemas import RolloutRequest

        request = RolloutRequest(
            rollout_id="test-2",
            server_url="http://localhost:8080",
            messages=[{"role": "user", "content": "What is 5+3?"}],
            completion_params={},
        )

        mock_llm = MagicMock()
        mock_llm.get_metrics.return_value = MagicMock(
            llm_latency_ms=0, num_llm_calls=0, prompt_tokens=0, response_tokens=0
        )

        # First call: tool call; second call: final response
        tool_call_response = MagicMock()
        tool_call_response.message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 5, "b": 3}'},
                }
            ],
        }
        tool_call_response.has_tool_calls = True
        tool_call_response.tool_calls = tool_call_response.message["tool_calls"]

        final_response = MagicMock()
        final_response.message = {"role": "assistant", "content": "The answer is 8."}
        final_response.has_tool_calls = False
        final_response.tool_calls = None

        mock_llm.chat_completions = AsyncMock(
            side_effect=[tool_call_response, final_response]
        )

        ctx = RolloutContext(
            request=request,
            tools=agent.get_tools(request),
            llm=mock_llm,
        )

        result = await agent.run(ctx)

        assert result.status == "COMPLETED"
        # user + assistant(tool_call) + tool_result + assistant(final)
        assert len(result.final_messages) == 4
        assert result.final_messages[2]["role"] == "tool"
        assert result.final_messages[2]["content"] == "8"
        server._tool_manager.call_tool.assert_awaited_once_with("add", {"a": 5, "b": 3})

    async def test_tool_call_exception_handled(self):
        """Tool execution error should be captured as error message, loop continues."""
        server = _make_mock_mcp_server({
            "fail_tool": {
                "description": "Always fails",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        })
        server._tool_manager.call_tool = AsyncMock(side_effect=RuntimeError("boom"))

        agent = MCPAgentLoop(server)

        from osmosis_ai.rollout.core.base import RolloutContext
        from osmosis_ai.rollout.core.schemas import RolloutRequest

        request = RolloutRequest(
            rollout_id="test-3",
            server_url="http://localhost:8080",
            messages=[{"role": "user", "content": "Do something"}],
            completion_params={},
            max_turns=2,
        )

        mock_llm = MagicMock()
        mock_llm.get_metrics.return_value = MagicMock(
            llm_latency_ms=0, num_llm_calls=0, prompt_tokens=0, response_tokens=0
        )

        # First: tool call, second: final response
        tc_resp = MagicMock()
        tc_resp.message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_err", "type": "function", "function": {"name": "fail_tool", "arguments": "{}"}},
            ],
        }
        tc_resp.has_tool_calls = True
        tc_resp.tool_calls = tc_resp.message["tool_calls"]

        final_resp = MagicMock()
        final_resp.message = {"role": "assistant", "content": "Sorry, that failed."}
        final_resp.has_tool_calls = False
        final_resp.tool_calls = None

        mock_llm.chat_completions = AsyncMock(side_effect=[tc_resp, final_resp])

        ctx = RolloutContext(
            request=request,
            tools=agent.get_tools(request),
            llm=mock_llm,
        )

        result = await agent.run(ctx)

        assert result.status == "COMPLETED"
        # Check error message was created
        tool_msg = result.final_messages[2]
        assert tool_msg["role"] == "tool"
        assert "Error:" in tool_msg["content"]
        assert "boom" in tool_msg["content"]


# ---------------------------------------------------------------------------
# _extract_tool_value tests
# ---------------------------------------------------------------------------


class TestExtractToolValue:
    def test_structured_content_with_result_key(self):
        tr = MagicMock()
        tr.structured_content = {"result": 42}
        assert _extract_tool_value(tr) == 42

    def test_structured_content_dict(self):
        tr = MagicMock()
        tr.structured_content = {"key": "val", "count": 1}
        assert _extract_tool_value(tr) == {"key": "val", "count": 1}

    def test_fallback_to_content_text(self):
        tr = MagicMock()
        tr.structured_content = None
        item = MagicMock()
        item.text = "hello"
        tr.content = [item]
        assert _extract_tool_value(tr) == "hello"

    def test_empty_result(self):
        tr = MagicMock()
        tr.structured_content = None
        tr.content = []
        assert _extract_tool_value(tr) == ""


# ---------------------------------------------------------------------------
# load_mcp_server tests
# ---------------------------------------------------------------------------


class TestLoadMCPServer:
    @pytest.fixture(autouse=True)
    def _fake_fastmcp(self, monkeypatch: pytest.MonkeyPatch):
        _install_fake_fastmcp(monkeypatch)

    def test_directory_not_found(self):
        with pytest.raises(MCPLoadError, match="does not exist"):
            load_mcp_server("/nonexistent/path/to/mcp")

    def test_no_main_py(self, tmp_path):
        with pytest.raises(MCPLoadError, match="No main.py found"):
            load_mcp_server(str(tmp_path))

    def test_no_fastmcp_instance(self, tmp_path):
        main_py = tmp_path / "main.py"
        main_py.write_text("x = 1\n")
        with pytest.raises(MCPLoadError, match="No FastMCP instance found"):
            load_mcp_server(str(tmp_path))

    def test_successful_load(self, tmp_path):
        main_py = tmp_path / "main.py"
        main_py.write_text(textwrap.dedent("""\
            from fastmcp import FastMCP

            mcp = FastMCP("test_server")

            @mcp.tool()
            def add(a: int, b: int) -> int:
                \"\"\"Add two numbers\"\"\"
                return a + b
        """))

        server = load_mcp_server(str(tmp_path))

        from fastmcp import FastMCP
        assert isinstance(server, FastMCP)
        assert "add" in server._tool_manager._tools

    def test_import_error_in_main(self, tmp_path):
        main_py = tmp_path / "main.py"
        main_py.write_text("import nonexistent_module_xyz\n")
        with pytest.raises(MCPLoadError, match="Error importing"):
            load_mcp_server(str(tmp_path))

    def test_restores_sys_path_after_load(self, tmp_path):
        original_sys_path = list(sys.path)

        main_py = tmp_path / "main.py"
        main_py.write_text(textwrap.dedent("""\
            from fastmcp import FastMCP
            mcp = FastMCP("path_test")
        """))

        _ = load_mcp_server(str(tmp_path))

        assert sys.path == original_sys_path

    def test_supports_relative_imports_in_main(self, tmp_path):
        helpers_py = tmp_path / "helpers.py"
        helpers_py.write_text(textwrap.dedent("""\
            from fastmcp import FastMCP

            def build_server():
                mcp = FastMCP("relative_imports")

                @mcp.tool()
                def ping() -> str:
                    return "pong"

                return mcp
        """))

        main_py = tmp_path / "main.py"
        main_py.write_text(textwrap.dedent("""\
            from .helpers import build_server

            mcp = build_server()
        """))

        server = load_mcp_server(str(tmp_path))
        assert "ping" in server._tool_manager._tools


# ---------------------------------------------------------------------------
# CLI mutual exclusion tests
# ---------------------------------------------------------------------------


class TestCLIMutualExclusion:
    def test_eval_both_module_and_tools_error(self):
        from osmosis_ai.rollout.eval.evaluation.cli import EvalCommand

        cmd = EvalCommand()
        args = argparse.Namespace(
            module="foo:bar",
            mcp="./mcp",
            n_runs=1,
            batch_size=1,
            max_turns=10,
            offset=0,
            limit=None,
            baseline_model=None,
            baseline_base_url=None,
            baseline_api_key=None,
        )
        error = cmd._validate_args(args)
        assert error is not None
        assert "mutually exclusive" in error

    def test_eval_neither_module_nor_tools_error(self):
        from osmosis_ai.rollout.eval.evaluation.cli import EvalCommand

        cmd = EvalCommand()
        args = argparse.Namespace(
            module=None,
            mcp=None,
            n_runs=1,
            batch_size=1,
            max_turns=10,
            offset=0,
            limit=None,
            baseline_model=None,
            baseline_base_url=None,
            baseline_api_key=None,
        )
        error = cmd._validate_args(args)
        assert error is not None
        assert "required" in error

    def test_eval_module_only_ok(self):
        from osmosis_ai.rollout.eval.evaluation.cli import EvalCommand

        cmd = EvalCommand()
        args = argparse.Namespace(
            module="foo:bar",
            mcp=None,
            n_runs=1,
            batch_size=1,
            max_turns=10,
            offset=0,
            limit=None,
            baseline_model=None,
            baseline_base_url=None,
            baseline_api_key=None,
        )
        assert cmd._validate_args(args) is None

    def test_eval_tools_only_ok(self):
        from osmosis_ai.rollout.eval.evaluation.cli import EvalCommand

        cmd = EvalCommand()
        args = argparse.Namespace(
            module=None,
            mcp="./mcp",
            n_runs=1,
            batch_size=1,
            max_turns=10,
            offset=0,
            limit=None,
            baseline_model=None,
            baseline_base_url=None,
            baseline_api_key=None,
        )
        assert cmd._validate_args(args) is None

    def test_test_both_module_and_tools_error(self):
        from osmosis_ai.rollout.eval.test_mode.cli import TestCommand

        cmd = TestCommand()
        args = argparse.Namespace(
            module="foo:bar",
            mcp="./mcp",
            row=None,
            interactive=False,
        )
        error = cmd._validate_args(args)
        assert error is not None
        assert "mutually exclusive" in error

    def test_test_neither_module_nor_tools_error(self):
        from osmosis_ai.rollout.eval.test_mode.cli import TestCommand

        cmd = TestCommand()
        args = argparse.Namespace(
            module=None,
            mcp=None,
            row=None,
            interactive=False,
        )
        error = cmd._validate_args(args)
        assert error is not None
        assert "required" in error

    def test_test_tools_only_ok(self):
        from osmosis_ai.rollout.eval.test_mode.cli import TestCommand

        cmd = TestCommand()
        args = argparse.Namespace(
            module=None,
            mcp="./mcp",
            row=None,
            interactive=False,
        )
        assert cmd._validate_args(args) is None


class TestEvalCLIBatchSize:
    async def _run_eval_cli_and_capture_run_eval_kwargs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        batch_size: int,
    ) -> Dict[str, Any]:
        from osmosis_ai.rollout.eval.evaluation.cli import EvalCommand

        captured_kwargs: Dict[str, Any] = {}

        class DummyLLMClient:
            async def __aenter__(self) -> "DummyLLMClient":
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

            async def close(self) -> None:
                return None

        class FakeEvalRunner:
            def __init__(self, *args, **kwargs) -> None:
                pass

            async def run_eval(self, **kwargs):
                captured_kwargs.update(kwargs)
                return object()

        async def _verify_llm_client(*args, **kwargs) -> Optional[str]:
            return None

        monkeypatch.setattr(
            "osmosis_ai.rollout.eval.evaluation.cli.create_llm_client",
            lambda **kwargs: (DummyLLMClient(), None),
        )
        monkeypatch.setattr(
            "osmosis_ai.rollout.eval.evaluation.cli.verify_llm_client",
            _verify_llm_client,
        )
        monkeypatch.setattr(
            "osmosis_ai.rollout.eval.evaluation.cli.load_agent",
            lambda **kwargs: (MagicMock(), None),
        )
        monkeypatch.setattr(
            "osmosis_ai.rollout.eval.evaluation.cli.load_dataset_rows",
            lambda **kwargs: (
                [{"user_prompt": "hi", "system_prompt": None, "ground_truth": "ok"}],
                None,
            ),
        )
        monkeypatch.setattr(
            EvalCommand,
            "_load_eval_fns",
            lambda self, args: ([MagicMock(name="simple_eval")], None),
        )
        monkeypatch.setattr(
            "osmosis_ai.rollout.eval.evaluation.runner.EvalRunner",
            FakeEvalRunner,
        )

        cmd = EvalCommand()
        args = argparse.Namespace(
            module="my_agent:MyAgentLoop",
            mcp=None,
            dataset="data.jsonl",
            model="openai/gpt-4o",
            eval_fns=["rewards:score"],
            n_runs=1,
            pass_threshold=1.0,
            max_turns=10,
            temperature=None,
            max_tokens=None,
            api_key=None,
            base_url=None,
            baseline_model=None,
            baseline_base_url=None,
            baseline_api_key=None,
            output=None,
            debug=False,
            quiet=True,
            limit=None,
            offset=0,
            batch_size=batch_size,
        )

        exit_code = await cmd._run_async(args)
        assert exit_code == 0
        return captured_kwargs

    async def test_batch_size_gt_one_forwards_batch_size_only(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        kwargs = await self._run_eval_cli_and_capture_run_eval_kwargs(
            monkeypatch,
            batch_size=4,
        )
        assert kwargs["batch_size"] == 4
        assert "fail_fast" not in kwargs

    async def test_batch_size_one_forwards_batch_size_only(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        kwargs = await self._run_eval_cli_and_capture_run_eval_kwargs(
            monkeypatch,
            batch_size=1,
        )
        assert kwargs["batch_size"] == 1
        assert "fail_fast" not in kwargs
