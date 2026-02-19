"""MCPAgentLoop — built-in RolloutAgentLoop for git-sync MCP users.

Wraps a FastMCP server instance so that ``osmosis eval`` and ``osmosis test``
can run MCP tools without the user having to write any AgentLoop code.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from osmosis_ai.rollout.core.base import RolloutAgentLoop, RolloutContext, RolloutResult
from osmosis_ai.rollout.core.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    RolloutRequest,
)
from osmosis_ai.rollout.tools import (
    create_tool_error_result,
    create_tool_result,
    get_tool_call_info,
    serialize_tool_result,
)

logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MCP Tool → OpenAI Schema conversion
# ---------------------------------------------------------------------------


def _resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any]:
    """Resolve a ``$ref`` pointer like ``#/$defs/Color`` against *defs*."""
    parts = ref.lstrip("#/").split("/")
    node: Any = {"$defs": defs}
    for part in parts:
        node = node[part]
    result: dict[str, Any] = node
    return result


def _json_schema_to_property(
    schema: dict[str, Any],
    defs: dict[str, Any],
) -> OpenAIFunctionPropertySchema:
    """Convert a single JSON Schema property to an OpenAIFunctionPropertySchema.

    Handles ``$ref``, ``anyOf`` (for Optional types), ``enum``, and plain types.
    Complex nested types (arrays, objects) are simplified to their top-level type.
    """
    # Resolve $ref
    if "$ref" in schema:
        schema = _resolve_ref(schema["$ref"], defs)

    # Handle anyOf (e.g. Optional[X] → [X, null])
    if "anyOf" in schema:
        non_null = [s for s in schema["anyOf"] if s.get("type") != "null"]
        if non_null:
            return _json_schema_to_property(non_null[0], defs)
        return OpenAIFunctionPropertySchema(type="string")

    prop_type_raw = schema.get("type", "string")
    if isinstance(prop_type_raw, list):
        # JSON Schema allows union types like ["integer", "null"].
        non_null_types = [
            t for t in prop_type_raw if isinstance(t, str) and t != "null"
        ]
        prop_type = non_null_types[0] if non_null_types else "string"
    elif isinstance(prop_type_raw, str):
        prop_type = prop_type_raw
    else:
        prop_type = "string"

    description = schema.get("description")
    enum = schema.get("enum")

    # Normalise JSON Schema integer → number for OpenAI
    if prop_type == "integer":
        prop_type = "number"

    return OpenAIFunctionPropertySchema(
        type=prop_type,
        description=description,
        enum=enum,
    )


def _mcp_tool_to_openai(
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> OpenAIFunctionToolSchema:
    """Convert an MCP FunctionTool's metadata to an OpenAI tool schema."""
    defs = parameters.get("$defs", {})
    raw_props = parameters.get("properties", {})
    required = parameters.get("required", [])

    properties: dict[str, OpenAIFunctionPropertySchema] = {}
    for prop_name, prop_schema in raw_props.items():
        properties[prop_name] = _json_schema_to_property(prop_schema, defs)

    return OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name=name,
            description=description or f"Tool: {name}",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties=properties,
                required=required,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# MCPAgentLoop
# ---------------------------------------------------------------------------


class MCPAgentLoop(RolloutAgentLoop):
    """Agent loop that executes MCP tools from a FastMCP server instance.

    This is the built-in loop used by ``osmosis eval --mcp <dir>`` and
    ``osmosis test --mcp <dir>`` so that git-sync users don't need to
    write their own ``RolloutAgentLoop``.
    """

    name: str = "mcp_agent"

    def __init__(self, mcp_server: Any, agent_name: str = "mcp_agent") -> None:
        self.name = agent_name
        self._tool_manager = mcp_server._tool_manager
        self._openai_schemas: list[OpenAIFunctionToolSchema] = []

        # Build cached OpenAI schemas from registered MCP tools
        for _tool_name, tool in self._tool_manager._tools.items():
            self._openai_schemas.append(
                _mcp_tool_to_openai(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=tool.parameters,
                )
            )

    def get_tools(self, request: RolloutRequest) -> list[OpenAIFunctionToolSchema]:
        return list(self._openai_schemas)

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        messages = list(ctx.request.messages)

        for _turn in range(ctx.request.max_turns):
            result = await ctx.chat(messages, **ctx.request.completion_params)
            messages.append(result.message)

            if not result.has_tool_calls:
                break

            tool_calls = result.tool_calls or []
            for tool_call in tool_calls:
                start = time.monotonic()
                try:
                    call_id, fn_name, args = get_tool_call_info(tool_call)
                    tool_result = await self._tool_manager.call_tool(fn_name, args)

                    # Extract value: prefer structured_content, fallback to content text
                    value = _extract_tool_value(tool_result)
                    messages.append(
                        create_tool_result(call_id, serialize_tool_result(value))
                    )
                except Exception as e:
                    call_id = tool_call.get("id", "unknown")
                    logger.warning("Tool call %s failed: %s", call_id, e)
                    messages.append(create_tool_error_result(call_id, str(e)))
                finally:
                    latency_ms = (time.monotonic() - start) * 1000
                    ctx.record_tool_call(latency_ms=latency_ms)
        else:
            # Loop exhausted without breaking — LLM was still requesting
            # tool calls when max_turns was reached.
            return ctx.complete(messages, finish_reason="max_turns")

        return ctx.complete(messages)


def _extract_tool_value(tool_result: Any) -> Any:
    """Extract a Python value from a FastMCP ``ToolResult``."""
    # structured_content is a dict when available (e.g. {'result': 8})
    sc = getattr(tool_result, "structured_content", None)
    if sc is not None:
        # If there's a single 'result' key, unwrap it
        if isinstance(sc, dict) and list(sc.keys()) == ["result"]:
            return sc["result"]
        return sc

    # Fallback: concatenate text from content items
    content = getattr(tool_result, "content", None)
    if content:
        texts = [item.text for item in content if hasattr(item, "text") and item.text]
        return "\n".join(texts) if texts else ""

    return ""


__all__ = ["MCPAgentLoop", "_mcp_tool_to_openai"]
