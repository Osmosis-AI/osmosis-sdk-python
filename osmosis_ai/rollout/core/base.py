"""Base classes for Osmosis remote rollout agent loop implementations.

This module provides the abstract base class for implementing agent loops,
along with context and result classes for managing rollout execution.

Key Classes:
    - RolloutAgentLoop: Abstract base class for agent implementations
    - RolloutContext: Execution context with LLM client and utilities
    - RolloutResult: Result object returned from agent execution

Example:
    from osmosis_ai.rollout.core import (
        RolloutAgentLoop,
        RolloutContext,
        RolloutResult,
    )

    class MyAgent(RolloutAgentLoop):
        name = "my_agent"

        def get_tools(self, request):
            return [my_tool_schema]

        async def run(self, ctx: RolloutContext) -> RolloutResult:
            messages = list(ctx.request.messages)
            result = await ctx.chat(messages)
            messages.append(result.message)
            return ctx.complete(messages)
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from osmosis_ai.rollout.core.schemas import (
    OpenAIFunctionToolSchema,
    RolloutMetrics,
    RolloutRequest,
)

if TYPE_CHECKING:
    from osmosis_ai.rollout.client import CompletionsResult
    from osmosis_ai.rollout.core.llm_client import LLMClientProtocol

logger = logging.getLogger(__name__)


@dataclass
class RolloutResult:
    """Result returned from agent loop execution.

    This is the return type for RolloutAgentLoop.run(). It contains
    the final state of the conversation and execution status.

    Attributes:
        status: "COMPLETED" for success, "ERROR" for failure.
        final_messages: Final conversation messages (append-only from request.messages).
        finish_reason: Why the rollout ended ("stop", "max_turns", "error", etc.).
        error_message: Error description if status="ERROR".
        metrics: Optional execution metrics for monitoring.
        reward: Optional precomputed trajectory reward score.

    Example:
        # Success
        result = RolloutResult(
            status="COMPLETED",
            final_messages=[...],
            finish_reason="stop",
        )

        # Success with reward
        result = RolloutResult(
            status="COMPLETED",
            final_messages=[...],
            finish_reason="stop",
            reward=1.0,
        )

        # Error
        result = RolloutResult(
            status="ERROR",
            final_messages=[],
            finish_reason="error",
            error_message="Tool execution failed",
        )
    """

    status: str
    final_messages: List[Dict[str, Any]]
    finish_reason: str
    error_message: Optional[str] = None
    metrics: Optional[RolloutMetrics] = None
    reward: Optional[float] = None


@dataclass
class RolloutContext:
    """Context provided to agent loop during execution.

    Provides access to the rollout request, available tools, and LLM client.
    Also includes convenience methods for common operations like creating
    results and recording metrics.

    Attributes:
        request: The original RolloutRequest with messages and parameters.
        tools: List of tools available for this rollout.
        llm: LLM client implementing LLMClientProtocol (OsmosisLLMClient or ExternalLLMClient).

    Example:
        async def run(self, ctx: RolloutContext) -> RolloutResult:
            messages = list(ctx.request.messages)

            for _ in range(ctx.request.max_turns):
                result = await ctx.chat(messages)
                messages.append(result.message)

                if not result.has_tool_calls:
                    break

                # Execute tools and add results
                for tool_call in result.tool_calls:
                    tool_result = await self.execute_tool(tool_call)
                    messages.append(tool_result)
                    ctx.record_tool_call(latency_ms=...)

            return ctx.complete(messages)
    """

    request: RolloutRequest
    tools: List[OpenAIFunctionToolSchema]
    llm: "LLMClientProtocol"

    # Internal state for tracking metrics
    _start_time: float = field(default=0.0, repr=False)
    _tool_latency_ms: float = field(default=0.0, repr=False)
    _num_tool_calls: int = field(default=0, repr=False)

    # Debug logging configuration
    _debug_dir: Optional[str] = field(default=None, repr=False)

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> "CompletionsResult":
        """Make a chat completion call to the LLM.

        Shorthand for self.llm.chat_completions(). Passes through
        any additional keyword arguments.

        Args:
            messages: Full conversation message list.
            **kwargs: Additional arguments (temperature, max_tokens, etc.).

        Returns:
            CompletionsResult with the LLM response.

        Example:
            result = await ctx.chat(messages, temperature=0.7)
            if result.has_tool_calls:
                # Handle tool calls
                pass
        """
        return await self.llm.chat_completions(messages, **kwargs)

    def complete(
        self,
        final_messages: List[Dict[str, Any]],
        finish_reason: str = "stop",
        reward: Optional[float] = None,
    ) -> RolloutResult:
        """Create a successful completion result.

        Use this when the agent has finished successfully.

        Args:
            final_messages: Final conversation messages.
            finish_reason: Why the rollout ended (default: "stop").
            reward: Optional precomputed trajectory reward score.

        Returns:
            RolloutResult with status="COMPLETED".

        Example:
            return ctx.complete(messages, finish_reason="stop")

            # With reward
            return ctx.complete(messages, finish_reason="stop", reward=1.0)
        """
        return RolloutResult(
            status="COMPLETED",
            final_messages=final_messages,
            finish_reason=finish_reason,
            metrics=self._build_metrics(),
            reward=reward,
        )

    def error(
        self,
        error_message: str,
        final_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> RolloutResult:
        """Create an error result.

        Use this when the agent encounters an unrecoverable error.

        Args:
            error_message: Description of what went wrong.
            final_messages: Optional partial conversation messages.

        Returns:
            RolloutResult with status="ERROR".

        Example:
            return ctx.error("Tool execution failed: division by zero")
        """
        return RolloutResult(
            status="ERROR",
            final_messages=final_messages or [],
            finish_reason="error",
            error_message=error_message,
            metrics=self._build_metrics(),
        )

    def record_tool_call(self, latency_ms: float = 0.0) -> None:
        """Record a tool call for metrics tracking.

        Call this after each tool execution to track tool usage.

        Args:
            latency_ms: Time taken for the tool call in milliseconds.

        Example:
            start = time.monotonic()
            result = await execute_tool(tool_call)
            latency = (time.monotonic() - start) * 1000
            ctx.record_tool_call(latency_ms=latency)
        """
        self._num_tool_calls += 1
        self._tool_latency_ms += latency_ms

    @property
    def debug_enabled(self) -> bool:
        """Check if debug logging is enabled."""
        return self._debug_dir is not None

    def _get_debug_file_path(self) -> Optional[str]:
        """Get the debug file path for this rollout."""
        if not self._debug_dir:
            return None
        return os.path.join(self._debug_dir, f"{self.request.rollout_id}.jsonl")

    def log_event(self, event_type: str, **data: Any) -> None:
        """Log a debug event to the rollout's JSONL file.

        Events are written to {debug_dir}/{rollout_id}.jsonl in JSONL format.
        Each line is a JSON object with the event data.

        This method is a no-op if debug logging is not enabled.

        Args:
            event_type: Type of event (e.g., "pre_llm", "llm_response", "tool_results").
            **data: Additional event data to include.

        Example:
            # Log before LLM call
            ctx.log_event(
                "pre_llm",
                turn=0,
                num_messages=len(messages),
                messages_summary=[{"role": m["role"]} for m in messages],
            )

            # Log LLM response
            ctx.log_event(
                "llm_response",
                turn=0,
                has_tool_calls=result.has_tool_calls,
                finish_reason=result.finish_reason,
            )

            # Log tool execution results
            ctx.log_event(
                "tool_results",
                turn=0,
                num_tool_results=len(tool_results),
            )

            # Log rollout completion
            ctx.log_event(
                "rollout_complete",
                finish_reason="stop",
                reward=1.0,
                total_turns=5,
            )
        """
        if not self._debug_dir:
            return

        debug_file = self._get_debug_file_path()
        if not debug_file:
            return

        # Ensure debug directory exists
        try:
            os.makedirs(self._debug_dir, exist_ok=True)
        except OSError as e:
            logger.warning("Failed to create debug directory %s: %s", self._debug_dir, e)
            return

        # Build event entry
        entry = {
            "event": event_type,
            "rollout_id": self.request.rollout_id,
            **data,
        }

        # Write to JSONL file
        try:
            with open(debug_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError as e:
            logger.warning("Failed to write debug event to %s: %s", debug_file, e)

    def _build_metrics(self) -> RolloutMetrics:
        """Build metrics combining context tracking and LLM client metrics."""
        import time

        llm_metrics = self.llm.get_metrics()
        total_latency = (
            (time.monotonic() - self._start_time) * 1000 if self._start_time > 0 else 0
        )

        return RolloutMetrics(
            total_latency_ms=total_latency,
            llm_latency_ms=llm_metrics.llm_latency_ms,
            tool_latency_ms=self._tool_latency_ms,
            num_llm_calls=llm_metrics.num_llm_calls,
            num_tool_calls=self._num_tool_calls,
            prompt_tokens=llm_metrics.prompt_tokens,
            response_tokens=llm_metrics.response_tokens,
        )


class RolloutAgentLoop(ABC):
    """Abstract base class for remote rollout agent loop implementations.

    Subclass this to implement your agent logic. You must:
    1. Set the `name` class attribute to a unique identifier
    2. Implement `get_tools()` to return available tools
    3. Implement `run()` with your agent logic

    The SDK handles:
    - HTTP protocol with TrainGate
    - LLM client setup and management
    - Metrics collection
    - Error handling and retries

    Attributes:
        name: Unique identifier for this agent loop (class attribute).
              Must be set by the subclass.

    Example:
        class CalculatorAgent(RolloutAgentLoop):
            name = "calculator"

            def get_tools(self, request: RolloutRequest) -> list:
                return [
                    OpenAIFunctionToolSchema(
                        type="function",
                        function=OpenAIFunctionSchema(
                            name="add",
                            description="Add two numbers",
                            parameters=OpenAIFunctionParametersSchema(
                                type="object",
                                properties={
                                    "a": OpenAIFunctionPropertySchema(type="number"),
                                    "b": OpenAIFunctionPropertySchema(type="number"),
                                },
                                required=["a", "b"],
                            ),
                        ),
                    )
                ]

            async def run(self, ctx: RolloutContext) -> RolloutResult:
                messages = list(ctx.request.messages)

                for _ in range(ctx.request.max_turns):
                    result = await ctx.chat(messages, **ctx.request.completion_params)
                    messages.append(result.message)

                    if not result.has_tool_calls:
                        break

                    for tc in result.tool_calls:
                        args = json.loads(tc["function"]["arguments"])
                        answer = args["a"] + args["b"]
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": str(answer),
                        })

                return ctx.complete(messages)

        # Create server
        app = create_app(CalculatorAgent())
    """

    name: str  # Must be set by subclass

    @abstractmethod
    def get_tools(self, request: RolloutRequest) -> List[OpenAIFunctionToolSchema]:
        """Return the tools available for this rollout.

        Called when a new rollout request arrives. Can return different
        tools based on request.messages or request.metadata.

        Args:
            request: The rollout request with initial messages and metadata.

        Returns:
            List of OpenAIFunctionToolSchema objects. Return empty list
            if no tools are needed.

        Example:
            def get_tools(self, request):
                # Dynamic tool selection based on metadata
                if request.metadata.get("enable_search"):
                    return [self.calculator_tool, self.search_tool]
                return [self.calculator_tool]
        """
        raise NotImplementedError

    @abstractmethod
    async def run(self, ctx: RolloutContext) -> RolloutResult:
        """Execute the agent loop.

        This is the main entry point for your agent logic. The typical
        pattern is:

        1. Start with ctx.request.messages
        2. Loop until done or max_turns reached:
           a. Call ctx.chat() for LLM completion
           b. Append assistant message to messages
           c. If tool_calls, execute tools and append results
        3. Return ctx.complete() or ctx.error()

        Important: Messages must be append-only. Never modify or remove
        previous messages.

        Args:
            ctx: Execution context with LLM client, tools, and utilities.

        Returns:
            RolloutResult with final conversation and status.

        Raises:
            Exception: Any unhandled exception is caught by the server
                      and converted to an ERROR result.
        """
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate that concrete subclasses define the name attribute."""
        super().__init_subclass__(**kwargs)
        # Skip validation for abstract subclasses
        if getattr(cls, "__abstractmethods__", None):
            return
        # Check that name is defined and non-empty
        name = getattr(cls, "name", None)
        if not name:
            raise TypeError(
                f"Agent loop class {cls.__name__} must define a 'name' class attribute"
            )
