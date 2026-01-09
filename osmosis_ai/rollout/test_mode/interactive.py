"""Interactive mode for step-by-step agent loop testing.

Pause after each LLM call to inspect messages, tools, and intermediate state.
Commands: [n]ext, [c]ontinue, [m]essages, [t]ools, [q]uit
"""

from __future__ import annotations

import asyncio
import copy
import json
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from osmosis_ai.rollout.client import CompletionsResult
from osmosis_ai.rollout.core.base import (
    RolloutAgentLoop,
    RolloutContext,
    RolloutResult,
)
from osmosis_ai.rollout.core.schemas import OpenAIFunctionToolSchema, RolloutMetrics
from osmosis_ai.rollout.test_mode.dataset import DatasetRow, dataset_row_to_request
from osmosis_ai.rollout.test_mode.exceptions import ToolValidationError
from osmosis_ai.rollout.test_mode.external_llm_client import ExternalLLMClient
from osmosis_ai.rollout.test_mode.runner import validate_tools


@dataclass
class InteractiveStep:
    """Represents a single step in the interactive session.

    Attributes:
        turn: Turn number (1-indexed).
        step_type: Type of step (llm_response, tool_result, final).
        message: The message at this step.
        tool_calls: Tool calls if any.
        finish_reason: Finish reason if this is a completion.
    """

    turn: int
    step_type: str  # "llm_response", "tool_result", "final"
    message: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = None


class InteractiveLLMClient:
    """Wrapper around ExternalLLMClient that enables step-by-step execution.

    This client wraps the ExternalLLMClient and calls a callback after each
    LLM response, allowing the user to inspect and control execution.
    """

    def __init__(
        self,
        wrapped_client: ExternalLLMClient,
        on_step: Callable[[InteractiveStep], bool],
        on_messages_updated: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    ) -> None:
        """Initialize the interactive client wrapper.

        Args:
            wrapped_client: The underlying LLM client.
            on_step: Callback called after each step. Return True to continue,
                     False to abort.
            on_messages_updated: Optional callback called with the current messages
                     list after each LLM call, allowing the runner to track
                     conversation progress.
        """
        self._wrapped = wrapped_client
        self._on_step = on_step
        self._on_messages_updated = on_messages_updated
        self._turn = 0
        self._aborted = False

    def set_tools(self, tools: List[Any]) -> None:
        """Delegate to wrapped client."""
        self._wrapped.set_tools(tools)

    def clear_tools(self) -> None:
        """Delegate to wrapped client."""
        self._wrapped.clear_tools()

    def reset_metrics(self) -> None:
        """Reset metrics and turn counter."""
        self._wrapped.reset_metrics()
        self._turn = 0
        self._aborted = False

    def get_metrics(self) -> RolloutMetrics:
        """Delegate to wrapped client."""
        return self._wrapped.get_metrics()

    async def chat_completions(
        self,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> CompletionsResult:
        """Make a chat completion and pause for user interaction.

        Args:
            messages: Conversation messages.
            **kwargs: Additional parameters.

        Returns:
            CompletionsResult from the LLM.

        Raises:
            InterruptedError: If user aborts the execution.
        """
        if self._aborted:
            raise InterruptedError("Execution aborted by user")

        self._turn += 1

        # Update messages tracker before the LLM call (so user can see what's being sent)
        if self._on_messages_updated:
            self._on_messages_updated(messages)

        # Make the actual LLM call
        result = await self._wrapped.chat_completions(messages, **kwargs)

        # Extract tool calls if present
        tool_calls = None
        if result.message.get("tool_calls"):
            tool_calls = result.message["tool_calls"]

        # Create step info
        step = InteractiveStep(
            turn=self._turn,
            step_type="llm_response",
            message=result.message,
            tool_calls=tool_calls,
            finish_reason=result.finish_reason,
        )

        # Call the callback and check if we should continue
        should_continue = self._on_step(step)
        if not should_continue:
            self._aborted = True
            raise InterruptedError("Execution aborted by user")

        return result

    async def close(self) -> None:
        """Delegate to wrapped client."""
        await self._wrapped.close()

    async def __aenter__(self) -> "InteractiveLLMClient":
        """Async context manager entry."""
        await self._wrapped.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self._wrapped.__aexit__(*args)


class InteractiveRunner:
    """Interactive test runner with step-by-step execution.

    Pauses after each LLM call for user to inspect state and control execution.
    """

    # ANSI color codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    RED = "\033[31m"

    def __init__(
        self,
        agent_loop: RolloutAgentLoop,
        llm_client: ExternalLLMClient,
        debug: bool = False,
    ) -> None:
        """Initialize the interactive runner.

        Args:
            agent_loop: Agent loop instance to test.
            llm_client: External LLM client instance.
            debug: Enable debug output.
        """
        self.agent_loop = agent_loop
        self.llm_client = llm_client
        self.debug = debug

        # State for interactive session
        self._current_messages: List[Dict[str, Any]] = []
        self._current_tools: List[OpenAIFunctionToolSchema] = []
        self._auto_continue = False

    def _print_separator(self, title: str = "") -> None:
        """Print a separator line."""
        width = 60
        if title:
            padding = (width - len(title) - 2) // 2
            line = "─" * padding + f" {title} " + "─" * padding
            # Adjust if odd length
            if len(line) < width:
                line += "─"
        else:
            line = "─" * width
        print(f"{self.DIM}{line}{self.RESET}")

    def _print_message(self, msg: Dict[str, Any], prefix: str = "") -> None:
        """Print a message in a formatted way."""
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Color based on role
        if role == "system":
            color = self.MAGENTA
            label = "System"
        elif role == "user":
            color = self.GREEN
            label = "User"
        elif role == "assistant":
            color = self.BLUE
            label = "Assistant"
        elif role == "tool":
            color = self.CYAN
            label = f"Tool ({msg.get('name', 'unknown')})"
        else:
            color = self.RESET
            label = role.capitalize()

        print(f"{prefix}{color}[{label}]{self.RESET} {content}")

        # Print tool calls if present
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            print(f"\n{prefix}{self.YELLOW}Tool calls:{self.RESET}")
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "unknown")
                args = func.get("arguments", "{}")
                # Try to format arguments nicely
                try:
                    args_dict = json.loads(args) if isinstance(args, str) else args
                    args_str = json.dumps(args_dict, indent=2)
                except (json.JSONDecodeError, TypeError):
                    args_str = str(args)

                # Indent multi-line args
                args_lines = args_str.split("\n")
                if len(args_lines) > 1:
                    args_display = args_lines[0] + "\n" + "\n".join(
                        "    " + line for line in args_lines[1:]
                    )
                else:
                    args_display = args_str

                print(f"{prefix}  • {self.CYAN}{name}{self.RESET}({args_display})")

    def _print_step(self, step: InteractiveStep) -> None:
        """Print a step in the interactive session."""
        self._print_separator(f"Turn {step.turn}: LLM Response")
        print()

        if step.message:
            self._print_message(step.message)

        if step.finish_reason and step.finish_reason != "tool_calls":
            print(f"\n{self.DIM}Finish reason: {step.finish_reason}{self.RESET}")

        print()

    def _print_initial_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Print the initial conversation messages."""
        self._print_separator("Initial Messages")
        print()
        for msg in messages:
            self._print_message(msg)
            print()

    def _print_tools(self, tools: List[OpenAIFunctionToolSchema]) -> None:
        """Print available tools."""
        self._print_separator("Available Tools")
        print()
        for tool in tools:
            func = tool.function
            print(f"  • {self.CYAN}{func.name}{self.RESET}")
            if func.description:
                print(f"    {self.DIM}{func.description}{self.RESET}")
            if func.parameters and func.parameters.properties:
                props = func.parameters.properties
                print(f"    Parameters: {', '.join(props.keys())}")
        print()

    def _print_all_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Print all messages in the conversation."""
        self._print_separator("All Messages")
        print()
        for i, msg in enumerate(messages):
            print(f"{self.DIM}[{i}]{self.RESET}", end=" ")
            self._print_message(msg)
            print()

    def _print_result(
        self,
        result: Optional[RolloutResult],
        duration_ms: float,
        metrics: RolloutMetrics,
        error: Optional[str] = None,
    ) -> None:
        """Print the final result."""
        self._print_separator("Result")
        print()

        if error:
            print(f"{self.RED}Status: ERROR{self.RESET}")
            print(f"{self.RED}Error: {error}{self.RESET}")
        elif result:
            status_color = self.GREEN if result.status == "COMPLETED" else self.YELLOW
            print(f"{status_color}Status: {result.status}{self.RESET}")

            if result.reward is not None:
                print(f"Reward: {result.reward}")

            if result.finish_reason:
                print(f"Finish reason: {result.finish_reason}")

        print(f"Duration: {duration_ms/1000:.2f}s")
        total_tokens = metrics.prompt_tokens + metrics.response_tokens
        print(f"Tokens: {total_tokens:,} ({metrics.num_llm_calls} LLM calls)")
        print()

    def _get_user_input(self) -> str:
        """Get user input with command prompt."""
        print(
            f"{self.DIM}Commands: "
            f"[n]ext, [c]ontinue, [m]essages, [t]ools, [q]uit{self.RESET}"
        )
        try:
            return input(f"{self.BOLD}>{self.RESET} ").strip().lower()
        except EOFError:
            return "q"
        except KeyboardInterrupt:
            print()
            return "q"

    def _handle_step(self, step: InteractiveStep) -> bool:
        """Handle a step in interactive mode.

        Returns True to continue execution, False to abort.
        """
        self._print_step(step)

        # If auto-continue is enabled, just proceed
        if self._auto_continue:
            return True

        while True:
            user_input = self._get_user_input()

            if user_input in ("n", "next", ""):
                # Continue to next step
                return True

            elif user_input in ("c", "continue"):
                # Auto-continue to completion
                self._auto_continue = True
                print(f"{self.DIM}Continuing to completion...{self.RESET}")
                return True

            elif user_input in ("m", "messages"):
                # Show all messages
                self._print_all_messages(self._current_messages)

            elif user_input in ("t", "tools"):
                # Show tools
                self._print_tools(self._current_tools)

            elif user_input in ("q", "quit", "exit"):
                # Abort execution
                print(f"{self.YELLOW}Aborting execution...{self.RESET}")
                return False

            else:
                print(
                    f"{self.RED}Unknown command: {user_input}{self.RESET}"
                )
                print("  n/next  - Continue to next step")
                print("  c       - Continue to completion (no more pauses)")
                print("  m       - Show all messages")
                print("  t       - Show available tools")
                print("  q/quit  - Abort execution")

    async def run_single_interactive(
        self,
        row: DatasetRow,
        row_index: int,
        max_turns: int = 10,
        completion_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[RolloutResult], Optional[str]]:
        """Run a single row interactively.

        Args:
            row: Dataset row to test.
            row_index: Index of the row.
            max_turns: Maximum agent turns.
            completion_params: LLM sampling parameters.

        Returns:
            Tuple of (RolloutResult or None, error message or None).
        """
        self._auto_continue = False
        overall_start = time.monotonic()

        # Reset client state
        self.llm_client.reset_metrics()
        self.llm_client.clear_tools()

        try:
            # Convert row to request
            request = dataset_row_to_request(
                row=row,
                row_index=row_index,
                max_turns=max_turns,
                completion_params=completion_params,
            )

            # Store initial messages
            self._current_messages = list(request.messages)

            # Print initial messages
            self._print_initial_messages(self._current_messages)

            # Get and validate tools
            tools = self.agent_loop.get_tools(request)
            validate_tools(tools)
            self._current_tools = tools

            # Show tools
            if tools:
                self._print_tools(tools)

            # Create callback to update current messages for the 'm' command
            def update_messages(messages: List[Dict[str, Any]]) -> None:
                # Create a deep copy of the messages list to track conversation progress
                self._current_messages = copy.deepcopy(messages)

            # Create interactive client wrapper
            interactive_client = InteractiveLLMClient(
                wrapped_client=self.llm_client,
                on_step=self._handle_step,
                on_messages_updated=update_messages,
            )
            interactive_client.set_tools(tools)

            # Start timing after preparation
            agent_start_time = time.monotonic()

            # Create context with interactive client
            ctx = RolloutContext(
                request=request,
                tools=tools,
                llm=interactive_client,  # type: ignore
                _start_time=agent_start_time,
                _debug_dir="./test_debug" if self.debug else None,
            )

            # Run agent loop
            result = await self.agent_loop.run(ctx)

            # Print result
            duration_ms = (time.monotonic() - overall_start) * 1000
            metrics = self.llm_client.get_metrics()
            self._print_result(result, duration_ms, metrics)

            return result, None

        except InterruptedError:
            duration_ms = (time.monotonic() - overall_start) * 1000
            metrics = self.llm_client.get_metrics()
            self._print_result(None, duration_ms, metrics, error="Aborted by user")
            return None, "Aborted by user"

        except ToolValidationError as e:
            duration_ms = (time.monotonic() - overall_start) * 1000
            metrics = self.llm_client.get_metrics()
            error_msg = f"Tool validation error: {e}"
            self._print_result(None, duration_ms, metrics, error=error_msg)
            return None, error_msg

        except Exception as e:
            duration_ms = (time.monotonic() - overall_start) * 1000
            metrics = self.llm_client.get_metrics()
            error_msg = str(e)
            self._print_result(None, duration_ms, metrics, error=error_msg)
            return None, error_msg

        finally:
            self.llm_client.clear_tools()

    async def run_interactive_session(
        self,
        rows: List[DatasetRow],
        max_turns: int = 10,
        completion_params: Optional[Dict[str, Any]] = None,
        initial_row: Optional[int] = None,
        row_offset: int = 0,
    ) -> None:
        """Run an interactive testing session.

        Args:
            rows: List of dataset rows.
            max_turns: Maximum agent turns per row.
            completion_params: LLM sampling parameters.
            initial_row: Initial row to test (absolute index in dataset file, optional).
                         With --offset 50 --limit 10, valid range is 50-59.
            row_offset: Offset for calculating absolute row indices (from --offset).
                        Used for correct row numbering in display, rollout IDs, and metadata.
        """
        total_rows = len(rows)

        if total_rows == 0:
            print(f"{self.RED}No rows available to test.{self.RESET}")
            return

        # Calculate display range (absolute row indices in the dataset file)
        start_row = row_offset
        end_row = row_offset + total_rows - 1

        # If initial row specified, run it first
        # initial_row is an absolute index in the dataset file (e.g., 55 with --offset 50)
        if initial_row is not None:
            # Convert absolute index to relative index in loaded rows
            relative_row = initial_row - row_offset
            if 0 <= relative_row < total_rows:
                self._print_separator(f"Row {initial_row}")
                print()
                await self.run_single_interactive(
                    rows[relative_row],
                    initial_row,  # Pass absolute index for display/rollout_id
                    max_turns,
                    completion_params,
                )
            else:
                print(
                    f"{self.RED}Invalid row index: {initial_row}. "
                    f"Valid range: {start_row}-{end_row}{self.RESET}"
                )

        # Interactive loop
        while True:
            print()
            print(
                f"Select a row to test [{self.CYAN}{start_row}-{end_row}{self.RESET}] "
                f"or '{self.CYAN}q{self.RESET}' to quit:"
            )

            try:
                user_input = input(f"{self.BOLD}>{self.RESET} ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if user_input in ("q", "quit", "exit"):
                break

            # Try to parse as row number (absolute index in dataset file)
            try:
                absolute_row = int(user_input)
                # Convert absolute row to relative index in loaded rows
                relative_row = absolute_row - row_offset
                if 0 <= relative_row < total_rows:
                    self._print_separator(f"Row {absolute_row}")
                    print()
                    await self.run_single_interactive(
                        rows[relative_row],
                        absolute_row,
                        max_turns,
                        completion_params,
                    )
                else:
                    print(
                        f"{self.RED}Invalid row index. "
                        f"Valid range: {start_row}-{end_row}{self.RESET}"
                    )
            except ValueError:
                print(f"{self.RED}Please enter a valid row number or 'q' to quit.{self.RESET}")

        print(f"\n{self.DIM}Interactive session ended.{self.RESET}")


__all__ = [
    "InteractiveLLMClient",
    "InteractiveRunner",
    "InteractiveStep",
]
