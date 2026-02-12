"""Eval function loading and normalization for eval mode.

Supports two eval function signatures, auto-detected via parameter inspection:

Simple (compatible with @osmosis_reward):
    def my_eval(solution_str: str, ground_truth: str, extra_info: dict = None, **kwargs) -> float

Full context:
    def my_eval(messages: list, ground_truth: str, metadata: dict, **kwargs) -> float
    # or async variant
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import sys
from typing import Any, Callable, Dict, List


class EvalFnError(Exception):
    """Error raised when eval function loading or execution fails."""

    pass


class EvalFnWrapper:
    """Normalizes both simple and full eval function signatures into a unified async interface.

    Detects mode via first parameter name:
    - "solution_str" -> simple mode (extracts last assistant content)
    - "messages" -> full mode (passes full conversation)
    """

    def __init__(self, fn: Callable, name: str) -> None:
        self.fn = fn
        self.name = name
        self._is_async = asyncio.iscoroutinefunction(fn)
        self._mode = self._detect_mode(fn)

    @staticmethod
    def _detect_mode(fn: Callable) -> str:
        """Detect eval function mode from first parameter name."""
        try:
            sig = inspect.signature(fn)
            params = list(sig.parameters.keys())
            if not params:
                raise EvalFnError(
                    f"Eval function has no parameters. "
                    f"Expected (solution_str, ...) or (messages, ...)"
                )
            first_param = params[0]
            if first_param == "solution_str":
                return "simple"
            elif first_param == "messages":
                return "full"
            else:
                raise EvalFnError(
                    f"Cannot detect eval function mode: first parameter is '{first_param}'. "
                    f"Expected 'solution_str' (simple mode) or 'messages' (full mode)."
                )
        except (ValueError, TypeError) as e:
            raise EvalFnError(f"Cannot inspect eval function signature: {e}")

    @staticmethod
    def _extract_last_assistant_content(messages: List[Dict[str, Any]]) -> str:
        """Extract content from the last assistant message."""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if content is not None:
                    return str(content)
        return ""

    async def __call__(
        self,
        messages: List[Dict[str, Any]],
        ground_truth: str,
        metadata: Dict[str, Any],
    ) -> float:
        """Call the eval function with normalized arguments.

        Args:
            messages: Full conversation messages.
            ground_truth: Expected ground truth.
            metadata: Row metadata dict.

        Returns:
            Float score from the eval function.
        """
        if self._mode == "simple":
            solution_str = self._extract_last_assistant_content(messages)
            kwargs = {
                "solution_str": solution_str,
                "ground_truth": ground_truth,
                "extra_info": metadata,
            }
        else:
            kwargs = {
                "messages": messages,
                "ground_truth": ground_truth,
                "metadata": metadata,
            }

        if self._is_async:
            result = await self.fn(**kwargs)
        else:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: self.fn(**kwargs))

        if not isinstance(result, (int, float)):
            raise EvalFnError(
                f"Eval function '{self.name}' returned {type(result).__name__}, expected float"
            )

        return float(result)


def load_eval_fn(module_path: str) -> Callable:
    """Load an eval function from a module:attr path.

    Args:
        module_path: Path in format "module.path:function_name"

    Returns:
        The callable eval function.

    Raises:
        EvalFnError: If loading fails.
    """
    if ":" not in module_path:
        raise EvalFnError(
            f"Invalid eval function path '{module_path}'. "
            f"Expected format: 'module:function' (e.g., 'rewards:compute_reward')"
        )

    module_name, attr_name = module_path.rsplit(":", 1)

    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise EvalFnError(f"Cannot import module '{module_name}': {e}")

    try:
        fn = getattr(module, attr_name)
    except AttributeError:
        raise EvalFnError(
            f"Module '{module_name}' has no attribute '{attr_name}'. "
            f"Available: {[a for a in dir(module) if not a.startswith('_')]}"
        )

    if not callable(fn):
        raise EvalFnError(f"'{attr_name}' in '{module_name}' is not callable")

    return fn


def load_eval_fns(module_paths: List[str]) -> List[EvalFnWrapper]:
    """Load multiple eval functions and wrap them.

    Args:
        module_paths: List of "module:function" paths.

    Returns:
        List of EvalFnWrapper instances.

    Raises:
        EvalFnError: If any function fails to load or has invalid signature.
    """
    wrappers = []
    seen_names = set()
    for path in module_paths:
        fn = load_eval_fn(path)
        # Use fully qualified module:function as the display key to avoid collisions.
        name = path
        if name in seen_names:
            raise EvalFnError(f"Duplicate eval function path: '{name}'")
        seen_names.add(name)
        wrapper = EvalFnWrapper(fn, name)
        wrappers.append(wrapper)
    return wrappers

__all__ = [
    "EvalFnError",
    "EvalFnWrapper",
    "load_eval_fn",
    "load_eval_fns",
]
