# Copyright 2025 Osmosis AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for osmosis_ai.rollout.cli_utils."""

from __future__ import annotations

import sys
import textwrap

import pytest

from osmosis_ai.rollout.cli_utils import CLIError, load_agent_loop
from osmosis_ai.rollout.core.base import RolloutAgentLoop

# =============================================================================
# CLIError Tests
# =============================================================================


def test_cli_error_is_exception() -> None:
    """Verify CLIError is a proper Exception subclass."""
    error = CLIError("something went wrong")
    assert isinstance(error, Exception)
    assert str(error) == "something went wrong"


def test_cli_error_can_be_raised_and_caught() -> None:
    """Verify CLIError can be raised and caught specifically."""
    with pytest.raises(CLIError, match="test message"):
        raise CLIError("test message")


# =============================================================================
# load_agent_loop - Input Validation
# =============================================================================


def test_load_agent_loop_rejects_missing_colon() -> None:
    """Verify load_agent_loop raises CLIError when ':' is missing from the module path."""
    with pytest.raises(CLIError, match="Invalid module path") as exc_info:
        load_agent_loop("my_module_without_colon")
    assert "Expected format: 'module.path:attribute_name'" in str(exc_info.value)


def test_load_agent_loop_rejects_plain_string() -> None:
    """Verify load_agent_loop rejects a bare module name with no attribute."""
    with pytest.raises(CLIError, match="Invalid module path 'agent'"):
        load_agent_loop("agent")


# =============================================================================
# load_agent_loop - Module Import Errors
# =============================================================================


def test_load_agent_loop_raises_cli_error_for_nonexistent_module() -> None:
    """Verify load_agent_loop wraps ImportError in CLIError for missing modules."""
    with pytest.raises(CLIError, match="Cannot import module") as exc_info:
        load_agent_loop("this_module_definitely_does_not_exist_xyz:some_attr")
    assert "this_module_definitely_does_not_exist_xyz" in str(exc_info.value)


def test_load_agent_loop_import_error_preserves_cause() -> None:
    """Verify the original ImportError is chained as the cause."""
    with pytest.raises(CLIError) as exc_info:
        load_agent_loop("nonexistent_module_abc:attr")
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, ImportError)


# =============================================================================
# load_agent_loop - Attribute Resolution
# =============================================================================


def test_load_agent_loop_raises_cli_error_for_missing_attribute() -> None:
    """Verify load_agent_loop raises CLIError when the attribute doesn't exist on the module."""
    # 'os' is always importable but doesn't have 'nonexistent_attr_xyz'
    with pytest.raises(
        CLIError, match="has no attribute 'nonexistent_attr_xyz'"
    ) as exc_info:
        load_agent_loop("os:nonexistent_attr_xyz")
    assert "Available attributes:" in str(exc_info.value)


def test_load_agent_loop_missing_attribute_lists_available_attrs() -> None:
    """Verify the error message includes available attributes when attribute is missing."""
    with pytest.raises(CLIError) as exc_info:
        load_agent_loop("os:nonexistent_thing")
    # os module has 'path' as a public attribute
    assert "path" in str(exc_info.value)


# =============================================================================
# load_agent_loop - Class vs Instance Handling
# =============================================================================


def test_load_agent_loop_accepts_valid_instance(tmp_path) -> None:
    """Verify load_agent_loop returns a pre-instantiated RolloutAgentLoop instance."""
    module_code = textwrap.dedent("""\
        from osmosis_ai.rollout.core.base import RolloutAgentLoop, RolloutContext, RolloutResult
        from osmosis_ai.rollout.core.schemas import RolloutRequest, OpenAIFunctionToolSchema

        class MyAgent(RolloutAgentLoop):
            name = "test_agent"

            def get_tools(self, request):
                return []

            async def run(self, ctx):
                return ctx.complete([])

        agent_instance = MyAgent()
    """)
    module_file = tmp_path / "test_agent_instance.py"
    module_file.write_text(module_code)

    # Temporarily add tmp_path to sys.path
    original_path = sys.path.copy()
    sys.path.insert(0, str(tmp_path))
    try:
        # Remove cached module if present
        sys.modules.pop("test_agent_instance", None)
        result = load_agent_loop("test_agent_instance:agent_instance")
        assert isinstance(result, RolloutAgentLoop)
        assert result.name == "test_agent"
    finally:
        sys.path[:] = original_path
        sys.modules.pop("test_agent_instance", None)


def test_load_agent_loop_instantiates_valid_class(tmp_path) -> None:
    """Verify load_agent_loop instantiates a RolloutAgentLoop subclass when given a class."""
    module_code = textwrap.dedent("""\
        from osmosis_ai.rollout.core.base import RolloutAgentLoop, RolloutContext, RolloutResult
        from osmosis_ai.rollout.core.schemas import RolloutRequest, OpenAIFunctionToolSchema

        class MyAgent(RolloutAgentLoop):
            name = "class_agent"

            def get_tools(self, request):
                return []

            async def run(self, ctx):
                return ctx.complete([])
    """)
    module_file = tmp_path / "test_agent_class.py"
    module_file.write_text(module_code)

    original_path = sys.path.copy()
    sys.path.insert(0, str(tmp_path))
    try:
        sys.modules.pop("test_agent_class", None)
        result = load_agent_loop("test_agent_class:MyAgent")
        assert isinstance(result, RolloutAgentLoop)
        assert result.name == "class_agent"
    finally:
        sys.path[:] = original_path
        sys.modules.pop("test_agent_class", None)


def test_load_agent_loop_rejects_non_rollout_class(tmp_path) -> None:
    """Verify load_agent_loop raises CLIError for a class that is not a RolloutAgentLoop subclass."""
    module_code = textwrap.dedent("""\
        class NotAnAgent:
            name = "not_agent"
    """)
    module_file = tmp_path / "test_non_agent_class.py"
    module_file.write_text(module_code)

    original_path = sys.path.copy()
    sys.path.insert(0, str(tmp_path))
    try:
        sys.modules.pop("test_non_agent_class", None)
        with pytest.raises(CLIError, match="not a RolloutAgentLoop subclass"):
            load_agent_loop("test_non_agent_class:NotAnAgent")
    finally:
        sys.path[:] = original_path
        sys.modules.pop("test_non_agent_class", None)


def test_load_agent_loop_rejects_non_rollout_instance(tmp_path) -> None:
    """Verify load_agent_loop raises CLIError for an object that is not a RolloutAgentLoop instance."""
    module_code = textwrap.dedent("""\
        class NotAnAgent:
            pass

        not_agent_instance = NotAnAgent()
    """)
    module_file = tmp_path / "test_non_agent_inst.py"
    module_file.write_text(module_code)

    original_path = sys.path.copy()
    sys.path.insert(0, str(tmp_path))
    try:
        sys.modules.pop("test_non_agent_inst", None)
        with pytest.raises(
            CLIError, match="must be a RolloutAgentLoop instance or subclass"
        ):
            load_agent_loop("test_non_agent_inst:not_agent_instance")
    finally:
        sys.path[:] = original_path
        sys.modules.pop("test_non_agent_inst", None)


def test_load_agent_loop_rejects_plain_value(tmp_path) -> None:
    """Verify load_agent_loop raises CLIError when attribute is a plain value (e.g. string, int)."""
    module_code = textwrap.dedent("""\
        some_string = "not an agent"
    """)
    module_file = tmp_path / "test_plain_value.py"
    module_file.write_text(module_code)

    original_path = sys.path.copy()
    sys.path.insert(0, str(tmp_path))
    try:
        sys.modules.pop("test_plain_value", None)
        with pytest.raises(
            CLIError, match="must be a RolloutAgentLoop instance or subclass"
        ):
            load_agent_loop("test_plain_value:some_string")
    finally:
        sys.path[:] = original_path
        sys.modules.pop("test_plain_value", None)


def test_load_agent_loop_class_instantiation_failure(tmp_path) -> None:
    """Verify load_agent_loop raises CLIError when class constructor raises an exception."""
    module_code = textwrap.dedent("""\
        from osmosis_ai.rollout.core.base import RolloutAgentLoop, RolloutContext, RolloutResult
        from osmosis_ai.rollout.core.schemas import RolloutRequest, OpenAIFunctionToolSchema

        class BrokenAgent(RolloutAgentLoop):
            name = "broken_agent"

            def __init__(self):
                raise RuntimeError("constructor exploded")

            def get_tools(self, request):
                return []

            async def run(self, ctx):
                return ctx.complete([])
    """)
    module_file = tmp_path / "test_broken_agent.py"
    module_file.write_text(module_code)

    original_path = sys.path.copy()
    sys.path.insert(0, str(tmp_path))
    try:
        sys.modules.pop("test_broken_agent", None)
        with pytest.raises(
            CLIError, match="Cannot instantiate 'BrokenAgent'"
        ) as exc_info:
            load_agent_loop("test_broken_agent:BrokenAgent")
        assert "constructor exploded" in str(exc_info.value)
    finally:
        sys.path[:] = original_path
        sys.modules.pop("test_broken_agent", None)


# =============================================================================
# load_agent_loop - sys.path Modification
# =============================================================================


def test_load_agent_loop_adds_cwd_to_sys_path_if_missing() -> None:
    """Verify load_agent_loop adds cwd to sys.path when it's not already present."""
    import os

    cwd = os.getcwd()

    # Temporarily remove cwd from sys.path
    original_path = sys.path.copy()
    sys.path = [p for p in sys.path if p != cwd]

    try:
        assert cwd not in sys.path
        # This will fail to import but that's fine - we just want to check sys.path
        with pytest.raises(CLIError):
            load_agent_loop("nonexistent_xyz_module:attr")
        assert cwd in sys.path
    finally:
        sys.path[:] = original_path


def test_load_agent_loop_does_not_duplicate_cwd_in_sys_path() -> None:
    """Verify load_agent_loop does not add cwd to sys.path if it's already there."""
    import os

    cwd = os.getcwd()

    # Ensure cwd IS in sys.path
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    count_before = sys.path.count(cwd)

    try:
        with pytest.raises(CLIError):
            load_agent_loop("nonexistent_xyz_module:attr")
        count_after = sys.path.count(cwd)
        assert count_after == count_before
    finally:
        pass


# =============================================================================
# load_agent_loop - Module Path Parsing
# =============================================================================


def test_load_agent_loop_rsplit_handles_multiple_colons() -> None:
    """Verify load_agent_loop uses rsplit so only the last colon is treated as separator."""
    # A path like "package.module:submodule:attr" should split as
    # module_name="package.module:submodule", attr_name="attr"
    # This will fail at import (no such module) but demonstrates rsplit behavior.
    with pytest.raises(
        CLIError, match=r"Cannot import module 'package\.module:submodule'"
    ):
        load_agent_loop("package.module:submodule:attr")


def test_load_agent_loop_with_dotted_module_path() -> None:
    """Verify load_agent_loop handles dotted module paths correctly."""
    # os.path is a valid module; 'join' is a valid attribute (but not an agent)
    with pytest.raises(
        CLIError, match="must be a RolloutAgentLoop instance or subclass"
    ):
        load_agent_loop("os.path:join")
