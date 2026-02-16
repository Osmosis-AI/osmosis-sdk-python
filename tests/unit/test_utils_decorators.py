"""Tests for osmosis_ai.utils decorator validation, async wrapping, and type checking.

Covers @osmosis_reward and @osmosis_rubric decorators with both valid and invalid
function signatures, async function support, return type enforcement, and the
_is_str_annotation helper.

NOTE: This file intentionally does NOT use ``from __future__ import annotations``
because osmosis_reward checks annotation identity (``annotation is not str``),
which breaks when annotations are stringified by PEP 563.
"""

import sys

import pytest

from osmosis_ai.utils import _is_str_annotation, osmosis_reward, osmosis_rubric

# =============================================================================
# _is_str_annotation Tests
# =============================================================================


class TestIsStrAnnotation:
    """Tests for the _is_str_annotation helper used by osmosis_rubric."""

    def test_str_type_is_recognized(self) -> None:
        """The builtin str type should be recognized."""
        assert _is_str_annotation(str) is True

    def test_str_string_literal_is_recognized(self) -> None:
        """String literal 'str' should be recognized."""
        assert _is_str_annotation("str") is True

    def test_builtins_str_string_is_recognized(self) -> None:
        """String literal 'builtins.str' should be recognized."""
        assert _is_str_annotation("builtins.str") is True

    def test_empty_annotation_rejected(self) -> None:
        """inspect.Parameter.empty (no annotation) should not match str."""
        import inspect

        assert _is_str_annotation(inspect.Parameter.empty) is False

    def test_non_str_type_rejected(self) -> None:
        """Other types (int, dict, etc.) should be rejected."""
        assert _is_str_annotation(int) is False
        assert _is_str_annotation(dict) is False
        assert _is_str_annotation(float) is False

    def test_non_str_string_rejected(self) -> None:
        """Arbitrary strings that are not 'str' or 'builtins.str' are rejected."""
        assert _is_str_annotation("int") is False
        assert _is_str_annotation("something") is False

    def test_str_subclass_is_recognized(self) -> None:
        """A subclass of str should be recognized via issubclass."""

        class MyStr(str):
            pass

        assert _is_str_annotation(MyStr) is True

    def test_none_type_rejected(self) -> None:
        """NoneType should be rejected."""
        assert _is_str_annotation(type(None)) is False


# =============================================================================
# @osmosis_reward - Valid Classic Signatures
# =============================================================================


class TestOsmosisRewardValidClassic:
    """Tests for valid classic-style reward functions."""

    def test_minimal_classic_signature(self) -> None:
        """Classic signature with exactly 3 required params is accepted."""

        @osmosis_reward
        def reward(
            solution_str: str,
            ground_truth: str,
            extra_info: dict = None,  # noqa: RUF013
        ) -> float:
            return 1.0

        assert reward("answer", "truth", None) == 1.0

    def test_classic_with_kwargs(self) -> None:
        """Classic signature with **kwargs is accepted and kwargs are passed through."""

        @osmosis_reward
        def reward(
            solution_str: str,
            ground_truth: str,
            extra_info: dict | None = None,
            **kwargs,
        ) -> float:
            return 1.0

        # data_source should be passed through when **kwargs is present
        assert reward("a", "b", None, data_source="test") == 1.0

    def test_classic_with_optional_dict_annotation(self) -> None:
        """Classic signature with Optional[dict] for extra_info is accepted."""

        @osmosis_reward
        def reward(
            solution_str: str,
            ground_truth: str,
            extra_info: dict | None = None,
        ) -> float:
            return 0.5

        assert reward("a", "b") == 0.5

    def test_classic_with_union_dict_none_annotation(self) -> None:
        """Classic signature with Union[dict, None] for extra_info is accepted."""

        @osmosis_reward
        def reward(
            solution_str: str,
            ground_truth: str,
            extra_info: dict | None = None,
        ) -> float:
            return 0.5

        assert reward("a", "b") == 0.5

    @pytest.mark.skipif(
        sys.version_info < (3, 10), reason="PEP 604 syntax requires Python 3.10+"
    )
    def test_classic_with_pep604_union(self) -> None:
        """Classic signature with dict | None (PEP 604) for extra_info is accepted."""
        # Use exec to avoid SyntaxError on Python < 3.10
        code = """
def _reward(
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> float:
    return 0.5
"""
        local_ns: dict = {}
        exec(code, local_ns)
        wrapped = osmosis_reward(local_ns["_reward"])
        assert wrapped("a", "b") == 0.5

    def test_return_value_is_float(self) -> None:
        """Classic reward function should enforce float return."""

        @osmosis_reward
        def reward(
            solution_str: str,
            ground_truth: str,
            extra_info: dict | None = None,
        ) -> float:
            return 0.75

        result = reward("a", "b")
        assert isinstance(result, float)
        assert result == 0.75


# =============================================================================
# @osmosis_reward - Invalid Classic Signatures
# =============================================================================


class TestOsmosisRewardInvalidClassic:
    """Tests for classic-style reward functions that should be rejected at decoration time."""

    def test_too_few_parameters(self) -> None:
        """Classic signature with fewer than 3 params should raise TypeError."""
        with pytest.raises(TypeError, match="must have at least 3 parameters"):

            @osmosis_reward
            def reward(solution_str: str, ground_truth: str) -> float:
                return 1.0

    def test_solution_str_wrong_type(self) -> None:
        """First parameter annotated as non-str should raise TypeError."""
        with pytest.raises(TypeError, match="must be annotated as str"):

            @osmosis_reward
            def reward(
                solution_str: int,
                ground_truth: str,
                extra_info: dict | None = None,
            ) -> float:
                return 1.0

    def test_ground_truth_wrong_name(self) -> None:
        """Second parameter with wrong name should raise TypeError."""
        with pytest.raises(TypeError, match="must be named 'ground_truth'"):

            @osmosis_reward
            def reward(
                solution_str: str,
                wrong_name: str,
                extra_info: dict | None = None,
            ) -> float:
                return 1.0

    def test_ground_truth_wrong_type(self) -> None:
        """Second parameter annotated as non-str should raise TypeError."""
        with pytest.raises(TypeError, match="must be annotated as str"):

            @osmosis_reward
            def reward(
                solution_str: str,
                ground_truth: int,
                extra_info: dict | None = None,
            ) -> float:
                return 1.0

    def test_extra_info_wrong_name(self) -> None:
        """Third parameter with wrong name should raise TypeError."""
        with pytest.raises(TypeError, match="must be named 'extra_info'"):

            @osmosis_reward
            def reward(
                solution_str: str,
                ground_truth: str,
                wrong_name: dict | None = None,
            ) -> float:
                return 1.0

    def test_extra_info_wrong_type(self) -> None:
        """Third parameter annotated as non-dict should raise TypeError."""
        with pytest.raises(TypeError, match="must be annotated as dict"):

            @osmosis_reward
            def reward(
                solution_str: str,
                ground_truth: str,
                extra_info: str = None,  # noqa: RUF013
            ) -> float:
                return 1.0

    def test_extra_info_no_default(self) -> None:
        """Third parameter without default value should raise TypeError."""
        with pytest.raises(TypeError, match="must have a default value"):

            @osmosis_reward
            def reward(
                solution_str: str,
                ground_truth: str,
                extra_info: dict,
            ) -> float:
                return 1.0

    def test_extra_info_union_with_wrong_type(self) -> None:
        """Third parameter with Union[str, None] should raise TypeError."""
        with pytest.raises(TypeError, match="must be annotated as dict"):

            @osmosis_reward
            def reward(
                solution_str: str,
                ground_truth: str,
                extra_info: str | None = None,
            ) -> float:
                return 1.0


# =============================================================================
# @osmosis_reward - Non-Classic Signatures
# =============================================================================


class TestOsmosisRewardNonClassic:
    """Tests for non-classic reward functions (first param is NOT solution_str)."""

    def test_arbitrary_signature_accepted(self) -> None:
        """Non-classic signatures are accepted without validation."""

        @osmosis_reward
        def custom_reward(messages: list, metadata: dict) -> float:
            return 0.5

        assert custom_reward([], {}) == 0.5

    def test_no_params_accepted(self) -> None:
        """Zero-param functions are accepted in non-classic mode."""

        @osmosis_reward
        def no_params() -> float:
            return 1.0

        assert no_params() == 1.0

    def test_non_classic_no_return_type_enforcement(self) -> None:
        """Non-classic functions have no return type enforcement."""

        @osmosis_reward
        def reward(x: int) -> str:
            return "not a float"

        # No TypeError is raised even though return is not float
        assert reward(42) == "not a float"


# =============================================================================
# @osmosis_reward - Return Type Enforcement (Classic)
# =============================================================================


class TestOsmosisRewardReturnType:
    """Tests for classic return type enforcement at call time."""

    def test_non_float_return_raises_typeerror(self) -> None:
        """Classic reward returning non-float raises TypeError at call time."""

        @osmosis_reward
        def bad_return(
            solution_str: str,
            ground_truth: str,
            extra_info: dict | None = None,
        ) -> float:
            return "not a float"  # type: ignore[return-value]

        with pytest.raises(TypeError, match="must return a float"):
            bad_return("a", "b")

    def test_int_return_raises_typeerror(self) -> None:
        """Classic reward returning int (not float) raises TypeError."""

        @osmosis_reward
        def int_return(
            solution_str: str,
            ground_truth: str,
            extra_info: dict | None = None,
        ) -> float:
            return 1  # type: ignore[return-value]

        with pytest.raises(TypeError, match="must return a float"):
            int_return("a", "b")


# =============================================================================
# @osmosis_reward - data_source Handling
# =============================================================================


class TestOsmosisRewardDataSource:
    """Tests for the data_source kwarg stripping behavior."""

    def test_data_source_stripped_without_kwargs(self) -> None:
        """When function has no **kwargs and no data_source param, data_source is stripped."""

        @osmosis_reward
        def reward(
            solution_str: str,
            ground_truth: str,
            extra_info: dict | None = None,
        ) -> float:
            return 1.0

        # This should not raise even though data_source is passed
        assert reward("a", "b", data_source="test") == 1.0

    def test_data_source_passed_when_kwargs_present(self) -> None:
        """When function has **kwargs, data_source is NOT stripped."""

        @osmosis_reward
        def reward(
            solution_str: str,
            ground_truth: str,
            extra_info: dict | None = None,
            **kwargs,
        ) -> float:
            return 1.0 if "data_source" in kwargs else 0.0

        assert reward("a", "b", data_source="test") == 1.0

    def test_data_source_passed_when_explicit_param(self) -> None:
        """When function has explicit data_source param, it is NOT stripped."""

        @osmosis_reward
        def reward(
            solution_str: str,
            ground_truth: str,
            extra_info: dict | None = None,
            data_source: str = None,  # noqa: RUF013
        ) -> float:
            return 1.0 if data_source == "test" else 0.0

        assert reward("a", "b", data_source="test") == 1.0


# =============================================================================
# @osmosis_reward - Async Support
# =============================================================================


class TestOsmosisRewardAsync:
    """Tests for async reward function wrapping."""

    async def test_async_classic_reward(self) -> None:
        """Async classic reward function is properly wrapped and awaitable."""

        @osmosis_reward
        async def async_reward(
            solution_str: str,
            ground_truth: str,
            extra_info: dict | None = None,
        ) -> float:
            return 0.9

        result = await async_reward("a", "b")
        assert result == 0.9

    async def test_async_return_type_enforcement(self) -> None:
        """Async classic reward returning non-float raises TypeError."""

        @osmosis_reward
        async def bad_async(
            solution_str: str,
            ground_truth: str,
            extra_info: dict | None = None,
        ) -> float:
            return "bad"  # type: ignore[return-value]

        with pytest.raises(TypeError, match="must return a float"):
            await bad_async("a", "b")

    async def test_async_data_source_stripped(self) -> None:
        """Async reward function correctly strips data_source."""

        @osmosis_reward
        async def async_reward(
            solution_str: str,
            ground_truth: str,
            extra_info: dict | None = None,
        ) -> float:
            return 1.0

        # Should not raise even with data_source
        result = await async_reward("a", "b", data_source="x")
        assert result == 1.0

    async def test_async_non_classic(self) -> None:
        """Async non-classic reward is properly wrapped."""

        @osmosis_reward
        async def async_custom(x: int) -> int:
            return x * 2

        result = await async_custom(5)
        assert result == 10


# =============================================================================
# @osmosis_reward - functools.wraps Preservation
# =============================================================================


class TestOsmosisRewardWraps:
    """Tests that functools.wraps preserves function metadata."""

    def test_function_name_preserved(self) -> None:
        """Wrapped function retains its original __name__."""

        @osmosis_reward
        def my_reward(
            solution_str: str,
            ground_truth: str,
            extra_info: dict | None = None,
        ) -> float:
            return 1.0

        assert my_reward.__name__ == "my_reward"

    def test_docstring_preserved(self) -> None:
        """Wrapped function retains its docstring."""

        @osmosis_reward
        def documented_reward(
            solution_str: str,
            ground_truth: str,
            extra_info: dict | None = None,
        ) -> float:
            """This is my reward function."""
            return 1.0

        assert documented_reward.__doc__ == "This is my reward function."


# =============================================================================
# @osmosis_rubric - Valid Classic Signatures
# =============================================================================


class TestOsmosisRubricValidClassic:
    """Tests for valid classic-style rubric functions."""

    def test_minimal_classic_signature(self) -> None:
        """Classic signature with exactly 3 required params is accepted."""

        @osmosis_rubric
        def rubric(
            solution_str: str,
            ground_truth: str,
            extra_info: dict,
        ) -> float:
            return 1.0

        result = rubric("answer", "truth", {})
        assert result == 1.0

    def test_classic_with_kwargs(self) -> None:
        """Classic signature with **kwargs is accepted."""

        @osmosis_rubric
        def rubric(
            solution_str: str,
            ground_truth: str,
            extra_info: dict,
            **kwargs,
        ) -> float:
            return 1.0

        assert rubric("a", "b", {}) == 1.0

    def test_ground_truth_optional_str(self) -> None:
        """ground_truth annotated as Optional[str] is accepted."""

        @osmosis_rubric
        def rubric(
            solution_str: str,
            ground_truth: str | None,
            extra_info: dict,
        ) -> float:
            return 1.0

        assert rubric("a", None, {}) == 1.0

    def test_ground_truth_union_str_none(self) -> None:
        """ground_truth annotated as Union[str, None] is accepted."""

        @osmosis_rubric
        def rubric(
            solution_str: str,
            ground_truth: str | None,
            extra_info: dict,
        ) -> float:
            return 1.0

        assert rubric("a", None, {}) == 1.0


# =============================================================================
# @osmosis_rubric - Invalid Classic Signatures
# =============================================================================


class TestOsmosisRubricInvalidClassic:
    """Tests for classic-style rubric functions that should be rejected."""

    def test_too_few_parameters(self) -> None:
        """Classic signature with fewer than 3 params should raise TypeError."""
        with pytest.raises(TypeError, match="must have at least 3 parameters"):

            @osmosis_rubric
            def rubric(solution_str: str, ground_truth: str) -> float:
                return 1.0

    def test_solution_str_wrong_type(self) -> None:
        """First parameter annotated as non-str should raise TypeError."""
        with pytest.raises(TypeError, match="must be annotated as str"):

            @osmosis_rubric
            def rubric(
                solution_str: int,
                ground_truth: str,
                extra_info: dict,
            ) -> float:
                return 1.0

    def test_solution_str_with_default_rejected(self) -> None:
        """First parameter with a default value should raise TypeError."""
        with pytest.raises(TypeError, match="cannot have a default value"):

            @osmosis_rubric
            def rubric(
                solution_str: str = "default",
                ground_truth: str = "",
                extra_info: dict | None = None,
            ) -> float:
                return 1.0

    def test_ground_truth_wrong_name(self) -> None:
        """Second parameter with wrong name should raise TypeError."""
        with pytest.raises(TypeError, match="must be named 'ground_truth'"):

            @osmosis_rubric
            def rubric(
                solution_str: str,
                wrong_name: str,
                extra_info: dict,
            ) -> float:
                return 1.0

    def test_ground_truth_wrong_type(self) -> None:
        """Second parameter annotated as non-str (and not Optional[str]) raises TypeError."""
        with pytest.raises(TypeError, match="must be annotated as str"):

            @osmosis_rubric
            def rubric(
                solution_str: str,
                ground_truth: int,
                extra_info: dict,
            ) -> float:
                return 1.0

    def test_ground_truth_with_default_rejected(self) -> None:
        """Second parameter with a default value should raise TypeError."""
        with pytest.raises(TypeError, match="cannot have a default value"):

            @osmosis_rubric
            def rubric(
                solution_str: str,
                ground_truth: str = "default",
                extra_info: dict | None = None,
            ) -> float:
                return 1.0

    def test_extra_info_wrong_name(self) -> None:
        """Third parameter with wrong name should raise TypeError."""
        with pytest.raises(TypeError, match="must be named 'extra_info'"):

            @osmosis_rubric
            def rubric(
                solution_str: str,
                ground_truth: str,
                wrong_name: dict,
            ) -> float:
                return 1.0


# =============================================================================
# @osmosis_rubric - Runtime Argument Validation (Classic)
# =============================================================================


class TestOsmosisRubricRuntimeValidation:
    """Tests for runtime argument validation in classic rubric functions."""

    def test_solution_str_must_be_string(self) -> None:
        """Passing a non-string for solution_str raises TypeError at call time."""

        @osmosis_rubric
        def rubric(
            solution_str: str,
            ground_truth: str,
            extra_info: dict,
        ) -> float:
            return 1.0

        with pytest.raises(TypeError, match="must be a string"):
            rubric(123, "truth", {})  # type: ignore[arg-type]

    def test_ground_truth_must_be_string_or_none(self) -> None:
        """Passing non-string/non-None for ground_truth raises TypeError."""

        @osmosis_rubric
        def rubric(
            solution_str: str,
            ground_truth: str | None,
            extra_info: dict,
        ) -> float:
            return 1.0

        with pytest.raises(TypeError, match="must be a string or None"):
            rubric("answer", 123, {})  # type: ignore[arg-type]

    def test_ground_truth_none_is_accepted(self) -> None:
        """None is a valid value for ground_truth."""

        @osmosis_rubric
        def rubric(
            solution_str: str,
            ground_truth: str | None,
            extra_info: dict,
        ) -> float:
            return 1.0

        assert rubric("a", None, {}) == 1.0

    def test_extra_info_required(self) -> None:
        """extra_info is required for classic rubric functions."""

        @osmosis_rubric
        def rubric(
            solution_str: str,
            ground_truth: str,
            extra_info: dict,
        ) -> float:
            return 1.0

        with pytest.raises(TypeError, match="'extra_info' argument is required"):
            rubric("a", "b")  # type: ignore[call-arg]

    def test_solution_str_required(self) -> None:
        """solution_str is required for classic rubric functions."""

        @osmosis_rubric
        def rubric(
            solution_str: str,
            ground_truth: str,
            extra_info: dict,
        ) -> float:
            return 1.0

        with pytest.raises(TypeError):
            rubric()  # type: ignore[call-arg]


# =============================================================================
# @osmosis_rubric - Return Type Enforcement (Classic)
# =============================================================================


class TestOsmosisRubricReturnType:
    """Tests for classic rubric return type enforcement."""

    def test_non_float_return_raises_typeerror(self) -> None:
        """Classic rubric returning non-float raises TypeError."""

        @osmosis_rubric
        def bad_return(
            solution_str: str,
            ground_truth: str,
            extra_info: dict,
        ) -> float:
            return "not a float"  # type: ignore[return-value]

        with pytest.raises(TypeError, match="must return a float"):
            bad_return("a", "b", {})

    def test_int_return_raises_typeerror(self) -> None:
        """Classic rubric returning int raises TypeError."""

        @osmosis_rubric
        def int_return(
            solution_str: str,
            ground_truth: str,
            extra_info: dict,
        ) -> float:
            return 1  # type: ignore[return-value]

        with pytest.raises(TypeError, match="must return a float"):
            int_return("a", "b", {})


# =============================================================================
# @osmosis_rubric - Non-Classic Signatures
# =============================================================================


class TestOsmosisRubricNonClassic:
    """Tests for non-classic rubric functions."""

    def test_arbitrary_signature_accepted(self) -> None:
        """Non-classic signatures are accepted without validation."""

        @osmosis_rubric
        def custom(messages: list, context: dict) -> float:
            return 0.5

        assert custom([], {}) == 0.5

    def test_no_params_accepted(self) -> None:
        """Zero-param functions are accepted in non-classic mode."""

        @osmosis_rubric
        def no_params() -> float:
            return 1.0

        assert no_params() == 1.0

    def test_non_classic_no_return_type_enforcement(self) -> None:
        """Non-classic functions have no return type enforcement."""

        @osmosis_rubric
        def rubric(x: int) -> str:
            return "string"

        assert rubric(1) == "string"

    def test_non_classic_no_runtime_arg_validation(self) -> None:
        """Non-classic functions skip runtime argument validation."""

        @osmosis_rubric
        def rubric(x: int) -> int:
            return x * 2

        # No TypeError even though int is passed
        assert rubric(5) == 10


# =============================================================================
# @osmosis_rubric - data_source Handling
# =============================================================================


class TestOsmosisRubricDataSource:
    """Tests for the data_source kwarg stripping behavior in rubric."""

    def test_data_source_stripped_without_kwargs(self) -> None:
        """When function has no **kwargs, data_source is stripped."""

        @osmosis_rubric
        def rubric(
            solution_str: str,
            ground_truth: str,
            extra_info: dict,
        ) -> float:
            return 1.0

        assert rubric("a", "b", {}, data_source="test") == 1.0

    def test_data_source_passed_when_kwargs_present(self) -> None:
        """When function has **kwargs, data_source is NOT stripped."""

        @osmosis_rubric
        def rubric(
            solution_str: str,
            ground_truth: str,
            extra_info: dict,
            **kwargs,
        ) -> float:
            return 1.0 if "data_source" in kwargs else 0.0

        assert rubric("a", "b", {}, data_source="test") == 1.0


# =============================================================================
# @osmosis_rubric - Async Support
# =============================================================================


class TestOsmosisRubricAsync:
    """Tests for async rubric function wrapping."""

    async def test_async_classic_rubric(self) -> None:
        """Async classic rubric function is properly wrapped and awaitable."""

        @osmosis_rubric
        async def async_rubric(
            solution_str: str,
            ground_truth: str,
            extra_info: dict,
        ) -> float:
            return 0.8

        result = await async_rubric("a", "b", {})
        assert result == 0.8

    async def test_async_return_type_enforcement(self) -> None:
        """Async classic rubric returning non-float raises TypeError."""

        @osmosis_rubric
        async def bad_async(
            solution_str: str,
            ground_truth: str,
            extra_info: dict,
        ) -> float:
            return "bad"  # type: ignore[return-value]

        with pytest.raises(TypeError, match="must return a float"):
            await bad_async("a", "b", {})

    async def test_async_runtime_validation(self) -> None:
        """Async rubric validates arguments at call time."""

        @osmosis_rubric
        async def async_rubric(
            solution_str: str,
            ground_truth: str,
            extra_info: dict,
        ) -> float:
            return 1.0

        with pytest.raises(TypeError, match="must be a string"):
            await async_rubric(42, "b", {})  # type: ignore[arg-type]

    async def test_async_data_source_stripped(self) -> None:
        """Async rubric correctly strips data_source."""

        @osmosis_rubric
        async def async_rubric(
            solution_str: str,
            ground_truth: str,
            extra_info: dict,
        ) -> float:
            return 1.0

        result = await async_rubric("a", "b", {}, data_source="x")
        assert result == 1.0

    async def test_async_non_classic(self) -> None:
        """Async non-classic rubric is properly wrapped."""

        @osmosis_rubric
        async def async_custom(x: int) -> int:
            return x * 2

        result = await async_custom(5)
        assert result == 10


# =============================================================================
# @osmosis_rubric - functools.wraps Preservation
# =============================================================================


class TestOsmosisRubricWraps:
    """Tests that functools.wraps preserves function metadata."""

    def test_function_name_preserved(self) -> None:
        """Wrapped rubric retains its original __name__."""

        @osmosis_rubric
        def my_rubric(
            solution_str: str,
            ground_truth: str,
            extra_info: dict,
        ) -> float:
            return 1.0

        assert my_rubric.__name__ == "my_rubric"

    def test_docstring_preserved(self) -> None:
        """Wrapped rubric retains its docstring."""

        @osmosis_rubric
        def documented_rubric(
            solution_str: str,
            ground_truth: str,
            extra_info: dict,
        ) -> float:
            """Evaluate quality."""
            return 1.0

        assert documented_rubric.__doc__ == "Evaluate quality."
