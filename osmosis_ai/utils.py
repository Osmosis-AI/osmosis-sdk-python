
import asyncio
import functools
import inspect
import types
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints

_UNION_TYPE = getattr(types, "UnionType", None)
_ALLOWED_UNION_ORIGINS = (Union,) + ((_UNION_TYPE,) if _UNION_TYPE is not None else ())


def osmosis_reward(func: Callable) -> Callable:
    """
    Decorator for reward functions.

    When the first parameter is named ``solution_str``, the classic Osmosis
    signature is enforced:
        (solution_str: str, ground_truth: str, extra_info: dict = None) -> float

    Any other signature is accepted without parameter or return-type validation.

    Args:
        func: The reward function to be wrapped

    Returns:
        The wrapped function

    Raises:
        TypeError: If a classic-style function doesn't have the required signature or return type
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    # Classic mode is identified by the first parameter being named 'solution_str'.
    is_classic = len(params) >= 1 and params[0].name == 'solution_str'

    if is_classic:
        if len(params) < 3:
            raise TypeError(f"Function {func.__name__} must have at least 3 parameters, got {len(params)}")

        # Check first parameter: solution_str: str
        if params[0].annotation != str:
            raise TypeError(f"First parameter 'solution_str' must be annotated as str, got {params[0].annotation}")

        # Check second parameter: ground_truth: str
        if params[1].name != 'ground_truth':
            raise TypeError(f"Second parameter must be named 'ground_truth', got '{params[1].name}'")
        if params[1].annotation != str:
            raise TypeError(f"Second parameter 'ground_truth' must be annotated as str, got {params[1].annotation}")

        # Check third parameter if present: extra_info: dict = None
        if len(params) >= 3:
            if params[2].name != 'extra_info':
                raise TypeError(f"Third parameter must be named 'extra_info', got '{params[2].name}'")
            if params[2].annotation != dict:
                raise TypeError(f"Third parameter 'extra_info' must be annotated as dict, got {params[2].annotation}")
            if params[2].default is inspect.Parameter.empty:
                raise TypeError("Third parameter 'extra_info' must have a default value of None")

    # Only strip data_source when the function can't accept it.
    _drop_data_source = (
        "data_source" not in sig.parameters
        and not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    )

    def _check_classic_return(result):
        if is_classic and not isinstance(result, float):
            raise TypeError(f"Function {func.__name__} must return a float, got {type(result).__name__}")
        return result

    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if _drop_data_source:
                kwargs.pop("data_source", None)
            return _check_classic_return(await func(*args, **kwargs))
    else:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if _drop_data_source:
                kwargs.pop("data_source", None)
            return _check_classic_return(func(*args, **kwargs))

    return wrapper

def _is_str_annotation(annotation: Any) -> bool:
    if annotation is inspect.Parameter.empty:
        return False
    if annotation is str:
        return True
    if isinstance(annotation, str):
        return annotation in {"str", "builtins.str"}
    if isinstance(annotation, type):
        try:
            return issubclass(annotation, str)
        except TypeError:
            return False
    forward_arg = getattr(annotation, "__forward_arg__", None)
    if isinstance(forward_arg, str):
        return forward_arg in {"str", "builtins.str"}
    return False


def osmosis_rubric(func: Callable) -> Callable:
    """
    Decorator for rubric functions.

    When the first parameter is named ``solution_str``, the classic Osmosis
    signature is enforced:
        (solution_str: str, ground_truth: str, extra_info: dict) -> float

    Any other signature is accepted without parameter or return-type validation.

    Args:
        func: The rubric function to be wrapped.

    Returns:
        The wrapped function.

    Raises:
        TypeError: If a classic-style function doesn't have the required signature or return type.

    Example:
        @osmosis_rubric
        def evaluate_response(
            solution_str: str,
            ground_truth: str,
            extra_info: dict,
        ) -> float:
            return some_evaluation(solution_str, ground_truth, extra_info)
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    # Classic mode is identified by the first parameter being named 'solution_str'.
    is_classic = len(params) >= 1 and params[0].name == "solution_str"

    if is_classic:
        try:
            resolved_annotations = get_type_hints(
                func,
                globalns=getattr(func, "__globals__", {}),
                include_extras=True,
            )
        except Exception:  # pragma: no cover - best effort for forward refs
            resolved_annotations = {}

        if len(params) < 3:
            raise TypeError(f"Function {func.__name__} must have at least 3 parameters, got {len(params)}")

        solution_param = params[0]
        solution_annotation = resolved_annotations.get(solution_param.name, solution_param.annotation)
        if not _is_str_annotation(solution_annotation):
            raise TypeError(f"First parameter 'solution_str' must be annotated as str, got {solution_annotation}")
        if solution_param.default is not inspect.Parameter.empty:
            raise TypeError("First parameter 'solution_str' cannot have a default value")

        ground_truth_param = params[1]
        if ground_truth_param.name != "ground_truth":
            raise TypeError(f"Second parameter must be named 'ground_truth', got '{ground_truth_param.name}'")
        ground_truth_annotation = resolved_annotations.get(ground_truth_param.name, ground_truth_param.annotation)
        if not _is_str_annotation(ground_truth_annotation):
            union_origin = get_origin(ground_truth_annotation)
            if union_origin not in _ALLOWED_UNION_ORIGINS:
                raise TypeError(
                    f"Second parameter 'ground_truth' must be annotated as str or Optional[str], got {ground_truth_annotation}"
                )
            union_args = tuple(arg for arg in get_args(ground_truth_annotation) if arg is not type(None))  # noqa: E721
            if len(union_args) != 1 or not _is_str_annotation(union_args[0]):
                raise TypeError(
                    f"Second parameter 'ground_truth' must be annotated as str or Optional[str], got {ground_truth_annotation}"
                )
        if ground_truth_param.default is not inspect.Parameter.empty:
            raise TypeError("Second parameter 'ground_truth' cannot have a default value")

        extra_info_param = params[2]
        if extra_info_param.name != "extra_info":
            raise TypeError(f"Third parameter must be named 'extra_info', got '{extra_info_param.name}'")

    # Only strip data_source when the function can't accept it.
    _drop_data_source = (
        "data_source" not in sig.parameters
        and not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    )

    def _validate_classic_args(*args, **kwargs):
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()

        if "solution_str" not in bound.arguments:
            raise TypeError("'solution_str' argument is required")
        solution_value = bound.arguments["solution_str"]
        if not isinstance(solution_value, str):
            raise TypeError(f"'solution_str' must be a string, got {type(solution_value).__name__}")

        if "ground_truth" not in bound.arguments:
            raise TypeError("'ground_truth' argument is required")
        ground_truth_value = bound.arguments["ground_truth"]
        if ground_truth_value is not None and not isinstance(ground_truth_value, str):
            raise TypeError(
                f"'ground_truth' must be a string or None, got {type(ground_truth_value).__name__}"
            )

        if "extra_info" not in bound.arguments:
            raise TypeError("'extra_info' argument is required")

    def _check_classic_return(result):
        if is_classic and not isinstance(result, float):
            raise TypeError(f"Function {func.__name__} must return a float, got {type(result).__name__}")
        return result

    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if _drop_data_source:
                kwargs.pop("data_source", None)
            if is_classic:
                _validate_classic_args(*args, **kwargs)
            return _check_classic_return(await func(*args, **kwargs))
    else:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if _drop_data_source:
                kwargs.pop("data_source", None)
            if is_classic:
                _validate_classic_args(*args, **kwargs)
            return _check_classic_return(func(*args, **kwargs))

    return wrapper
