
import functools
import inspect
import types
from typing import Any, Callable, Mapping, Union, get_args, get_origin, get_type_hints


def osmosis_reward(func: Callable) -> Callable:
    """
    Decorator for reward functions that enforces the signature:
    (solution_str: str, ground_truth: str, extra_info: dict = None) -> float

    Args:
        func: The reward function to be wrapped

    Returns:
        The wrapped function

    Raises:
        TypeError: If the function doesn't have the required signature or doesn't return a float

    Example:
        @osmosis_reward
        def calculate_reward(solution_str: str, ground_truth: str, extra_info: dict = None) -> float:
            return some_calculation(solution_str, ground_truth)
    """
    # Validate function signature
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    if len(params) < 3:
        raise TypeError(f"Function {func.__name__} must have at least 3 parameters, got {len(params)}")

    # Check first parameter: solution_str: str
    if params[0].name != 'solution_str':
        raise TypeError(f"First parameter must be named 'solution_str', got '{params[0].name}'")
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

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs.pop("data_source", None)
        result = func(*args, **kwargs)
        if not isinstance(result, float):
            raise TypeError(f"Function {func.__name__} must return a float, got {type(result).__name__}")
        return result

    return wrapper


ALLOWED_ROLES = {"user", "system", "assistant", "developer", "tool", "function"}

_UNION_TYPES = {Union}
_types_union_type = getattr(types, "UnionType", None)
if _types_union_type is not None:
    _UNION_TYPES.add(_types_union_type)


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


def _is_optional_str(annotation: Any) -> bool:
    if _is_str_annotation(annotation):
        return True
    if isinstance(annotation, str):
        normalized = annotation.replace(" ", "")
        if normalized in {
            "Optional[str]",
            "typing.Optional[str]",
            "Str|None",
            "str|None",
            "builtins.str|None",
            "None|str",
            "None|builtins.str",
        }:
            return True
    origin = get_origin(annotation)
    if origin in _UNION_TYPES:
        args = tuple(arg for arg in get_args(annotation) if arg is not type(None))  # noqa: E721
        return len(args) == 1 and _is_str_annotation(args[0])
    return False


def _is_list_annotation(annotation: Any) -> bool:
    if annotation is list:
        return True
    if isinstance(annotation, str):
        normalized = annotation.replace(" ", "")
        return (
            normalized in {"list", "builtins.list", "typing.List", "List"}
            or normalized.startswith("list[")
            or normalized.startswith("builtins.list[")
            or normalized.startswith("typing.List[")
            or normalized.startswith("List[")
        )
    origin = get_origin(annotation)
    return origin is list


def _is_float_annotation(annotation: Any) -> bool:
    if annotation in {inspect.Parameter.empty, float}:
        return True
    if isinstance(annotation, str):
        return annotation in {"float", "builtins.float"}
    origin = get_origin(annotation)
    return origin is float


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_dict_annotation(annotation: Any) -> bool:
    if annotation in {dict, Mapping}:
        return True
    origin = get_origin(annotation)
    if origin in {dict, Mapping}:
        return True
    if isinstance(annotation, type):
        try:
            return issubclass(annotation, dict)
        except TypeError:
            return False
    if isinstance(annotation, str):
        normalized = annotation.replace(" ", "")
        return (
            normalized in {"dict", "builtins.dict", "typing.Mapping", "collections.abc.Mapping", "Mapping"}
            or normalized.startswith("dict[")
            or normalized.startswith("builtins.dict[")
            or normalized.startswith("typing.Dict[")
            or normalized.startswith("Dict[")
            or normalized.startswith("typing.Mapping[")
            or normalized.startswith("Mapping[")
        )
    return False


def osmosis_rubric(func: Callable) -> Callable:
    """
    Decorator for rubric functions that enforces the signature:
    (solution_str: str, ground_truth: str, extra_info: dict) -> float

    The `extra_info` mapping must include the current `provider`, `model`, and `rubric`
    values, and may optionally provide `system_prompt`, `score_min`, and `score_max`.

    Args:
        func: The rubric function to be wrapped.

    Returns:
        The wrapped function.

    Raises:
        TypeError: If the function doesn't have the required signature or doesn't return a float.

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
    if solution_param.name != "solution_str":
        raise TypeError(f"First parameter must be named 'solution_str', got '{solution_param.name}'")
    solution_annotation = resolved_annotations.get(solution_param.name, solution_param.annotation)
    if not _is_str_annotation(solution_annotation):
        raise TypeError(f"First parameter 'solution_str' must be annotated as str, got {solution_annotation}")
    if solution_param.default is not inspect.Parameter.empty:
        raise TypeError("First parameter 'solution_str' cannot have a default value")

    ground_truth_param = params[1]
    if ground_truth_param.name != "ground_truth":
        raise TypeError(f"Second parameter must be named 'ground_truth', got '{ground_truth_param.name}'")
    ground_truth_annotation = resolved_annotations.get(ground_truth_param.name, ground_truth_param.annotation)
    if not _is_optional_str(ground_truth_annotation):
        raise TypeError(
            f"Second parameter 'ground_truth' must be annotated as str or Optional[str], got {ground_truth_annotation}"
        )
    if ground_truth_param.default is not inspect.Parameter.empty:
        raise TypeError("Second parameter 'ground_truth' cannot have a default value")

    extra_info_param = params[2]
    if extra_info_param.name != "extra_info":
        raise TypeError(f"Third parameter must be named 'extra_info', got '{extra_info_param.name}'")
    extra_info_annotation = resolved_annotations.get(extra_info_param.name, extra_info_param.annotation)
    if not _is_dict_annotation(extra_info_annotation):
        raise TypeError(
            f"Third parameter 'extra_info' must be annotated as a dict or mapping, got {extra_info_annotation}"
        )
    if extra_info_param.default is not inspect.Parameter.empty:
        raise TypeError("Third parameter 'extra_info' cannot have a default value")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs.pop("data_source", None)
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
        extra_info_value = bound.arguments["extra_info"]
        if not isinstance(extra_info_value, Mapping):
            raise TypeError(f"'extra_info' must be a mapping, got {type(extra_info_value).__name__}")

        provider_value = extra_info_value.get("provider")
        if not isinstance(provider_value, str) or not provider_value.strip():
            raise TypeError("'extra_info[\"provider\"]' must be a non-empty string")

        model_value = extra_info_value.get("model")
        if not isinstance(model_value, str) or not model_value.strip():
            raise TypeError("'extra_info[\"model\"]' must be a non-empty string")

        if "rubric" not in extra_info_value:
            raise TypeError("'extra_info' must include a 'rubric' string")
        rubric_value = extra_info_value["rubric"]
        if not isinstance(rubric_value, str):
            raise TypeError(f"'extra_info[\"rubric\"]' must be a string, got {type(rubric_value).__name__}")

        api_key_value = extra_info_value.get("api_key")
        api_key_env_value = extra_info_value.get("api_key_env")
        has_api_key = isinstance(api_key_value, str) and bool(api_key_value.strip())
        has_api_key_env = isinstance(api_key_env_value, str) and bool(api_key_env_value.strip())
        if not (has_api_key or has_api_key_env):
            raise TypeError(
                "'extra_info' must include either a non-empty 'api_key' or 'api_key_env' string"
            )

        system_prompt_value = extra_info_value.get("system_prompt")
        if system_prompt_value is not None and not isinstance(system_prompt_value, str):
            raise TypeError(
                f"'extra_info[\"system_prompt\"]' must be a string or None, got {type(system_prompt_value).__name__}"
            )

        score_min_value = extra_info_value.get("score_min")
        if score_min_value is not None and not _is_numeric(score_min_value):
            raise TypeError(
                f"'extra_info[\"score_min\"]' must be numeric, got {type(score_min_value).__name__}"
            )

        score_max_value = extra_info_value.get("score_max")
        if score_max_value is not None and not _is_numeric(score_max_value):
            raise TypeError(
                f"'extra_info[\"score_max\"]' must be numeric, got {type(score_max_value).__name__}"
            )

        if score_min_value is not None and score_max_value is not None:
            if float(score_max_value) <= float(score_min_value):
                raise ValueError("'extra_info[\"score_max\"]' must be greater than 'extra_info[\"score_min\"]'")

        result = func(*args, **kwargs)
        if not isinstance(result, float):
            raise TypeError(f"Function {func.__name__} must return a float, got {type(result).__name__}")
        return result

    return wrapper
