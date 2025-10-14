
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

    # Check parameter count
    if len(params) < 2 or len(params) > 3:
        raise TypeError(f"Function {func.__name__} must have 2-3 parameters, got {len(params)}")

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
    if len(params) == 3:
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
    (model_info: dict, rubric: str, messages: list, ground_truth: Optional[str] = None,
     system_message: Optional[str] = None, extra_info: dict = None,
     score_min: float = 0.0, score_max: float = 1.0) -> float

    The `model_info` mapping must provide non-empty string entries for both `provider` and `model`.

    Args:
        func: The rubric function to be wrapped

    Returns:
        The wrapped function

    Raises:
        TypeError: If the function doesn't have the required signature or doesn't return a float

    Example:
        @osmosis_rubric
        def evaluate_response(
            model_info: dict,
            rubric: str,
            messages: list,
            ground_truth: str | None = None,
            system_message: str | None = None,
            extra_info: dict = None,
            score_min: float = 0.0,
            score_max: float = 1.0,
        ) -> float:
            return some_evaluation(model_info, messages, ground_truth)
    """
    # Validate function signature
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

    # Check parameter count
    if len(params) < 3 or len(params) > 8:
        raise TypeError(f"Function {func.__name__} must have between 3 and 8 parameters, got {len(params)}")

    # Check first parameter: model_info: dict
    model_info_param = params[0]
    if model_info_param.name != "model_info":
        raise TypeError(f"First parameter must be named 'model_info', got '{model_info_param.name}'")
    model_info_annotation = resolved_annotations.get(model_info_param.name, model_info_param.annotation)
    if not _is_dict_annotation(model_info_annotation):
        raise TypeError(
            f"First parameter 'model_info' must be annotated as a dict or mapping, got {model_info_annotation}"
        )
    if model_info_param.default is not inspect.Parameter.empty:
        raise TypeError("First parameter 'model_info' cannot have a default value")

    # Check second parameter: rubric: str
    rubric_param = params[1]
    if rubric_param.name != "rubric":
        raise TypeError(f"Second parameter must be named 'rubric', got '{rubric_param.name}'")
    rubric_annotation = resolved_annotations.get(rubric_param.name, rubric_param.annotation)
    if not _is_str_annotation(rubric_annotation):
        raise TypeError(f"Second parameter 'rubric' must be annotated as str, got {rubric_annotation}")
    if rubric_param.default is not inspect.Parameter.empty:
        raise TypeError("Second parameter 'rubric' cannot have a default value")

    # Check third parameter: messages: list
    messages_param = params[2]
    if messages_param.name != "messages":
        raise TypeError(f"Third parameter must be named 'messages', got '{messages_param.name}'")
    messages_annotation = resolved_annotations.get(messages_param.name, messages_param.annotation)
    if messages_annotation is inspect.Parameter.empty:
        raise TypeError("Third parameter 'messages' must be annotated as list")
    if not _is_list_annotation(messages_annotation):
        raise TypeError(f"Third parameter 'messages' must be annotated as list, got {messages_annotation}")
    if messages_param.default is not inspect.Parameter.empty:
        raise TypeError("Third parameter 'messages' cannot have a default value")

    optional_params = params[3:]

    if optional_params:
        ground_truth_param = optional_params[0]
        # Check fourth parameter: ground_truth: Optional[str]
        if ground_truth_param.name != "ground_truth":
            raise TypeError(f"Fourth parameter must be named 'ground_truth', got '{ground_truth_param.name}'")
        ground_truth_annotation = resolved_annotations.get(
            ground_truth_param.name,
            ground_truth_param.annotation,
        )
        if ground_truth_annotation is inspect.Parameter.empty or not _is_optional_str(ground_truth_annotation):
            raise TypeError(
                "Fourth parameter 'ground_truth' must be annotated as Optional[str] or str"
            )
        if ground_truth_param.default is inspect.Parameter.empty:
            raise TypeError("Fourth parameter 'ground_truth' must have a default value of None")
        if ground_truth_param.default is not None:
            raise TypeError("Fourth parameter 'ground_truth' must default to None")
        optional_params = optional_params[1:]

    if optional_params:
        system_message_param = optional_params[0]
        # Check fifth parameter: system_message: Optional[str]
        if system_message_param.name != "system_message":
            raise TypeError(f"Fifth parameter must be named 'system_message', got '{system_message_param.name}'")
        system_message_annotation = resolved_annotations.get(
            system_message_param.name,
            system_message_param.annotation,
        )
        if system_message_annotation is inspect.Parameter.empty or not _is_optional_str(system_message_annotation):
            raise TypeError(
                "Fifth parameter 'system_message' must be annotated as Optional[str] or str"
            )
        if system_message_param.default is inspect.Parameter.empty:
            raise TypeError("Fifth parameter 'system_message' must have a default value of None")
        if system_message_param.default is not None:
            raise TypeError("Fifth parameter 'system_message' must default to None")
        optional_params = optional_params[1:]

    if optional_params:
        extra_info_param = optional_params[0]
        # Check sixth parameter: extra_info: dict = None
        if extra_info_param.name != "extra_info":
            raise TypeError(f"Sixth parameter must be named 'extra_info', got '{extra_info_param.name}'")
        extra_info_annotation = resolved_annotations.get(
            extra_info_param.name,
            extra_info_param.annotation,
        )
        if extra_info_annotation is inspect.Parameter.empty or not _is_dict_annotation(extra_info_annotation):
            raise TypeError(
                f"Sixth parameter 'extra_info' must be annotated as dict, got {extra_info_annotation}"
            )
        if extra_info_param.default is inspect.Parameter.empty:
            raise TypeError("Sixth parameter 'extra_info' must have a default value of None")
        if extra_info_param.default is not None:
            raise TypeError("Sixth parameter 'extra_info' must default to None")
        optional_params = optional_params[1:]

    if optional_params:
        score_min_param = optional_params[0]
        # Check seventh parameter: score_min: float = 0.0
        if score_min_param.name != "score_min":
            raise TypeError(f"Seventh parameter must be named 'score_min', got '{score_min_param.name}'")
        score_min_annotation = resolved_annotations.get(
            score_min_param.name,
            score_min_param.annotation,
        )
        if not _is_float_annotation(score_min_annotation):
            raise TypeError(
                f"Seventh parameter 'score_min' must be annotated as float, got {score_min_annotation}"
            )
        if score_min_param.default is inspect.Parameter.empty:
            raise TypeError("Seventh parameter 'score_min' must have a default value of 0.0")
        if not _is_numeric(score_min_param.default):
            raise TypeError("Seventh parameter 'score_min' must default to a numeric value")
        if float(score_min_param.default) != 0.0:
            raise TypeError("Seventh parameter 'score_min' must default to 0.0")
        optional_params = optional_params[1:]

    if optional_params:
        score_max_param = optional_params[0]
        # Check eighth parameter: score_max: float = 1.0
        if score_max_param.name != "score_max":
            raise TypeError(f"Eighth parameter must be named 'score_max', got '{score_max_param.name}'")
        score_max_annotation = resolved_annotations.get(
            score_max_param.name,
            score_max_param.annotation,
        )
        if not _is_float_annotation(score_max_annotation):
            raise TypeError(
                f"Eighth parameter 'score_max' must be annotated as float, got {score_max_annotation}"
            )
        if score_max_param.default is inspect.Parameter.empty:
            raise TypeError("Eighth parameter 'score_max' must have a default value of 1.0")
        if not _is_numeric(score_max_param.default):
            raise TypeError("Eighth parameter 'score_max' must default to a numeric value")
        if float(score_max_param.default) != 1.0:
            raise TypeError("Eighth parameter 'score_max' must default to 1.0")
        optional_params = optional_params[1:]

    if optional_params:
        unexpected_param = optional_params[0]
        raise TypeError(f"Function {func.__name__} has unexpected parameter '{unexpected_param.name}'")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Remove unsupported kwargs
        kwargs.pop("data_source", None)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()

        # Validate model_info argument
        if "model_info" not in bound.arguments:
            raise TypeError("'model_info' argument is required")
        model_info_value = bound.arguments["model_info"]
        if not isinstance(model_info_value, Mapping):
            raise TypeError(f"'model_info' must be a mapping, got {type(model_info_value).__name__}")
        required_model_fields = {"provider", "model"}
        missing_model_fields = required_model_fields - set(model_info_value.keys())
        if missing_model_fields:
            raise ValueError(f"'model_info' is missing required fields: {sorted(missing_model_fields)}")
        provider_value = model_info_value.get("provider")
        if not isinstance(provider_value, str) or not provider_value.strip():
            raise TypeError("'model_info[\"provider\"]' must be a non-empty string")
        model_value = model_info_value.get("model")
        if not isinstance(model_value, str) or not model_value.strip():
            raise TypeError("'model_info[\"model\"]' must be a non-empty string")

        # Validate rubric argument
        if "rubric" not in bound.arguments:
            raise TypeError("'rubric' argument is required")
        rubric_value = bound.arguments["rubric"]
        if not isinstance(rubric_value, str):
            raise TypeError(f"'rubric' must be a string, got {type(rubric_value).__name__}")

        # Validate messages argument
        if "messages" not in bound.arguments:
            raise TypeError("'messages' argument is required")
        messages_value = bound.arguments["messages"]
        if not isinstance(messages_value, list):
            raise TypeError(f"'messages' must be a list, got {type(messages_value).__name__}")

        # Validate optional ground_truth argument
        ground_truth_value = bound.arguments.get("ground_truth")
        if ground_truth_value is not None and not isinstance(ground_truth_value, str):
            raise TypeError(
                f"'ground_truth' must be a string or None, got {type(ground_truth_value).__name__}"
            )

        # Validate optional system_message argument
        system_message_value = bound.arguments.get("system_message")
        if system_message_value is not None and not isinstance(system_message_value, str):
            raise TypeError(
                f"'system_message' must be a string or None, got {type(system_message_value).__name__}"
            )

        # Validate messages structure
        for index, message in enumerate(messages_value):
            if not isinstance(message, dict):
                raise TypeError(f"'messages[{index}]' must be a dict, got {type(message).__name__}")
            missing_fields = {"type", "role", "content"} - message.keys()
            if missing_fields:
                raise ValueError(f"'messages[{index}]' is missing required fields: {missing_fields}")
            if message["role"] not in ALLOWED_ROLES:
                raise ValueError(
                    f"'messages[{index}]['role']' must be one of {sorted(ALLOWED_ROLES)}, "
                    f"got '{message['role']}'"
                )
            if not isinstance(message["content"], list):
                raise TypeError(f"'messages[{index}]['content']' must be a list")

        score_min_present = "score_min" in bound.arguments
        score_max_present = "score_max" in bound.arguments

        if score_min_present:
            score_min_value = bound.arguments["score_min"]
            if not _is_numeric(score_min_value):
                raise TypeError(
                    f"'score_min' must be a numeric type, got {type(score_min_value).__name__}"
                )
        else:
            score_min_value = None

        if score_max_present:
            score_max_value = bound.arguments["score_max"]
            if not _is_numeric(score_max_value):
                raise TypeError(
                    f"'score_max' must be a numeric type, got {type(score_max_value).__name__}"
                )
        else:
            score_max_value = None

        if score_min_present and score_max_present:
            if float(score_max_value) <= float(score_min_value):
                raise ValueError("'score_max' must be greater than 'score_min'")

        # Validate return type
        result = func(*args, **kwargs)
        if not isinstance(result, float):
            raise TypeError(f"Function {func.__name__} must return a float, got {type(result).__name__}")
        return result

    return wrapper
