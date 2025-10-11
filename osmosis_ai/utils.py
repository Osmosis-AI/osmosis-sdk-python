
import functools
import inspect
import types
from typing import Any, Callable, Union, get_args, get_origin


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


ALLOWED_ROLES = {"user", "system", "assistant", "developer"}


def _is_optional_str(annotation: Any) -> bool:
    if annotation is str:
        return True
    origin = get_origin(annotation)
    if origin in {Union, types.UnionType}:
        args = tuple(arg for arg in get_args(annotation) if arg is not type(None))  # noqa: E721
        return len(args) == 1 and args[0] is str
    if isinstance(annotation, type):
        return issubclass(annotation, str)
    return False


def _is_list_annotation(annotation: Any) -> bool:
    if annotation is list:
        return True
    origin = get_origin(annotation)
    return origin is list


def osmosis_rubric(func: Callable) -> Callable:
    """
    Decorator for rubric functions that enforces the signature:
    (rubric: str, messages: list, ground_truth: Optional[str] = None,
     system_message: Optional[str] = None, extra_info: dict = None) -> float

    Args:
        func: The rubric function to be wrapped

    Returns:
        The wrapped function

    Raises:
        TypeError: If the function doesn't have the required signature or doesn't return a float

    Example:
        @osmosis_rubric
        def evaluate_response(
            rubric: str,
            messages: list,
            ground_truth: str | None = None,
            system_message: str | None = None,
            extra_info: dict = None,
        ) -> float:
            return some_evaluation(messages, ground_truth)
    """
    # Validate function signature
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    # Check parameter count
    if len(params) < 2 or len(params) > 5:
        raise TypeError(f"Function {func.__name__} must have between 2 and 5 parameters, got {len(params)}")

    # Check first parameter: rubric: str
    rubric_param = params[0]
    if rubric_param.name != "rubric":
        raise TypeError(f"First parameter must be named 'rubric', got '{rubric_param.name}'")
    if rubric_param.annotation != str:
        raise TypeError(f"First parameter 'rubric' must be annotated as str, got {rubric_param.annotation}")
    if rubric_param.default is not inspect.Parameter.empty:
        raise TypeError("First parameter 'rubric' cannot have a default value")

    # Check second parameter: messages: list
    messages_param = params[1]
    if messages_param.name != "messages":
        raise TypeError(f"Second parameter must be named 'messages', got '{messages_param.name}'")
    if messages_param.annotation is inspect.Parameter.empty:
        raise TypeError("Second parameter 'messages' must be annotated as list")
    if not _is_list_annotation(messages_param.annotation):
        raise TypeError(f"Second parameter 'messages' must be annotated as list, got {messages_param.annotation}")
    if messages_param.default is not inspect.Parameter.empty:
        raise TypeError("Second parameter 'messages' cannot have a default value")

    optional_params = params[2:]

    if optional_params:
        ground_truth_param = optional_params[0]
        # Check third parameter: ground_truth: Optional[str]
        if ground_truth_param.name != "ground_truth":
            raise TypeError(f"Third parameter must be named 'ground_truth', got '{ground_truth_param.name}'")
        if ground_truth_param.annotation is inspect.Parameter.empty or not _is_optional_str(ground_truth_param.annotation):
            raise TypeError(
                "Third parameter 'ground_truth' must be annotated as Optional[str] or str"
            )
        if ground_truth_param.default is inspect.Parameter.empty:
            raise TypeError("Third parameter 'ground_truth' must have a default value of None")
        if ground_truth_param.default is not None:
            raise TypeError("Third parameter 'ground_truth' must default to None")
        optional_params = optional_params[1:]

    if optional_params:
        system_message_param = optional_params[0]
        # Check fourth parameter: system_message: Optional[str]
        if system_message_param.name != "system_message":
            raise TypeError(f"Fourth parameter must be named 'system_message', got '{system_message_param.name}'")
        if system_message_param.annotation is inspect.Parameter.empty or not _is_optional_str(system_message_param.annotation):
            raise TypeError(
                "Fourth parameter 'system_message' must be annotated as Optional[str] or str"
            )
        if system_message_param.default is inspect.Parameter.empty:
            raise TypeError("Fourth parameter 'system_message' must have a default value of None")
        if system_message_param.default is not None:
            raise TypeError("Fourth parameter 'system_message' must default to None")
        optional_params = optional_params[1:]

    if optional_params:
        extra_info_param = optional_params[0]
        # Check fifth parameter: extra_info: dict = None
        if extra_info_param.name != "extra_info":
            raise TypeError(f"Fifth parameter must be named 'extra_info', got '{extra_info_param.name}'")
        if extra_info_param.annotation != dict:
            raise TypeError(f"Fifth parameter 'extra_info' must be annotated as dict, got {extra_info_param.annotation}")
        if extra_info_param.default is inspect.Parameter.empty:
            raise TypeError("Fifth parameter 'extra_info' must have a default value of None")
        if extra_info_param.default is not None:
            raise TypeError("Fifth parameter 'extra_info' must default to None")
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

        # Validate return type
        result = func(*args, **kwargs)
        if not isinstance(result, float):
            raise TypeError(f"Function {func.__name__} must return a float, got {type(result).__name__}")
        return result

    return wrapper
