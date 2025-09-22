
import functools
import inspect
from typing import Callable


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
        result = func(*args, **kwargs)
        if not isinstance(result, float):
            raise TypeError(f"Function {func.__name__} must return a float, got {type(result).__name__}")
        return result

    return wrapper
