"""
Reward function examples using the @osmosis_reward decorator.

This file demonstrates correct and incorrect usage of the decorator,
which enforces the signature: (solution_str: str, ground_truth: str, extra_info: dict = None) -> float
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'osmosis_ai'))

from utils import osmosis_reward


# CORRECT USAGE EXAMPLES

@osmosis_reward
def simple_exact_match(solution_str: str, ground_truth: str, extra_info: dict = None) -> float:
    """Basic exact match reward function."""
    return 1.0 if solution_str.strip() == ground_truth.strip() else 0.0


@osmosis_reward
def case_insensitive_match(solution_str: str, ground_truth: str, extra_info: dict = None) -> float:
    """Case-insensitive string matching with optional extra info."""
    match = solution_str.lower().strip() == ground_truth.lower().strip()

    # Use extra_info if provided
    if extra_info and 'partial_credit' in extra_info:
        if not match and extra_info['partial_credit']:
            # Give partial credit for similar length
            len_diff = abs(len(solution_str) - len(ground_truth))
            if len_diff <= 2:
                return 0.5

    return 1.0 if match else 0.0


@osmosis_reward
def numeric_tolerance(solution_str: str, ground_truth: str, extra_info: dict = None) -> float:
    """Numeric comparison with tolerance."""
    try:
        solution_num = float(solution_str.strip())
        truth_num = float(ground_truth.strip())

        tolerance = 0.01  # default
        if extra_info and 'tolerance' in extra_info:
            tolerance = extra_info['tolerance']

        return 1.0 if abs(solution_num - truth_num) <= tolerance else 0.0
    except ValueError:
        return 0.0


# Only two parameters (extra_info is optional)
@osmosis_reward
def minimal_reward(solution_str: str, ground_truth: str) -> float:
    """Minimal reward function with just required parameters."""
    return float(solution_str == ground_truth)


# INCORRECT USAGE EXAMPLES (these will raise TypeError)

# Uncomment these to see the validation errors:

# @osmosis_reward  # Wrong parameter names
# def wrong_names(input_str: str, expected: str, extra: dict = None):
#     return input_str == expected

# @osmosis_reward  # Missing type annotations
# def missing_annotations(solution_str, ground_truth, extra_info=None):
#     return solution_str == ground_truth

# @osmosis_reward  # Wrong type annotations
# def wrong_types(solution_str: int, ground_truth: str, extra_info: dict = None):
#     return str(solution_str) == ground_truth

# @osmosis_reward  # Too many parameters
# def too_many_params(solution_str: str, ground_truth: str, extra_info: dict = None, bonus: float = 0.0):
#     return (solution_str == ground_truth) + bonus

# @osmosis_reward  # Missing default value for extra_info
# def no_default(solution_str: str, ground_truth: str, extra_info: dict):
#     return solution_str == ground_truth


if __name__ == "__main__":
    # Test the reward functions
    test_cases = [
        ("hello", "hello", None),
        ("Hello", "hello", None),
        ("123.45", "123.44", {"tolerance": 0.1}),
        ("wrong", "right", {"partial_credit": True}),
    ]

    functions = [
        simple_exact_match,
        case_insensitive_match,
        numeric_tolerance,
        minimal_reward
    ]

    for func in functions:
        print(f"\n{func.__name__}:")
        for solution, truth, extra in test_cases:
            try:
                result = func(solution, truth, extra)
                print(f"  {solution!r} vs {truth!r} (extra={extra}) -> {result}")
            except Exception as e:
                print(f"  {solution!r} vs {truth!r} (extra={extra}) -> ERROR: {e}")