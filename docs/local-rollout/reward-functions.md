# Reward Functions

Reward functions provide deterministic, code-based scoring for LLM outputs during training. They are used in Local Rollout mode to drive reinforcement learning -- the training loop calls your reward function after each rollout to compute a scalar score.

## @osmosis_reward

All functions decorated with `@osmosis_reward` must have exactly this signature:

```python
from osmosis_ai import osmosis_reward

@osmosis_reward
def your_function(solution_str: str, ground_truth: str, extra_info: dict = None, **kwargs) -> float:
    # Your reward logic here
    return float_score
```

### Parameters

- **`solution_str: str`** -- The solution string to evaluate (required).
- **`ground_truth: str`** -- The correct/expected answer (required).
- **`extra_info: dict = None`** -- Optional dictionary for additional configuration.
- **`**kwargs`** -- Required for platform compatibility.

### Return Value

- **`-> float`** -- Must return a float value representing the reward score.

The decorator raises a `TypeError` if the function doesn't match this exact signature or doesn't return a float.

## Examples

### Exact Match

```python
@osmosis_reward
def exact_match(solution_str: str, ground_truth: str, extra_info: dict = None, **kwargs) -> float:
    """Simple exact string matching."""
    return 1.0 if solution_str.strip() == ground_truth.strip() else 0.0
```

### Case-Insensitive Match with Partial Credit

```python
@osmosis_reward
def case_insensitive_match(solution_str: str, ground_truth: str, extra_info: dict = None, **kwargs) -> float:
    """Case-insensitive string matching with partial credit."""
    match = solution_str.lower().strip() == ground_truth.lower().strip()

    if extra_info and 'partial_credit' in extra_info:
        if not match and extra_info['partial_credit']:
            len_diff = abs(len(solution_str) - len(ground_truth))
            if len_diff <= 2:
                return 0.5

    return 1.0 if match else 0.0
```

### Numeric Tolerance

```python
@osmosis_reward
def numeric_tolerance(solution_str: str, ground_truth: str, extra_info: dict = None, **kwargs) -> float:
    """Numeric comparison with configurable tolerance."""
    try:
        solution_num = float(solution_str.strip())
        truth_num = float(ground_truth.strip())

        tolerance = extra_info.get('tolerance', 0.01) if extra_info else 0.01
        return 1.0 if abs(solution_num - truth_num) <= tolerance else 0.0
    except ValueError:
        return 0.0
```

### Extracting Solutions from Agent Output

A common pattern for tool-use agents is extracting a final numeric answer from a markdown-formatted response:

```python
import re
from osmosis_ai import osmosis_reward

def extract_solution(solution_str):
    """Extract the first number after a #### heading."""
    solution = re.search(r'####\s*([-+]?\d*\.?\d+)', solution_str)
    if not solution:
        return None
    return solution.group(1)

@osmosis_reward
def numbers_match_reward(solution_str: str, ground_truth: str, extra_info: dict = None, **kwargs) -> float:
    extracted = extract_solution(solution_str)
    try:
        sol_val = float(extracted)
    except (TypeError, ValueError):
        return 0.0

    gt_val = float(ground_truth)
    if abs(gt_val - sol_val) < 1e-7:
        return 1.0
    return 0.0
```

This example is from the [osmosis-git-sync-example](https://github.com/Osmosis-AI/osmosis-git-sync-example) repository (`reward_fn/compute_reward.py`).

## File Placement

In a Local Rollout repository, place reward functions in the `reward_fn/` directory:

```
reward_fn/
└── compute_reward.py    # Contains @osmosis_reward functions
```

Osmosis discovers and syncs all `@osmosis_reward`-decorated functions from this directory.

## See Also

- [Reward Rubrics](./reward-rubrics.md) -- LLM-as-judge scoring with `@osmosis_rubric`
- [Rewards API Reference](../rewards-api.md) -- full API reference for decorators and `evaluate_rubric`
- [Local Rollout Overview](./overview.md) -- repository structure and setup
- [Example Code](../../examples/) -- reward functions, rubric configs, sample data
