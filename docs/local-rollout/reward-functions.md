# Reward Functions

Reward functions provide deterministic, code-based scoring for LLM outputs during training.

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

The decorator raises a `TypeError` if the function doesn't match this exact signature or doesn't return a float.

## Example

```python
@osmosis_reward
def exact_match(solution_str: str, ground_truth: str, extra_info: dict = None, **kwargs) -> float:
    """Simple exact string matching."""
    return 1.0 if solution_str.strip() == ground_truth.strip() else 0.0
```

For more examples, see:
- [osmosis-remote-rollout-example](https://github.com/Osmosis-AI/osmosis-remote-rollout-example) -- `rewards.py`
- [osmosis-git-sync-example](https://github.com/Osmosis-AI/osmosis-git-sync-example) -- `reward_fn/compute_reward.py`

## See Also

- [Reward Rubrics](./reward-rubrics.md) -- LLM-as-judge scoring with `@osmosis_rubric`
- [Local Rollout Overview](./overview.md) -- repository structure and setup
