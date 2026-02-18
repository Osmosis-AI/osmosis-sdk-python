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

## Example

```python
@osmosis_reward
def exact_match(solution_str: str, ground_truth: str, extra_info: dict = None, **kwargs) -> float:
    """Simple exact string matching."""
    return 1.0 if solution_str.strip() == ground_truth.strip() else 0.0
```

For more examples (numeric tolerance, solution extraction, partial credit), see:
- [osmosis-remote-rollout-example](https://github.com/Osmosis-AI/osmosis-remote-rollout-example) -- `rewards.py`
- [osmosis-git-sync-example](https://github.com/Osmosis-AI/osmosis-git-sync-example) -- `reward_fn/compute_reward.py`

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
