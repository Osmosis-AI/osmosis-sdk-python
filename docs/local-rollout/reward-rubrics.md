# Reward Rubrics

Rubric evaluations use an external LLM as a judge to score agent outputs against natural-language criteria. This is useful when correctness is subjective or hard to express as a deterministic function.

## @osmosis_rubric

Rubric functions decorated with `@osmosis_rubric` must match this signature:

```python
from osmosis_ai import osmosis_rubric

@osmosis_rubric
def your_rubric(solution_str: str, ground_truth: str | None, extra_info: dict) -> float:
    # Your rubric logic here
    return float_score
```

> **Note:** The runtime forwards `None` for `ground_truth` when no reference answer exists.

### Required `extra_info` Fields

- **`provider`** -- Non-empty string identifying the judge provider.
- **`model`** -- Non-empty string naming the provider model to call.
- **`rubric`** -- Natural-language rubric instructions for the judge model.
- **`api_key` / `api_key_env`** -- Supply either the raw key or the environment variable name that exposes it.

### Optional `extra_info` Fields

- **`system_prompt`** -- Optional string prepended to the provider's base system prompt.
- **`score_min` / `score_max`** -- Optional numeric overrides for the expected score range.
- **`model_info_overrides`** -- Optional dict merged into the provider configuration.

> **Note:** `extra_info` must be annotated as `dict` **without** a default value, unlike `@osmosis_reward`.

## evaluate_rubric()

Evaluate a solution against a natural-language rubric using an external LLM judge via [LiteLLM](https://github.com/BerriAI/litellm).

```python
from osmosis_ai import evaluate_rubric

score = evaluate_rubric(
    rubric="Assistant must mention the verified capital city.",
    solution_str=solution,
    model_info={
        "provider": "openai",
        "model": "gpt-5",
        "api_key_env": "OPENAI_API_KEY",
    },
    ground_truth="Paris",
    metadata=None,          # optional structured context quoted in the judge prompt
    return_details=False,   # True returns RewardRubricRunResult with score, explanation, raw
)
```

When `return_details=True`, returns a `RewardRubricRunResult` dict:

```python
class RewardRubricRunResult(TypedDict):
    score: float        # The clamped rubric score
    explanation: str    # The judge model's explanation
    raw: Any            # Full raw response from the LLM provider
```

### Credentials

Credentials are resolved in order: `api_key` (direct) > `api_key_env` (env var name) > provider default:

| Provider | Default Environment Variable |
|----------|------------------------------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` |
| xAI | `XAI_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Cerebras | `CEREBRAS_API_KEY` |

Any provider supported by LiteLLM can be used without additional configuration beyond setting the appropriate API key environment variable.

## Example

```python
from osmosis_ai import osmosis_rubric, evaluate_rubric

@osmosis_rubric
def compute_rubric_score(solution_str: str, ground_truth: str | None, extra_info: dict) -> float:
    return evaluate_rubric(
        rubric="Evaluate whether the solution correctly matches the expected answer.",
        solution_str=solution_str,
        ground_truth=ground_truth,
        model_info={
            "provider": "openai",
            "model": "gpt-5-mini",
            "api_key_env": "OPENAI_API_KEY",
        },
    )
```

## Error Types

| Exception | Description |
|-----------|-------------|
| `MissingAPIKeyError` | Required API key not found in environment. Message explains which env var to export. |
| `ProviderRequestError` | LLM provider returned an error (auth, rate limit, timeout, invalid response). |
| `ModelNotFoundError` | Model identifier not recognized by the provider. Subclass of `ProviderRequestError`. |

## File Placement

In a Local Rollout repository, place rubric functions in the `reward_rubric/` directory. Osmosis discovers and syncs all `@osmosis_rubric`-decorated functions from this directory. See the [osmosis-git-sync-example](https://github.com/Osmosis-AI/osmosis-git-sync-example) for working examples.

## See Also

- [Reward Functions](./reward-functions.md) -- deterministic scoring with `@osmosis_reward`
- [Local Rollout Overview](./overview.md) -- repository structure and setup
