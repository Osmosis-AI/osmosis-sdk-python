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

> **Note:** The runtime forwards `None` for `ground_truth` when no reference answer exists. Annotate the parameter as `Optional[str]` (or handle `None` explicitly) if your rubric logic expects to run in that scenario.

### Required `extra_info` Fields

- **`provider`** -- Non-empty string identifying the judge provider.
- **`model`** -- Non-empty string naming the provider model to call.
- **`rubric`** -- Natural-language rubric instructions for the judge model.
- **`api_key` / `api_key_env`** -- Supply either the raw key or the environment variable name that exposes it.

### Optional `extra_info` Fields

- **`system_prompt`** -- Optional string prepended to the provider's base system prompt.
- **`score_min` / `score_max`** -- Optional numeric overrides for the expected score range.
- **`model_info_overrides`** -- Optional dict merged into the provider configuration.

Additional keys are passthrough and can be used for custom configuration. The decorator enforces the parameter names/annotations, validates the embedded configuration at call time, and ensures the wrapped function returns a `float`.

> **Note:** Annotation quirk: `extra_info` must be annotated as `dict` **without** a default value, unlike `@osmosis_reward`.

## evaluate_rubric

`evaluate_rubric` talks to hosted LLM providers through [LiteLLM](https://github.com/BerriAI/litellm), a unified interface supporting 100+ providers. Every provider returns a strict JSON object with `{"score": number, "explanation": string}`.

### Basic Usage

```python
from osmosis_ai import evaluate_rubric

solution = "The capital of France is Paris."

rubric_score = evaluate_rubric(
    rubric="Assistant must mention the verified capital city.",
    solution_str=solution,
    model_info={
        "provider": "openai",
        "model": "gpt-5",
        "api_key_env": "OPENAI_API_KEY",
    },
    ground_truth="Paris",
)

print(rubric_score)  # -> 1.0 (full payload available via return_details=True)
```

### Credentials

Credentials are resolved from environment variables by default:

- `OPENAI_API_KEY` for OpenAI
- `ANTHROPIC_API_KEY` for Anthropic
- `GEMINI_API_KEY` for Google Gemini
- `XAI_API_KEY` for xAI
- `OPENROUTER_API_KEY` for OpenRouter
- `CEREBRAS_API_KEY` for Cerebras

Override the environment variable name with `model_info={"api_key_env": "CUSTOM_ENV_NAME"}` when needed, or supply an inline secret with `model_info={"api_key": "sk-..."}` for ephemeral credentials.

`api_key` and `api_key_env` are mutually exclusive. When `api_key` is present and non-empty it is used directly, skipping any environment lookup. Otherwise the resolver falls back to `api_key_env` (or the provider default) and pulls the value from your local environment with `os.getenv`.

## Provider Examples

### OpenAI

```python
from osmosis_ai import osmosis_rubric, evaluate_rubric

@osmosis_rubric
def compute_rubric_score_openai(solution_str: str, ground_truth: str | None, extra_info: dict) -> float:
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

### Anthropic

```python
@osmosis_rubric
def compute_rubric_score_anthropic(solution_str: str, ground_truth: str | None, extra_info: dict) -> float:
    return evaluate_rubric(
        rubric="Evaluate whether the solution correctly matches the expected answer.",
        solution_str=solution_str,
        ground_truth=ground_truth,
        model_info={
            "provider": "anthropic",
            "model": "claude-sonnet-4-5-20250929",
            "api_key_env": "ANTHROPIC_API_KEY",
        },
    )
```

Any provider supported by LiteLLM can be used without additional configuration beyond setting the appropriate API key environment variable.

## File Placement

In a Local Rollout repository, place rubric functions in the `reward_rubric/` directory:

```
reward_rubric/
├── reward_rubric_openai.py
├── reward_rubric_anthropic.py
└── reward_rubric_xai.py
```

Osmosis discovers and syncs all `@osmosis_rubric`-decorated functions from this directory. See the [osmosis-git-sync-example](https://github.com/Osmosis-AI/osmosis-git-sync-example) for working examples.

## See Also

- [Reward Functions](./reward-functions.md) -- deterministic scoring with `@osmosis_reward`
- [Rewards API Reference](../rewards-api.md) -- full API reference for decorators and `evaluate_rubric`
- [Local Rollout Overview](./overview.md) -- repository structure and setup
