# Rewards API Reference

API reference for reward function and rubric evaluation decorators. These are used in both Local Rollout and Remote Rollout modes for scoring LLM outputs during training.

## @osmosis_reward

Decorator for deterministic reward functions. Validates the function signature at decoration time and ensures a `float` return value at call time.

### Required Signature

```python
from osmosis_ai import osmosis_reward

@osmosis_reward
def your_function(solution_str: str, ground_truth: str, extra_info: dict = None, **kwargs) -> float:
    return float_score
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `solution_str` | `str` | Yes | The solution string to evaluate |
| `ground_truth` | `str` | Yes | The correct/expected answer |
| `extra_info` | `dict` | No (default `None`) | Optional dictionary for additional configuration |
| `**kwargs` | - | Yes | Required for platform compatibility |

### Return Value

Must return a `float`. The decorator raises `TypeError` if the return type check fails.

### Validation Rules

- Function must accept exactly `solution_str`, `ground_truth`, `extra_info` as positional parameters
- `extra_info` must have a default value of `None`
- Must include `**kwargs` for platform compatibility
- Return value must be a `float` (checked at call time)

---

## @osmosis_rubric

Decorator for LLM-as-judge rubric functions. Validates the function signature and ensures required `extra_info` fields are present at call time.

### Required Signature

```python
from osmosis_ai import osmosis_rubric

@osmosis_rubric
def your_rubric(solution_str: str, ground_truth: str | None, extra_info: dict) -> float:
    return float_score
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `solution_str` | `str` | Yes | The solution string to evaluate |
| `ground_truth` | `str \| None` | Yes | The correct/expected answer (may be `None`) |
| `extra_info` | `dict` | Yes (no default) | Configuration dict with provider, model, rubric |

### Required `extra_info` Fields

| Field | Type | Description |
|-------|------|-------------|
| `provider` | `str` | Non-empty string identifying the judge provider |
| `model` | `str` | Non-empty string naming the provider model |
| `rubric` | `str` | Natural-language rubric instructions |
| `api_key` or `api_key_env` | `str` | Raw API key or environment variable name |

### Optional `extra_info` Fields

| Field | Type | Description |
|-------|------|-------------|
| `system_prompt` | `str` | Prepended to the provider's base system prompt |
| `score_min` | `float` | Override minimum score bound |
| `score_max` | `float` | Override maximum score bound |
| `model_info_overrides` | `dict` | Merged into provider configuration |

### Validation Rules

- `extra_info` must be annotated as `dict` **without** a default value
- `ground_truth` may be `None` (annotate as `Optional[str]` if needed)
- Return type must be `float`

---

## evaluate_rubric()

Evaluate a solution against a natural-language rubric using an external LLM judge via [LiteLLM](https://github.com/BerriAI/litellm).

### Signature

```python
from osmosis_ai import evaluate_rubric

score = evaluate_rubric(
    rubric: str,
    solution_str: str,
    model_info: dict,
    ground_truth: str = None,
    metadata: dict = None,
    return_details: bool = False,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rubric` | `str` | required | Natural-language rubric instructions for the judge |
| `solution_str` | `str` | required | The solution to evaluate |
| `model_info` | `dict` | required | Provider configuration (see below) |
| `ground_truth` | `str` | `None` | Optional reference answer |
| `metadata` | `dict` | `None` | Optional structured context quoted in the judge prompt |
| `return_details` | `bool` | `False` | Return full `RewardRubricRunResult` payload |

### `model_info` Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `provider` | `str` | Yes | Provider name (e.g., `"openai"`, `"anthropic"`) |
| `model` | `str` | Yes | Model name (e.g., `"gpt-5"`, `"claude-sonnet-4-5-20250929"`) |
| `api_key` | `str` | No | Raw API key (mutually exclusive with `api_key_env`) |
| `api_key_env` | `str` | No | Environment variable name for API key |
| `score_min` | `float` | No | Minimum score bound (default `0.0`) |
| `score_max` | `float` | No | Maximum score bound (default `1.0`) |
| `system_prompt` | `str` | No | Optional context prepended to judge prompt |
| `original_input` | `str` | No | Optional original user input for context |
| `timeout` | `float` | No | Provider timeout in seconds |
| `reasoning_effort` | `str \| None` | No | Reasoning effort hint passed to the provider (e.g., `"low"`, `"medium"`, `"high"`). Silently dropped for models that do not support it. |

### Return Value

- When `return_details=False`: Returns a `float` score clamped to `[score_min, score_max]`
- When `return_details=True`: Returns a `RewardRubricRunResult` dict with the following structure:

```python
class RewardRubricRunResult(TypedDict):
    score: float        # The clamped rubric score
    explanation: str    # The judge model's explanation for the score
    raw: Any            # The full raw response from the LLM provider
```

### Credential Resolution

Credentials are resolved in this order:

1. `api_key` in `model_info` (used directly if present and non-empty)
2. `api_key_env` in `model_info` (environment variable name to look up)
3. Provider default environment variable:

| Provider | Default Environment Variable |
|----------|------------------------------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` |
| xAI | `XAI_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Cerebras | `CEREBRAS_API_KEY` |

### Provider Architecture

All provider routing is handled by [LiteLLM](https://github.com/BerriAI/litellm). Any provider supported by LiteLLM can be used without additional configuration beyond setting the appropriate API key environment variable.

Every provider returns a strict JSON object with `{"score": number, "explanation": string}`. The helper clamps the score into the configured range and validates the structure.

---

## Error Types

### MissingAPIKeyError

Raised when the required API key is not found in the environment.

```python
from osmosis_ai import MissingAPIKeyError

try:
    score = evaluate_rubric(...)
except MissingAPIKeyError as e:
    print(f"Missing key: {e}")
    # Message explains which env var to export
```

### ProviderRequestError

Raised when the LLM provider returns an error.

```python
from osmosis_ai import ProviderRequestError

try:
    score = evaluate_rubric(...)
except ProviderRequestError as e:
    print(f"Provider error: {e}")
```

### ModelNotFoundError

Raised when the specified model identifier is not recognized by the provider. Subclass of `ProviderRequestError`.

```python
from osmosis_ai import ModelNotFoundError

try:
    score = evaluate_rubric(...)
except ModelNotFoundError as e:
    print(f"Model not found: {e}")
    # Check provider dashboard for latest model names
```

> Provider model snapshot names change frequently. Check each vendor's dashboard for the latest identifier if you encounter a "model not found" error.

---

## See Also

- [Reward Functions Guide](./local-rollout/reward-functions.md) -- usage guide with examples
- [Reward Rubrics Guide](./local-rollout/reward-rubrics.md) -- usage guide with provider examples
- [CLI Reference](./cli.md) -- `osmosis preview` and `osmosis eval-rubric` commands
