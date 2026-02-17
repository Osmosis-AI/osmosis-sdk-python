# Rewards & Rubrics

The Osmosis SDK provides two mechanisms for scoring LLM outputs during training: **reward functions** for deterministic, code-based scoring, and **rubric evaluations** for LLM-as-judge assessments powered by external providers. Both integrate with the Osmosis training platform to drive reinforcement learning workflows.

## @osmosis_reward

All functions decorated with `@osmosis_reward` must have exactly this signature:

```python
from osmosis_ai import osmosis_reward

@osmosis_reward
def your_function(solution_str: str, ground_truth: str, extra_info: dict = None) -> float:
    # Your reward logic here
    return float_score
```

### Parameters

- **`solution_str: str`** -- The solution string to evaluate (required).
- **`ground_truth: str`** -- The correct/expected answer (required).
- **`extra_info: dict = None`** -- Optional dictionary for additional configuration.

### Return Value

- **`-> float`** -- Must return a float value representing the reward score.

The decorator will raise a `TypeError` if the function doesn't match this exact signature or doesn't return a float.

## @osmosis_rubric

Rubric functions decorated with `@osmosis_rubric` must match this signature:

```python
from osmosis_ai import osmosis_rubric

@osmosis_rubric
def your_rubric(solution_str: str, ground_truth: str | None, extra_info: dict) -> float:
    # Your rubric logic here
    return float_score
```

> The runtime forwards `None` for `ground_truth` when no reference answer exists. Annotate the parameter as `Optional[str]` (or handle `None` explicitly) if your rubric logic expects to run in that scenario.

### Required `extra_info` fields

- **`provider`** -- Non-empty string identifying the judge provider.
- **`model`** -- Non-empty string naming the provider model to call.
- **`rubric`** -- Natural-language rubric instructions for the judge model.
- **`api_key` / `api_key_env`** -- Supply either the raw key or the environment variable name that exposes it.

### Optional `extra_info` fields

- **`system_prompt`** -- Optional string prepended to the provider's base system prompt when invoking the judge; include it inside `extra_info` rather than as a separate argument.
- **`score_min` / `score_max`** -- Optional numeric overrides for the expected score range.
- **`model_info_overrides`** -- Optional dict merged into the provider configuration passed to the judge.

Additional keys are passthrough and can be used for custom configuration. If you need to extend the provider payload (for example adding `api_key_env`), add a dict under `model_info_overrides` and it will be merged with the required `provider`/`model` pair before invoking `evaluate_rubric`. The decorator enforces the parameter names/annotations, validates the embedded configuration at call time, and ensures the wrapped function returns a `float`.

> Annotation quirk: `extra_info` must be annotated as `dict` **without** a default value, unlike `@osmosis_reward`.

> Tip: When delegating to `evaluate_rubric`, pass the raw `solution_str` directly and include any extra context inside the `metadata` payload.

## evaluate_rubric

`evaluate_rubric` talks to hosted LLM providers through [LiteLLM](https://github.com/BerriAI/litellm), a unified interface supporting 100+ providers (OpenAI, Anthropic, Google Gemini, xAI, OpenRouter, Cerebras, Azure, Bedrock, Vertex AI, and more). Every provider returns a strict JSON object with `{"score": number, "explanation": string}`. The helper clamps the score into your configured range, validates the structure, and exposes the raw payload when `return_details=True`.

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

Override the environment variable name with `model_info={"api_key_env": "CUSTOM_ENV_NAME"}` when needed, or supply an inline secret with `model_info={"api_key": "sk-..."}` for ephemeral credentials. Missing API keys raise a `MissingAPIKeyError` that explains how to export the secret before trying again.

`api_key` and `api_key_env` are mutually exclusive ways to provide the same credential. When `api_key` is present and non-empty it is used directly, skipping any environment lookup. Otherwise the resolver falls back to `api_key_env` (or the provider default) and pulls the value from your local environment with `os.getenv`.

### Configuration Options

`model_info` accepts additional rubric-specific knobs:

- **`score_min` / `score_max`** -- Change the default `[0.0, 1.0]` scoring bounds.
- **`system_prompt` / `original_input`** -- Provide optional context strings that will be quoted in the judging prompt.
- **`timeout`** -- Customise the provider timeout in seconds.

Pass `metadata={...}` to `evaluate_rubric` when you need structured context quoted in the judge prompt, and set `return_details=True` to receive the full `RewardRubricRunResult` payload (including the provider's raw response).

### Provider Architecture

All provider routing is handled by [LiteLLM](https://github.com/BerriAI/litellm). To use any supported provider, pass its name and model in `model_info`:

```python
result = evaluate_rubric(
    rubric="...",
    solution_str="...",
    model_info={"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
)
```

Any provider supported by LiteLLM can be used without additional configuration beyond setting the appropriate API key environment variable.

### Error Handling

Remote failures surface as `ProviderRequestError` instances, with `ModelNotFoundError` reserved for missing model identifiers so you can retry with a new snapshot.

> Provider model snapshot names change frequently. Check each vendor's dashboard for the latest identifier if you encounter a "model not found" error.

## Examples

See the [`examples/`](../examples/) directory for complete examples:

```python
@osmosis_reward
def case_insensitive_match(solution_str: str, ground_truth: str, extra_info: dict = None) -> float:
    """Case-insensitive string matching with partial credit."""
    match = solution_str.lower().strip() == ground_truth.lower().strip()

    if extra_info and 'partial_credit' in extra_info:
        if not match and extra_info['partial_credit']:
            len_diff = abs(len(solution_str) - len(ground_truth))
            if len_diff <= 2:
                return 0.5

    return 1.0 if match else 0.0

@osmosis_reward
def numeric_tolerance(solution_str: str, ground_truth: str, extra_info: dict = None) -> float:
    """Numeric comparison with configurable tolerance."""
    try:
        solution_num = float(solution_str.strip())
        truth_num = float(ground_truth.strip())

        tolerance = extra_info.get('tolerance', 0.01) if extra_info else 0.01
        return 1.0 if abs(solution_num - truth_num) <= tolerance else 0.0
    except ValueError:
        return 0.0
```

- `examples/reward_functions.py` keeps local reward helpers that showcase the decorator contract without external calls.
- `examples/rubric_functions.py` demonstrates `evaluate_rubric` with OpenAI, Anthropic, Gemini, xAI, OpenRouter, and Cerebras via LiteLLM's unified interface.
- `examples/rubric_configs.yaml` bundles two rubric definitions with provider configuration and scoring bounds.
- `examples/sample_data.jsonl` contains two rubric-aligned solution strings so you can trial dataset validation.

```yaml
# examples/rubric_configs.yaml (excerpt)
version: 1
rubrics:
  - id: support_followup
    model_info:
      provider: openai
      model: gpt-5-mini
      api_key_env: OPENAI_API_KEY
```

```jsonl
{"conversation_id": "ticket-001", "rubric_id": "support_followup", "original_input": "...", "solution_str": "..."}
{"conversation_id": "ticket-047", "rubric_id": "policy_grounding", "original_input": "...", "solution_str": "..."}
```

## See Also

- [CLI Reference](./cli.md) -- `osmosis preview` and `osmosis eval-rubric` commands
- [Rollout Evaluation](./rollout/eval.md) -- evaluating agents against datasets
