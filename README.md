# osmosis-ai

A Python library that provides reward and rubric validation helpers for LLM applications with strict type enforcement.

## Installation

```bash
pip install osmosis-ai
```

For development:
```bash
git clone https://github.com/Osmosis-AI/osmosis-sdk-python
cd osmosis-sdk-python
pip install -e .
```

## Quick Start

```python
from osmosis_ai import osmosis_reward

@osmosis_reward
def simple_reward(solution_str: str, ground_truth: str, extra_info: dict = None) -> float:
    """Basic exact match reward function."""
    return 1.0 if solution_str.strip() == ground_truth.strip() else 0.0

# Use the reward function
score = simple_reward("hello world", "hello world")  # Returns 1.0
```

```python
from osmosis_ai import evaluate_rubric

messages = [
    {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "What is the capital of France?"}],
    },
    {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "The capital of France is Paris."}],
    },
]

# Export OPENAI_API_KEY in your shell before running this snippet.
rubric_score = evaluate_rubric(
    rubric="Assistant must mention the verified capital city.",
    messages=messages,
    model_info={
        "provider": "openai",
        "model": "gpt-5",
        "api_key_env": "OPENAI_API_KEY",
    },
    ground_truth="Paris",
)

print(rubric_score)  # -> 1.0 (full payload available via return_details=True)
```

## Remote Rubric Evaluation

`evaluate_rubric` talks to each provider through its official Python SDK while enforcing the same JSON schema everywhere:

- **OpenAI / xAI** – Uses `OpenAI(...).responses.create` (or `chat.completions.create`) with `response_format={"type": "json_schema"}` and falls back to `json_object` when needed.
- **Anthropic** – Forces a tool call with a JSON schema via `Anthropic(...).messages.create`, extracting the returned tool arguments.
- **Google Gemini** – Invokes `google.genai.Client(...).models.generate_content` with `response_mime_type="application/json"` and `response_schema`.

Every provider therefore returns a strict JSON object with `{"score": number, "explanation": string}`. The helper clamps the score into your configured range, validates the structure, and exposes the raw payload when `return_details=True`.

Credentials are resolved from environment variables by default:

- `OPENAI_API_KEY` for OpenAI
- `ANTHROPIC_API_KEY` for Anthropic
- `GOOGLE_API_KEY` for Google Gemini
- `XAI_API_KEY` for xAI

Override the environment variable name with `model_info={"api_key_env": "CUSTOM_ENV_NAME"}` when needed, or supply an inline secret with `model_info={"api_key": "sk-..."}` for ephemeral credentials. Missing API keys raise a `MissingAPIKeyError` that explains how to export the secret before trying again.

`model_info` accepts additional rubric-specific knobs:

- `score_min` / `score_max` – change the default `[0.0, 1.0]` scoring bounds.
- `system_prompt` / `original_input` – override the helper’s transcript inference when those entries are absent.
- `timeout` – customise the provider timeout in seconds.

Pass `extra_info={...}` to `evaluate_rubric` when you need structured context quoted in the judge prompt, and set `return_details=True` to receive the full `RewardRubricRunResult` payload (including the provider’s raw response).

Remote failures surface as `ProviderRequestError` instances, with `ModelNotFoundError` reserved for missing model identifiers so you can retry with a new snapshot.

> Older SDK versions that lack schema parameters automatically fall back to instruction-only JSON; the helper still validates the response payload before returning.
> Provider model snapshot names change frequently. Check each vendor's dashboard for the latest identifier if you encounter a “model not found” error.

### Provider Architecture

All remote integrations live in `osmosis_ai/providers/` and implement the `RubricProvider` interface. At import time the default registry registers OpenAI, xAI, Anthropic, and Google Gemini so `evaluate_rubric` can route requests without additional configuration. The request/response plumbing is encapsulated in each provider module, keeping `evaluate_rubric` focused on prompt construction, payload validation, and credential resolution.

Add your own provider by subclassing `RubricProvider`, implementing `run()` with the vendor SDK, and calling `register_provider()` during start-up. A step-by-step guide is available in [`osmosis_ai/providers/README.md`](osmosis_ai/providers/README.md).

## Required Function Signature

All functions decorated with `@osmosis_reward` must have exactly this signature:

```python
@osmosis_reward
def your_function(solution_str: str, ground_truth: str, extra_info: dict = None) -> float:
    # Your reward logic here
    return float_score
```

### Parameters

- **`solution_str: str`** - The solution string to evaluate (required)
- **`ground_truth: str`** - The correct/expected answer (required)
- **`extra_info: dict = None`** - Optional dictionary for additional configuration

### Return Value

- **`-> float`** - Must return a float value representing the reward score

The decorator will raise a `TypeError` if the function doesn't match this exact signature or doesn't return a float.

## Rubric Function Signature

Rubric functions decorated with `@osmosis_rubric` must accept the parameters:

- `model_info: dict`
- `rubric: str`
- `messages: list`
- `ground_truth: Optional[str] = None`
- `system_message: Optional[str] = None`
- `extra_info: dict = None`
- `score_min: float = 0.0` *(optional lower bound; must default to 0.0 and stay below `score_max`)*
- `score_max: float = 1.0` *(optional upper bound; must default to 1.0 and stay above `score_min`)*

and must return a `float`. The decorator validates the signature and runtime payload (including message role validation and return type) before delegating to your custom logic.

> Required fields: `model_info` must contain non-empty `provider` and `model` string entries.

> Annotation quirk: `extra_info` must be annotated as a plain `dict` with a default of `None` to satisfy the validator.

> Tip: You can call `evaluate_rubric` from inside a rubric function (or any other orchestrator) to outsource judging to a hosted model while still benefiting from the decorator’s validation.

## Examples

See the [`examples/`](examples/) directory for complete examples:

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

- `examples/rubric_functions.py` demonstrates `evaluate_rubric` with OpenAI, Anthropic, Gemini, and xAI using the schema-enforced SDK integrations.
- `examples/reward_functions.py` keeps local reward helpers that showcase the decorator contract without external calls.

## Running Examples

```bash
PYTHONPATH=. python examples/reward_functions.py
PYTHONPATH=. python examples/rubric_functions.py  # Uncomment the provider you need before running
```

## Testing

Run `python -m pytest tests/test_rubric_eval.py` to exercise the guards that ensure rubric prompts ignore message metadata (for example `tests/test_rubric_eval.py::test_collect_text_skips_metadata_fields`) while still preserving nested tool output. Add additional tests under `tests/` as you extend the library.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and examples
5. Submit a pull request

## Links

- [Homepage](https://github.com/Osmosis-AI/osmosis-sdk-python)
- [Issues](https://github.com/Osmosis-AI/osmosis-sdk-python/issues)
