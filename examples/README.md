# Examples

This directory contains example usage of the osmosis-ai library.

## Reward Functions (`reward_functions.py`)

Demonstrates how to use the `@osmosis_reward` decorator with various reward function patterns:

- **Simple exact match**: Basic string comparison
- **Case-insensitive match**: Flexible string matching with optional partial credit
- **Numeric tolerance**: Float comparison with configurable tolerance
- **Minimal reward**: Using only the required parameters

### Required Function Signature

All functions decorated with `@osmosis_reward` must have this exact signature:

```python
@osmosis_reward
def your_function(solution_str: str, ground_truth: str, extra_info: dict = None) -> float:
    # Your reward logic here
    return float_score
```

### Parameters

- `solution_str: str` - The solution string to evaluate
- `ground_truth: str` - The correct/expected answer
- `extra_info: dict = None` - Optional dictionary for additional configuration

### Return Value

- `-> float` - Must return a float value representing the reward score

### Running Examples

```bash
cd examples
PYTHONPATH=.. python reward_functions.py
```

This will run test cases through all the example reward functions and show their outputs.

## Remote Rubric Evaluation (`rubric_functions.py`)

Shows how to call `osmosis_ai.evaluate_rubric` against different hosted judge providers using their official Python SDKs. The example:

- Builds a single rubric and candidate response string.
- Invokes OpenAI, Anthropic, Google Gemini, xAI, OpenRouter, and Cerebras (toggle which providers run inside `__main__`).
- Prints the numeric score and explanation returned by each provider's schema-enforced response.
- Gracefully skips providers whose API keys are not present (`MissingAPIKeyError`) and surfaces known remote failures via `ModelNotFoundError` / `ProviderRequestError`.

The helper uses the new provider registry in `osmosis_ai.providers`; each example call only needs to supply `provider` and `model` because the integrations are registered at import time. To experiment with your own provider implementation, create a subclass of `RubricProvider`, call `register_provider(...)`, and then pass the new provider name in `model_info`. See [`../osmosis_ai/providers/README.md`](../osmosis_ai/providers/README.md) for the detailed checklist.

## Rubric Configs and Dataset

Use `rubric_configs.yaml` for a pair of ready-to-run rubric configurations (OpenAI and Anthropic) and `sample_data.jsonl` for two matching solution strings that exercise those rubrics. They are designed to work with `osmosis preview --path examples/<file>` and can be adapted when building your own evaluation suites.

### Prerequisites

Export the relevant secrets before running:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
export XAI_API_KEY=...
export OPENROUTER_API_KEY=...
export CEREBRAS_API_KEY=...
```

### Running the Rubric Example

```bash
cd examples
PYTHONPATH=.. python rubric_functions.py  # uncomment providers you want to exercise
```
