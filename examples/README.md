# Examples

This directory contains example usage of the osmosis-ai library.

## Full Example Repos

Complete, runnable example projects:

- **Local Rollout**: [osmosis-git-sync-example](https://github.com/Osmosis-AI/osmosis-git-sync-example) — reward functions, rubrics, and MCP tools
- **Remote Rollout**: [osmosis-remote-rollout-example](https://github.com/Osmosis-AI/osmosis-remote-rollout-example) — custom `RolloutAgentLoop` server

## SDK API Examples

### Reward Functions (`reward_functions.py`)

Demonstrates how to use the `@osmosis_reward` decorator with various reward function patterns:

- **Simple exact match**: Basic string comparison
- **Case-insensitive match**: Flexible string matching with optional partial credit
- **Numeric tolerance**: Float comparison with configurable tolerance
- **Minimal reward**: Using only the required parameters

### Required Function Signature

All functions decorated with `@osmosis_reward` must have this exact signature:

```python
@osmosis_reward
def your_function(solution_str: str, ground_truth: str, extra_info: dict = None, **kwargs) -> float:
    # Your reward logic here
    return float_score
```

> **Note:** Including `**kwargs` is required for platform compatibility. The Osmosis platform passes additional keyword arguments to reward functions.

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

### Remote Rubric Evaluation (`rubric_functions.py`)

Shows how to call `osmosis_ai.evaluate_rubric` against different hosted judge providers using their official Python SDKs. The example:

- Builds a single rubric and candidate response string.
- Invokes OpenAI, Anthropic, Google Gemini, xAI, OpenRouter, and Cerebras (toggle which providers run inside `__main__`).
- Prints the numeric score and explanation returned by each provider's schema-enforced response.
- Gracefully skips providers whose API keys are not present (`MissingAPIKeyError`) and surfaces known remote failures via `ModelNotFoundError` / `ProviderRequestError`.

The helper uses [LiteLLM](https://github.com/BerriAI/litellm) under the hood; each example call only needs to supply `provider` and `model` because LiteLLM routes to the correct API automatically. Any provider supported by LiteLLM can be used by passing its name in `model_info`.

### Rubric Configs and Dataset

Use `rubric_configs.yaml` for a pair of ready-to-run rubric configurations (OpenAI and Anthropic) and `sample_data.jsonl` for two matching solution strings that exercise those rubrics. They are designed to work with `osmosis preview --path examples/<file>` and can be adapted when building your own evaluation suites.

### Prerequisites

Export the relevant secrets before running:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export XAI_API_KEY=...
export OPENROUTER_API_KEY=...
export CEREBRAS_API_KEY=...
```

### Running the Rubric Example

```bash
cd examples
PYTHONPATH=.. python rubric_functions.py  # uncomment providers you want to exercise
```
