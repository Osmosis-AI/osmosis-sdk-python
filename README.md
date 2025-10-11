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
from osmosis_ai import osmosis_reward, osmosis_rubric

@osmosis_reward
def simple_reward(solution_str: str, ground_truth: str, extra_info: dict = None) -> float:
    """Basic exact match reward function."""
    return 1.0 if solution_str.strip() == ground_truth.strip() else 0.0

# Use the reward function
score = simple_reward("hello world", "hello world")  # Returns 1.0
```

```python
@osmosis_rubric
def simple_rubric(
    rubric: str,
    messages: list,
    ground_truth: str | None = None,
    system_message: str | None = None,
    extra_info: dict = None,
) -> float:
    """Rubric that checks whether the assistant used the provided fact."""
    assistant_turn = next((m for m in reversed(messages) if m["role"] == "assistant"), None)
    if not assistant_turn:
        return 0.0

    assistant_text = " ".join(
        block["text"]
        for block in assistant_turn["content"]
        if isinstance(block, dict) and block.get("type") == "output_text"
    )

    if ground_truth and ground_truth.lower() in assistant_text.lower():
        return 1.0
    if extra_info and extra_info.get("partial_credit"):
        return 0.5
    return 0.0

# Use the rubric function
rubric_score = simple_rubric(
    rubric="Assistant must mention the verified capital city.",
    messages=[
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
    ],
    ground_truth="Paris",
)
```

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

Rubric functions decorated with `@osmosis_rubric` must follow this structure:

```python
@osmosis_rubric
def your_rubric(
    rubric: str,
    messages: list,
    ground_truth: str | None = None,
    system_message: str | None = None,
    extra_info: dict = None,
):
    # Your rubric logic here
    return float_score
```

### Parameters

- **`rubric: str`** - Description of the evaluation you are performing (required)
- **`messages: list`** - Provide a list of structured message dicts. Each dict must include `type`, `role`, and `content`, and `role` must be one of `user`, `system`, `assistant`, or `developer`.
- **`ground_truth: str | None = None`** - Optional ground truth string the assistant response should align with
- **`system_message: str | None = None`** - Optional system instruction used to guide the conversation
- **`extra_info: dict = None`** - Optional dictionary for additional configuration (same behavior as `@osmosis_reward`)

The decorator validates the parameter names, type annotations, and runtime payload for `messages`, raising a `TypeError` or `ValueError` when constraints are not satisfied. It also enforces that the wrapped function returns a `float`.

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

- `examples/rubric_functions.py` walks through rubric validation using realistic marketing compliance conversations and scoring logic.

## Running Examples

```bash
PYTHONPATH=. python examples/reward_functions.py
PYTHONPATH=. python examples/rubric_functions.py
```

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
