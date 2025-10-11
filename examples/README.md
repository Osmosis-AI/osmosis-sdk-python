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

## Rubric Functions (`rubric_functions.py`)

Highlights how the `@osmosis_rubric` decorator can be used to validate conversation transcripts for compliance with a rubric:

- Ensures the function signature follows the required ordering (`rubric`, `messages`, optional `ground_truth`, optional `system_message`, optional `extra_info`)
- Demonstrates checking that the assistant response references an approved marketing claim
- Shows how to flag forbidden terminology via `extra_info`

### Running the Rubric Example

```bash
cd examples
PYTHONPATH=.. python rubric_functions.py
```

The script prints evaluation results for both a passing and a failing conversation.
