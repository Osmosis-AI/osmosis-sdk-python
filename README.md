# osmosis-ai

A Python library that provides reward functionality for LLM applications with strict type enforcement.

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

## Running Examples

```bash
python examples/reward_functions.py
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