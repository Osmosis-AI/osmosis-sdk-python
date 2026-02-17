# osmosis-ai

A Python SDK for Osmosis LLM training workflows:
- Reward/rubric validation helpers with strict type enforcement
- Remote Rollout SDK for integrating agent frameworks with Osmosis training
- Agent testing with external LLMs (`osmosis test`)
- Model evaluation with custom eval functions and pass@k metrics (`osmosis eval`)
- MCP tools support for git-sync workflows (`--mcp`)

## Installation

```bash
pip install osmosis-ai
```

Requires Python 3.10 or newer. For development setup, see [CONTRIBUTING.md](CONTRIBUTING.md).

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

solution = "The capital of France is Paris."

# Export OPENAI_API_KEY in your shell before running this snippet.
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

## Remote Rollout SDK

If you're integrating an agent loop with Osmosis remote rollout / TrainGate, see the [Rollout Quick Start](docs/rollout/README.md).

## Documentation

| Topic | Description |
|-------|-------------|
| [Rewards & Rubrics](docs/rewards.md) | `@osmosis_reward`, `@osmosis_rubric`, `evaluate_rubric` |
| [CLI Reference](docs/cli.md) | All `osmosis` commands: auth, serve, test, eval, rubric tools |
| **Remote Rollout SDK** | |
| [Quick Start](docs/rollout/README.md) | Get a rollout server running in minutes |
| [Architecture](docs/rollout/architecture.md) | Protocol design and agent lifecycle |
| [API Reference](docs/rollout/api-reference.md) | Endpoints, schemas, types |
| [Examples](docs/rollout/examples.md) | Agent implementations and utilities |
| [Dataset Format](docs/rollout/dataset-format.md) | Supported formats and required columns |
| [Test Mode](docs/rollout/test-mode.md) | Test agents with cloud LLMs |
| [Evaluation](docs/rollout/eval.md) | Evaluate agents with eval functions and pass@k |
| [Testing](docs/rollout/testing.md) | Unit tests and mock trainer |
| [Deployment](docs/rollout/deployment.md) | Docker, health checks, production config |

Full documentation index: [docs/README.md](docs/README.md)

## Running Examples

```bash
PYTHONPATH=. python examples/reward_functions.py
PYTHONPATH=. python examples/rubric_functions.py  # Uncomment the provider you need before running
```

See the [`examples/`](examples/) directory for all examples.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, linting, and PR guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Links

- [Homepage](https://github.com/Osmosis-AI/osmosis-sdk-python)
- [Issues](https://github.com/Osmosis-AI/osmosis-sdk-python/issues)
