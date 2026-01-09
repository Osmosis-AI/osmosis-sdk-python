"""Test mode for RolloutServer agent loop validation.

This package provides functionality to test RolloutAgentLoop implementations
locally without requiring the full TrainGate infrastructure. It uses cloud
LLM providers (OpenAI, Anthropic) to simulate the LLM calls that would
normally be routed through TrainGate.

Key Components:
    - DatasetReader: Read and validate test datasets (JSON, JSONL, Parquet)
    - TestLLMClient: Base class for test mode LLM providers
    - TestRunner: Execute agent loops against test data
    - CLI: Command-line interface for running tests

Supported Dataset Formats:
    - .json - Array of objects
    - .jsonl - JSON Lines (one object per line)
    - .parquet - Apache Parquet

Required Dataset Columns:
    - ground_truth: Expected output for reward calculation
    - user_prompt: User's input message
    - system_prompt: System prompt for the agent

Example:
    # Programmatic usage
    from osmosis_ai.rollout.test_mode import (
        DatasetReader,
        TestRunner,
    )
    from osmosis_ai.rollout.test_mode.providers import get_provider

    # Load dataset
    reader = DatasetReader("./test_data.jsonl")
    rows = reader.read(limit=10)

    # Create provider
    client_class = get_provider("openai")
    client = client_class(model="gpt-4o-mini")

    # Run tests
    runner = TestRunner(agent_loop=MyAgent(), llm_client=client)
    results = await runner.run_batch(rows)

    # Check results
    print(f"Passed: {results.passed}/{results.total}")

CLI Usage:
    # Basic usage
    osmosis test \\
        --agent my_agent:MyAgentLoop \\
        --dataset ./test_data.jsonl \\
        --provider openai \\
        --model gpt-4o

    # With options
    osmosis test \\
        --agent my_agent:MyAgentLoop \\
        --dataset ./test_data.jsonl \\
        --provider openai \\
        --model gpt-4o-mini \\
        --limit 10 \\
        --max-turns 5 \\
        --output results.json
"""

from osmosis_ai.rollout.test_mode.dataset import (
    REQUIRED_COLUMNS,
    DatasetReader,
    DatasetRow,
    dataset_row_to_request,
)
from osmosis_ai.rollout.test_mode.exceptions import (
    DatasetParseError,
    DatasetValidationError,
    ProviderError,
    TestModeError,
    ToolValidationError,
)
from osmosis_ai.rollout.test_mode.runner import (
    TestBatchResult,
    TestRunResult,
    TestRunner,
)

__all__ = [
    # Dataset
    "DatasetReader",
    "DatasetRow",
    "REQUIRED_COLUMNS",
    "dataset_row_to_request",
    # Runner
    "TestRunner",
    "TestRunResult",
    "TestBatchResult",
    # Exceptions
    "TestModeError",
    "DatasetValidationError",
    "DatasetParseError",
    "ProviderError",
    "ToolValidationError",
]
