"""Test mode for RolloutAgentLoop validation.

Test agent loops locally without TrainGate using external LLM providers
(OpenAI, Anthropic, Groq, Ollama, etc.) via LiteLLM.

Components:
    - DatasetReader: Read datasets (JSON, JSONL, Parquet)
    - ExternalLLMClient: Call external LLM APIs via LiteLLM
    - LocalTestRunner: Batch test execution
    - InteractiveRunner: Step-by-step debugging

Dataset columns: ground_truth, user_prompt, system_prompt

Example:
    from osmosis_ai.rollout.test_mode import LocalTestRunner, ExternalLLMClient, DatasetReader

    reader = DatasetReader("./test_data.jsonl")
    client = ExternalLLMClient("gpt-4o")  # or "anthropic/claude-sonnet-4-20250514"
    runner = LocalTestRunner(agent_loop=MyAgent(), llm_client=client)
    results = await runner.run_batch(reader.read())

CLI:
    osmosis test --agent my_agent:MyAgentLoop --dataset data.jsonl --model gpt-4o
    osmosis test ... --model anthropic/claude-sonnet-4-20250514
    osmosis test ... --interactive  # Step-by-step mode
    osmosis test ... --interactive --row 5  # Test specific row

See https://docs.litellm.ai/docs/providers for supported providers.
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
from osmosis_ai.rollout.test_mode.external_llm_client import ExternalLLMClient
from osmosis_ai.rollout.test_mode.interactive import (
    InteractiveLLMClient,
    InteractiveRunner,
    InteractiveStep,
)
from osmosis_ai.rollout.test_mode.runner import (
    LocalTestBatchResult,
    LocalTestRunResult,
    LocalTestRunner,
)

__all__ = [
    # Dataset
    "DatasetReader",
    "DatasetRow",
    "REQUIRED_COLUMNS",
    "dataset_row_to_request",
    # Runner
    "LocalTestRunner",
    "LocalTestRunResult",
    "LocalTestBatchResult",
    # Interactive
    "InteractiveRunner",
    "InteractiveLLMClient",
    "InteractiveStep",
    # LLM Client
    "ExternalLLMClient",
    # Exceptions
    "TestModeError",
    "DatasetValidationError",
    "DatasetParseError",
    "ProviderError",
    "ToolValidationError",
]
