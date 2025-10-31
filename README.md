# osmosis-ai

A Python library that provides reward and rubric validation helpers for LLM applications with strict type enforcement.

## Installation

```bash
pip install osmosis-ai
```

Requires Python 3.9 or newer.

This installs the Osmosis CLI and pulls in the required provider SDKs (`openai`, `anthropic`, `google-genai`, `xai-sdk`) along with supporting utilities such as `PyYAML`, `python-dotenv`, `requests`, and `xxhash`.

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

`api_key` and `api_key_env` are mutually exclusive ways to provide the same credential. When `api_key` is present and non-empty it is used directly, skipping any environment lookup. Otherwise the resolver falls back to `api_key_env` (or the provider default) and pulls the value from your local environment with `os.getenv`.

`model_info` accepts additional rubric-specific knobs:

- `score_min` / `score_max` – change the default `[0.0, 1.0]` scoring bounds.
- `system_prompt` / `original_input` – provide optional context strings that will be quoted in the judging prompt.
- `timeout` – customise the provider timeout in seconds.

Pass `metadata={...}` to `evaluate_rubric` when you need structured context quoted in the judge prompt, and set `return_details=True` to receive the full `RewardRubricRunResult` payload (including the provider’s raw response).

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

Rubric functions decorated with `@osmosis_rubric` must match this signature:

```python
@osmosis_rubric
def your_rubric(solution_str: str, ground_truth: str | None, extra_info: dict) -> float:
    # Your rubric logic here
    return float_score
```

> The runtime forwards `None` for `ground_truth` when no reference answer exists. Annotate the parameter as `Optional[str]` (or handle `None` explicitly) if your rubric logic expects to run in that scenario.

### Required `extra_info` fields

- **`provider`** – Non-empty string identifying the judge provider.
- **`model`** – Non-empty string naming the provider model to call.
- **`rubric`** – Natural-language rubric instructions for the judge model.
- **`api_key` / `api_key_env`** – Supply either the raw key or the environment variable name that exposes it.

### Optional `extra_info` fields

- **`system_prompt`** – Optional string prepended to the provider’s base system prompt when invoking the judge; include it inside `extra_info` rather than as a separate argument.
- **`score_min` / `score_max`** – Optional numeric overrides for the expected score range.
- **`model_info_overrides`** – Optional dict merged into the provider configuration passed to the judge.

Additional keys are passthrough and can be used for custom configuration. If you need to extend the provider payload (for example adding `api_key_env`), add a dict under `model_info_overrides` and it will be merged with the required `provider`/`model` pair before invoking `evaluate_rubric`. The decorator enforces the parameter names/annotations, validates the embedded configuration at call time, and ensures the wrapped function returns a `float`.

> Annotation quirk: `extra_info` must be annotated as `dict` **without** a default value, unlike `@osmosis_reward`.

> Tip: When delegating to `evaluate_rubric`, pass the raw `solution_str` directly and include any extra context inside the `metadata` payload.

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

## CLI Tools

Installing the SDK also provides a lightweight CLI available as `osmosis` (aliases: `osmosis_ai`, `osmosis-ai`) for inspecting rubric YAML files and JSONL test payloads.

Preview a rubric file and print every configuration discovered, including nested entries:

```bash
osmosis preview --path path/to/rubric.yaml
```

Preview a dataset of rubric-scored solutions stored as JSONL:

```bash
osmosis preview --path path/to/data.jsonl
```

Evaluate a dataset against a hosted rubric configuration and print the returned scores:

```bash
osmosis eval --rubric support_followup --data examples/sample_data.jsonl
```

- Supply the dataset with `-d`/`--data path/to/data.jsonl`; the path is resolved relative to the current working directory.
- Use `--config path/to/rubric_configs.yaml` when the rubric definitions are not located alongside the dataset.
- Pass `-n`/`--number` to sample the provider multiple times per record; the CLI prints every run along with aggregate statistics (average, variance, standard deviation, and min/max).
- Provide `--output path/to/dir` to create the directory (if needed) and emit `rubric_eval_result_<unix_timestamp>.json`, or supply a full file path (any extension) to control the filename; each file captures every run, provider payloads, timestamps, and aggregate statistics for downstream analysis.
- Skip `--output` to collect results under `~/.cache/osmosis/eval_result/<rubric_id>/rubric_eval_result_<identifier>.json`; the CLI writes this JSON whether the evaluation finishes cleanly or hits provider/runtime errors so you can inspect failures later (only a manual Ctrl+C interrupt leaves no file behind).
- Dataset rows whose `rubric_id` does not match the requested rubric are skipped automatically.
- Each dataset record must provide a non-empty `solution_str`; optional fields such as `original_input`, `ground_truth`, and `extra_info` travel with the record and are forwarded to the evaluator when present.
- When delegating to a custom `@osmosis_rubric` function, the CLI enriches `extra_info` with the active `provider`, `model`, `rubric`, score bounds, any configured `system_prompt`, the resolved `original_input`, and the record’s metadata/extra fields so the decorator’s required entries are always present.
- Rubric configuration files intentionally reject `extra_info`; provide per-example context through the dataset instead.

Both commands validate the file, echo a short summary (`Loaded <n> ...`), and pretty-print the parsed records so you can confirm that new rubrics or test fixtures look correct before committing them. Invalid files raise a descriptive error and exit with a non-zero status code.

## Running Examples

```bash
PYTHONPATH=. python examples/reward_functions.py
PYTHONPATH=. python examples/rubric_functions.py  # Uncomment the provider you need before running
```

## Testing

Run `python -m pytest` (or any subset under `tests/`) to exercise the updated helpers:

- `tests/test_rubric_eval.py` covers prompt construction for `solution_str` evaluations.
- `tests/test_cli_services.py` validates dataset parsing, extra-info enrichment, and engine interactions.
- `tests/test_cli.py` ensures the CLI pathways surface the new fields end to end.

Add additional tests under `tests/` as you extend the library.

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
