# Troubleshooting

Common errors and their resolutions when working with the Osmosis SDK.

## Installation Issues

### Missing extras

The SDK ships optional dependency groups. If you see an `ImportError` for a
package that should be available, you likely need to install the correct extra.

| Error | Install command |
|-------|----------------|
| `No module named 'fastmcp'` | `pip install osmosis-ai[mcp]` |
| `No module named 'fastapi'` / `No module named 'uvicorn'` | `pip install osmosis-ai[server]` |
| `No module named 'pydantic_settings'` | `pip install osmosis-ai[server]` |
| `No module named 'pyarrow'` | `pip install osmosis-ai[server]` (or `pip install pyarrow`) |
| `No module named 'rich'` | `pip install osmosis-ai[server]` |
| `No module named 'litellm'` | `pip install osmosis-ai` (included in core dependencies) |

To install everything at once:

```bash
pip install osmosis-ai[full]    # server + mcp
pip install osmosis-ai[dev]     # full + testing/linting tools
```

### Python version requirements

The SDK requires **Python 3.10 or later** (3.10, 3.11, 3.12, 3.13). If you see
syntax errors or compatibility issues, verify your Python version:

```bash
python --version
```

## Authentication Issues

Credentials are saved to `~/.config/osmosis/credentials.json` after a
successful `osmosis login`.

### `osmosis login` fails

1. **Browser does not open** -- pass `--no-browser` to print the authentication
   URL so you can open it manually.
2. **Network / firewall** -- the login flow requires outbound HTTPS to the
   Osmosis platform. Ensure your network allows it.
3. **Force re-login** -- if your session is in an inconsistent state, run:

   ```bash
   osmosis login --force
   ```

### Token expiration

Tokens have an expiration time. When a token expires, commands that require
authentication will fail silently or report that you are not logged in.

```
Not logged in. Please run 'osmosis login' first, or use skip_register=True for local testing.
```

Resolution:

```bash
osmosis login
```

To check your current session:

```bash
osmosis whoami
```

### Multiple workspaces

If you are logged in to the wrong workspace, commands may fail with permission
errors. List and switch workspaces with:

```bash
osmosis workspace list
osmosis workspace switch <workspace-name>
```

## Reward Function Errors

### Signature validation errors from `@osmosis_reward`

The `@osmosis_reward` decorator enforces the classic Osmosis signature when the
first parameter is named `solution_str`:

```
(solution_str: str, ground_truth: str, extra_info: dict = None, **kwargs) -> float
```

Common `TypeError` messages and their causes:

| Error message | Cause |
|---------------|-------|
| `Function <name> must have at least 3 parameters, got <n>` | The function has fewer than three parameters. |
| `First parameter 'solution_str' must be annotated as str, got <type>` | Missing or wrong type annotation on `solution_str`. |
| `Second parameter must be named 'ground_truth', got '<name>'` | The second parameter has the wrong name. |
| `Second parameter 'ground_truth' must be annotated as str, got <type>` | Missing or wrong type annotation on `ground_truth`. |
| `Third parameter must be named 'extra_info', got '<name>'` | The third parameter has the wrong name. |
| `Third parameter 'extra_info' must be annotated as dict or dict \| None, got <type>` | Wrong annotation on `extra_info`. Use `dict` or `dict \| None`. |
| `Third parameter 'extra_info' must have a default value of None` | `extra_info` does not have a default value. |

### Return type must be float

When using the classic signature, the decorated function **must** return a plain
`float`. Any other return type raises:

```
TypeError: Function <name> must return a float, got <type>
```

Ensure you return `float(value)` rather than `int`, `Decimal`, or another
numeric type.

## Rubric Evaluation Errors

These errors originate from `osmosis_ai.rubric_eval` and
`osmosis_ai.rubric_types`.

### MissingAPIKeyError

Raised when the LLM provider API key cannot be found. The error message
includes a hint showing the expected environment variable.

Example:

```
MissingAPIKeyError: Environment variable 'OPENAI_API_KEY' is not set.
Export it with your openai API key before calling evaluate_rubric.
Set the required API key before running:

    export OPENAI_API_KEY="..."
```

Resolution -- set the appropriate environment variable for your provider:

| Provider | Environment variable |
|----------|---------------------|
| openai | `OPENAI_API_KEY` |
| anthropic | `ANTHROPIC_API_KEY` |
| xai | `XAI_API_KEY` |
| gemini / google | `GEMINI_API_KEY` |
| openrouter | `OPENROUTER_API_KEY` |
| cerebras | `CEREBRAS_API_KEY` |
| azure | `AZURE_API_KEY` |
| bedrock | `AWS_ACCESS_KEY_ID` |
| vertex_ai | `GOOGLE_APPLICATION_CREDENTIALS` |

You can also provide the key directly in `model_info` via the `api_key` or
`api_key_env` fields.

### ProviderRequestError

Raised when the LLM provider call fails. The error includes the provider name,
model name, and a detail string.

```
ProviderRequestError: Provider 'openai' request for model 'gpt-5-mini' failed. <detail>
```

Common causes:

- **Authentication failure** -- your API key is invalid or expired.
- **Rate limiting** -- you have exceeded the provider's rate limit. Wait and
  retry.
- **Timeout** -- the request took too long. Try increasing the `timeout`
  parameter or use a faster model.
- **Connection error** -- network issue between you and the provider. Check
  connectivity.
- **Invalid JSON response** -- the model returned content that could not be
  parsed. Refine rubric instructions so the model returns valid JSON.

### ModelNotFoundError

A subclass of `ProviderRequestError` raised when the requested model does not
exist.

```
ModelNotFoundError: Provider 'openai' request for model 'gpt-99' failed.
Model 'gpt-99' was not found. Confirm the model identifier is correct and your openai account has access to it.
```

Resolution -- verify the model name and that your account has access. Use the
`provider/model` format, e.g. `openai/gpt-5-mini`, `anthropic/claude-sonnet-4-5`.

## Test Mode Errors

These errors occur when running `osmosis test` or `osmosis eval`.

### DatasetParseError

Raised when the dataset file cannot be read or parsed.

```
DatasetParseError: Unsupported file format: .txt. Supported formats: .parquet (recommended), .jsonl, .csv
```

```
DatasetParseError: Invalid JSON at line 5: Expecting ',' delimiter
```

```
DatasetParseError: Parquet support requires pyarrow. Install with: pip install pyarrow
```

Resolution:

- Ensure your file uses a supported format: `.parquet`, `.jsonl`, or `.csv`.
- For JSONL files, verify each line is valid JSON.
- For Parquet files, install `pyarrow` (included in the `server` extra).

### DatasetValidationError

Raised when dataset rows are missing required columns or have invalid values.

```
DatasetValidationError: Row 0: Missing required columns: ['user_prompt']
```

```
DatasetValidationError: Row 3: 'ground_truth' cannot be null
```

```
DatasetValidationError: Row 7: 'system_prompt' must be a string, got int
```

Every dataset row must include these columns:

| Column | Type | Description |
|--------|------|-------------|
| `ground_truth` | `str` | Reference answer |
| `user_prompt` | `str` | User message |
| `system_prompt` | `str` | System prompt |

All values must be non-empty strings.

### ToolValidationError

Raised when tool schemas returned by your agent are invalid for the provider's
API.

```
ToolValidationError: <detail>
```

Resolution -- ensure your `get_tools()` method returns valid OpenAI-compatible
function tool schemas. Each tool must have a `type` field set to `"function"` and
a `function` object with at least a `name`.

### Provider connection errors

`SystemicProviderError` is raised when a provider error affects all rows (e.g.
authentication failure, budget exhausted, network unreachable). The batch aborts
early instead of retrying each row.

Resolution -- fix the underlying credential or connectivity issue and re-run.

## Remote Rollout Errors

### AgentLoopValidationError

Raised by `osmosis validate` or when starting a server with `validate=True`
(the default). The error lists all validation failures.

```
AgentLoopValidationError: Agent loop validation failed with 2 error(s):
  - [MISSING_NAME] name: Agent loop must have a 'name' attribute
  - [RUN_NOT_ASYNC] run: 'run' method must be an async function (async def)
```

Common validation error codes:

| Code | Meaning |
|------|---------|
| `MISSING_NAME` | Agent loop class has no `name` attribute. |
| `INVALID_NAME_TYPE` | `name` is not a string. |
| `EMPTY_NAME` | `name` is empty or whitespace. |
| `MISSING_RUN_METHOD` | No `run` method defined. |
| `RUN_NOT_CALLABLE` | `run` exists but is not callable. |
| `RUN_NOT_ASYNC` | `run` is not an `async def` function. |
| `GET_TOOLS_RETURNS_NONE` | `get_tools()` returned `None` instead of a list. |
| `GET_TOOLS_INVALID_TYPE` | `get_tools()` returned a non-list type. |
| `GET_TOOLS_EXCEPTION` | `get_tools()` raised an exception. |
| `MISSING_TOOL_TYPE` | A tool dict is missing the `type` field. |
| `MISSING_FUNCTION` | A tool dict is missing the `function` field. |
| `MISSING_FUNCTION_NAME` | A function definition has no `name`. |
| `INVALID_FUNCTION_NAME` | Function name is empty or not a string. |

Run validation before serving to catch these early:

```bash
osmosis validate -m my_agent:agent_loop
```

### ServeError

Raised when `serve_agent_loop()` or `osmosis serve` cannot start the server.

**Not logged in:**

```
ServeError: Not logged in. Please run 'osmosis login' first, or use skip_register=True for local testing.
```

Resolution -- run `osmosis login`, or pass `--skip-register` / `--local` if you
do not need platform registration.

**Conflicting options:**

```
ServeError: local_debug=True disables API key authentication; do not provide api_key in local debug mode.
```

Resolution -- do not combine `--local` with `--api-key`. Local debug mode
intentionally disables API key authentication.

**Missing dependencies:**

```
ImportError: FastAPI is required for serve_agent_loop(). Install it with: pip install osmosis-ai[server]
```

```
ImportError: uvicorn is required for serve_agent_loop(). Install it with: pip install osmosis-ai[server]
```

Resolution:

```bash
pip install osmosis-ai[server]
```

### Connection / timeout issues

Rollout protocol exceptions are defined in
`osmosis_ai.rollout.core.exceptions`:

| Exception | Description | Retryable? |
|-----------|-------------|------------|
| `OsmosisTransportError` | Network-level failure (connection refused, DNS error). | Yes |
| `OsmosisServerError` | Server returned HTTP 5xx. Includes `status_code` attribute. | Yes |
| `OsmosisValidationError` | Server returned HTTP 4xx. Includes `status_code` attribute. | No |
| `OsmosisTimeoutError` | Request exceeded configured timeout. | Yes |
| `AgentLoopNotFoundError` | Registry lookup for agent name failed. Includes `name` and `available` attributes. | No |
| `ToolExecutionError` | A tool call failed during execution. Includes `tool_call_id` and `tool_name`. | Depends |
| `ToolArgumentError` | Tool arguments could not be parsed (subclass of `ToolExecutionError`). | No |

For retryable errors, use exponential backoff. The SDK client settings provide
configurable retry parameters (see [Configuration](./configuration.md)).

### PublicIPDetectionError

Raised when the server cannot detect its public IP address. This happens when
binding to `0.0.0.0` and all detection methods fail.

```
PublicIPDetectionError: Failed to detect public IP address. All detection methods failed:
  1. Cloud metadata (AWS/GCP/Azure): unavailable or returned no public IP
  2. External IP services: all failed or timed out

To fix this, provide an explicit IP/hostname to your application.
```

Resolution -- the SDK tries cloud metadata services (AWS, GCP, Azure) first,
then external IP services (checkip.amazonaws.com, ipify, icanhazip, ifconfig.me)
as a fallback. If all fail:

- Check network connectivity and firewall rules.
- Provide an explicit host via the `OSMOSIS_PUBLIC_HOST` environment variable
  or the `--host` flag.
- If running locally, use `--local` mode which skips IP detection for
  platform registration.

## Debug Tips

### Using `--verbose` flag

The `osmosis serve` command accepts `-v` / `--verbose` to increase output
verbosity, which prints detailed validation warnings and server configuration.

```bash
osmosis serve -m my_agent:agent_loop --verbose
```

### Using `--debug` flag

The `osmosis test` and `osmosis eval` commands accept `--debug` to enable debug
logging, which prints detailed information about each step including provider
requests and responses.

```bash
osmosis test -m my_agent:agent_loop -d data.jsonl --model openai/gpt-5-mini --debug
osmosis eval -m my_agent:agent_loop -d data.jsonl --eval-fn rewards:fn --model openai/gpt-5-mini --debug
```

### Interactive mode in test mode

Use `--interactive` with `osmosis test` to step through each row one at a time.
This is useful for debugging agent logic and inspecting intermediate messages:

```bash
osmosis test -m my_agent:agent_loop -d data.jsonl --interactive

# Start at a specific row
osmosis test -m my_agent:agent_loop -d data.jsonl --interactive --row 5
```

### Debug directory for rollout server

When serving an agent, use `--debug-dir` to write detailed execution traces
for each rollout to JSONL files:

```bash
osmosis serve -m my_agent:agent_loop --debug-dir ./debug-logs
```

Each rollout will produce a file at `{debug_dir}/{timestamp}/{rollout_id}.jsonl`.

### Checking logs

The SDK uses Python's standard `logging` module. Increase the log level to see
more detail:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or set the uvicorn log level when serving:

```bash
osmosis serve -m my_agent:agent_loop --log-level debug
```

### Checking credentials

To verify your authentication state and credential file:

```bash
# Show current user and workspace
osmosis whoami

# Credentials are stored at:
#   ~/.config/osmosis/credentials.json
```

## See Also

- [CLI Reference](./cli.md) -- all `osmosis` commands and options
- [Configuration](./configuration.md) -- environment variables and settings
- [Dataset Format](./datasets.md) -- supported formats and required columns
- [Test Mode](./test-mode.md) -- full `osmosis test` documentation
- [Eval Mode](./eval-mode.md) -- full `osmosis eval` documentation
- [Rewards API Reference](./rewards-api.md) -- `@osmosis_reward`, `@osmosis_rubric`, `evaluate_rubric`
