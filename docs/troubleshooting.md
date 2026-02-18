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

### MissingAPIKeyError

Raised when the LLM provider API key cannot be found. The error message
includes a hint showing the expected environment variable.

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

Raised when the LLM provider call fails. Common causes:

- **Authentication failure** -- your API key is invalid or expired.
- **Rate limiting** -- you have exceeded the provider's rate limit. Wait and retry.
- **Timeout** -- the request took too long. Try increasing the `timeout` parameter or use a faster model.
- **Invalid JSON response** -- the model returned content that could not be parsed.

### ModelNotFoundError

A subclass of `ProviderRequestError` raised when the requested model does not
exist. Verify the model name and that your account has access.

## Test Mode Errors

These errors occur when running `osmosis test` or `osmosis eval`.

### DatasetParseError

Raised when the dataset file cannot be read or parsed.

Resolution:

- Ensure your file uses a supported format: `.parquet`, `.jsonl`, or `.csv`.
- For JSONL files, verify each line is valid JSON.
- For Parquet files, install `pyarrow` (included in the `server` extra).

### DatasetValidationError

Raised when dataset rows are missing required columns or have invalid values.

Every dataset row must include these columns:

| Column | Type | Description |
|--------|------|-------------|
| `ground_truth` | `str` | Reference answer |
| `user_prompt` | `str` | User message |
| `system_prompt` | `str` | System prompt |

All values must be non-empty strings.

### ToolValidationError

Ensure your `get_tools()` method returns valid OpenAI-compatible function tool
schemas. Each tool must have a `type` field set to `"function"` and a `function`
object with at least a `name`.

### Provider connection errors

`SystemicProviderError` is raised when a provider error affects all rows (e.g.
authentication failure, budget exhausted). The batch aborts early instead of
retrying each row. Fix the underlying credential or connectivity issue and re-run.

## Remote Rollout Errors

### AgentLoopValidationError

Raised by `osmosis validate` or when starting a server with `validate=True`.

Common validation error codes:

| Code | Meaning |
|------|---------|
| `MISSING_NAME` | Agent loop class has no `name` attribute. |
| `INVALID_NAME_TYPE` | `name` is not a string. |
| `EMPTY_NAME` | `name` is empty or whitespace. |
| `RUN_NOT_ASYNC` | `run` is not an `async def` function. |
| `GET_TOOLS_RETURNS_NONE` | `get_tools()` returned `None` instead of a list. |
| `GET_TOOLS_EXCEPTION` | `get_tools()` raised an exception. |
| `MISSING_TOOL_TYPE` | A tool dict is missing the `type` field. |
| `MISSING_FUNCTION_NAME` | A function definition has no `name`. |

Run validation before serving to catch these early:

```bash
osmosis validate -m server:agent_loop
```

### ServeError

**Not logged in:**

```
ServeError: Not logged in. Please run 'osmosis login' first, or use skip_register=True for local testing.
```

Resolution -- run `osmosis login`, or pass `--skip-register` / `--local` if you
do not need platform registration.

**Missing dependencies:**

```
ImportError: FastAPI is required for serve_agent_loop(). Install it with: pip install osmosis-ai[server]
```

### Connection / timeout issues

Rollout protocol exceptions:

| Exception | Description | Retryable? |
|-----------|-------------|------------|
| `OsmosisTransportError` | Network-level failure (connection refused, DNS error). | Yes |
| `OsmosisServerError` | Server returned HTTP 5xx. | Yes |
| `OsmosisValidationError` | Server returned HTTP 4xx. | No |
| `OsmosisTimeoutError` | Request exceeded configured timeout. | Yes |
| `AgentLoopNotFoundError` | Registry lookup for agent name failed. | No |
| `ToolExecutionError` | A tool call failed during execution. | Depends |
| `ToolArgumentError` | Tool arguments could not be parsed. | No |

For retryable errors, use exponential backoff. See [Configuration](./configuration.md) for client retry settings.

### PublicIPDetectionError

Raised when the server cannot detect its public IP address. Resolution:

- Check network connectivity and firewall rules.
- Provide an explicit host via `OSMOSIS_PUBLIC_HOST` environment variable or the `--host` flag.
- If running locally, use `--local` mode which skips IP detection.

## See Also

- [CLI Reference](./cli.md) -- all `osmosis` commands and options
- [Configuration](./configuration.md) -- environment variables and settings
- [Dataset Format](./datasets.md) -- supported formats and required columns
- [Test Mode](./test-mode.md) -- full `osmosis test` documentation
- [Eval Mode](./eval-mode.md) -- full `osmosis eval` documentation
