# CLI Reference

Installing the SDK provides a lightweight CLI available as `osmosis` (aliases: `osmosis_ai`, `osmosis-ai`). The CLI automatically loads `.env` from the current working directory via `python-dotenv`.

## Authentication

Log in to Osmosis AI and manage credentials. Credentials are saved to `~/.config/osmosis/credentials.json` and include workspace information and token expiration.

### osmosis login

Open a browser-based authentication flow to log in to Osmosis AI:

```bash
# Log in (opens browser for authentication)
osmosis login

# Force re-login, clearing existing credentials
osmosis login --force

# Print the authentication URL without opening browser
osmosis login --no-browser
```

### osmosis logout

End your session and revoke stored credentials:

```bash
# Logout (interactive workspace selection)
osmosis logout

# Logout from all workspaces
osmosis logout --all

# Skip confirmation prompt
osmosis logout -y
```

### osmosis whoami

Display the current user and all workspaces:

```bash
osmosis whoami
```

### osmosis workspace

Manage multiple workspaces after logging in. You can log in to multiple workspaces and switch between them.

```bash
# List all logged-in workspaces
osmosis workspace list

# Show the current active workspace
osmosis workspace current

# Switch to a different workspace
osmosis workspace switch <workspace-name>
```

## Testing Your Agent

### osmosis test

Test your agent locally against a dataset using external LLMs. Works with both Local Rollout (MCP tools) and Remote Rollout (RolloutAgentLoop) agents.

```bash
# Remote Rollout: test an AgentLoop implementation
osmosis test -m my_agent:agent_loop -d data.jsonl --model openai/gpt-5-mini

# Local Rollout: test MCP tools directly (no AgentLoop needed)
osmosis test --mcp ./mcp -d data.jsonl --model openai/gpt-5-mini

# Interactive step-by-step debugging
osmosis test -m my_agent:agent_loop -d data.jsonl --interactive

# Interactive with MCP tools
osmosis test --mcp ./mcp -d data.jsonl --interactive
```

See [Test Mode](./test-mode.md) for full documentation on dataset format, interactive mode, and all options.

## Evaluating Models

### osmosis eval

Evaluate trained models with custom eval functions and pass@k metrics. Works with both Local Rollout (MCP tools) and Remote Rollout (RolloutAgentLoop) agents.

```bash
# Remote Rollout: benchmark a trained model at a serving endpoint
osmosis eval -m my_agent:agent_loop -d data.jsonl \
    --eval-fn rewards:compute_reward \
    --model my-finetuned-model --base-url http://localhost:8000/v1

# Local Rollout: evaluate MCP tools directly
osmosis eval --mcp ./mcp -d data.jsonl \
    --eval-fn rewards:compute_reward \
    --model openai/gpt-5-mini

# pass@k with 5 runs per row
osmosis eval -m my_agent:agent_loop -d data.jsonl \
    --eval-fn rewards:compute_reward --n 5 \
    --model my-finetuned-model --base-url http://localhost:8000/v1
```

See [Eval Mode](./eval-mode.md) for full documentation on eval functions, pass@k metrics, and output formats.

## Remote Rollout Server

### osmosis serve

Start a RolloutServer for an agent loop implementation. The module path format is `module:attribute`, e.g., `server:agent_loop` or `mypackage.agents:MyAgentClass`.

```bash
# Start server with Platform registration (requires `osmosis login`)
osmosis login
osmosis serve -m my_agent:agent_loop

# Specify host and port
osmosis serve -m my_agent:agent_loop -H 127.0.0.1 -p 8080

# Local / container mode: skip Platform registration (no login required).
# NOTE: API key auth is still enabled by default.
osmosis serve -m my_agent:agent_loop --skip-register

# Local debug mode: disable API key auth AND skip Platform registration
osmosis serve -m my_agent:agent_loop --local

# Provide a stable API key (otherwise one is generated and printed on startup)
osmosis serve -m my_agent:agent_loop --skip-register --api-key "$MY_API_KEY"

# Enable auto-reload for development
osmosis serve -m my_agent:agent_loop --local --reload

# Set log level and write execution traces to a directory
osmosis serve -m my_agent:agent_loop --log-level debug --log ./traces

# Skip validation (not recommended)
osmosis serve -m my_agent:agent_loop --no-validate
```

**Options:**

| Option | Description |
|--------|-------------|
| `-m`/`--module` | Module path to the agent loop (required) |
| `-p`/`--port` | Port to bind to (default: 9000) |
| `-H`/`--host` | Host to bind to (default: 0.0.0.0) |
| `--skip-register` | Skip registering with Osmosis Platform (for local testing) |
| `--local` | Local debug mode: disable API key auth and skip Platform registration |
| `--api-key` | API key used by TrainGate to authenticate requests to this server |
| `--no-validate` | Skip agent loop validation before starting |
| `--reload` | Enable auto-reload for development |
| `--log-level` | Uvicorn log level: debug, info, warning, error, critical (default: info) |
| `--log DIR` | Enable logging and write per-rollout execution traces (as `{rollout_id}.jsonl`) to DIR |

> **Note:** The `--api-key` option sets the API key for this RolloutServer. It is used by TrainGate to authenticate its requests *to* your server. This key is **not** the same as your `osmosis login` token (which is for authenticating with the Osmosis Platform), nor is it used for callbacks *from* your server back to TrainGate.

See [Remote Rollout Overview](./remote-rollout/overview.md) for architecture details and the full agent lifecycle.

### osmosis validate

Validate an agent loop before starting the server (checks tools, async run method, etc.):

```bash
# Basic validation
osmosis validate -m my_agent:agent_loop

# Verbose output with detailed warnings
osmosis validate -m my_agent:agent_loop -v
```

**Options:**

| Option | Description |
|--------|-------------|
| `-m`/`--module` | Module path to the agent loop (required) |
| `-v`/`--verbose` | Show detailed validation output including warnings |

See [Remote Rollout Overview](./remote-rollout/overview.md) for architecture details and the full agent lifecycle.

## Rubric Tools

### osmosis preview

Preview a rubric file and print every configuration discovered, including nested entries:

```bash
osmosis preview --path path/to/rubric.yaml
```

Preview a dataset of rubric-scored solutions stored as JSONL:

```bash
osmosis preview --path path/to/data.jsonl
```

Both formats validate the file, echo a short summary (`Loaded <n> ...`), and pretty-print the parsed records so you can confirm that new rubrics or test fixtures look correct before committing them. Invalid files raise a descriptive error and exit with a non-zero status code.

### osmosis eval-rubric

Evaluate a dataset against a hosted rubric configuration and print the returned scores:

```bash
osmosis eval-rubric --rubric support_followup --data examples/sample_data.jsonl
```

**Command split** (development-stage breaking change):
- `osmosis eval-rubric` evaluates JSONL conversations against hosted rubrics.
- `osmosis eval` runs rollout eval functions against agent datasets.

**Options:**

- `-d`/`--data path/to/data.jsonl` -- Supply the dataset; the path is resolved relative to the current working directory.
- `--config path/to/rubric_configs.yaml` -- Provide rubric definitions when they are not located alongside the dataset.
- `-n`/`--number` -- Sample the provider multiple times per record; the CLI prints every run along with aggregate statistics (average, variance, standard deviation, and min/max).
- `--output path/to/dir` -- Create the directory (if needed) and emit `rubric_eval_result_<unix_timestamp>.json`, or supply a full file path (any extension) to control the filename. Each file captures every run, provider payloads, timestamps, and aggregate statistics for downstream analysis.

**Behavior notes:**

- Skip `--output` to collect results under `~/.cache/osmosis/eval_result/<rubric_id>/rubric_eval_result_<identifier>.json`. The CLI writes this JSON whether the evaluation finishes cleanly or hits provider/runtime errors so you can inspect failures later (only a manual Ctrl+C interrupt leaves no file behind).
- Dataset rows whose `rubric_id` does not match the requested rubric are skipped automatically.
- Each dataset record must provide a non-empty `solution_str`; optional fields such as `original_input`, `ground_truth`, and `extra_info` travel with the record and are forwarded to the evaluator when present.
- When delegating to a custom `@osmosis_rubric` function, the CLI enriches `extra_info` with the active `provider`, `model`, `rubric`, score bounds, any configured `system_prompt`, the resolved `original_input`, and the record's metadata/extra fields so the decorator's required entries are always present.
- Rubric configuration files intentionally reject `extra_info`; provide per-example context through the dataset instead.

## See Also

- [Test Mode](./test-mode.md) -- full `osmosis test` documentation
- [Eval Mode](./eval-mode.md) -- full `osmosis eval` documentation
- [Rewards API Reference](./rewards-api.md) -- `@osmosis_reward`, `@osmosis_rubric`, `evaluate_rubric`
- [Dataset Format](./datasets.md) -- supported formats and required columns
