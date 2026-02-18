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
osmosis test -m server:agent_loop -d data.jsonl --model openai/gpt-5-mini
osmosis test --mcp ./mcp -d data.jsonl --model openai/gpt-5-mini
osmosis test -m server:agent_loop -d data.jsonl --interactive
```

See [Test Mode](./test-mode.md) for full documentation on dataset format, interactive mode, and all options.

## Evaluating Models

### osmosis eval

Evaluate trained models with custom eval functions and pass@k metrics. Works with both Local Rollout (MCP tools) and Remote Rollout (RolloutAgentLoop) agents.

```bash
osmosis eval -m server:agent_loop -d data.jsonl \
    --eval-fn rewards:compute_reward \
    --model my-finetuned-model --base-url http://localhost:8000/v1

osmosis eval --mcp ./mcp -d data.jsonl \
    --eval-fn rewards:compute_reward --model openai/gpt-5-mini
```

See [Eval Mode](./eval-mode.md) for full documentation on eval functions, pass@k metrics, and output formats.

## Remote Rollout Server

### osmosis serve

Start a RolloutServer for an agent loop implementation. The module path format is `module:attribute`, e.g., `server:agent_loop` or `mypackage.agents:MyAgentClass`.

```bash
# Start server with Platform registration (requires `osmosis login`)
osmosis serve -m server:agent_loop

# Local debug mode: disable API key auth AND skip Platform registration
osmosis serve -m server:agent_loop --local

# Enable auto-reload for development
osmosis serve -m server:agent_loop --local --reload

# Set log level and write execution traces to a directory
osmosis serve -m server:agent_loop --log-level debug --log ./traces
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
| `--log DIR` | Enable logging and write per-rollout execution traces to DIR |

> **Note:** The `--api-key` option sets the API key for this RolloutServer. It is used by TrainGate to authenticate its requests *to* your server. This is **not** the same as your `osmosis login` token.

See [Remote Rollout Overview](./remote-rollout/overview.md) for architecture details and the full agent lifecycle.

### osmosis validate

Validate an agent loop before starting the server:

```bash
osmosis validate -m server:agent_loop
osmosis validate -m server:agent_loop -v  # Verbose with warnings
```

| Option | Description |
|--------|-------------|
| `-m`/`--module` | Module path to the agent loop (required) |
| `-v`/`--verbose` | Show detailed validation output including warnings |

## Rubric Tools

### osmosis preview

Preview a rubric file or dataset:

```bash
osmosis preview --path path/to/rubric.yaml
osmosis preview --path path/to/data.jsonl
```

Both formats validate the file, echo a short summary, and pretty-print the parsed records.

### osmosis eval-rubric

Evaluate a dataset against a hosted rubric configuration:

```bash
osmosis eval-rubric --rubric support_followup --data examples/sample_data.jsonl
```

**Options:**

- `-d`/`--data path/to/data.jsonl` -- Supply the dataset
- `--config path/to/rubric_configs.yaml` -- Provide rubric definitions
- `-n`/`--number` -- Sample multiple times per record with aggregate statistics
- `--output path/to/dir` -- Write results to a file or directory

**Command split** (development-stage breaking change): `osmosis eval-rubric` evaluates JSONL conversations against hosted rubrics, while `osmosis eval` runs rollout eval functions against agent datasets.

## See Also

- [Test Mode](./test-mode.md) -- full `osmosis test` documentation
- [Eval Mode](./eval-mode.md) -- full `osmosis eval` documentation
- [Rewards API Reference](./rewards-api.md) -- `@osmosis_reward`, `@osmosis_rubric`, `evaluate_rubric`
- [Dataset Format](./datasets.md) -- supported formats and required columns
