# Configuration

The Osmosis rollout SDK is configured through environment variables, `.env` files, or programmatic Python calls. All settings use [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) when the `pydantic-settings` package is installed (included in the `server` extra).

## Environment Variables

Settings are loaded automatically from environment variables and from a `.env` file in the current working directory. The tables below list every recognized variable.

### Client Settings (`OSMOSIS_ROLLOUT_CLIENT_*`)

HTTP client configuration for communication between the RolloutServer and TrainGate.

| Variable | Type | Default | Range | Description |
|----------|------|---------|-------|-------------|
| `OSMOSIS_ROLLOUT_CLIENT_TIMEOUT_SECONDS` | `float` | `300.0` | 1.0 -- 3600.0 | HTTP request timeout in seconds. |
| `OSMOSIS_ROLLOUT_CLIENT_MAX_RETRIES` | `int` | `3` | 0 -- 10 | Maximum retry attempts for 5xx errors. |
| `OSMOSIS_ROLLOUT_CLIENT_COMPLETE_ROLLOUT_RETRIES` | `int` | `2` | 0 -- 10 | Maximum retries for the completion callback (`/v1/rollout/completed`). |
| `OSMOSIS_ROLLOUT_CLIENT_RETRY_BASE_DELAY` | `float` | `1.0` | 0.1 -- 60.0 | Base delay for exponential backoff in seconds. |
| `OSMOSIS_ROLLOUT_CLIENT_RETRY_MAX_DELAY` | `float` | `30.0` | 1.0 -- 300.0 | Maximum delay between retries in seconds. |
| `OSMOSIS_ROLLOUT_CLIENT_MAX_CONNECTIONS` | `int` | `100` | 1 -- 1000 | Maximum number of HTTP connections. |
| `OSMOSIS_ROLLOUT_CLIENT_MAX_KEEPALIVE_CONNECTIONS` | `int` | `20` | 1 -- 100 | Maximum keepalive connections in the pool. |

### Server Settings (`OSMOSIS_ROLLOUT_SERVER_*`)

Server-side configuration for the RolloutServer process.

| Variable | Type | Default | Range | Description |
|----------|------|---------|-------|-------------|
| `OSMOSIS_ROLLOUT_SERVER_MAX_CONCURRENT_ROLLOUTS` | `int` | `100` | 1 -- 10000 | Maximum number of concurrent rollouts. |
| `OSMOSIS_ROLLOUT_SERVER_RECORD_TTL_SECONDS` | `float` | `3600.0` | 60.0 -- 86400.0 | How long to keep completed rollout records (seconds). |
| `OSMOSIS_ROLLOUT_SERVER_CLEANUP_INTERVAL_SECONDS` | `float` | `60.0` | 10.0 -- 3600.0 | Interval for the cleanup task (seconds). |
| `OSMOSIS_ROLLOUT_SERVER_REQUEST_TIMEOUT_SECONDS` | `float` | `600.0` | 10.0 -- 3600.0 | Timeout for individual requests (seconds). |
| `OSMOSIS_ROLLOUT_SERVER_REGISTRATION_READINESS_TIMEOUT_SECONDS` | `float` | `10.0` | 1.0 -- 60.0 | Maximum wait for the server to become ready before platform registration. The server polls its own health endpoint to confirm readiness. |
| `OSMOSIS_ROLLOUT_SERVER_REGISTRATION_READINESS_POLL_INTERVAL_SECONDS` | `float` | `0.2` | 0.05 -- 5.0 | Interval between health check polls during server readiness check. |
| `OSMOSIS_ROLLOUT_SERVER_REGISTRATION_SHUTDOWN_TIMEOUT_SECONDS` | `float` | `30.0` | 1.0 -- 300.0 | Timeout for waiting for platform registration to complete during shutdown. |

### Global Settings (`OSMOSIS_ROLLOUT_*`)

Top-level settings that apply across the SDK.

| Variable | Type | Default | Range | Description |
|----------|------|---------|-------|-------------|
| `OSMOSIS_ROLLOUT_MAX_METADATA_SIZE_BYTES` | `int` | `1048576` (1 MB) | 1024 -- 104857600 (100 MB) | Maximum allowed size for rollout metadata in bytes. |

## Settings Classes

Configuration is organized into three Pydantic Settings classes. When `pydantic-settings` is installed, these classes automatically read from environment variables and `.env` files. Without `pydantic-settings`, they fall back to plain `pydantic.BaseModel` with only programmatic configuration.

### `RolloutClientSettings`

HTTP client settings for communication with TrainGate.

```python
from osmosis_ai.rollout.config import RolloutClientSettings

client = RolloutClientSettings()
print(client.timeout_seconds)     # 300.0
print(client.max_retries)         # 3
print(client.retry_base_delay)    # 1.0
```

Env prefix: `OSMOSIS_ROLLOUT_CLIENT_`

### `RolloutServerSettings`

Server-side settings for the RolloutServer process.

```python
from osmosis_ai.rollout.config import RolloutServerSettings

server = RolloutServerSettings()
print(server.max_concurrent_rollouts)   # 100
print(server.request_timeout_seconds)   # 600.0
```

Env prefix: `OSMOSIS_ROLLOUT_SERVER_`

### `RolloutSettings`

Top-level settings that aggregates `RolloutClientSettings` and `RolloutServerSettings` as nested objects.

```python
from osmosis_ai.rollout.config import RolloutSettings

settings = RolloutSettings()
print(settings.client.timeout_seconds)              # 300.0
print(settings.server.max_concurrent_rollouts)       # 100
print(settings.max_metadata_size_bytes)              # 1048576
```

Env prefix: `OSMOSIS_ROLLOUT_`

## Programmatic Configuration

### `get_settings()`

Returns the global settings singleton. On first call, settings are loaded from
environment variables and `.env` files.

```python
from osmosis_ai.rollout.config import get_settings

settings = get_settings()
timeout = settings.client.timeout_seconds
```

### `configure()`

Replaces the global settings singleton with a custom instance. Call this early
in your application to override environment variable defaults.

```python
from osmosis_ai.rollout.config import (
    configure,
    RolloutSettings,
    RolloutClientSettings,
    RolloutServerSettings,
)

configure(RolloutSettings(
    client=RolloutClientSettings(
        timeout_seconds=120.0,
        max_retries=5,
    ),
    server=RolloutServerSettings(
        max_concurrent_rollouts=200,
    ),
))
```

### `reset_settings()`

Resets the global settings singleton to `None`. Primarily used in tests to
ensure a clean state.

```python
from osmosis_ai.rollout.config import reset_settings

reset_settings()
```

## Metadata Size Limits

Rollout requests include a `metadata` dict that is validated against a
configurable size limit. The default limit is **1 MB**. The size is measured as
the byte length of the JSON-serialized metadata.

### `get_max_metadata_size_bytes()`

Returns the current maximum metadata size limit in bytes. Thread-safe.

```python
from osmosis_ai.rollout.core.schemas import get_max_metadata_size_bytes

limit = get_max_metadata_size_bytes()  # 1048576 (1 MB)
```

### `set_max_metadata_size_bytes()`

Sets the maximum metadata size limit in bytes. Thread-safe. The value must be
positive; otherwise a `ValueError` is raised.

```python
from osmosis_ai.rollout.core.schemas import set_max_metadata_size_bytes

# Increase to 2 MB
set_max_metadata_size_bytes(2 * 1024 * 1024)
```

The limit is also configurable via the `max_metadata_size_bytes` field on
`RolloutSettings` (env: `OSMOSIS_ROLLOUT_MAX_METADATA_SIZE_BYTES`), though
the runtime functions above provide thread-safe access used during request
validation.

## Server Constants

The `osmosis_ai.rollout.server.serve` module defines the following defaults
for `serve_agent_loop()`:

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_PORT` | `9000` | Default port the RolloutServer binds to. |
| `DEFAULT_HOST` | `"0.0.0.0"` | Default host the RolloutServer binds to (all interfaces). |

Override these when starting the server:

```bash
osmosis serve -m my_agent:agent_loop -p 8080
```

Or programmatically:

```python
from osmosis_ai.rollout.server import serve_agent_loop

serve_agent_loop(my_agent, host="127.0.0.1", port=8080)
```

## Credential File Location

Authentication tokens from `osmosis login` are stored at:

```
~/.config/osmosis/credentials.json
```

The configuration directory (`~/.config/osmosis/`) is created with mode `0700`
(owner-only access). The credentials file is written with mode `0600`
(owner read/write only).

The credentials file uses a multi-workspace format. It stores credentials for
each workspace you have logged in to, along with the currently active workspace
name. Use `osmosis workspace list` to see all stored workspaces and
`osmosis workspace switch <name>` to change the active one.

## `.env` File Support

Both the CLI and the settings classes load environment variables from a `.env`
file in the current working directory via `python-dotenv`. This is useful for
local development:

```bash
# .env
OSMOSIS_ROLLOUT_CLIENT_TIMEOUT_SECONDS=120
OSMOSIS_ROLLOUT_SERVER_MAX_CONCURRENT_ROLLOUTS=50
OPENAI_API_KEY=sk-...
```

## See Also

- [CLI Reference](./cli.md) -- all `osmosis` commands and options
- [Troubleshooting](./troubleshooting.md) -- common errors and resolutions
- [Deployment](./remote-rollout/deployment.md) -- production configuration for remote rollout
