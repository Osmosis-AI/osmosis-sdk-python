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
| `OSMOSIS_ROLLOUT_SERVER_REGISTRATION_READINESS_TIMEOUT_SECONDS` | `float` | `10.0` | 1.0 -- 60.0 | Maximum wait for the server to become ready before platform registration. |
| `OSMOSIS_ROLLOUT_SERVER_REGISTRATION_READINESS_POLL_INTERVAL_SECONDS` | `float` | `0.2` | 0.05 -- 5.0 | Interval between health check polls during server readiness check. |
| `OSMOSIS_ROLLOUT_SERVER_REGISTRATION_SHUTDOWN_TIMEOUT_SECONDS` | `float` | `30.0` | 1.0 -- 300.0 | Timeout for waiting for platform registration to complete during shutdown. |

### Global Settings (`OSMOSIS_ROLLOUT_*`)

| Variable | Type | Default | Range | Description |
|----------|------|---------|-------|-------------|
| `OSMOSIS_ROLLOUT_MAX_METADATA_SIZE_BYTES` | `int` | `1048576` (1 MB) | 1024 -- 104857600 (100 MB) | Maximum allowed size for rollout metadata in bytes. |

### Eval Cache Settings

These environment variables control the behavior of `osmosis eval` result caching.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OSMOSIS_CACHE_DIR` | `str` | - | Override the eval cache root directory. When set, cache files are stored under `$OSMOSIS_CACHE_DIR/eval/`. |
| `OSMOSIS_EVAL_LOCK_TIMEOUT` | `int` | `30` | Timeout in seconds for acquiring the cache file lock. If another eval with the same config is running, the process waits up to this duration before failing. Must be a positive integer. |

When `OSMOSIS_CACHE_DIR` is not set, the cache follows the XDG Base Directory convention: `$XDG_CACHE_HOME/osmosis/eval/` (defaults to `~/.cache/osmosis/eval/`).

## Programmatic Configuration

Configuration is organized into three Pydantic Settings classes. When the `pydantic-settings` package is installed (included in the `server` extra), these classes automatically read from environment variables and `.env` files. Without `pydantic-settings`, they fall back to plain Pydantic models that only accept values passed programmatically:

| Class | Env Prefix | Description |
|-------|-----------|-------------|
| `RolloutClientSettings` | `OSMOSIS_ROLLOUT_CLIENT_` | HTTP client settings (timeout, retries, connection pool) |
| `RolloutServerSettings` | `OSMOSIS_ROLLOUT_SERVER_` | Server settings (concurrency, TTL, timeouts) |
| `RolloutSettings` | `OSMOSIS_ROLLOUT_` | Top-level aggregator with nested `client` and `server` objects |

```python
from osmosis_ai.rollout.config import get_settings, configure, RolloutSettings, RolloutClientSettings

# Read current settings (loaded from env on first call)
settings = get_settings()
print(settings.client.timeout_seconds)  # 300.0

# Override settings programmatically
configure(RolloutSettings(
    client=RolloutClientSettings(timeout_seconds=120.0, max_retries=5),
))
```

Use `reset_settings()` in tests to clear the singleton.

## Server Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_PORT` | `9000` | Default port the RolloutServer binds to. |
| `DEFAULT_HOST` | `"0.0.0.0"` | Default host the RolloutServer binds to (all interfaces). |

## Credential File Location

Authentication tokens from `osmosis login` are stored at `~/.config/osmosis/credentials.json` with owner-only permissions (`0600`). The file uses a multi-workspace format -- use `osmosis workspace list` and `osmosis workspace switch <name>` to manage stored workspaces.

## `.env` File Support

Both the CLI and the settings classes load environment variables from a `.env` file in the current working directory via `python-dotenv`:

```bash
# .env
OSMOSIS_ROLLOUT_CLIENT_TIMEOUT_SECONDS=120
OSMOSIS_ROLLOUT_SERVER_MAX_CONCURRENT_ROLLOUTS=50
OPENAI_API_KEY=sk-...
```

## See Also

- [CLI Reference](./cli.md) -- all `osmosis` commands and options
- [Troubleshooting](./troubleshooting.md) -- common errors and resolutions
