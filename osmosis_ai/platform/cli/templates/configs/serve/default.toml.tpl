# Osmosis Serve Configuration
# Usage: osmosis rollout serve configs/serve/<your-config>.toml

[serve]
rollout = "<your-rollout>"                     # Rollout name (directory under rollouts/)
entrypoint = "<your-entrypoint-file>"          # Entrypoint file (relative to rollout dir)

[server]
# port = 9000                                  # Port to bind to
# host = "0.0.0.0"                             # Host to bind to
# log_level = "info"                           # Uvicorn log level

[debug]
# no_validate = false                          # Skip startup validation
# trace_dir =                                  # Directory for execution traces (disabled if omitted)
