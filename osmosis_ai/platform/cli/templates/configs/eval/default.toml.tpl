# Osmosis Eval Configuration
# Usage: osmosis --json eval run configs/eval/<your-config>.toml

[eval]
rollout = "<your-rollout>"                 # Rollout name (directory under rollouts/)
entrypoint = "<your-entrypoint-file>"      # Entrypoint file (relative to rollout dir)
dataset = "<your-dataset>.jsonl"           # Dataset path (relative to workspace root)
# limit =                                  # Max rows to evaluate
# offset = 0                               # Skip first N rows
# fresh = false                            # Discard cached results
# retry_failed = false                     # Re-run only failed

[llm]
model = "openai/gpt-5.4"                   # LiteLLM model name (required)
# base_url =                               # Custom OpenAI-compatible endpoint
# api_key_env = "OPENAI_API_KEY"           # Env var name for API key

# Grader is auto-discovered from the rollout package.
# No [grader] section needed — define a Grader subclass in your rollout.

[runs]
# n = 1                                    # Runs per row (for pass@n)
# batch_size = 1                           # Concurrent batch size
# pass_threshold = 1.0                     # Score threshold for pass@k

[output]
log_samples = false                        # Save conversations to JSONL (or use --log-samples flag)
# output_path =                            # Structured output directory
# quiet = false                            # Suppress progress output
# debug = false                            # Enable debug logging + trace

# [baseline]
# model = "openai/gpt-3.5-turbo"          # Baseline model for comparison
# base_url =                               # Baseline endpoint
# api_key_env =                            # Baseline API key env var
