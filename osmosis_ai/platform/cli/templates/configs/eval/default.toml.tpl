# Osmosis Eval Configuration
# Copy and customize this file for a local eval run.
#
# Usage: osmosis eval run configs/eval/<your-config>.toml
# For AI agents or automation, use `osmosis --json ...` or `osmosis --plain ...`.

[eval]
rollout = "<your-rollout>"                 # Rollout name (directory under rollouts/)
entrypoint = "<your-entrypoint-file>"      # Entrypoint file relative to the rollout
dataset = "data/<your-dataset>.jsonl"      # Dataset path under data/
# limit = 100                              # Evaluate at most N rows
# offset = 0                               # Skip the first N rows
# fresh = false                            # Ignore cached results and rerun
# retry_failed = false                     # Retry cached failed rows

[llm]
model = "openai/gpt-5-mini"                # Model passed to LiteLLM
# base_url = "https://api.openai.com/v1"    # Override the provider base URL
# api_key_env = "OPENAI_API_KEY"            # Require this env var before eval

[runs]
# Uncomment and adjust as needed. Defaults are shown.
#
# n = 1                                    # Attempts per dataset row
# batch_size = 1                           # Rows evaluated concurrently
# pass_threshold = 1.0                     # Fraction of attempts required to pass

[timeouts]
# agent_workflow_timeout_s = 450           # Agent rollout timeout per row
# grader_timeout_s = 150                   # Grader timeout per row

[output]
# log_samples = false                      # Persist full sample records in cache
# output_path = "eval-results.json"         # Optional summary JSON output path
# quiet = false                            # Reduce console output
# debug = false                            # Show debug details
