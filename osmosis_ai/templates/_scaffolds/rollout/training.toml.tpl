# Osmosis Training Configuration
# Copy and customize this file for your training run.
#
# Command: osmosis train submit configs/training/<your-rollout>.toml
# For AI agents or automation, use `osmosis --json ...` or `osmosis --plain ...`.
#
# Supported models:
#   - Qwen/Qwen3.6-35B-A3B
#   - Qwen/Qwen3.5-122B-A10B

[experiment]
rollout = "<your-rollout>"  # Rollout name (directory under rollouts/)
entrypoint = "main.py"  # Entrypoint file name
model_path = "<your-model-path>"  # Must be a supported model (see above)
dataset = "<your-dataset-name>"  # Platform dataset name from `osmosis dataset list`
# commit_sha =                        # Pin to a specific commit (default: latest on default branch)

[training]
# Optional. Omit values to use platform defaults.
# Uncomment a field only when you want to override platform behavior.
# lr = 1e-6                           # Learning rate
# total_epochs = 1                    # Number of training epochs
# n_samples_per_prompt = 8            # Rollout samples per prompt
# rollout_batch_size = 32             # Rollout batch size
# max_prompt_length = 8192            # Max prompt tokens
# max_response_length = 8192          # Max response tokens
# agent_workflow_timeout_s = 450      # Agent rollout timeout per row
# grader_timeout_s = 150              # Grader timeout per row

[sampling]
# Optional. Omit values to use platform defaults.
# rollout_temperature = 1.0           # Sampling temperature during rollouts
# rollout_top_p = 1.0                 # Top-p (nucleus) sampling

[checkpoints]
# Optional. Omit values to use platform defaults.
# eval_interval = 10                  # Evaluate every N rollouts
# checkpoint_save_freq = 20           # Save checkpoint every N rollouts

# Environment variables & secrets
# Both sections are optional. Omit them entirely if your rollout needs no
# additional environment configuration.
#
# [rollout.env]
# Literal key = "value" pairs injected verbatim into the rollout container.
# Values are visible in this file and in CLI output - do NOT put secrets here.
# Keys must match ^[A-Z_][A-Z0-9_]*$ and must not be reserved names (see below).
#
# Example:
#   LOG_LEVEL = "INFO"
#   DEFAULT_REGION = "us-west-2"
#
# [rollout.secrets]
# Maps env-var names to workspace environment_secret record *names*.
# Values are resolved server-side from the workspace's encrypted secret store
# and injected into the container - they never appear in this file or in transit.
# Pre-register secrets at /:orgName/secrets in the platform UI before submitting.
#
# Example:
#   OPENAI_API_KEY = "openai-api-key"   # value is the *name* of the secret record
#
# Reserved names (cannot appear in either section - managed by the platform):
#   GITHUB_CLONE_URL  GITHUB_TOKEN  ENTRYPOINT_SCRIPT  REPOSITORY_PATH
#   TRAINING_RUN_ID   ROLLOUT_NAME  ROLLOUT_PORT
