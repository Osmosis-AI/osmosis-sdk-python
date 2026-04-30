# Osmosis Training Configuration
# Copy and customize this file for your training run.
#
# Usage: osmosis --json train submit configs/training/<your-config>.toml
#
# Supported models:
#   - Qwen/Qwen3.6-35B-A3B
#   - Qwen/Qwen3.5-122B-A10B

[experiment]
rollout = "<your-rollout>"        # Rollout name (directory under rollouts/)
entrypoint = "<your-entrypoint-file>" # Entrypoint file name
model_path = "<your-model-path>"     # Must be a supported model (see above)
dataset = "<your-dataset-name>"        # Dataset name from `osmosis --json dataset list`
# commit_sha =                       # Pin to a specific commit (default: latest on default branch)

[training]
# Uncomment and adjust as needed. Defaults are shown.
#
# lr = 1e-6                          # Learning rate
# total_epochs = 1                   # Number of training epochs
# n_samples_per_prompt = 8           # Rollout samples per prompt
# global_batch_size = 64             # Training batch size
# max_prompt_length = 8192           # Max prompt tokens
# max_response_length = 8192         # Max response tokens

[sampling]
# rollout_temperature = 1.0          # Sampling temperature during rollouts
# rollout_top_p = 1.0                # Top-p (nucleus) sampling

[checkpoints]
# eval_interval =                    # Evaluate every N rollouts (optional)
# checkpoint_save_freq = 20          # Save checkpoint every N rollouts
