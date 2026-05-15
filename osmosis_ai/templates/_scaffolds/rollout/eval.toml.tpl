# Osmosis eval config reference.
# Command: osmosis eval run configs/eval/<your-rollout>.toml

[eval]
# Rollout name must match a directory under rollouts/.
rollout = "<your-rollout>"
entrypoint = "main.py"
# Local eval datasets must live under data/.
dataset = "data/<your-dataset>.jsonl"
# Optional: cap the number of rows read from the dataset.
# limit = 200

[llm]
model = "openai/gpt-5-mini"
# api_key_env = "OPENAI_API_KEY"
# base_url = "http://localhost:8080"

[runs]
n = 1
batch_size = 1
pass_threshold = 1.0

[timeouts]
agent_workflow_timeout_s = 450
grader_timeout_s = 150

[output]
log_samples = false
