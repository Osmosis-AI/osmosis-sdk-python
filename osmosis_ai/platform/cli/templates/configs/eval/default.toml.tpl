# Osmosis Eval Configuration
# Usage: osmosis eval run configs/eval/<your-config>.toml

[eval]
rollout = "<your-rollout>"
entrypoint = "main.py"
dataset = "data/<your-dataset>.jsonl"

[llm]
model = "openai/gpt-5-mini"
api_key_env = "OPENAI_API_KEY"

[runs]
n = 1
batch_size = 1
pass_threshold = 1.0

[timeouts]
agent_sec = 450
grader_sec = 150

[output]
log_samples = false
