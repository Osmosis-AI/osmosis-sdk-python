# Osmosis eval config reference.
# Command: osmosis eval submit configs/eval/<your-rollout>.toml

[experiment]
rollout = "<your-rollout>"            # Rollout name
entrypoint = "main.py"                # Entrypoint file name
dataset = "<your-dataset-name>"       # Platform dataset name
# commit_sha =                        # Pin to a specific commit

[llm]
model_path = "openai/gpt-5-mini"
base_url = "https://api.openai.com/v1"

[evaluation]
# limit = 200                         # Optional row cap
n = 1                                 # Number of eval runs
batch_size = 1                        # Rows evaluated per batch
pass_threshold = 1.0                  # Minimum passing score
agent_workflow_timeout_s = 450        # Agent workflow timeout per row
grader_timeout_s = 150                # Grader timeout per row

[env]
LOG_LEVEL = "INFO"

[secrets]
OPENAI_API_KEY = "openai-api-key"
