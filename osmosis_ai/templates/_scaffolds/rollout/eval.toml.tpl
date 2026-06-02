# Osmosis evaluation config reference.
# Command: osmosis eval submit configs/eval/<your-rollout>.toml
#
[experiment]
rollout = "<your-rollout>"            # Rollout name
entrypoint = "main.py"                # Entrypoint file name
model_path = "openai/gpt-5-mini"      # LiteLLM-style model name
dataset = "<your-dataset-name>"       # Platform dataset name
# commit_sha =                        # Pin to a specific commit

[evaluation]
# Optional. Omit values to use platform defaults.
# Uncomment a field only when you want to override platform behavior.
# limit = 200                         # Optional row cap
# n = 1                               # Number of evaluation attempts
# batch_size = 1                      # Rows evaluated per batch
# pass_threshold = 1.0                # Minimum passing score
# agent_workflow_timeout_s = 450      # Agent workflow timeout per row
# grader_timeout_s = 150              # Grader timeout per row

# [env]
# Literal key = "value" pairs injected verbatim into the rollout container.
# Values are visible in this file and in CLI output - do NOT put secrets here.
# Keys must match ^[A-Z_][A-Z0-9_]*$ and must not start with _OSMOSIS_.
# LOG_LEVEL = "INFO"

[secrets]
# Default OpenAI eval models need this platform secret.
# Use [] only when this evaluation needs no secret refs.
# Register secret records first with `osmosis secret set NAME` (personal is the default),
# or pass `--scope workspace` for workspace-shared secrets.
required = ["OPENAI_API_KEY"]
