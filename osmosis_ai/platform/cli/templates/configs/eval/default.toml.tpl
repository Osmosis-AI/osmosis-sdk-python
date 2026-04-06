# Osmosis Eval Configuration
# Usage: osmosis eval run configs/eval/<your-config>.toml

[eval]
module = "<your-module>:<YourWorkflow>"    # AgentWorkflow class (required)
dataset = "<your-dataset>.jsonl"           # Dataset path (required)

[llm]
model = "openai/gpt-5.4"                   # LiteLLM model name (required)
# base_url =                               # Custom OpenAI-compatible endpoint
# api_key_env = "OPENAI_API_KEY"           # Env var name for API key

# [grader]
# Include this section to enable grading. Omit for smoke-test mode.
# Grader is auto-discovered from the workflow module.
# module = "<module>:<YourGrader>"         # Explicit grader (if auto-discovery fails)
# config = "<module>:grader_config"        # Explicit grader config

[runs]
# n = 1                                    # Runs per row (for pass@n)
# batch_size = 1                           # Concurrent batch size
# pass_threshold = 1.0                     # Score threshold for pass@k

[output]
# log_samples = false                      # Save conversations to JSONL
# output_path =                            # Structured output directory

# [baseline]
# model = "openai/gpt-3.5-turbo"          # Baseline model for comparison
# base_url =                               # Baseline endpoint
# api_key_env =                            # Baseline API key env var
