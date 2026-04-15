# Skill: Create Rollout

## Goal
Help the user create a new rollout in this Osmosis workspace.

## Steps

1. Understand the user's task/domain (what should the agent learn to do?)
2. Choose a rollout name (lowercase, hyphens or underscores)
3. Create `rollouts/<rollout_name>/` manually
4. Design tools the agent will need (as plain Python functions)
5. Design grader functions (how to score agent performance)
6. Decide on system prompt and max_turns
7. Implement in `rollouts/<rollout_name>/main.py` with a concrete `AgentWorkflow` and a concrete `Grader`
8. Create `configs/eval/<rollout_name>.toml` (see Eval Config Template below)
9. Validate: `osmosis eval run configs/eval/<rollout_name>.toml`

## Rollout Structure

Each rollout must have:
```
rollouts/<rollout_name>/
├── main.py          # Define a concrete AgentWorkflow and Grader
├── pyproject.toml   # Per-rollout dependencies
└── README.md        # Description
```

## Tool Design Guidelines

- Tools are plain async Python functions with type hints
- SDK auto-generates OpenAI tool schemas from signature + docstring
- Keep tools focused: one action per function
- Return strings (model reads the output)
- Use descriptive docstrings (the model sees them)

## Grader Design Guidelines

- Grader functions receive `messages` (full conversation) and `answer` (ground truth)
- Return float between 0.0 and 1.0
- Always include `**kwargs` for forward compatibility
- Consider partial credit (not just 0/1)

## Eval Config Template

Create `configs/eval/<rollout_name>.toml` with:

```toml
[eval]
rollout = "<rollout_name>"
entrypoint = "main.py"
dataset = "path/to/dataset.jsonl"

[llm]
model = "openai/gpt-4o"

[runs]
n = 1
batch_size = 1
```

## Validation

After creating the rollout and its eval config, always run:
```bash
osmosis eval run configs/eval/<rollout_name>.toml
```
