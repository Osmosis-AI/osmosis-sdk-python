# Skill: Create Environment

## Goal
Help the user create a new environment in this Osmosis workspace.

## Steps

1. Understand the user's task/domain (what should the agent learn to do?)
2. Choose an environment name (lowercase, hyphens or underscores)
3. Create `environments/<env_name>/` via `osmosis env init <env_name>` or manually
4. Design tools the agent will need (as plain Python functions)
5. Design grader functions (how to score agent performance)
6. Decide on system prompt and max_turns
7. Implement in `environments/<env_name>/main.py` with `load_environment()`
8. Validate: `osmosis test <env_name> -m gpt-4.1-mini`

## Environment Structure

Each environment must have:
```
environments/<env_name>/
├── main.py          # Must export load_environment() -> Environment
├── pyproject.toml   # Per-env dependencies
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

## Validation

After creating the environment, always run:
```bash
osmosis test <env_name> -m gpt-4.1-mini
```
