# `<your-rollout>`

Placeholder rollout created by `osmosis rollout init`. Edit `main.py` to fill in
`MyAgentWorkflow.run()` and `MyGrader.grade()`, then run its declarative server
config from the workspace directory:

```bash
uv run --project rollouts/<your-rollout> \
  osmosis rollout serve rollouts/<your-rollout>/rollout.toml
```

Submit an evaluation run from the workspace directory:

```bash
pip install -e rollouts/<your-rollout>
osmosis eval submit configs/eval/<your-rollout>.toml
```

Once your rollout is ready, submit a training run:

```bash
git push   # Git Sync must be connected in the Osmosis Platform
osmosis train submit configs/training/<your-rollout>.toml
```
