# `<your-rollout>`

Placeholder rollout created by `osmosis rollout init`. Edit `main.py` to fill in
`MyAgentWorkflow.run()` and `MyGrader.grade()`.

Local dev loop from the project root:

```bash
pip install -e rollouts/<your-rollout>
osmosis eval run configs/eval/<your-rollout>.toml --limit 1
```

Once your rollout produces non-zero rewards locally, submit a training run:

```bash
git push   # Git Sync must be connected in the Osmosis Platform
osmosis train submit configs/training/<your-rollout>.toml
```
