# `<your-rollout>`

Placeholder rollout created by `osmosis rollout init`. Edit `main.py` to fill in
`MyAgentWorkflow.run()` and `MyGrader.grade()`.

Submit a cloud eval from the workspace directory:

```bash
pip install -e rollouts/<your-rollout>
osmosis eval submit configs/eval/<your-rollout>.toml
```

Once your rollout is ready, submit a training run:

```bash
git push   # Git Sync must be connected in the Osmosis Platform
osmosis train submit configs/training/<your-rollout>.toml
```
