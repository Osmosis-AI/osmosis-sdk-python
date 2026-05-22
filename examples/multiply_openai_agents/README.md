# multiply_openai_agents

Minimal end-to-end Osmosis rollout server using the OpenAI Agents SDK.

A single-file FastAPI server (`main.py`) that exposes the standard rollout
contract (`POST /rollout`, `GET /health`). On each rollout the workflow
spins up an `OsmosisAgent` with one `multiply` tool, runs it against the
trainer's session URL via LiteLLM, and the grader scores the final
`Answer: <number>` against the dataset label.

Use this as the smoke test when you bring up multi-LoRA + remote rollout
on a fresh cluster.

## Run standalone (sanity check)

From this directory:

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e '../..[server]' 'openai-agents>=0.14,<0.15' uvicorn

python main.py &
curl http://localhost:8080/health     # -> {"status":"ok", ...}
kill %1
```

Standalone the server boots but won't drive a rollout end-to-end — there's
no chat-completions URL to call. The full flow is driven by the trainer
once you register an adapter that points at it.

## Connect to the multi-LoRA + remote-rollout cluster

Assumes you've already done `traingate multi-lora start ...` and the head
pod is reachable. Get the head pod name:

```bash
POD=$(kubectl get pods -l 'skypilot-cluster=<CLUSTER>' -o name | head -n1)
```

Copy the example into the pod and start it on `localhost:8080`:

```bash
tar czf /tmp/multiply.tgz -C .. multiply_openai_agents
kubectl cp /tmp/multiply.tgz "$POD":/tmp/multiply.tgz
kubectl exec "$POD" -- bash -lc '
  set -e
  mkdir -p /tmp/multiply && tar xzf /tmp/multiply.tgz -C /tmp/multiply
  cd /tmp/multiply/multiply_openai_agents
  pip install -q -e /osmosis/workspace/sdk-python"[server]" \
              "openai-agents>=0.14,<0.15"
  OPENAI_API_KEY='"$OPENAI_API_KEY"' ROLLOUT_PORT=8080 \
    nohup python main.py > /tmp/multiply.log 2>&1 &
'
kubectl exec "$POD" -- curl -fsS http://localhost:8080/health
```

Register an adapter pointing at it (run from `osmosis-traingate/`):

```toml
# /tmp/multiply.toml
[config]
data = "/osmosis/data/datasets/multiply/train.parquet"
input_key = "messages"
label_key = "label"
rm_type = "math"
rank = 16
alpha = 16
num_row = 20

[config.metadata]
rollout_server_url = "http://localhost:8080"
```

```bash
uv run traingate multi-lora register multiply \
  --cluster <CLUSTER> --spec /tmp/multiply.toml
```

Tail both sides; you should see chat completions flowing within seconds:

```bash
# rollout server
kubectl exec "$POD" -- tail -f /tmp/multiply.log

# trainer
sky logs <CLUSTER> --follow | rg -i 'rollout|adapter|session-server'
```

## Dataset shape

Each row in the parquet file needs the keys named in the adapter spec
(`input_key`, `label_key`). For this example:

```python
{
    "messages": [
        {"role": "user", "content": "What is 23 times 41?"},
    ],
    "label": "943",
}
```

The workflow treats `ctx.prompt` as the verbatim message list to seed the
agent's session.
