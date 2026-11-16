# Rollout ownership

Implementation: [server/observability.py](../osmosis_ai/rollout/server/observability.py), wired by [server/app.py](../osmosis_ai/rollout/server/app.py). Contract tests: [test_server_observability.py](../tests/unit/rollout/test_server_observability.py). For client, workflow, and platform responsibilities, see [runtime boundaries](architecture.md#runtime-boundaries).

Servers built with `create_rollout_server` can export ownership events over
OTLP/HTTP. Install the `server` extra and set `OSMOSIS_ROLLOUT_OTLP_ENDPOINT`
to the collector base URL (without `/v1/logs`). No events are exported when
the endpoint is unset.

The platform sets `_OSMOSIS_ROLLOUT_OTLP_ENDPOINT`, `_OSMOSIS_ROLLOUT_SERVER_ID`, `_OSMOSIS_ROLLOUT_NAMESPACE` and `_OSMOSIS_ROLLOUT_NAME`. A standalone server can set `OSMOSIS_ROLLOUT_SERVER_ID` and `OSMOSIS_ROLLOUT_NAMESPACE`; otherwise the server ID falls back to its instance ID or hostname. Client metadata cannot override server identity.

Clients can attach a training run to each request:

```python
metadata = {
    "osmosis_observability": {
        "run_id": "training-run-id",
        "run_name": "GLM DeepSWE",
    }
}
```

The `osmosis-rollout-server` service emits `event=rollout.ownership` with
`rollout_id`, `server_id`, `server_name`, `namespace`, `run_id`, `run_name` and
`status`. Admission, running, grading and terminal states are recorded, including
cancellation and polling-lease expiry. Missing run metadata stays absent.
Prompts, arbitrary metadata, callback URLs, credentials and error messages are
never included. Run fields must be nonempty strings of at most 512 characters.

The terminal event (`success`, `failure` or `cancelled`) also summarizes where the rollout spent its time, so slow or failing stages can be aggregated by `run_id` without reading trajectories:

| Attribute | Meaning |
|-----------|---------|
| `err_category` | The rollout's error category, when it has one. Never the error message. |
| `total_sec` | Admission to the terminal event. |
| `startup_sec` | Admission to agent start: concurrency queueing plus, for Harbor, sandbox and agent setup. |
| `agent_sec` | Agent start to grading start, or to the terminal event when grading never started. |
| `grading_sec` | Grading start to the terminal event, including trajectory saving. |
| `phase_<name>_sec` | Backend-reported phase durations. Harbor reports `environment_setup`, `agent_setup`, `agent`, `verifier` and `total` from its trial timing records. |

The server computes `total_sec` through `grading_sec` from wall-clock UTC timestamps. Each is absent when its stage never started, or when a backend reported a timestamp without a timezone. Backends report `phase_<name>_sec` values directly as durations in seconds through `RolloutContext.record_phase_timings`. Only identifier names with finite numeric values are exported. All durations are clamped at zero and rounded to milliseconds.

Export runs in a background batch processor with a 1,024-record queue. Ownership
logs are best-effort diagnostics; an exporter outage can drop events and never
blocks rollout execution. A status is the last observed event, not proof that a
server is still alive. Use the gateway health and in-flight metrics alongside it.
