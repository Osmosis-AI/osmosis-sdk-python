# Rollout ownership

Servers built with `create_rollout_server` can export ownership events over
OTLP/HTTP. Install the `server` extra and set `OSMOSIS_ROLLOUT_OTLP_ENDPOINT`
to the collector base URL (without `/v1/logs`). No events are exported when
the endpoint is unset.

The platform sets `OSMOSIS_ROLLOUT_SERVER_ID`, `OSMOSIS_ROLLOUT_NAMESPACE`
and `_OSMOSIS_ROLLOUT_NAME`. A standalone server can set those too; otherwise
the server ID falls back to its instance ID or hostname. Client metadata cannot
override server identity.

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

Export runs in a background batch processor with a 1,024-record queue. Ownership
logs are best-effort diagnostics; an exporter outage can drop events and never
blocks rollout execution. A status is the last observed event, not proof that a
server is still alive. Use the gateway health and in-flight metrics alongside it.
