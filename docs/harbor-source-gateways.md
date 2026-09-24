# Managed Harbor source gateways

Build the source environments first with the [image-build CLI](image-builds.md).
Start the gateway at the full `source.revision` returned in its verified JSON.

```bash
# Use source.revision from images.json for the gateway.
osmosis dev server up --url https://github.com/acme/training-tasks \
  --path tasks --ref FULL_COMMIT_SHA --backend gke \
  --sandbox-environment opensandbox

osmosis dev server list --url https://github.com/acme/training-tasks
osmosis dev server logs SERVER_ID --url https://github.com/acme/training-tasks
osmosis dev server down SERVER_ID --url https://github.com/acme/training-tasks
```

The gateway runs built-in SDK code and fetches the original pinned source;
it needs no local `main.py`, dataset download, or task-image JSON file. Builder and gateway use
the same GitHub archive fetcher so checkout filters cannot change build bytes. It
independently computes image identities, resolves all required GAR tags to
verified manifest digests, and prewarms representative agent environments
before accepting rollouts. Each trial gets a temporary task copy bound to those
digests. Source files stay unchanged. A gateway serves one pinned source;
requests select tasks using `metadata.harbor_task_id`.

Registry lookup uses a short-lived read credential delivered through the
managed gateway's secret bundle. Once startup resolves the source, trials use
cached immutable digests and GKE's image-pull identity. Missing images or failed
prewarm prevent readiness. The managed service must support source gateways
and have access to the source image repository.

Managed credentials take precedence over local Docker configuration. Direct
resolver use supports inline Docker `auth` entries; Docker `credHelpers` and
`credsStore` configurations are rejected with an explicit instruction to use a
managed source gateway.

Use `dev server up --config gateway.json` to configure the managed Harbor agent
without adding Python code or changing the task source. For example:

```json
{
  "agent": "opencode",
  "concurrency": 4,
  "native_agent_kwargs": {"version": "1.18.27"},
  "environment_kwargs": {"use_server_proxy": true},
  "cleanup_successful_trials": false
}
```

`native_agent_kwargs` and `environment_kwargs` are passed to Harbor's native
agent and OpenSandbox configuration. Source gateways use OpenSandbox's server
proxy by default and derive its protocol from the configured service URL;
explicit environment options take precedence. Optional `environment_healthcheck` uses
Harbor's `command`, timing and retry fields; it adds readiness requirements to
the agent environment in temporary trial copies, preserving the original task
and separate verifier healthchecks. Configuration defaults and validation live
in `HarborGatewayConfig`. Keep configuration containing private routing or
credentials outside Git. The managed service stores it in the encrypted gateway
secret bundle. Configuration requires the source `--url` path; custom-code
gateways continue to configure their own backend.
