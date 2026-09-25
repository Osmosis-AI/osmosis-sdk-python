# Rollout lifecycle and native evidence

`RolloutClient` supports authenticated health checks and bounded draining. The
server owns lifecycle counters, so clients can use these operations with any
execution backend.

```python
from osmosis_ai.rollout.client import RolloutClient

client = RolloutClient(url="https://rollout.example", api_key="server-bearer-key")
try:
    health = await client.health()
    drained = await client.drain(timeout_sec=30)
    if not drained.drained:
        drained = await client.drain(timeout_sec=30)
    # Persist process_id and rollout_ids with any export request.
finally:
    await client.aclose()
```

Pass the same key to `create_rollout_server(backend=backend, api_key=...)` to
protect all server routes, or authenticate them in the hosting gateway. The
client's server key is separate from the per-rollout `llm_api_key`. Supplying a
custom `httpx.AsyncClient` preserves its transport settings and existing headers;
the rollout client adds its own bearer header per request and refuses redirects.
TLS verification remains enabled by default.

`GET /health` preserves backend health fields and adds:

```json
{
  "instance_id": "deployment-or-instance-identifier",
  "process_id": "unique-process-uuid",
  "lifecycle": {"accepting_rollouts": true, "active_rollouts": 0}
}
```

`active_rollouts` covers admitted work through execution, native evidence
retention, trajectory saving and result finalization. It is independent of a
backend's queue counters. `await client.wait_idle(timeout_sec=120)` observes this
counter without changing admissions. Older servers without lifecycle metadata
are rejected. Pass `process_id=` to bind the wait to a known process.

Backend outcome status and cancellation dispositions describe execution, so a
terminal backend outcome or `not_found` cancellation can precede archive
completion. Use the server's lifecycle counters and drain response to establish
that retained evidence has finished writing.

`POST /drain` accepts `{"timeout_sec": 30}` with a finite timeout from 0 to 300
seconds. It fences new admissions before taking the active-work snapshot. It
returns HTTP 200 with `accepting_rollouts: false`, `active_rollouts`, `drained`,
`instance_id`, `process_id`, and sorted `rollout_ids` containing every admitted ID
in that process. New rollout submissions receive HTTP 503. A timeout returns
`drained: false` and leaves the fence set; repeating drain is safe. Result polling
and cancellation remain available, and polling leases still require renewal.
The drain request does not cancel admitted work.

There is no reopen operation. A new process starts accepting again, and has a new
`process_id` even when its configured `instance_id` is unchanged. A completed
drain is valid only for that process: check its identity again before publishing
an export or retiring its resources. Shutdown also fences admissions before its
bounded drain and cancellation cleanup.

## Native Harbor evidence

The backend retains a sanitized copy of Harbor's native evidence
at `<artifact_root>/<rollout_id>/harbor/`. Harbor rejects rollout IDs containing
colons or control characters before starting trial work so their evidence paths
remain portable. Other backends keep their existing rollout ID contract.
The artifact root uses the existing
`_OSMOSIS_ROLLOUT_ARTIFACT_ROOT` setting. The copy contains `result.json`, selected
native logs under `logs/`, and `manifest.json`. Configuration and task source
files are excluded. This is an execution-evidence projection, not a backup of
application databases, snapshots, configuration or caches. Existing `artifacts/`,
`logs/` and `trajectory.json` keep their established locations.

The `harbor-native-logs-v1` selection policy excludes only these recognized
OpenCode application-state entries beneath a root or step-local `agent/` or
`user-agent/` directory:

- `opencode/xdg-data/opencode/opencode.db`, `opencode.db-wal` and `opencode.db-shm`.
- `opencode/xdg-data/opencode/snapshot/` (the Git snapshot store).

Sibling text logs, session exports, canonical trajectories, stdout/stderr and
verifier records remain selected. Other binary files do not become exclusions
merely because they are beneath an XDG directory. The manifest's
`selection_policy` names this policy; `excluded` counts recognized database files
and snapshot directories using fixed category names, without recording dynamic
source paths.

The manifest uses `schema_version: "harbor-evidence-v1"` and includes
`rollout_id`, `process_id`, `complete`, `errors`, and `files`. Each file entry contains its
relative `path`, `size_bytes`, and hexadecimal `sha256`. Publication replaces a
staged directory containing its finished manifest, so consumers must verify the
inventory before transferring it and publish the manifest after all files.

Credential fields, the supplied model key, bearer tokens, recognizable secret
assignments and URL user credentials are redacted. Only UTF-8 files up to 64 MiB
per file can be sanitized; links, special files, unreadable files, invalid native
results and larger/binary **selected** files produce an explicitly incomplete
manifest. Completeness covers every selected native record; it does not claim
to contain the excluded application state.

The existing private `logs/` copy retains application state outside the managed
native export after ordinary completion and upstream credential scrubbing.
Incomplete retention prevents deleting that original trial working copy.
Cancelled or interrupted submissions may skip the upstream scrub, so they export
only the sanitized native surface and do not create private raw-log copies.
Cancelled working copies, including application state, are removed after the
retention attempt; their manifest records any selected-evidence incompleteness.
Interrupted submissions keep their scrubbed working copy. Credential-bearing
staging is always removed, and failed cancellation cleanup is surfaced.
Private logs and working copies have their existing server-storage lifetime;
they are not a durable application-state backup. Retention never changes reward.

Use the portable integrity helpers without importing Harbor:

```python
from pathlib import Path
from osmosis_ai.rollout.utils.evidence import (
    trial_evidence_inventory,
    verify_trial_evidence,
)

manifest = verify_trial_evidence(
    Path("download/rollout-1/harbor"), "rollout-1", process_id=drained.process_id
)
inventory = trial_evidence_inventory(
    Path("download"), drained.rollout_ids, process_id=drained.process_id
)
assert inventory["complete"]  # Missing or partial rollouts are listed explicitly.
```

Evidence inventories do not decide which rollouts a trainer consumed. Keep the
trainer's consumed-ID list separately and join it to the retained rollout IDs;
prefetched or cancelled work remains distinguishable. A missing native result
is incomplete evidence, even if the server successfully finalized a failure or
cancellation response.

The server propagates its process identity through `RolloutContext` into native
manifests. Verify it against the successful drain to reject evidence left by an
earlier process, including reused rollout IDs. Direct backend executions without
a server context have `process_id: null`.

## Source image values

`osmosis_ai.source_images.SourceImageManifest` validates a published `source-v1`
manifest whose task bindings reference image keys. Its `.resolved()` method
returns `SourceImageBuildResult`, whose bindings contain complete image records.
Both validate the existing `TaskSource`, pinned digests and inventory consistency.
The resolved result's `.select_task_ids()` validates a requested subset and
returns sorted unique IDs. These types contain no authentication, transport or
workspace lookup behavior. Native gateway options use the existing
`osmosis_ai.rollout.types.harbor.HarborGatewayConfig`.
