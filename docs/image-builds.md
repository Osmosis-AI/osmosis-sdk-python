# Repository image builds

The [command shell](../osmosis_ai/cli/commands/images.py) delegates to
[images.py](../osmosis_ai/platform/cli/images.py) and
[OsmosisClient](../osmosis_ai/platform/api/client.py). Monolith owns source
fetching, task discovery, Cloud Build, batching, and image publication.

```bash
osmosis images build --repo https://github.com/acme/training-job
osmosis images build --repo git@github.com:acme/training-job.git \
  --ref experiment --tasks-dir harbor/tasks --output-dir .osmosis/experiment
osmosis images info REQUEST_UUID --repo https://github.com/acme/training-job
```

The base SDK install is sufficient; no local Docker, Google credentials, or
Harbor extra is required for these commands. The repository must be connected
to a workspace accessible to the authenticated internal FDE account, with
`dev_servers` write permission for submission and read permission for status.
Monolith's GitHub App must be installed on that repository.

## Submission contract

`POST /api/cli/image-builds` accepts:

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "repository": "https://github.com/acme/training-job",
  "ref": "main",
  "tasks_dir": "tasks"
}
```

`ref` defaults to `HEAD` (the repository's default branch); `tasks_dir` defaults
to `tasks`. The SDK normalizes HTTPS and SSH URLs and sends the same repository
identity in the body and `X-Osmosis-Git`. The API can derive that scope from the
repository body for callers without a workspace header. It checks workspace
membership and repository identity before forwarding a server-derived
organization and GitHub installation ID. The request body contains no task array,
repository credentials or registry credentials. The CLI sends your Osmosis token
in the Authorization header to authenticate to the platform.

Submission returns HTTP 202 with `request_id`, `status_url` and the initial
status. `GET /api/cli/image-builds/{request_id}` reports `phase`,
`source_revision`, `task_count`, `image_count`, `completed_image_count`,
`failed_image_count`, errors and the completed result. The final manifest
contains the full task/environment-to-image mapping and Cloud Build links;
large inventories are not returned in Temporal workflow payloads.

## Repository layout

Monolith finds task roots recursively under `tasks_dir`, stopping at each
`task.toml` so nested fixtures are not additional tasks. A task's
`environment.docker_image` is mirrored; otherwise its `environment/Dockerfile`
is built with the environment directory as context. Separately configured
verifier images are built or mirrored too. A prebuilt image does not incorporate
local context files; use a Dockerfile with `FROM` and `COPY` for that.

An optional repository-root `image-build.toml` declares a trainer image:

```toml
[trainer]
context = "images/trainer"
build_timeout_sec = 7200
```

Alternatively set `docker_image` to a registry image, preferably pinned by
digest, to reuse it. The trainer appears in `named_images.trainer` and is not
included in the training tasks. The builder must have access to private images.
Git submodules, Git LFS task files, task symlinks, and Compose environments are
unsupported. Archives are limited to 2 GiB compressed/extracted and 100,000
entries; the Git path has no 1,000-task request cap.

## Resume and artifacts

The CLI writes `request.json` before submitting, so an ambiguous response can
be retried with the same request UUID. Monolith pins the commit before fetching
the snapshot and keeps that commit even if the branch moves. Reusing an output
directory resumes that build. Use a new directory for a new revision or after
fixing a terminal build failure.

`--no-wait` submits and returns. Otherwise the CLI waits up to `--timeout`
seconds (10,800 by default), then reports that remote work continues. Rerun
the same command to resume; local timeout or interruption does not cancel builds.

During submission and polling, `status.json` records the latest progress or failure
details, including with `--no-wait`. After completion the CLI obtains fresh
download capabilities and validates
checksums, source revision, task count, and agent/verifier image bindings. It
writes `manifest.json`, `bundle.tar.gz`, `images.json` and prepared `tasks/`
under `--output-dir` (default `.osmosis/images`). It refuses to replace locally
edited task files. Signed URLs and registry credentials are not saved in state.

The one-hour transfer/pull capability lifetime is independent of batch duration.
The Python client's `get_image_pull_credentials` returns a short-lived token
for a published image; callers must keep it out of output and refresh it before
later pulls. It is repository-wide read access to the managed registry, so the
API remains internal-FDE-only.
