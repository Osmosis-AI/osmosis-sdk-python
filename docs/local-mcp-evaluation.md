# Optional local stdio MCP bridge evaluation

## Decision and scope

Keep direct CLI JSON as the default for coding agents with shell access. The local bridge is a **go for a source-only, read-only experiment, and no-go for a public installed feature at this stage**. Its useful difference is discoverable, typed tools with bounded pagination and fixed local context; the evidence does not establish better agent task completion than CLI JSON. The hosted Platform MCP remains an independent server and release: it must never launch, install, or depend on this process.

The prototype builds on the SDK’s existing explicit workspace selection, platform-scoped persistent authentication, and keyring/file credential storage. It does not change SDK runtime code, default dependencies, console scripts, or the lockfile.

## Prototype contract

The entry point is [bridge.py](../prototypes/stdio_mcp/bridge.py). Launch it with an absolute repository root and an explicit Platform workspace name. These are separate concepts: the local directory only supplies config filenames; the remote workspace selects the authorized tenant. A repository name or Git remote never supplies authorization. The human chooses the workspace at process launch; tools cannot switch it, change the directory, change the platform, pass credentials, choose URLs, or run commands.

| Tool | Input | Output |
| --- | --- | --- |
| `list_local_configs` | `kind`: `training`, `eval`, or `benchmark` | The lexicographically first 100 UTF-8-encodable immediate `.toml` filenames under `configs/<kind>/`, selected workspace, and a truncation flag; no file contents |
| `list_datasets` | `limit`: 1–50, `offset`: 0–1,000,000 | Dataset ID, filename, status, row count, total count, `has_more`, and `next_offset` |
| `list_training_runs` | Same bounded page input | Run ID, name, status, creation time, and the same pagination fields |

Input models reject unknown fields and type coercion. Each tool advertises an output JSON schema and read-only/idempotent annotations. Results have `{ "data": ..., "error": null }`; failures have `data: null`, a fixed error code, and MCP `isError: true`. Codes are `INVALID_ARGUMENTS`, `AUTH_REQUIRED`, `CREDENTIALS_UNAVAILABLE`, `FORBIDDEN`, `UPGRADE_REQUIRED`, `PLATFORM_UNAVAILABLE`, `INVALID_RESPONSE`, and `LOCAL_CONTEXT_UNAVAILABLE`. Error bodies, tracebacks, credential objects, dataset previews, run config/environment, upload URLs, creator emails, and arbitrary upstream URLs are never projected into tool results. Names and other allowed text remain untrusted data. Output serialization is validated before the result reaches the protocol writer; non-serializable upstream text returns `INVALID_RESPONSE`.

The bridge directly reuses SDK credential loading, token-to-platform binding, CLI version headers, and `PaginatedDatasets` / `PaginatedTrainingRuns` parsing. It deliberately supplies a small GET-only HTTP transport rather than calling CLI handlers or subprocesses. The current general SDK request client also owns CLI diagnostics and redirect behavior; an isolated transport lets the experiment reject redirects and ambient proxies, enforce a 1 MiB response cap, and return fixed errors without changing existing callers. The two endpoint names and the projection are the explicit additional maintenance cost. A production bridge would justify extracting a shared noninteractive transport instead of growing this adapter.

Platform reads use cancellable async HTTP streaming with a ten-second total deadline covering admission, credential lookup, connection, headers, and the complete response body. At most four remote calls run concurrently per bridge process; queued calls share the same total deadline and local listing remains independent. Deadline expiry returns `PLATFORM_UNAVAILABLE`; MCP cancellation and stdin EOF close active HTTP responses and connections. Filesystem operations use `asyncio.to_thread`; credential loading uses at most one separate daemon thread per settings instance. Concurrent calls share an unfinished credential read, shielded from individual request cancellation, so repeated timeouts cannot accumulate workers or exhaust the filesystem executor. A synchronous keyring read cannot be forcibly interrupted; it may remain blocked until the backend recovers or the process exits, but it cannot hold the process open on stdin EOF. Each call retains its own deadline, and calls after the read completes perform a fresh lookup through the existing env → platform-specific keyring/file flow. A read-only 401 never starts login, refreshes, revokes, removes, or rewrites them. Warnings from credential loading use the existing plain, noninteractive output context on stderr. The protocol uses the low-level MCP SDK server with fixed, redacted MCP protocol messages on stderr and needs no Rich, prompts, or TTY.

Local listing opens directory descriptors with `O_NOFOLLOW`, rejects descendant directory symlinks, and skips symlink files and filenames containing surrogate characters that cannot be encoded as UTF-8. It scans all immediate entries while retaining only the smallest 101 matching paths, making both the first 100 results and the truncation flag independent of directory enumeration order with bounded memory. It has no arbitrary path parameter and never reads TOML values, `.env`, Git config, or repository instructions. The explicit root is resolved once, and no tool derives context from process CWD. The experiment is limited to macOS/Linux and ASCII workspace names matching `[A-Za-z0-9][A-Za-z0-9_.-]{0,127}`. Windows and other existing workspace-name forms are outside this prototype's supported surface.

## Run the experiment

Use a configured repository, an existing `osmosis auth login`, and an isolated Python environment. Set `OSMOSIS_PLATFORM_URL` in the launcher if using a non-default platform; repository `.env` discovery is intentionally disabled. Normal non-production environment-token binding rules still apply. Remote platforms require HTTPS; loopback HTTP is permitted for local development. Never place a token in an MCP tool argument or a committed client config.

```bash
uv run --no-project \
  --with-editable /absolute/path/to/osmosis-sdk-python \
  --with-requirements /absolute/path/to/osmosis-sdk-python/prototypes/stdio_mcp/requirements.txt \
  python /absolute/path/to/osmosis-sdk-python/prototypes/stdio_mcp/bridge.py \
  --workspace-directory /absolute/path/to/configured-repository \
  --workspace acme
```

Use this executable and argument vector as a stdio MCP server in a client. No HTTP port is opened. A missing login or rejected authentication is returned as `AUTH_REQUIRED`; complete login in a separate terminal and retry the tool, and check platform binding when using an environment token. Credential storage failures return `CREDENTIALS_UNAVAILABLE`; unlock or repair the keyring, or check the credential file’s permissions and format, before retrying. Storage failures do not start login or modify stored credentials. Normal process termination is the rollback; remove its client registration to stop using it. No global registration is installed by this evaluation.

## CLI comparison

| Concern | Direct CLI JSON | Optional bridge |
| --- | --- | --- |
| Dataset/run reads | Already supported: `osmosis --json --workspace acme dataset list --limit 20` and `train list --limit 20` | Same existing API; narrow fixed-field output |
| Agent discovery | Agent reads CLI help and JSON envelope | MCP advertises input/output schemas and annotations |
| Pagination | List accepts `--limit` or `--all`; JSON exposes `next_offset`, but CLI has no offset option | Explicit bounded offset input, no unbounded `--all` |
| Local context | Most platform-only commands work outside a Git checkout with `--workspace` | Explicit absolute directory is mandatory; local tool only lists filenames |
| Workspace discovery | No `workspace list` command is registered today; `doctor`/`auth whoami` are different flows | Deliberately omitted from this three-tool prototype; the API already has `list_workspaces()` if later needed |
| Authentication | Existing human login and environment/keyring/file storage | Same local login, no new credential scheme |
| Maintenance | Existing command contracts and `schema_version: 1` JSON envelope | Additional MCP transport lifecycle, tool schemas, projection/error policies, dependency major, and client matrix |

This is an engineering comparison, not a model-quality benchmark. No claim is made that agents chose tools more accurately or completed tasks faster. Before changing the no-go recommendation, run the same dataset/run inspection tasks in fresh coding-agent sessions with CLI JSON and the bridge, record success, incorrect workspace attempts, sensitive-data exposure, latency, and setup failures, and require a concrete usability advantage. Do not add CLI parity to compensate for an inconclusive experiment.

## Packaging and versioning plan

The prototype stays under `prototypes/`, outside the installed `osmosis_ai` package and console scripts. Its isolated requirement is `mcp>=1.28.1,<2.0.0`, reusing the repository's existing MCP security floor. The lower bound and currently locked version 1.29.1 are tested; newer 1.x patches still require validation. MCP 1.29.1 advertises protocol `2025-11-25`; this experiment does not claim implementation of the hosted workstream's `2026-07-28` profile. This bound applies only to the experiment, not to the SDK's existing Strands or OpenAI Agents extras.

The official [v1 SDK documentation](https://github.com/modelcontextprotocol/python-sdk/blob/v1.x/README.md) identifies v1 as a maintenance line and v2 as current stable; its migration guide describes a changed client/server API. A public release therefore requires a separate v2 migration decision and the same transport/auth/output tests, rather than silently widening this range. If the experiment later earns a go, choose a separate optional `local-mcp` extra and lazy entry point, keep base installs and other extras unchanged, test combination resolution, and version the tool contract independently from the Python package. Additive optional fields can be minor changes; renamed tools, changed field meanings, or auth behavior require a major contract change and documented migration. Do not promise hosted/local tool-name parity or tie their release schedules together.

The stdio framing and logging policy follows the [official transport specification](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio). Dependency research used the Mintlify Index first, then the official Python SDK source for the exact v1 API.

## Verification and remaining acceptance

Run the [contract suite](../tests/unit/prototypes/test_stdio_mcp.py) and [stdio lifecycle suite](../tests/integration/test_stdio_mcp_lifecycle.py) from the repository root in an environment with the SDK development dependencies and the [isolated prototype requirements](../prototypes/stdio_mcp/requirements.txt):

```bash
python -m pytest tests/unit/prototypes/test_stdio_mcp.py tests/integration/test_stdio_mcp_lifecycle.py
```

Repeat validation with the supported MCP lower bound and the repository’s locked MCP version. Record the exact commit, Python/MCP versions, commands, and results in the PR rather than keeping test counts or revision-relative results in this reference. Tests marked `slow` deliberately exercise the real ten-second deadlines; `-m "not slow"` is useful for a quick local pass but does not establish deadline or repeated-timeout behavior.

The suites own the prototype’s wire, auth, containment, and lifecycle contracts, which existing CLI tests cannot observe:

- The real stdio executable initializes from an unrelated working directory, advertises typed tools, and returns structured successes or fixed errors without UI imports. Malformed wire messages and invalid upstream Unicode cannot expose raw details on stderr or prevent subsequent valid calls.
- File and keyring credentials remain unchanged on read-only failures. Credential storage errors remain distinct from missing authentication; workspace/platform binding, scope headers, redirect denial, bounded pages, response size limits, and sensitive-field projection are enforced.
- Directory and file symlinks cannot expose external config names or contents. Unencodable filenames are omitted before deterministic sorting and truncation. Tool arguments cannot introduce arbitrary paths or workspace scope.
- A trickling HTTP response reaches the total deadline; cancellation and stdin EOF close active responses. Remote admission is bounded, queued requests share the deadline, and local calls remain available. Repeated timeouts share one stalled keyring read; backend recovery restores remote reads, and the process exits on EOF even while the backend remains blocked.

Keep the existing credential and API suites as owners of their shared SDK contracts. Follow [repository verification guidance](../CONTRIBUTING.md), check the standalone prototype with Pyright, and inspect built wheel/sdist contents to ensure the prototype remains outside the installed SDK and does not add an entry point. Tests must not require production-only flags or exports.

Fixture HTTP tests do not establish live Platform RBAC or authenticated tenant isolation. Promotion to a supported feature still requires real-account acceptance in Codex, Claude Code, Cursor, and other intended clients, plus native macOS/Linux execution. A simulated surrogate-escaped filename case does not establish native Linux filesystem acceptance. Windows remains unsupported. Deployment, SDK release, credential migration, and human-client login are separate acceptance work; none follows from a passing fixture suite.
