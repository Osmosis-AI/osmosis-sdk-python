# CLI internals (for contributors)

> The user-facing command + flag reference lives at [docs.osmosis.ai/cli/command-reference](https://docs.osmosis.ai/cli/command-reference). This page explains how the CLI is wired so you can add or change commands correctly.

## Entry point and registration

The console script is `osmosis_ai.cli.main:main` (aliases: `osmosis`, `osmosis-ai`, `osmosis_ai`). [../osmosis_ai/cli/main.py](../osmosis_ai/cli/main.py):

- `main()` calls `_register_commands()` once, then runs the Typer `app` with `standalone_mode=False` so it can map exceptions to exit codes itself.
- `_register_commands()` imports each command group **lazily inside the function**. Groups attach via `app.add_typer(...)`; the standalone `quickstart` / `doctor` / `upgrade` commands attach via `app.command(...)`. Two `rich_help_panel`s split the help: `Workflow Commands` (`quickstart`, `dataset`, `train`, `model`, `eval`, `benchmark`, `rollout`, `template`, `doctor`) and `Platform Commands` (`auth`, `secret`, `upgrade`).
- The root `_callback` resolves `--json` / `--plain`, builds an `OutputContext`, installs it on the Typer context, and registers `verify_output_emitted` on close. The CLI loads an explicit `--env-file` / `OSMOSIS_ENV_FILE`, or otherwise discovers the nearest `.env` from the working directory upward. Non-empty process variables take precedence, overlapping dotenv auth values must agree, and `--platform` overrides `OSMOSIS_PLATFORM_URL` for one invocation. Non-HTTPS non-loopback platform URLs are refused unless `OSMOSIS_ALLOW_INSECURE_PLATFORM_URL=1`. `hoist_format_selectors` lets the format flags appear anywhere on the line.

## Authentication storage and environments

`OSMOSIS_TOKEN` is the highest-priority implicit authentication source and is intended for CI/CD, agents, and other non-interactive processes; an explicit `auth login --token` overrides it and persists the supplied token. Set `OSMOSIS_TOKEN_PLATFORM_URL` to the same normalized origin as `OSMOSIS_PLATFORM_URL`; this binding is required whenever the active platform is not the default platform and recommended in all environments. For backward compatibility, an unbound environment token is accepted only for the default platform URL. Interactive/device login and `auth login --token` persist credentials in the operating-system keyring; the JSON file under `~/.config/osmosis/` contains only non-secret metadata for new logins. Existing file-backed tokens remain readable for migration, but new plaintext tokens are never written there.

Persistent credentials are keyed by the normalized platform URL, so credentials for multiple platform environments can coexist without overwriting one another. Read-only commands preserve and reject corrupt or unknown shared metadata; explicit login/logout recover by moving it to a non-overwriting `credentials.json.bak[.N]` backup. Logout removes local metadata even when the current host cannot access the system keyring and emits a cleanup warning. An HTTP 401 reports an expired or revoked session but never deletes local credentials; only explicit `auth logout` changes stored state. `auth whoami` reports both the effective source and whether a persistent login also exists, while logout makes it explicit when `OSMOSIS_TOKEN` remains active.

The nearest `.env` is loaded automatically, so a workspace can set only `OSMOSIS_PLATFORM_URL` and select its platform-scoped keyring login without exporting it in every shell. Use `--env-file` / `OSMOSIS_ENV_FILE` to select another file explicitly; non-empty process variables still win, and `--platform` has the final override. If a dotenv file contains `OSMOSIS_TOKEN`, that token intentionally takes precedence over the keyring for the process, so keep `OSMOSIS_TOKEN`, `OSMOSIS_PLATFORM_URL`, and `OSMOSIS_TOKEN_PLATFORM_URL` together. Profiles already loaded by `uv` are accepted when overlapping values agree; missing values are filled from dotenv instead of being treated as a conflict.

## Command shells delegate; they don't do work

Files in [../osmosis_ai/cli/commands/](../osmosis_ai/cli/commands/) are thin Typer shells. Each command parses options and delegates to business logic:

- platform-facing logic lives in [../osmosis_ai/platform/cli/](../osmosis_ai/platform/cli/) (e.g. `dataset.py`, `train.py`, `eval.py`, `secret.py`);
- eval/rubric logic lives in [../osmosis_ai/eval/](../osmosis_ai/eval/).

Module-level imports in `commands/` are kept light: `typer`, `cli.errors`, the lightweight `osmosis_ai.platform.constants` (pagination limits), and stdlib. Everything heavy (`rollout.*`, `platform.api.*`, `platform.cli.*`, `eval.*`, `cli.console`) must be imported **inside the function** to keep CLI startup fast — see the lazy-loading section of [architecture.md](./architecture.md).

`osmosis benchmark` puts the benchmark first, mirroring the platform's
Benchmarks pages: top-level `list` and `info` act on benchmarks (the workspace
list and one benchmark's page - `info` shows its metadata, leaderboard, and
runs), `submit` starts a run, and run lifecycle lives under the nested
`benchmark runs list|info|logs|stop|download` namespace.

List output includes a shell-safe benchmark `Key`, such as
`terminal-bench-2-1`. Pass that key to `osmosis benchmark info <key>`;
exact display names and UUIDs remain supported for compatibility.

## Commands return results; they don't print

The Typer app is created with `result_callback=render_command_result` ([../osmosis_ai/cli/main.py](../osmosis_ai/cli/main.py)). A command function **returns** a `CommandResult`; the callback renders it in the active format. Do not `print()` from a command — return a typed result instead.

Result types ([../osmosis_ai/cli/output/result.py](../osmosis_ai/cli/output/result.py)):

| Type | Use |
|------|-----|
| `ListResult` | A single list/table |
| `SectionedListResult` | Multiple named lists (e.g. base + LoRA models) |
| `DetailResult` | One resource's fields/sections |
| `OperationResult` | A mutation's outcome |
| `MessageResult` | A plain message |

Serializers that turn API models into these shapes live in [../osmosis_ai/cli/output/serializers.py](../osmosis_ai/cli/output/serializers.py).

## Output envelopes

[../osmosis_ai/cli/output/renderer.py](../osmosis_ai/cli/output/renderer.py) builds the machine contract. Every JSON success envelope carries `schema_version: 1` and a shape matching the result type (`_envelope_list`, `_envelope_sectioned_list`, `_envelope_detail`, `_envelope_operation`, `_envelope_message`). Rich is the default for humans; `--plain` is intentionally low-noise text (not a strict schema).

The output context, format enum, and selector resolution live in [../osmosis_ai/cli/output/context.py](../osmosis_ai/cli/output/context.py); the full output surface is re-exported from [../osmosis_ai/cli/output/__init__.py](../osmosis_ai/cli/output/__init__.py). Serializer names are lazy (PEP 562) so importing the package does not load platform API models.

Hidden `osmosis dev server logs` is a streaming exception to the envelope contract: `--json` writes NDJSON lines (`{"timestamp", "message"}`) to stdout and exits via `typer.Exit(0)` rather than a `schema_version: 1` list envelope. `--follow` off is still that stream, not a final list envelope.

## Errors

Raise `CLIError` ([../osmosis_ai/cli/errors.py](../osmosis_ai/cli/errors.py)) — the single error type shared by every domain. `CLIError.code` is a `CLIErrorCode` `StrEnum`. `main()` funnels all exceptions through `_handle_cli_error`:

- in JSON mode, `classify_error()` + `emit_structured_error_to_stderr()` write a structured error envelope (with a CLI error `code`, command path, and SDK version) to **stderr** ([../osmosis_ai/cli/output/error.py](../osmosis_ai/cli/output/error.py));
- otherwise a plain `Error: …` line is printed.

Command path prefers Click's `command_path` when the context is already inside a subcommand; otherwise argv is parsed against the same name catalog `_register_commands` uses ([../osmosis_ai/cli/command_registry.py](../osmosis_ai/cli/command_registry.py)), including three-token prefixes such as `dev server up` and `benchmark runs download`.

Not-logged-in failures use `AUTH_REQUIRED`. `SubscriptionRequiredError` maps to `SUBSCRIPTION_REQUIRED` or `BILLING_REQUIRED` from the platform `error_code`; a generic HTTP 403 stays `PLATFORM_ERROR`. The error object has `code`, `message`, and `details` only — `request_id` is omitted because the platform client does not expose one.

JSON success and error envelopes are encoded with `allow_nan=False`. Non-finite metric values are sanitized to `null` in train/eval metrics exports (including the file written by `train info --output`); any remaining non-finite float fails the command rather than emitting invalid JSON.

Unknown exceptions become `INTERNAL` with a generic message. Set `OSMOSIS_DEBUG=1` to append the original exception and traceback to stderr (the JSON envelope is unchanged).

`KeyboardInterrupt` / `click.Abort` exit `130`; `typer.Exit` / `SystemExit` preserve their code.

## Conventions when adding a command

1. Put the Typer shell in `cli/commands/`; put the logic in `platform/cli/` or `eval/`.
2. Keep module-level imports minimal; lazy-import heavy deps inside the function.
3. Return a `CommandResult`; never print directly. Annotate handlers as `-> CommandResult`. Reuse [../osmosis_ai/cli/options.py](../osmosis_ai/cli/options.py) for `--limit` / `--all` / `--cursor`.
4. Raise `CLIError` for user-facing failures.
5. Support non-interactive flows (`--yes`, `--token`, `--env`) so `--json` / `--plain` don't dead-end on a prompt (`INTERACTIVE_REQUIRED`).

## See also

- [architecture.md](./architecture.md) — package layout + lazy loading
- [CONTRIBUTING.md](../CONTRIBUTING.md) — dev workflow, tests, lint
- [docs.osmosis.ai/cli/command-reference](https://docs.osmosis.ai/cli/command-reference) — user-facing reference
