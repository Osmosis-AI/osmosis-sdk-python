# CLI internals (for contributors)

> The user-facing command + flag reference lives at [docs.osmosis.ai/cli/command-reference](https://docs.osmosis.ai/cli/command-reference). This page explains how the CLI is wired so you can add or change commands correctly.

## Entry point and registration

The console script is `osmosis_ai.cli.main:main` (aliases: `osmosis`, `osmosis-ai`, `osmosis_ai`). [../osmosis_ai/cli/main.py](../osmosis_ai/cli/main.py):

- `main()` calls `_register_commands()` once, then runs the Typer `app` with `standalone_mode=False` so it can map exceptions to exit codes itself.
- `_register_commands()` imports each command group **lazily inside the function**. Groups attach via `app.add_typer(...)`; the standalone `doctor` / `upgrade` commands attach via `app.command(...)`. Two `rich_help_panel`s split the help: `Workflow Commands` (`dataset`, `train`, `model`, `eval`, `rollout`, `template`, `doctor`) and `Platform Commands` (`auth`, `secret`, `upgrade`).
- The root `_callback` resolves `--json` / `--plain`, builds an `OutputContext`, installs it on the Typer context, registers `verify_output_emitted` on close, and loads `.env` via `python-dotenv`. `hoist_format_selectors` lets the format flags appear anywhere on the line.

## Command shells delegate; they don't do work

Files in [../osmosis_ai/cli/commands/](../osmosis_ai/cli/commands/) are thin Typer shells. Each command parses options and delegates to business logic:

- platform-facing logic lives in [../osmosis_ai/platform/cli/](../osmosis_ai/platform/cli/) (e.g. `dataset.py`, `train.py`, `eval.py`, `secret.py`);
- eval/rubric logic lives in [../osmosis_ai/eval/](../osmosis_ai/eval/).

Module-level imports in `commands/` are kept light: `typer`, `cli.console`, `cli.errors`, the lightweight `osmosis_ai.platform.constants` (pagination limits), and stdlib. Everything heavy (`rollout.*`, `platform.api.*`, `platform.cli.*`, `eval.*`) must be imported **inside the function** to keep CLI startup fast — see the lazy-loading section of [architecture.md](./architecture.md).

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

The output context, format enum, and selector resolution live in [../osmosis_ai/cli/output/context.py](../osmosis_ai/cli/output/context.py); the full output surface is re-exported from [../osmosis_ai/cli/output/__init__.py](../osmosis_ai/cli/output/__init__.py).

## Errors

Raise `CLIError` ([../osmosis_ai/cli/errors.py](../osmosis_ai/cli/errors.py)) — the single error type shared by every domain. `main()` funnels all exceptions through `_handle_cli_error`:

- in JSON mode, `classify_error()` + `emit_structured_error_to_stderr()` write a structured error envelope (with a CLI error `code`, command path, and SDK version) to **stderr** ([../osmosis_ai/cli/output/error.py](../osmosis_ai/cli/output/error.py));
- otherwise a plain `Error: …` line is printed.

`KeyboardInterrupt` / `click.Abort` exit `130`; `typer.Exit` / `SystemExit` preserve their code.

## Conventions when adding a command

1. Put the Typer shell in `cli/commands/`; put the logic in `platform/cli/` or `eval/`.
2. Keep module-level imports minimal; lazy-import heavy deps inside the function.
3. Return a `CommandResult`; never print directly.
4. Raise `CLIError` for user-facing failures.
5. Support non-interactive flows (`--yes`, `--token`, `--env`) so `--json` / `--plain` don't dead-end on a prompt (`INTERACTIVE_REQUIRED`).

## See also

- [architecture.md](./architecture.md) — package layout + lazy loading
- [CONTRIBUTING.md](../CONTRIBUTING.md) — dev workflow, tests, lint
- [docs.osmosis.ai/cli/command-reference](https://docs.osmosis.ai/cli/command-reference) — user-facing reference
