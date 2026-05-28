"""Osmosis AI CLI — built with Typer."""

import sys
import warnings

import typer
from dotenv import find_dotenv, load_dotenv

from osmosis_ai.cli.output.context import (
    OutputContext,
    OutputFormat,
    _argv_format_prescan,
    _output_context_var,
    get_output_context,
    install_output_context,
    resolve_format_selectors,
)
from osmosis_ai.cli.output.error import (
    ClickException,
    classify_error,
    command_path_for_error,
    emit_structured_error_to_stderr,
    is_cli_usage_error,
)
from osmosis_ai.cli.output.renderer import render_command_result, verify_output_emitted
from osmosis_ai.consts import PACKAGE_VERSION, package_name

app: typer.Typer = typer.Typer(
    name="osmosis",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    result_callback=render_command_result,
)


@app.callback(invoke_without_command=True)
def _callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "-V",
        "--version",
        help="Show version and exit.",
        is_eager=True,
    ),
    json_alias: bool = typer.Option(
        False,
        "--json",
        help="Emit structured JSON; recommended for AI agents and CI/CD.",
    ),
    plain_alias: bool = typer.Option(
        False,
        "--plain",
        help="Emit low-noise text for shell pipelines.",
    ),
) -> None:
    """Osmosis AI CLI.

    Rich output is the default for humans. For AI agents, CI/CD, and scripts, put a global output flag before the command, for example: `osmosis --json dataset list` or `osmosis --plain dataset list`.
    """
    warnings.filterwarnings("ignore")
    if version:
        typer.echo(f"{package_name} {PACKAGE_VERSION}")
        raise typer.Exit()

    selected_format = resolve_format_selectors(
        json_alias=json_alias,
        plain_alias=plain_alias,
    )
    output = OutputContext(
        format=selected_format,
        interactive=selected_format is OutputFormat.rich and sys.stdin.isatty(),
    )
    install_output_context(ctx, output)
    ctx.call_on_close(verify_output_emitted)
    load_dotenv(find_dotenv(usecwd=True))


_registered = False


def _print_error(message: str) -> None:
    from osmosis_ai.cli.console import Console

    Console(file=sys.stderr).print_error(f"Error: {message}", soft_wrap=True)


def _output_context_for_error(argv: list[str] | None) -> OutputContext:
    # The active context survives in the ContextVar; if it was already reset
    # (parse error before the callback installed it), recover the format from
    # the explicit argv. Mirrors get_output_context()'s fallback chain.
    stored = _output_context_var.get()
    if stored is not None:
        return stored

    pre = _argv_format_prescan(argv if argv is not None else sys.argv[1:])
    if pre is not None:
        return OutputContext(format=pre, interactive=False)
    return get_output_context()


def _handle_cli_error(
    exc: BaseException,
    *,
    argv: list[str] | None,
    exit_code: int = 1,
) -> int:
    output = _output_context_for_error(argv)
    if output.format is OutputFormat.json:
        command_argv = argv if argv is not None else sys.argv[1:]
        emit_structured_error_to_stderr(
            classify_error(exc),
            command=command_path_for_error(None, argv=command_argv),
        )
    else:
        _print_error(str(exc))
    return exit_code


def _register_commands() -> None:
    """Register all subcommands. Called once before app() runs."""
    global _registered
    if _registered:
        return
    _registered = True
    # Typer's documented ``add_completion=True`` path initializes shell classes
    # through this public helper, but also exposes install/show completion
    # options. Keep those options hidden while preserving Typer's zsh/fish env
    # contract instead of Click's COMP_WORDS-based default.
    from typer.completion import get_completion_inspect_parameters

    get_completion_inspect_parameters()
    # -- Command groups --
    from osmosis_ai.cli.commands.auth import app as auth_app
    from osmosis_ai.cli.commands.dataset import app as dataset_app
    from osmosis_ai.cli.commands.deployment import app as deployment_app
    from osmosis_ai.cli.commands.eval import app as eval_app
    from osmosis_ai.cli.commands.model import app as model_app
    from osmosis_ai.cli.commands.rollout import app as rollout_app
    from osmosis_ai.cli.commands.secret import app as secret_app
    from osmosis_ai.cli.commands.template import app as template_app
    from osmosis_ai.cli.commands.train import app as train_app

    _WORKFLOW = "Workflow Commands"
    _PLATFORM = "Platform Commands"

    app.add_typer(dataset_app, name="dataset", rich_help_panel=_WORKFLOW)
    app.add_typer(train_app, name="train", rich_help_panel=_WORKFLOW)
    app.add_typer(model_app, name="model", rich_help_panel=_WORKFLOW)
    app.add_typer(deployment_app, name="deployment", rich_help_panel=_WORKFLOW)
    app.add_typer(eval_app, name="eval", rich_help_panel=_WORKFLOW)
    app.add_typer(rollout_app, name="rollout", rich_help_panel=_WORKFLOW)
    app.add_typer(template_app, name="template", rich_help_panel=_WORKFLOW)

    app.add_typer(auth_app, name="auth", rich_help_panel=_PLATFORM)
    app.add_typer(secret_app, name="secret", rich_help_panel=_PLATFORM)

    # `deploy` and `undeploy` are verbs, not CRUD on the deployment resource,
    # so they are promoted to top-level to avoid `osmosis deployment deploy`.
    from osmosis_ai.cli.commands.deployment import deploy, undeploy
    from osmosis_ai.cli.commands.workspace import doctor

    app.command("deploy", rich_help_panel=_WORKFLOW)(deploy)
    app.command("doctor", rich_help_panel=_WORKFLOW)(doctor)
    app.command("undeploy", rich_help_panel=_WORKFLOW)(undeploy)

    from osmosis_ai.cli.upgrade import upgrade

    app.command("upgrade", rich_help_panel=_PLATFORM)(upgrade)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the osmosis CLI."""
    _register_commands()
    try:
        result = app(argv, standalone_mode=False)
        # standalone_mode=False returns None on normal completion and re-raises
        # ClickException / Abort for the except arms below to handle.
        if isinstance(result, int) and result != 0:
            return result
        return 0
    except typer.Exit as e:
        return e.exit_code
    except (KeyboardInterrupt, typer.Abort):
        # typer.Abort is a RuntimeError (not a ClickException); under
        # standalone_mode=False Click does not convert it to an exit, so handle
        # it here instead of letting it fall through and read as INTERNAL.
        return 130
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0
    except ClickException as exc:
        # Bundled Click base — covers every parser/usage error. An empty message
        # means no_args_is_help already printed help, so exit cleanly. Usage
        # errors are exit 2 (POSIX), other Click errors exit 1.
        if not str(exc):
            return 0
        exit_code = 2 if is_cli_usage_error(exc) else 1
        return _handle_cli_error(exc, argv=argv, exit_code=exit_code)
    except Exception as exc:
        return _handle_cli_error(exc, argv=argv)


if __name__ == "__main__":
    sys.exit(main())
