"""Osmosis AI CLI — built with Typer."""

import difflib
import sys
import warnings

import click
import typer
import typer.core
from dotenv import find_dotenv, load_dotenv

from osmosis_ai.cli.errors import CLIError
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
    classify_error,
    command_path_for_error,
    emit_structured_error_to_stderr,
)
from osmosis_ai.cli.output.renderer import render_command_result, verify_output_emitted
from osmosis_ai.consts import PACKAGE_VERSION, package_name
from osmosis_ai.platform.auth.platform_client import (
    AuthenticationExpiredError,
    PlatformAPIError,
)


class OsmosisGroup(typer.core.TyperGroup):
    """Typer group with fuzzy command suggestion."""

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            if args:
                cmd_name = args[0]
                matches = difflib.get_close_matches(
                    cmd_name, self.list_commands(ctx), n=1, cutoff=0.5
                )
                if matches:
                    raise click.UsageError(
                        f"No such command '{cmd_name}'. Did you mean '{matches[0]}'?"
                    ) from None
            raise


app: typer.Typer = typer.Typer(
    name="osmosis",
    cls=OsmosisGroup,
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
    output_format: OutputFormat | None = typer.Option(
        None,
        "--format",
        help=(
            "Global output format: rich, json, or plain. Use json/plain for "
            "AI agents, CI/CD, and scripts."
        ),
        case_sensitive=False,
    ),
    json_alias: bool = typer.Option(
        False,
        "--json",
        help="Shortcut for --format json; recommended for AI agents and CI/CD.",
    ),
    plain_alias: bool = typer.Option(
        False,
        "--plain",
        help="Shortcut for --format plain; low-noise text for shell pipelines.",
    ),
) -> None:
    """Osmosis AI CLI.

    Rich output is the default for humans. For AI agents, CI/CD, and scripts,
    prefer global output flags before the command, for example:
    `osmosis --json dataset list` or `osmosis --format plain dataset list`.
    """
    warnings.filterwarnings("ignore")
    if version:
        typer.echo(f"{package_name} {PACKAGE_VERSION}")
        raise typer.Exit()

    selected_format = resolve_format_selectors(
        output_format,
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


def _output_context_for_error(
    exc: BaseException,
    argv: list[str] | None,
) -> OutputContext:
    ctx = getattr(exc, "ctx", None)
    if isinstance(exc, click.ClickException) and isinstance(ctx, click.Context):
        root_obj = ctx.find_root().obj
        if isinstance(root_obj, OutputContext):
            return root_obj

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
    output = _output_context_for_error(exc, argv)
    if output.format is OutputFormat.json:
        raw_ctx = getattr(exc, "ctx", None)
        ctx = raw_ctx if isinstance(raw_ctx, click.Context) else None
        emit_structured_error_to_stderr(
            classify_error(exc),
            command=command_path_for_error(ctx),
        )
    else:
        if isinstance(exc, AuthenticationExpiredError):
            _print_error(str(exc))
        else:
            _print_error(str(exc))
    return exit_code


def _register_commands() -> None:
    """Register all subcommands. Called once before app() runs."""
    global _registered
    if _registered:
        return
    _registered = True

    # -- Command groups --
    from osmosis_ai.cli.commands.auth import app as auth_app
    from osmosis_ai.cli.commands.dataset import app as dataset_app
    from osmosis_ai.cli.commands.deployment import app as deployment_app
    from osmosis_ai.cli.commands.eval import app as eval_app
    from osmosis_ai.cli.commands.model import app as model_app
    from osmosis_ai.cli.commands.project import app as project_app
    from osmosis_ai.cli.commands.rollout import app as rollout_app
    from osmosis_ai.cli.commands.template import app as template_app
    from osmosis_ai.cli.commands.train import app as train_app
    from osmosis_ai.cli.commands.workspace import app as workspace_app

    _WORKFLOW = "Workflow Commands"
    _PLATFORM = "Platform Commands"

    app.add_typer(project_app, name="project", rich_help_panel=_WORKFLOW)
    app.add_typer(dataset_app, name="dataset", rich_help_panel=_WORKFLOW)
    app.add_typer(train_app, name="train", rich_help_panel=_WORKFLOW)
    app.add_typer(model_app, name="model", rich_help_panel=_WORKFLOW)
    app.add_typer(deployment_app, name="deployment", rich_help_panel=_WORKFLOW)
    app.add_typer(eval_app, name="eval", rich_help_panel=_WORKFLOW)
    app.add_typer(rollout_app, name="rollout", rich_help_panel=_WORKFLOW)
    app.add_typer(template_app, name="template", rich_help_panel=_WORKFLOW)

    app.add_typer(auth_app, name="auth", rich_help_panel=_PLATFORM)
    app.add_typer(workspace_app, name="workspace", rich_help_panel=_PLATFORM)

    # -- Top-level commands --
    from osmosis_ai.cli.commands.init import init

    app.command("init", rich_help_panel=_WORKFLOW)(init)

    # `deploy` and `undeploy` are verbs, not CRUD on the deployment resource,
    # so they are promoted to top-level to avoid `osmosis deployment deploy`.
    from osmosis_ai.cli.commands.deployment import deploy, undeploy

    app.command("deploy", rich_help_panel=_WORKFLOW)(deploy)
    app.command("undeploy", rich_help_panel=_WORKFLOW)(undeploy)

    from osmosis_ai.cli.upgrade import upgrade

    app.command("upgrade", rich_help_panel=_PLATFORM)(upgrade)

    # -- Transitional: deprecated aliases (hidden from help) --
    from osmosis_ai.cli.commands.auth import login as auth_login
    from osmosis_ai.cli.commands.auth import logout as auth_logout
    from osmosis_ai.cli.commands.auth import whoami as auth_whoami

    app.command("login", hidden=True)(auth_login)
    app.command("logout", hidden=True)(auth_logout)
    app.command("whoami", hidden=True)(auth_whoami)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the osmosis CLI."""
    _register_commands()
    try:
        result = app(argv, standalone_mode=False)
        # standalone_mode=False returns None on normal completion;
        # typer.Exit() still raises SystemExit, caught by the except block below.
        if isinstance(result, int) and result != 0:
            return result
        return 0
    except click.exceptions.Exit as e:
        return e.exit_code
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0
    except click.UsageError as exc:
        # NoArgsIsHelpError (from no_args_is_help=True) has an empty message
        # after help is already printed — just exit cleanly.
        if not str(exc):
            return 0
        return _handle_cli_error(exc, argv=argv, exit_code=exc.exit_code)
    except AuthenticationExpiredError as exc:
        return _handle_cli_error(exc, argv=argv)
    except PlatformAPIError as exc:
        return _handle_cli_error(exc, argv=argv)
    except CLIError as exc:
        return _handle_cli_error(exc, argv=argv)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        return _handle_cli_error(exc, argv=argv)


if __name__ == "__main__":
    sys.exit(main())
