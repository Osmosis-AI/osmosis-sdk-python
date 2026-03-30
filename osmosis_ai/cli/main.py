"""Osmosis AI CLI — built with Typer."""

import difflib
import sys
import warnings

import click
import typer
import typer.core
from dotenv import find_dotenv, load_dotenv

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.consts import PACKAGE_VERSION, package_name
from osmosis_ai.platform.auth.platform_client import (
    AuthenticationExpiredError,
    PlatformAPIError,
)
from osmosis_ai.platform.cli.constants import MSG_SESSION_EXPIRED


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
) -> None:
    """Osmosis AI CLI"""
    warnings.filterwarnings("ignore")
    if version:
        typer.echo(f"{package_name} {PACKAGE_VERSION}")
        raise typer.Exit()
    load_dotenv(find_dotenv(usecwd=True))


_registered = False


def _register_commands() -> None:
    """Register all subcommands. Called once before app() runs."""
    global _registered
    if _registered:
        return
    _registered = True

    # -- Command groups --
    from osmosis_ai.cli.commands.auth import app as auth_app
    from osmosis_ai.cli.commands.dataset import app as dataset_app
    from osmosis_ai.cli.commands.eval import app as eval_app
    from osmosis_ai.cli.commands.model import app as model_app
    from osmosis_ai.cli.commands.rollout import app as rollout_app
    from osmosis_ai.cli.commands.train import app as train_app
    from osmosis_ai.cli.commands.workspace import app as workspace_app

    _WORKFLOW = "Workflow Commands"
    _PLATFORM = "Platform Commands"

    app.add_typer(dataset_app, name="dataset", rich_help_panel=_WORKFLOW)
    app.add_typer(train_app, name="train", rich_help_panel=_WORKFLOW)
    app.add_typer(model_app, name="model", rich_help_panel=_WORKFLOW)
    app.add_typer(eval_app, name="eval", rich_help_panel=_WORKFLOW)
    app.add_typer(rollout_app, name="rollout", rich_help_panel=_WORKFLOW)

    app.add_typer(auth_app, name="auth", rich_help_panel=_PLATFORM)
    app.add_typer(workspace_app, name="workspace", rich_help_panel=_PLATFORM)

    # -- Top-level commands --
    from osmosis_ai.cli.commands.init import init

    app.command("init", rich_help_panel=_WORKFLOW)(init)

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
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0
    except click.UsageError as exc:
        # NoArgsIsHelpError (from no_args_is_help=True) has an empty message
        # after help is already printed — just exit cleanly.
        if not str(exc):
            return 0
        print(f"Error: {exc}", file=sys.stderr)
        return exc.exit_code
    except AuthenticationExpiredError:
        print(f"Error: {MSG_SESSION_EXPIRED}", file=sys.stderr)
        return 1
    except PlatformAPIError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except CLIError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
