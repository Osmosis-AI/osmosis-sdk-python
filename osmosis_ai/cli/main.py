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

    def resolve_command(self, ctx, args):
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


app = typer.Typer(
    name="osmosis",
    cls=OsmosisGroup,
    no_args_is_help=True,
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
    load_dotenv(find_dotenv(usecwd=True))
    if version:
        typer.echo(f"{package_name} {PACKAGE_VERSION}")
        raise typer.Exit()


_registered = False


def _register_commands() -> None:
    """Register all subcommands. Called once before app() runs."""
    global _registered
    if _registered:
        return
    _registered = True

    # ── Platform (leaf commands) ──
    from osmosis_ai.platform.cli.login import login_cmd
    from osmosis_ai.platform.cli.logout import logout
    from osmosis_ai.platform.cli.whoami import whoami

    app.command("login", rich_help_panel="Platform")(login_cmd)
    app.command("whoami", rich_help_panel="Platform")(whoami)
    app.command("logout", rich_help_panel="Platform")(logout)

    # ── Platform (sub-command groups) ──
    from osmosis_ai.platform.cli.dataset import app as dataset_app
    from osmosis_ai.platform.cli.model import app as model_app
    from osmosis_ai.platform.cli.run import app as run_app
    from osmosis_ai.platform.cli.workspace import app as workspace_app

    app.add_typer(workspace_app, name="workspace", rich_help_panel="Platform")
    app.add_typer(dataset_app, name="dataset", rich_help_panel="Platform")
    app.add_typer(run_app, name="run", rich_help_panel="Platform")
    app.add_typer(model_app, name="model", rich_help_panel="Platform")

    # ── Evaluation ──
    from osmosis_ai.rollout.eval.evaluation.cli import app as eval_app
    from osmosis_ai.rubric.cli.eval_rubric import app as eval_rubric_app
    from osmosis_ai.rubric.cli.preview import preview

    app.command("preview", rich_help_panel="Evaluation")(preview)
    app.add_typer(eval_rubric_app, name="eval-rubric", rich_help_panel="Evaluation")
    app.add_typer(eval_app, name="eval", rich_help_panel="Evaluation")

    # ── Rollout ──
    from osmosis_ai.rollout.cli import serve, validate
    from osmosis_ai.rollout.eval.test_mode.cli import app as test_app

    app.command("serve", rich_help_panel="Rollout")(serve)
    app.command("validate", rich_help_panel="Rollout")(validate)
    app.add_typer(test_app, name="test", rich_help_panel="Rollout")

    # ── Tools (leaf) ──
    from osmosis_ai.cli.upgrade import upgrade

    app.command("upgrade", rich_help_panel="Tools")(upgrade)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the osmosis CLI."""
    _register_commands()
    try:
        result = app(argv, standalone_mode=False)
        # standalone_mode=False makes typer.Exit() return the code instead of raising SystemExit
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
