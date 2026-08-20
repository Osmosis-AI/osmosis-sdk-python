"""Osmosis AI CLI — built with Typer."""

import difflib
import os
import sys
import warnings
from pathlib import Path

import typer
import typer.core

from osmosis_ai.cli._click_compat import (
    ClickException,
    Command,
    Context,
    NoArgsIsHelpError,
    UsageError,
    get_current_context,
)
from osmosis_ai.cli.output.context import (
    OutputContext,
    OutputFormat,
    _argv_format_prescan,
    _output_context_var,
    get_output_context,
    hoist_format_selectors,
    install_output_context,
    resolve_format_selectors,
)
from osmosis_ai.cli.output.error import (
    classify_error,
    command_path_for_error,
    emit_internal_debug,
    emit_structured_error_to_stderr,
)
from osmosis_ai.cli.output.renderer import render_command_result, verify_output_emitted
from osmosis_ai.consts import PACKAGE_VERSION, package_name


class OsmosisGroup(typer.core.TyperGroup):
    """Typer group with fuzzy command suggestion."""

    def resolve_command(
        self, ctx: Context, args: list[str]
    ) -> tuple[str | None, Command | None, list[str]]:
        try:
            return super().resolve_command(ctx, args)
        except UsageError:
            if args:
                cmd_name = args[0]
                if cmd_name == "help":
                    raise UsageError(
                        "No such command 'help'. Use 'osmosis --help', "
                        "or 'osmosis <command> --help' for a specific command."
                    ) from None
                candidates = []
                for name in self.list_commands(ctx):
                    command = self.get_command(ctx, name)
                    if command is None or getattr(command, "hidden", False):
                        continue
                    candidates.append(name)
                matches = difflib.get_close_matches(
                    cmd_name, candidates, n=1, cutoff=0.5
                )
                if matches:
                    raise UsageError(
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
    # OsmosisGroup owns suggestions: single candidate, hidden commands
    # filtered, and the 'help' nudge. Typer's built-in suggester would
    # append its own (unfiltered, multi-candidate) line on top.
    suggest_commands=False,
)

_AUTH_PROFILE_ENV_VARS = {
    "OSMOSIS_PLATFORM_URL",
    "OSMOSIS_TOKEN",
    "OSMOSIS_TOKEN_PLATFORM_URL",
}
_PLATFORM_URL_ENV_VARS = {
    "OSMOSIS_PLATFORM_URL",
    "OSMOSIS_TOKEN_PLATFORM_URL",
}


def _auth_value_matches(name: str, existing: str | None, dotenv_value: str) -> bool:
    """Return whether an existing auth value matches its dotenv value."""
    if existing is None:
        return True
    if name not in _PLATFORM_URL_ENV_VARS:
        return existing == dotenv_value

    from osmosis_ai.platform.auth.config import normalize_platform_url

    return normalize_platform_url(existing) == normalize_platform_url(dotenv_value)


def _load_env_file(env_file: Path, *, platform_overridden: bool = False) -> None:
    """Load a dotenv file without combining unrelated auth profiles."""
    from dotenv import dotenv_values, load_dotenv

    values = dotenv_values(env_file)
    ambient_token = os.environ.get("OSMOSIS_TOKEN")
    auth_values = {
        name: value
        for name, value in values.items()
        if name in _AUTH_PROFILE_ENV_VARS
        and isinstance(value, str)
        and not (platform_overridden and name == "OSMOSIS_PLATFORM_URL")
    }
    if ambient_token and auth_values:
        file_token_matches = auth_values.get("OSMOSIS_TOKEN") == ambient_token
        existing_values_do_not_conflict = all(
            _auth_value_matches(name, os.environ.get(name), value)
            for name, value in auth_values.items()
            if name != "OSMOSIS_TOKEN"
        )
        safe_existing_profile = file_token_matches and existing_values_do_not_conflict
        if not safe_existing_profile:
            from osmosis_ai.cli.errors import CLIError

            raise CLIError(
                "OSMOSIS_TOKEN is already set, so a different auth profile "
                f"from {env_file} cannot be merged with it. Unset the ambient "
                "token or keep the complete auth profile in one source.",
                code="CONFLICT",
            )
    load_dotenv(env_file, override=False)


def _find_env_file() -> Path | None:
    """Find the nearest .env from the current directory upward."""
    from dotenv import find_dotenv

    discovered = find_dotenv(usecwd=True)
    return Path(discovered) if discovered else None


def _emit_version() -> None:
    typer.echo(f"{package_name} {PACKAGE_VERSION}")


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
    env_file: Path | None = typer.Option(
        None,
        "--env-file",
        envvar="OSMOSIS_ENV_FILE",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Load environment variables from this dotenv file.",
    ),
    platform: str | None = typer.Option(
        None,
        "--platform",
        help="Platform base URL for this command.",
    ),
) -> None:
    """Osmosis AI CLI.

    Rich output is the default for humans. For AI agents, CI/CD, and scripts, pass `--json` or `--plain` anywhere on the command line, for example: `osmosis dataset list --json` or `osmosis --plain dataset list`.
    """
    warnings.filterwarnings("ignore")
    if version:
        _emit_version()
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

    selected_env_file = env_file or _find_env_file()
    if selected_env_file is not None:
        _load_env_file(
            selected_env_file,
            platform_overridden=platform is not None,
        )
    if platform is not None:
        os.environ["OSMOSIS_PLATFORM_URL"] = platform
    _refuse_insecure_platform_url()


_registered = False


def _refuse_insecure_platform_url() -> None:
    """Fail closed on non-HTTPS non-loopback platform URLs.

    Explicit environment configuration can point ``OSMOSIS_PLATFORM_URL`` at
    http:// while ``OSMOSIS_TOKEN`` is already exported. Opt in with
    ``OSMOSIS_ALLOW_INSECURE_PLATFORM_URL=1``.
    """
    if os.environ.get("OSMOSIS_PLATFORM_URL") is None:
        return

    from osmosis_ai.cli.errors import CLIError
    from osmosis_ai.platform.auth.config import (
        get_platform_url,
        is_insecure_platform_url,
    )

    platform_url = get_platform_url()
    if not is_insecure_platform_url(platform_url):
        return
    from osmosis_ai.cli.console import console

    console.print_warning(
        f"OSMOSIS_PLATFORM_URL is not HTTPS ({platform_url}). "
        "Tokens will be transmitted in plaintext.",
        code="INSECURE_PLATFORM_URL",
    )
    if os.environ.get("OSMOSIS_ALLOW_INSECURE_PLATFORM_URL") == "1":
        return
    raise CLIError(
        f"Refusing non-HTTPS platform URL {platform_url}. "
        "Set OSMOSIS_ALLOW_INSECURE_PLATFORM_URL=1 to override.",
        code="VALIDATION",
    )


def _print_error(message: str) -> None:
    from osmosis_ai.cli.console import Console

    Console(file=sys.stderr).print_error(f"Error: {message}", soft_wrap=True)


def _output_context_for_error(
    exc: BaseException,
    argv: list[str] | None,
) -> OutputContext:
    ctx = getattr(exc, "ctx", None)
    if isinstance(exc, ClickException) and isinstance(ctx, Context):
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
    classified = None
    if output.format is OutputFormat.json:
        classified = classify_error(exc)
        raw_ctx = getattr(exc, "ctx", None)
        ctx = raw_ctx if isinstance(raw_ctx, Context) else None
        if ctx is None:
            ctx = get_current_context(silent=True)
        command_argv = argv if argv is not None else sys.argv[1:]
        emit_structured_error_to_stderr(
            classified,
            command=command_path_for_error(ctx, argv=command_argv),
        )
    else:
        _print_error(str(exc))
    emit_internal_debug(exc, classified)
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
    from osmosis_ai.cli import command_registry as cmdreg
    from osmosis_ai.cli.commands.auth import app as auth_app
    from osmosis_ai.cli.commands.benchmark import app as benchmark_app
    from osmosis_ai.cli.commands.dataset import app as dataset_app
    from osmosis_ai.cli.commands.eval import app as eval_app
    from osmosis_ai.cli.commands.model import app as model_app
    from osmosis_ai.cli.commands.rollout import app as rollout_app
    from osmosis_ai.cli.commands.secret import app as secret_app
    from osmosis_ai.cli.commands.template import app as template_app
    from osmosis_ai.cli.commands.train import app as train_app

    _WORKFLOW = "Workflow Commands"
    _PLATFORM = "Platform Commands"

    from osmosis_ai.cli.commands.quickstart import HELP as QUICKSTART_HELP
    from osmosis_ai.cli.commands.quickstart import quickstart

    app.command(
        cmdreg.STANDALONE_QUICKSTART, help=QUICKSTART_HELP, rich_help_panel=_WORKFLOW
    )(quickstart)

    app.add_typer(dataset_app, name=cmdreg.GROUP_DATASET, rich_help_panel=_WORKFLOW)
    app.add_typer(train_app, name=cmdreg.GROUP_TRAIN, rich_help_panel=_WORKFLOW)
    app.add_typer(model_app, name=cmdreg.GROUP_MODEL, rich_help_panel=_WORKFLOW)
    app.add_typer(eval_app, name=cmdreg.GROUP_EVAL, rich_help_panel=_WORKFLOW)
    app.add_typer(benchmark_app, name=cmdreg.GROUP_BENCHMARK, rich_help_panel=_WORKFLOW)
    app.add_typer(rollout_app, name=cmdreg.GROUP_ROLLOUT, rich_help_panel=_WORKFLOW)
    app.add_typer(template_app, name=cmdreg.GROUP_TEMPLATE, rich_help_panel=_WORKFLOW)

    from osmosis_ai.cli.commands.dev import app as dev_app

    app.add_typer(dev_app, name=cmdreg.GROUP_DEV, hidden=True)

    app.add_typer(auth_app, name=cmdreg.GROUP_AUTH, rich_help_panel=_PLATFORM)
    app.add_typer(secret_app, name=cmdreg.GROUP_SECRET, rich_help_panel=_PLATFORM)

    from osmosis_ai.cli.commands.workspace import doctor

    app.command(cmdreg.STANDALONE_DOCTOR, rich_help_panel=_WORKFLOW)(doctor)

    from osmosis_ai.cli.upgrade import upgrade

    app.command(cmdreg.STANDALONE_UPGRADE, rich_help_panel=_PLATFORM)(upgrade)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the Osmosis CLI."""
    argv = argv if argv is not None else sys.argv[1:]
    # Bare -V/--version skips command registration. Combined with other flags
    # (e.g. --json --version) still goes through Typer so output stays identical.
    if argv == ["--version"] or argv == ["-V"]:
        _emit_version()
        return 0
    _register_commands()
    argv = hoist_format_selectors(argv)
    try:
        result = app(argv, standalone_mode=False)
        # standalone_mode=False returns the command result on success, or the
        # exit code when the run ended via typer.Exit (Typer also converts
        # Ctrl-C into Exit(130) internally).
        if isinstance(result, int) and result != 0:
            return result
        return 0
    except NoArgsIsHelpError:
        # no_args_is_help=True: instantiating this exception already rendered
        # rich help to stdout (via ctx.get_help), so just exit cleanly.
        return 0
    except typer.Exit as e:
        return e.exit_code
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0
    except UsageError as exc:
        return _handle_cli_error(exc, argv=argv, exit_code=exc.exit_code)
    except (KeyboardInterrupt, typer.Abort):
        return 130
    except Exception as exc:
        # CLIError, PlatformAPIError, AuthenticationExpiredError and anything
        # else funnel through classify_error() into the structured envelope.
        return _handle_cli_error(exc, argv=argv)


if __name__ == "__main__":
    sys.exit(main())
