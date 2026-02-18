from __future__ import annotations

import argparse
import sys
import warnings

from dotenv import find_dotenv, load_dotenv

from .cli_commands import (
    EvalRubricCommand,
    LoginCommand,
    LogoutCommand,
    PreviewCommand,
    WhoamiCommand,
    WorkspaceCommand,
)
from .cli_services import CLIError
from .consts import PACKAGE_VERSION, package_name


def main(argv: list[str] | None = None) -> int:
    """Entry point for the osmosis CLI."""
    # Suppress all Python warnings for a clean CLI experience.
    # This only affects the CLI process; library consumers are not impacted.
    # Meaningful user-facing messages should use logging or print instead.
    warnings.filterwarnings("ignore")

    # Load environment variables from .env file in current working directory.
    # find_dotenv(usecwd=True) is required because the default find_dotenv()
    # walks up from the caller's file path (site-packages or editable source),
    # which never reaches the user's project directory.
    load_dotenv(find_dotenv(usecwd=True))

    parser = _build_parser()
    args = parser.parse_args(argv)

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1

    try:
        exit_code: int = handler(args)
        return exit_code
    except CLIError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="osmosis",
        description="Osmosis AI SDK - rubric evaluation and remote rollout server.",
    )

    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"{package_name} {PACKAGE_VERSION}",
        help="Show version number.",
    )

    subparsers = parser.add_subparsers(dest="command")

    login_parser = subparsers.add_parser(
        "login",
        help="Authenticate with Osmosis AI.",
    )
    LoginCommand().configure_parser(login_parser)

    whoami_parser = subparsers.add_parser(
        "whoami",
        help="Show current authenticated user and workspace.",
    )
    WhoamiCommand().configure_parser(whoami_parser)

    logout_parser = subparsers.add_parser(
        "logout",
        help="Logout and revoke CLI token.",
    )
    LogoutCommand().configure_parser(logout_parser)

    workspace_parser = subparsers.add_parser(
        "workspace",
        help="Manage workspaces (list, switch, current).",
    )
    WorkspaceCommand().configure_parser(workspace_parser)

    preview_parser = subparsers.add_parser(
        "preview",
        help="Preview a rubric YAML file or test JSONL file and print its parsed contents.",
    )
    PreviewCommand().configure_parser(preview_parser)

    eval_rubric_parser = subparsers.add_parser(
        "eval-rubric",
        help="Evaluate JSONL conversations against a rubric using remote providers.",
    )
    EvalRubricCommand().configure_parser(eval_rubric_parser)

    # Rollout server commands
    from .rollout.cli import EvalCommand, ServeCommand, TestCommand, ValidateCommand

    serve_parser = subparsers.add_parser(
        "serve",
        help="Start a RolloutServer for an agent loop implementation.",
    )
    ServeCommand().configure_parser(serve_parser)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a RolloutAgentLoop implementation without starting the server.",
    )
    ValidateCommand().configure_parser(validate_parser)

    test_parser = subparsers.add_parser(
        "test",
        help="Test a RolloutAgentLoop against a dataset using cloud LLM providers.",
    )
    TestCommand().configure_parser(test_parser)

    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate agent against dataset with eval functions.",
    )
    EvalCommand().configure_parser(eval_parser)

    return parser


if __name__ == "__main__":
    sys.exit(main())
