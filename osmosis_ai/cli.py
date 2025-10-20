from __future__ import annotations

import argparse
import sys
from typing import Optional

from .cli_commands import EvalCommand, PreviewCommand
from .cli_services import CLIError


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for the osmosis CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except CLIError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="osmosis", description="Utilities for inspecting Osmosis rubric and test data files."
    )
    subparsers = parser.add_subparsers(dest="command")

    preview_parser = subparsers.add_parser(
        "preview",
        help="Preview a rubric YAML file or test JSONL file and print its parsed contents.",
    )
    PreviewCommand().configure_parser(preview_parser)

    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate JSONL conversations against a rubric using remote providers.",
    )
    EvalCommand().configure_parser(eval_parser)

    return parser


if __name__ == "__main__":
    sys.exit(main())
