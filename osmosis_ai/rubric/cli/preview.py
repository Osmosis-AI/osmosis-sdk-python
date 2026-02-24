from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.rubric.services import (
    ParsedItem,
    load_jsonl_records,
    load_rubric_configs,
    render_json_records,
    render_yaml_items,
)


class PreviewCommand:
    """Handler for `osmosis preview`."""

    def __init__(
        self,
        *,
        yaml_loader: Callable[[Path], list[ParsedItem]] = load_rubric_configs,
        json_loader: Callable[[Path], list[dict[str, Any]]] = load_jsonl_records,
    ):
        self._yaml_loader = yaml_loader
        self._json_loader = json_loader

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.set_defaults(handler=self.run)
        parser.add_argument(
            "-p",
            "--path",
            dest="path",
            required=True,
            help="Path to the YAML or JSONL file to inspect.",
        )

    def run(self, args: argparse.Namespace) -> int:
        path = Path(args.path).expanduser()
        if not path.exists():
            raise CLIError(f"Path '{path}' does not exist.")
        if path.is_dir():
            raise CLIError(f"Expected a file path but got directory '{path}'.")

        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            items = self._yaml_loader(path)
            print(f"Loaded {len(items)} rubric config(s) from {path}")
            print(render_yaml_items(items, label="Rubric config"))
        elif suffix == ".jsonl":
            records = self._json_loader(path)
            print(f"Loaded {len(records)} JSONL record(s) from {path}")
            print(render_json_records(records))
        else:
            raise CLIError(
                f"Unsupported file extension '{suffix}'. Expected .yaml, .yml, or .jsonl."
            )

        return 0
