from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

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

    def run(self, *, path: str) -> int:
        p = Path(path).expanduser()
        if not p.exists():
            raise CLIError(f"Path '{p}' does not exist.")
        if p.is_dir():
            raise CLIError(f"Expected a file path but got directory '{p}'.")

        suffix = p.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            items = self._yaml_loader(p)
            print(f"Loaded {len(items)} rubric config(s) from {p}")
            print(render_yaml_items(items, label="Rubric config"))
        elif suffix == ".jsonl":
            records = self._json_loader(p)
            print(f"Loaded {len(records)} JSONL record(s) from {p}")
            print(render_json_records(records))
        else:
            raise CLIError(
                f"Unsupported file extension '{suffix}'. Expected .yaml, .yml, or .jsonl."
            )

        return 0


def preview(
    path: str = typer.Argument(help="Path to the YAML or JSONL file to inspect."),
) -> None:
    """Preview a rubric YAML or test JSONL file."""
    PreviewCommand().run(path=path)
