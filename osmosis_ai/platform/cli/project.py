"""Local project commands (validate canonical project layout)."""

from __future__ import annotations

from typing import Any

from osmosis_ai.cli.console import console


def validate_project(path: Any) -> Any:
    """Validate the canonical Osmosis project structure."""
    from osmosis_ai.cli.output import (
        DetailField,
        DetailResult,
        OutputFormat,
        get_output_context,
    )
    from osmosis_ai.platform.cli.project_contract import (
        resolve_project_root,
        validate_project_contract,
    )

    output = get_output_context()
    project_root = resolve_project_root(path)
    validate_project_contract(project_root)
    rows = [
        ("Root", str(project_root)),
        ("Project metadata", ".osmosis/project.toml"),
        ("Research", ".osmosis/research/"),
        ("Rollouts", "rollouts/"),
        ("Training configs", "configs/training/"),
        ("Eval configs", "configs/eval/"),
        ("Datasets", "data/"),
    ]
    if output.format is OutputFormat.rich:
        console.table(
            [
                ("Root", console.format_text(project_root)),
                ("Project metadata", ".osmosis/project.toml"),
                ("Research", ".osmosis/research/"),
                ("Rollouts", "rollouts/"),
                ("Training configs", "configs/training/"),
                ("Eval configs", "configs/eval/"),
                ("Datasets", "data/"),
            ],
            title="Project Contract",
        )
        console.print("Project contract is valid.", style="green")
        return None

    return DetailResult(
        title="Project Contract",
        data={
            "root": str(project_root),
            "required_paths": [value for label, value in rows if label != "Root"],
            "valid": True,
        },
        fields=[DetailField(label=label, value=value) for label, value in rows],
    )


__all__ = ["validate_project"]
