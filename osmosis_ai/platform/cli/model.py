"""Handlers for `osmosis model` subcommands.

Models cover both base (foundation) models and LoRA models produced by
training runs. The ``cli/commands/model.py`` shell delegates here:

    osmosis model list                      -> list_models()
    osmosis model deploy   <lora-model>     -> deploy()
    osmosis model undeploy <lora-model>     -> undeploy()
"""

from __future__ import annotations

from typing import Any

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import (
    ListColumn,
    ListResult,
    ListSection,
    OperationResult,
    SectionedListResult,
    get_output_context,
    serialize_lora_model,
    serialize_model,
)
from osmosis_ai.cli.output.display import format_local_date
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.api.models import BaseModelInfo, LoraModelInfo
from osmosis_ai.platform.cli.utils import (
    format_deployment_status,
    format_reward,
    paginated_fetch,
    require_git_workspace_directory_context,
    validate_list_options,
)
from osmosis_ai.platform.cli.workspace_directory_context import git_result_context

_VALID_LIST_TYPES = ("all", "base", "lora")

_BASE_MODEL_COLUMNS = [
    ListColumn(key="model_name", label="Name", ratio=3, overflow="fold"),
    ListColumn(key="created_at", label="Created", no_wrap=True, ratio=1),
    ListColumn(key="creator_name", label="Created By", no_wrap=True, ratio=1),
]

_LORA_MODEL_COLUMNS = [
    ListColumn(key="model_name", label="Name", ratio=3, overflow="fold"),
    ListColumn(key="base_model", label="Base Model", ratio=2, overflow="fold"),
    ListColumn(key="training_run_name", label="Training Run", ratio=2, overflow="fold"),
    ListColumn(key="checkpoint_step", label="Checkpoint Step", no_wrap=True, ratio=1),
    ListColumn(key="reward", label="Training Reward", no_wrap=True, ratio=1),
    ListColumn(key="created_at", label="Created", no_wrap=True, ratio=1),
    ListColumn(
        key="deployment_status", label="Deployment Status", no_wrap=True, ratio=1
    ),
]


def _lora_model_summary_resource(result: Any) -> dict[str, Any]:
    return {
        "id": result.id,
        "model_name": result.model_name,
        "status": result.status,
    }


def _base_model_display_item(model: BaseModelInfo) -> dict[str, Any]:
    return {
        **serialize_model(model),
        "created_at": format_local_date(model.created_at),
        "creator_name": model.creator_name or "—",
    }


def _lora_model_display_item(model: LoraModelInfo) -> dict[str, Any]:
    return {
        **serialize_lora_model(model),
        "base_model": model.base_model or "—",
        "training_run_name": model.training_run_name or "—",
        "checkpoint_step": (
            str(model.checkpoint_step) if model.checkpoint_step is not None else "—"
        ),
        "reward": format_reward(model.reward),
        "created_at": format_local_date(model.created_at),
        "deployment_status": format_deployment_status(model.deployment_status),
    }


def list_models(
    *, limit: int, all_: bool, type_: str = "all"
) -> SectionedListResult | ListResult:
    """List base models and LoRA models for the current workspace directory.

    ``type_`` ``"all"`` returns a :class:`SectionedListResult` with a base
    models section followed by a LoRA models section, each independently
    paginated. ``"base"`` / ``"lora"`` return a single :class:`ListResult`.
    """
    if type_ not in _VALID_LIST_TYPES:
        raise CLIError(
            "Type must be 'all', 'base', or 'lora'.",
            code="VALIDATION",
        )
    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    context = require_git_workspace_directory_context()
    credentials = context.credentials
    git_identity = context.git_identity

    output = get_output_context()
    client = OsmosisClient()

    base_models: list[BaseModelInfo] = []
    lora_models: list[LoraModelInfo] = []
    base_total, base_has_more, base_next_offset = 0, False, None
    lora_total, lora_has_more, lora_next_offset = 0, False, None
    with output.status("Fetching models..."):
        if type_ in ("all", "base"):
            base_models, base_total, base_has_more, base_next_offset = paginated_fetch(
                lambda lim, off: client.list_base_models(
                    limit=lim,
                    offset=off,
                    credentials=credentials,
                    git_identity=git_identity,
                ),
                items_attr="models",
                limit=effective_limit,
                fetch_all=fetch_all,
            )
        if type_ in ("all", "lora"):
            lora_models, lora_total, lora_has_more, lora_next_offset = paginated_fetch(
                lambda lim, off: client.list_lora_models(
                    limit=lim,
                    offset=off,
                    credentials=credentials,
                    git_identity=git_identity,
                ),
                items_attr="models",
                limit=effective_limit,
                fetch_all=fetch_all,
            )

    base_section = ListSection(
        key="base_models",
        title="Base Models",
        items=[serialize_model(m) for m in base_models],
        total_count=base_total,
        has_more=base_has_more,
        next_offset=base_next_offset,
        columns=_BASE_MODEL_COLUMNS,
        display_items=[_base_model_display_item(m) for m in base_models],
    )
    lora_section = ListSection(
        key="lora_models",
        title="LoRA Models",
        items=[serialize_lora_model(m) for m in lora_models],
        total_count=lora_total,
        has_more=lora_has_more,
        next_offset=lora_next_offset,
        columns=_LORA_MODEL_COLUMNS,
        display_items=[_lora_model_display_item(m) for m in lora_models],
    )
    display_hints = (
        ["Deploy a LoRA model with: osmosis model deploy <name>"] if lora_models else []
    )

    if type_ == "base":
        return ListResult(
            title=base_section.title,
            items=base_section.items,
            total_count=base_section.total_count,
            has_more=base_section.has_more,
            next_offset=base_section.next_offset,
            extra=git_result_context(context),
            columns=base_section.columns,
            display_items=base_section.display_items,
        )
    if type_ == "lora":
        return ListResult(
            title=lora_section.title,
            items=lora_section.items,
            total_count=lora_section.total_count,
            has_more=lora_section.has_more,
            next_offset=lora_section.next_offset,
            extra=git_result_context(context),
            columns=lora_section.columns,
            display_items=lora_section.display_items,
            display_hints=display_hints,
        )

    return SectionedListResult(
        sections=[base_section, lora_section],
        extra=git_result_context(context),
        display_hints=display_hints,
    )


def deploy(lora_model_name: str) -> OperationResult:
    """Deploy (or reactivate) a LoRA model."""
    context = require_git_workspace_directory_context()
    client = OsmosisClient()
    output = get_output_context()

    with output.status(f'Deploying LoRA model "{console.escape(lora_model_name)}"...'):
        result = client.deploy_lora_model(
            lora_model_name,
            credentials=context.credentials,
            git_identity=context.git_identity,
        )

    resource = _lora_model_summary_resource(result)
    resource.update(git_result_context(context))
    return OperationResult(
        operation="model.deploy",
        status="success",
        resource=resource,
        message=f"LoRA model {result.model_name or '-'} {result.status}",
        display_next_steps=["Check status with: osmosis model list"],
        next_steps_structured=[{"action": "model_list"}],
    )


def undeploy(lora_model_name: str) -> OperationResult:
    """Undeploy a LoRA model (transition to ``inactive``)."""
    context = require_git_workspace_directory_context()
    client = OsmosisClient()
    output = get_output_context()

    with output.status(
        f'Undeploying LoRA model "{console.escape(lora_model_name)}"...'
    ):
        result = client.undeploy_lora_model(
            lora_model_name,
            credentials=context.credentials,
            git_identity=context.git_identity,
        )

    resource = _lora_model_summary_resource(result)
    resource.update(git_result_context(context))
    return OperationResult(
        operation="model.undeploy",
        status="success",
        resource=resource,
        message=f"LoRA model {result.model_name or '-'} {result.status}",
        display_next_steps=["Check status with: osmosis model list"],
        next_steps_structured=[{"action": "model_list"}],
    )
