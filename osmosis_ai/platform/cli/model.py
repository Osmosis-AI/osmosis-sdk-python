"""Handlers for `osmosis model` subcommands.

Models cover both base (foundation) models and LoRA models produced by
training runs. The ``cli/commands/model.py`` shell delegates here:

    osmosis model list                      -> list_models()
    osmosis model info     <lora-model>     -> info()
    osmosis model deploy   <lora-model>     -> deploy()
    osmosis model undeploy <lora-model>     -> undeploy()
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import (
    DetailResult,
    DetailSection,
    ListColumn,
    ListResult,
    ListSection,
    OperationResult,
    SectionedListResult,
    detail_fields,
    get_output_context,
    serialize_lora_model,
    serialize_model,
)
from osmosis_ai.cli.output.display import format_local_date, format_local_datetime
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.api.models import BaseModelInfo, LoraModelInfo
from osmosis_ai.platform.cli.utils import (
    format_deployment_status,
    format_reward,
    kv_section,
    paginated_fetch,
    require_git_workspace_directory_context,
    validate_list_options,
)
from osmosis_ai.platform.cli.workspace_directory_context import git_result_context
from osmosis_ai.platform.constants import INFERENCE_URL

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
]

# Shown only when the platform includes deployment info (inference available).
_LORA_DEPLOYMENT_COLUMN = ListColumn(
    key="deployment_status", label="Deployment Status", no_wrap=True, ratio=1
)


def _lora_model_summary_resource(result: Any) -> dict[str, Any]:
    return {
        "id": result.id,
        "model_name": result.model_name,
        "status": result.status,
    }


def _workspace_page_url(platform_url: str | None, page: str) -> str | None:
    """Derive a workspace page URL (e.g. ``/api-keys``) from a resource's
    ``platform_url``, whose first path segment is the workspace name."""
    if not platform_url:
        return None
    parts = urlsplit(platform_url)
    segments = [segment for segment in parts.path.split("/") if segment]
    if not segments:
        return None
    return f"{parts.scheme}://{parts.netloc}/{segments[0]}{page}"


def _base_model_display_item(model: BaseModelInfo) -> dict[str, Any]:
    return {
        **serialize_model(model),
        "created_at": format_local_date(model.created_at),
        "creator_name": model.creator_name or "–",
    }


def _lora_model_display_item(model: LoraModelInfo) -> dict[str, Any]:
    item = {
        **serialize_lora_model(model),
        "base_model": model.base_model or "–",
        "training_run_name": model.training_run_name or "–",
        "checkpoint_step": (
            str(model.checkpoint_step) if model.checkpoint_step is not None else "–"
        ),
        "reward": format_reward(model.reward),
        "created_at": format_local_date(model.created_at),
    }
    if model.has_deployment_info:
        item["deployment_status"] = format_deployment_status(model.deployment_status)
    return item


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

    # ``paginated_fetch`` returns only items + cursor fields, discarding the
    # page object, so capture the page-level deployment quota and info flag
    # from the first response via a closure.
    quota: dict[str, int] = {}
    deployment_info = {"present": True}

    def _fetch_lora(lim: int, off: int) -> Any:
        page = client.list_lora_models(
            limit=lim,
            offset=off,
            credentials=credentials,
            git_identity=git_identity,
        )
        quota.setdefault("active_deployments", page.active_deployments)
        quota.setdefault("max_active_deployments", page.max_active_deployments)
        deployment_info["present"] = page.has_deployment_info
        return page

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
                _fetch_lora,
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
    lora_columns = list(_LORA_MODEL_COLUMNS)
    if deployment_info["present"]:
        lora_columns.append(_LORA_DEPLOYMENT_COLUMN)
    lora_section = ListSection(
        key="lora_models",
        title="LoRA Models",
        items=[serialize_lora_model(m) for m in lora_models],
        total_count=lora_total,
        has_more=lora_has_more,
        next_offset=lora_next_offset,
        columns=lora_columns,
        display_items=[_lora_model_display_item(m) for m in lora_models],
    )
    active_deployments = quota.get("active_deployments", 0)
    max_active_deployments = quota.get("max_active_deployments", 0)
    display_hints: list[str] = []
    if max_active_deployments > 0:
        display_hints.append(
            f"{active_deployments} of {max_active_deployments} "
            "inference deployments used"
        )
    if lora_models:
        display_hints.append("Use osmosis model info <name> for details.")
    lora_extra = {**git_result_context(context)}
    if deployment_info["present"]:
        lora_extra["active_deployments"] = active_deployments
        lora_extra["max_active_deployments"] = max_active_deployments

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
            extra=lora_extra,
            columns=lora_section.columns,
            display_items=lora_section.display_items,
            display_hints=display_hints,
        )

    return SectionedListResult(
        sections=[base_section, lora_section],
        extra=lora_extra,
        display_hints=display_hints,
    )


def info(lora_model_name: str) -> DetailResult:
    """Show details for a single LoRA model."""
    context = require_git_workspace_directory_context()
    client = OsmosisClient()
    output = get_output_context()

    with output.status(f'Fetching LoRA model "{console.escape(lora_model_name)}"...'):
        model = client.get_lora_model(
            lora_model_name,
            credentials=context.credentials,
            git_identity=context.git_identity,
        )

    # Rows mirror the platform's model detail sidebar (fields and order);
    # Hugging Face and Deployment render as their own sections, like the
    # page's cards.
    rows: list[tuple[str, str]] = [
        ("Name", model.model_name),
        ("Base Model", model.base_model or "–"),
        ("Training Run", model.training_run_name or "–"),
        (
            "Checkpoint Step",
            str(model.checkpoint_step) if model.checkpoint_step is not None else "–",
        ),
        ("Training Reward", format_reward(model.reward)),
        ("Created", format_local_datetime(model.created_at)),
    ]

    sections: list[DetailSection] = []

    hf_rows: list[tuple[str, str]] = [
        (
            "Upload Status",
            model.hf_upload_status.title() if model.hf_upload_status else "–",
        ),
    ]
    if model.hf_url:
        hf_rows.append(("URL", model.hf_url))
    if model.uploaded_by:
        hf_rows.append(("Uploaded By", model.uploaded_by))
    hf_section = kv_section("Hugging Face", hf_rows)
    if hf_section is not None:
        sections.append(hf_section)

    if model.has_deployment_info:
        deployment_rows: list[tuple[str, str]] = [
            ("Status", format_deployment_status(model.deployment_status, plain=True)),
        ]
        if model.deployed_at:
            deployment_rows.append(
                ("Deployed", format_local_datetime(model.deployed_at))
            )
        if model.deployed_by:
            deployment_rows.append(("Deployed By", model.deployed_by))
        deployment_section = kv_section("Deployment", deployment_rows)
        if deployment_section is not None:
            sections.append(deployment_section)

    display_hints: list[str] = []
    if model.platform_url:
        display_hints.append(f"View: {model.platform_url}")
    if model.has_deployment_info:
        if model.deployment_status == "active":
            if model.inference_model:
                display_hints.append(
                    "Query it (OpenAI-compatible, requires a workspace API key): "
                    f"curl -X POST {INFERENCE_URL}/v1/chat/completions "
                    '-H "Authorization: Bearer $OSMOSIS_API_KEY" '
                    '-H "Content-Type: application/json" '
                    f'-d \'{{"model": "{model.inference_model}", '
                    '"messages": [{"role": "user", "content": "Hello!"}]}\''
                )
                api_keys_url = _workspace_page_url(model.platform_url, "/api-keys")
                if api_keys_url:
                    display_hints.append(f"Create an API key: {api_keys_url}")
            display_hints.append(
                f"Undeploy with: osmosis model undeploy {model.model_name}"
            )
        else:
            display_hints.append(
                f"Deploy with: osmosis model deploy {model.model_name}"
            )

    lora_model = {
        **serialize_lora_model(model),
        "hf_upload_status": model.hf_upload_status,
        "hf_url": model.hf_url,
        "uploaded_by": model.uploaded_by,
        "inference_model": model.inference_model,
    }

    return DetailResult(
        title="LoRA Model Info",
        data={
            "lora_model": lora_model,
            "platform_url": model.platform_url,
            **git_result_context(context),
        },
        fields=detail_fields(rows),
        sections=sections,
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
        message=f"LoRA model deployed: {result.model_name or lora_model_name}",
        display_next_steps=[
            f"Check status with: osmosis model info {result.model_name}",
        ],
        next_steps_structured=[{"action": "model_info", "name": result.model_name}],
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
        message=f"LoRA model undeployed: {result.model_name or lora_model_name}",
        display_next_steps=[
            f"Check status with: osmosis model info {result.model_name}",
        ],
        next_steps_structured=[{"action": "model_info", "name": result.model_name}],
    )
