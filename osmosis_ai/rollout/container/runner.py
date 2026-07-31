"""In-container entrypoints for bundled workflows and graders.

The bundle's generated shim calls ``agent_main(WorkflowClass, config)`` and
``grader_main(GraderClass, config)`` directly — the class is bound at package
time, so nothing is resolved at runtime. Both read the ContainerInput staged
by the backend; the grader additionally reads the agent phase's
ContainerResult.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import traceback
from typing import Any

from osmosis_ai.rollout.container.files import (
    AGENT_LOGS_DIR,
    INPUT_FILENAME,
    RESULT_FILENAME,
    VERIFIER_LOGS_DIR,
    ContainerInput,
    ContainerResult,
    write_reward,
)
from osmosis_ai.rollout.context import (
    AgentWorkflowContext,
    GraderContext,
    RolloutContext,
)
from osmosis_ai.rollout.types import RolloutSample, RolloutStatus
from osmosis_ai.rollout.types.output import AgentWorkflowOutput, coerce_output
from osmosis_ai.rollout.utils.file_artifacts import (
    GRADER_ARTIFACTS_SNAPSHOT_DIRNAME,
    HARBOR_ARTIFACTS_DIR,
    ArtifactFileState,
    artifact_tree_state,
    copy_artifact_tree,
)


async def run_agent(workflow_cls: Any, workflow_config: Any) -> ContainerResult:
    container_input = ContainerInput.read(AGENT_LOGS_DIR / INPUT_FILENAME)
    rollout_ctx = RolloutContext(
        chat_completions_url=container_input.chat_completions_url,
        api_key=container_input.api_key,
        rollout_id=container_input.rollout_id,
    )
    ctx = AgentWorkflowContext(
        prompt=container_input.prompt,
        config=workflow_config,
        metadata=container_input.metadata,
        artifacts_dir=HARBOR_ARTIFACTS_DIR,
    )
    workflow = workflow_cls(workflow_config)
    with rollout_ctx:
        returned = await workflow.run(ctx)
        output = coerce_output(returned)
        if output is None:
            sample = await rollout_ctx.get_sample()
            if sample is not None:
                output = AgentWorkflowOutput(
                    samples={"default": sample.messages},
                    metrics=sample.metrics or {},
                )
            else:
                output = AgentWorkflowOutput()
    return ContainerResult(status=RolloutStatus.SUCCESS, output=output)


def agent_main(workflow_cls: Any, workflow_config: Any = None) -> None:
    try:
        result = asyncio.run(run_agent(workflow_cls, workflow_config))
    except Exception as e:
        traceback.print_exc()
        result = ContainerResult(status=RolloutStatus.FAILURE, err_message=str(e))
    result.write(AGENT_LOGS_DIR / RESULT_FILENAME)
    print(f"Agent runner complete: status={result.status}")


def snapshot_grader_artifacts(baseline: ArtifactFileState) -> None:
    """Stage grader-authored artifact changes where Harbor returns verifier files."""
    snapshot_dir = VERIFIER_LOGS_DIR / GRADER_ARTIFACTS_SNAPSHOT_DIRNAME
    try:
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        if not HARBOR_ARTIFACTS_DIR.is_dir():
            return
        copied = copy_artifact_tree(
            HARBOR_ARTIFACTS_DIR,
            snapshot_dir,
            destination_root=VERIFIER_LOGS_DIR,
            baseline=baseline,
        )
        if not copied:
            shutil.rmtree(snapshot_dir, ignore_errors=True)
    except Exception as e:
        print(f"Failed to stage grader artifacts (best-effort): {e}", file=sys.stderr)


def grader_main(grader_cls: Any, grader_config: Any = None) -> None:
    container_input = ContainerInput.read(AGENT_LOGS_DIR / INPUT_FILENAME)
    result_path = AGENT_LOGS_DIR / RESULT_FILENAME
    result = ContainerResult.read(result_path) if result_path.exists() else None

    output = result.output if result else None
    messages = output.primary_messages() if output else None
    if messages is None:
        print("No transcript from the agent phase, skipping grading")
        write_reward(0.0)
        return

    sample = RolloutSample(id="default", messages=messages, metrics=output.metrics)
    try:
        baseline = artifact_tree_state(HARBOR_ARTIFACTS_DIR)
    except Exception:
        baseline = {}
    try:
        ctx = GraderContext(
            label=container_input.label,
            sample=sample,
            metadata=container_input.metadata,
            artifacts_dir=HARBOR_ARTIFACTS_DIR,
        )
        grader = grader_cls(grader_config)
        asyncio.run(grader.grade(ctx))
    finally:
        snapshot_grader_artifacts(baseline)

    if ctx.sample is None or ctx.sample.reward is None:
        raise RuntimeError("Sample has no reward after grading")
    write_reward(ctx.sample.reward)
    print(f"Grading complete: reward={ctx.sample.reward}")
