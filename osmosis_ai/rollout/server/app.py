import logging
import traceback
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException

from osmosis_ai.rollout.backend.base import ExecutionBackend
from osmosis_ai.rollout.context import RolloutContext
from osmosis_ai.rollout.server.auth import ControllerAuth
from osmosis_ai.rollout.trajectory import (
    TrajectoryReport,
    report_from_response,
    save_trajectories,
)
from osmosis_ai.rollout.types import (
    ExecutionRequest,
    ExecutionResult,
    GraderCompleteRequest,
    GraderStatus,
    RolloutCompleteRequest,
    RolloutInitRequest,
    RolloutInitResponse,
    RolloutStatus,
)
from osmosis_ai.rollout.utils.http import post_json_with_retry

logger: logging.Logger = logging.getLogger(__name__)


def _configure_default_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def create_rollout_server(
    *,
    backend: ExecutionBackend,
    lifespan: Any = None,
    configure_logging: bool = True,
) -> FastAPI:
    """Build the FastAPI app a rollout entrypoint serves.

    Uvicorn configures only its own loggers, so ``configure_logging`` installs a
    default INFO handler when the process has none. Pass ``False`` to keep your
    own logging setup.
    """
    if configure_logging:
        _configure_default_logging()
    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return backend.health()

    @app.post("/rollout")
    async def rollout(
        request: RolloutInitRequest, background_tasks: BackgroundTasks
    ) -> RolloutInitResponse:
        try:
            background_tasks.add_task(_handle_rollout, backend, request)
            return RolloutInitResponse()
        except Exception as e:
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


async def _handle_rollout(
    backend: ExecutionBackend, request: RolloutInitRequest
) -> None:
    auth = ControllerAuth(api_key=request.controller_api_key)

    rollout_ctx = RolloutContext(
        chat_completions_url=request.chat_completions_url,
        api_key=request.controller_api_key,
        rollout_id=request.rollout_id,
    )

    # Prefer grader (has rewards) unless it has no samples.
    result_to_save: ExecutionResult | None = None
    # Latest metrics from callback acks.
    report: TrajectoryReport | None = None

    def record_result_to_save(result: ExecutionResult) -> None:
        nonlocal result_to_save
        if result_to_save is None or result.samples:
            result_to_save = result

    async def on_workflow_complete(result: ExecutionResult) -> None:
        nonlocal report
        record_result_to_save(result)
        resp = await post_json_with_retry(
            url=request.completion_callback_url,
            payload=RolloutCompleteRequest(
                rollout_id=request.rollout_id,
                status=result.status,
                err_message=result.err_message,
                err_category=result.err_category,
            ).model_dump(),
            headers=auth.as_bearer_headers(),
        )
        report = report_from_response(resp) or report

    async def on_grader_complete(result: ExecutionResult) -> None:
        nonlocal report
        record_result_to_save(result)
        if not request.grader_callback_url:
            logger.info(
                "Skipping grader callback for %s: no grader_callback_url",
                request.rollout_id,
            )
            return
        logger.info(
            "Posting grader callback for %s to %s (status=%s, samples=%d)",
            request.rollout_id,
            request.grader_callback_url,
            result.status,
            len(result.samples),
        )
        resp = await post_json_with_retry(
            url=request.grader_callback_url,
            payload=GraderCompleteRequest(
                rollout_id=request.rollout_id,
                status=GraderStatus.SUCCESS
                if result.status == RolloutStatus.SUCCESS
                else GraderStatus.FAILURE,
                samples=result.samples,
                err_message=result.err_message,
                err_category=result.err_category,
            ).model_dump(exclude={"samples": {"__all__": {"trajectory_messages"}}}),
            headers=auth.as_bearer_headers(),
        )
        report = report_from_response(resp) or report
        logger.info(
            "Grader callback for %s completed: status=%d",
            request.rollout_id,
            resp.status_code,
        )

    try:
        with rollout_ctx:
            await backend.execute(
                ExecutionRequest(
                    id=request.rollout_id,
                    prompt=request.initial_messages,
                    label=request.label,
                    metadata=request.metadata,
                    agent_timeout_sec=request.agent_timeout_sec,
                    grader_timeout_sec=request.grader_timeout_sec,
                ),
                on_workflow_complete=on_workflow_complete,
                on_grader_complete=on_grader_complete
                if request.grader_callback_url
                else None,
            )
        logger.info("Rollout %s completed successfully", request.rollout_id)
    except Exception:
        logger.error(
            "Rollout %s failed: %s", request.rollout_id, traceback.format_exc()
        )
        try:
            resp = await post_json_with_retry(
                url=request.completion_callback_url,
                payload=RolloutCompleteRequest(
                    rollout_id=request.rollout_id,
                    status=RolloutStatus.FAILURE,
                    err_message="Internal server error",
                ).model_dump(),
                headers=auth.as_bearer_headers(),
            )
            report = report_from_response(resp) or report
        except Exception:
            logger.error("Failed to post error callback: %s", traceback.format_exc())
        if request.grader_callback_url:
            try:
                await on_grader_complete(ExecutionResult(status=RolloutStatus.FAILURE))
            except Exception:
                logger.error(
                    "Failed to post grader error callback: %s",
                    traceback.format_exc(),
                )
    finally:
        # Best-effort archive once execute() has finished.
        if result_to_save is not None:
            await save_trajectories(
                rollout_id=request.rollout_id,
                result=result_to_save,
                request_label=request.label,
                request_metadata=request.metadata,
                request_extra_fields=request.extra_fields,
                report=report,
            )
