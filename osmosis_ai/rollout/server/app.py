import logging
import threading
import traceback
import uuid
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException

from osmosis_ai.rollout.backend.base import ExecutionBackend
from osmosis_ai.rollout.context import RolloutContext
from osmosis_ai.rollout.server.auth import ControllerAuth
from osmosis_ai.rollout.server.native_harbor_gateway import (
    install_native_harbor_gateway_routes,
)
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

_ROLLOUT_BACKEND_STATE_ATTR = "_osmosis_execution_backend"
_ROLLOUT_QUEUE_FULL_DETAIL = "Rollout queue is full"


class _RolloutAdmission:
    """Process-local reservation accounting for accepted rollout requests."""

    def __init__(self, backend: ExecutionBackend) -> None:
        self._max_concurrent = backend.max_concurrency
        self._max_queue_depth = backend.max_queue_depth
        if self._max_queue_depth is not None and self._max_queue_depth < 0:
            raise ValueError("backend.max_queue_depth must be >= 0 or None")
        self._in_flight = 0
        self._lock = threading.Lock()

    @property
    def _limit(self) -> int | None:
        if self._max_concurrent <= 0 or self._max_queue_depth is None:
            return None
        return self._max_concurrent + self._max_queue_depth

    def reserve(self) -> bool:
        """Atomically reserve one accepted request, or reject a full queue."""
        with self._lock:
            limit = self._limit
            if limit is not None and self._in_flight >= limit:
                return False
            self._in_flight += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._in_flight <= 0:
                raise RuntimeError("rollout admission reservation underflow")
            self._in_flight -= 1

    def snapshot(self) -> dict[str, int | bool | None]:
        with self._lock:
            limit = self._limit
            queue_depth = (
                max(0, self._in_flight - self._max_concurrent)
                if self._max_concurrent > 0
                else 0
            )
            available = None if limit is None else max(0, limit - self._in_flight)
            return {
                "max_concurrent": self._max_concurrent,
                "max_queue_depth": self._max_queue_depth,
                "in_flight": self._in_flight,
                "queue_depth": queue_depth,
                "available": available,
                "accepting": available is None or available > 0,
            }


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
    setattr(app.state, _ROLLOUT_BACKEND_STATE_ATTR, backend)
    admission = _RolloutAdmission(backend)

    translation_gateway = getattr(backend, "translation_gateway", None)
    if translation_gateway is not None:
        install_native_harbor_gateway_routes(app, translation_gateway)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        payload = dict(backend.health())
        payload["capacity"] = admission.snapshot()
        return payload

    @app.post("/rollout")
    async def rollout(
        request: RolloutInitRequest, background_tasks: BackgroundTasks
    ) -> RolloutInitResponse:
        if not admission.reserve():
            raise HTTPException(status_code=429, detail=_ROLLOUT_QUEUE_FULL_DETAIL)
        try:
            response = RolloutInitResponse()
            background_tasks.add_task(
                _handle_admitted_rollout,
                admission,
                backend,
                request,
            )
        except Exception as e:
            admission.release()
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e)) from e
        return response

    return app


def _get_rollout_server_backend(app: Any) -> ExecutionBackend | None:
    """Return the backend recorded by ``create_rollout_server``, if any."""
    state = getattr(app, "state", None)
    backend = getattr(state, _ROLLOUT_BACKEND_STATE_ATTR, None)
    return backend if isinstance(backend, ExecutionBackend) else None


async def _handle_rollout(
    backend: ExecutionBackend, request: RolloutInitRequest
) -> None:
    # Routing identity is in the URLs; ``rollout_id`` in the body is debug
    # metadata. We prefer the caller's id (so logs/cache rows correlate
    # across systems) and synthesize one only if the caller omits it.
    rollout_id = request.rollout_id or uuid.uuid4().hex
    auth = ControllerAuth(api_key=request.controller_api_key)

    rollout_ctx = RolloutContext(
        chat_completions_url=request.chat_completions_url,
        api_key=request.controller_api_key,
        rollout_id=rollout_id,
    )

    # Prefer grader (has the reward) unless it carries no sample.
    result_to_save: ExecutionResult | None = None
    # Latest metrics from callback acks.
    report: TrajectoryReport | None = None

    def record_result_to_save(result: ExecutionResult) -> None:
        nonlocal result_to_save
        if result_to_save is None or result.sample is not None:
            result_to_save = result

    async def on_workflow_complete(result: ExecutionResult) -> None:
        nonlocal report
        record_result_to_save(result)
        resp = await post_json_with_retry(
            url=request.completion_callback_url,
            payload=RolloutCompleteRequest(
                status=result.status,
                rollout_id=rollout_id,
                extra_fields=result.extra_fields,
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
                rollout_id,
            )
            return
        logger.info(
            "Posting grader callback for %s to %s (status=%s, has_sample=%s)",
            rollout_id,
            request.grader_callback_url,
            result.status,
            result.sample is not None,
        )
        resp = await post_json_with_retry(
            url=request.grader_callback_url,
            payload=GraderCompleteRequest(
                rollout_id=rollout_id,
                status=GraderStatus.SUCCESS
                if result.status == RolloutStatus.SUCCESS
                else GraderStatus.FAILURE,
                sample=result.sample,
                extra_fields=result.extra_fields,
                err_message=result.err_message,
                err_category=result.err_category,
            ).model_dump(exclude={"sample": {"trajectory_messages"}}),
            headers=auth.as_bearer_headers(),
        )
        report = report_from_response(resp) or report
        logger.info(
            "Grader callback for %s completed: status=%d",
            rollout_id,
            resp.status_code,
        )

    try:
        with rollout_ctx:
            await backend.execute(
                ExecutionRequest(
                    id=rollout_id,
                    prompt=request.initial_messages,
                    label=request.label,
                    metadata=request.metadata,
                    agent_timeout_sec=request.agent_timeout_sec,
                    grader_timeout_sec=request.grader_timeout_sec,
                ),
                on_workflow_complete=on_workflow_complete,
                on_grader_complete=on_grader_complete
                if request.grader_callback_url or backend.capture_final_result
                else None,
            )
        logger.info("Rollout %s completed successfully", rollout_id)
    except Exception:
        logger.error("Rollout %s failed: %s", rollout_id, traceback.format_exc())
        try:
            resp = await post_json_with_retry(
                url=request.completion_callback_url,
                payload=RolloutCompleteRequest(
                    status=RolloutStatus.FAILURE,
                    rollout_id=rollout_id,
                    extra_fields=result_to_save.extra_fields
                    if result_to_save is not None
                    else None,
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
                rollout_id=rollout_id,
                result=result_to_save,
                request_label=request.label,
                request_metadata=request.metadata,
                request_extra_fields=request.extra_fields,
                report=report,
            )


async def _handle_admitted_rollout(
    admission: _RolloutAdmission,
    backend: ExecutionBackend,
    request: RolloutInitRequest,
) -> None:
    try:
        await _handle_rollout(backend, request)
    finally:
        admission.release()
