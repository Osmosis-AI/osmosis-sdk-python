from __future__ import annotations

import asyncio
import logging
import os
import traceback
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from functools import partial
from typing import Any

from fastapi import FastAPI, Header, HTTPException

from osmosis_ai.rollout.backend.base import ExecutionBackend
from osmosis_ai.rollout.context import RolloutContext
from osmosis_ai.rollout.server.lease import InvalidLeaseError
from osmosis_ai.rollout.server.result_registry import (
    DuplicateRolloutError,
    RolloutFutureRegistry,
    UnknownRolloutError,
)
from osmosis_ai.rollout.trajectory import save_trajectory
from osmosis_ai.rollout.types import (
    POLLING_LEASE_HEADER,
    CancelRolloutsRequest,
    CancelRolloutsResponse,
    ExecutionOutcome,
    ExecutionRequest,
    ExecutionResult,
    RolloutInitRequest,
    RolloutInitResponse,
    RolloutResultResponse,
    RolloutStatus,
)
from osmosis_ai.rollout.utils.errors import categorize_exception

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_RESULT_WAIT_TIMEOUT_SEC = 30.0
DEFAULT_POLLING_LEASE_TIMEOUT_SEC = 120.0
DEFAULT_RESULT_RETENTION_SEC = 900.0
_SHUTDOWN_DRAIN_SEC = 10.0


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
    result_wait_timeout_sec: float = DEFAULT_RESULT_WAIT_TIMEOUT_SEC,
    polling_lease_timeout_sec: float = DEFAULT_POLLING_LEASE_TIMEOUT_SEC,
    result_retention_sec: float = DEFAULT_RESULT_RETENTION_SEC,
) -> FastAPI:
    """Build the FastAPI app a rollout entrypoint serves.

    Uvicorn configures only its own loggers, so ``configure_logging`` installs a
    default INFO handler when the process has none. Pass ``False`` to keep your
    own logging setup.
    """
    if configure_logging:
        _configure_default_logging()

    scheduled_tasks: set[asyncio.Task[None]] = set()

    def _cancel_expired_rollout(rollout_id: str) -> None:
        backend.cancel_rollouts(ids=[rollout_id])

    registry = RolloutFutureRegistry(
        result_wait_timeout_sec=result_wait_timeout_sec,
        polling_lease_timeout_sec=polling_lease_timeout_sec,
        result_retention_sec=result_retention_sec,
        cancel_rollout=_cancel_expired_rollout,
    )

    async def _drain_scheduled_tasks() -> None:
        if not scheduled_tasks:
            return
        logger.info("Draining %d in-flight rollout(s)", len(scheduled_tasks))
        _done, pending = await asyncio.wait(
            set(scheduled_tasks), timeout=_SHUTDOWN_DRAIN_SEC
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    @asynccontextmanager
    async def _lifespan_with_drain(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            if lifespan is not None:
                await stack.enter_async_context(lifespan(app))
            try:
                yield
            finally:
                await _drain_scheduled_tasks()
                await registry.close()

    app = FastAPI(lifespan=_lifespan_with_drain)
    app.state.rollout_futures = registry
    instance_id = os.environ.get("_OSMOSIS_ROLLOUT_INSTANCE_ID")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        payload = dict(backend.health())
        if instance_id:
            payload["instance_id"] = instance_id
        return payload

    def _finish_rollout_task(rollout_id: str, task: asyncio.Task[None]) -> None:
        scheduled_tasks.discard(task)
        exc = None if task.cancelled() else task.exception()
        if exc is not None:
            logger.error("Rollout task for %s crashed", rollout_id, exc_info=exc)

    async def _run_rollout(request: RolloutInitRequest) -> None:
        await registry.set_status(request.rollout_id, RolloutStatus.RUNNING)
        try:
            response = await _handle_rollout(backend, request)
        except asyncio.CancelledError:
            response = RolloutResultResponse(
                rollout_id=request.rollout_id,
                status=RolloutStatus.CANCELLED,
                err_message="cancelled",
            )
            await registry.complete(request.rollout_id, response)
            raise
        await registry.complete(request.rollout_id, response)

    @app.post("/rollout", status_code=202)
    async def rollout(request: RolloutInitRequest) -> RolloutInitResponse:
        if not backend.has_capacity():
            raise HTTPException(
                status_code=429,
                detail="rollout queue is full; retry later",
                headers={"Retry-After": "5"},
            )
        try:
            lease_token = await registry.register(request.rollout_id)
        except DuplicateRolloutError as exc:
            raise HTTPException(
                status_code=409, detail="rollout_id is already registered"
            ) from exc
        try:
            task = asyncio.create_task(_run_rollout(request))
            await registry.bind_task(request.rollout_id, task)
        except Exception:
            await registry.discard(request.rollout_id)
            raise
        scheduled_tasks.add(task)
        task.add_done_callback(partial(_finish_rollout_task, request.rollout_id))
        return RolloutInitResponse(
            rollout_id=request.rollout_id,
            status=RolloutStatus.QUEUED,
            polling_lease_token=lease_token,
            result_wait_timeout_sec=result_wait_timeout_sec,
            polling_lease_timeout_sec=polling_lease_timeout_sec,
        )

    @app.get("/rollout/{rollout_id}/result")
    async def rollout_result(
        rollout_id: str,
        lease_token: str = Header(alias=POLLING_LEASE_HEADER, min_length=1),
    ) -> RolloutResultResponse:
        def _current_status() -> RolloutStatus:
            state = backend.rollout_status(rollout_id)
            if state is not None:
                try:
                    status = RolloutStatus(state.get("status"))
                except (TypeError, ValueError):
                    pass
                else:
                    if status in {
                        RolloutStatus.QUEUED,
                        RolloutStatus.RUNNING,
                        RolloutStatus.GRADING,
                    }:
                        return status
            return RolloutStatus.RUNNING

        try:
            return await registry.wait_for_result(
                rollout_id, lease_token, _current_status
            )
        except UnknownRolloutError as exc:
            raise HTTPException(status_code=404, detail="unknown rollout_id") from exc
        except InvalidLeaseError as exc:
            raise HTTPException(
                status_code=403, detail="invalid polling lease"
            ) from exc

    @app.post("/rollout/cancel")
    async def cancel(request: CancelRolloutsRequest) -> CancelRolloutsResponse:
        selectors = sum(
            [request.ids is not None, request.prefix is not None, request.all]
        )
        if selectors != 1:
            raise HTTPException(
                status_code=422,
                detail="pass exactly one selector: ids, prefix, or all",
            )
        dispositions = backend.cancel_rollouts(
            ids=request.ids, prefix=request.prefix, all=request.all
        )
        logger.info(
            "Cancelled %d rollout(s) (%s)",
            sum(
                1
                for disposition in dispositions.values()
                if disposition.startswith("cancelled")
            ),
            "all" if request.all else request.prefix or f"{len(request.ids or [])} ids",
        )
        return CancelRolloutsResponse(dispositions=dispositions)

    return app


async def _handle_rollout(
    backend: ExecutionBackend, request: RolloutInitRequest
) -> RolloutResultResponse:
    rollout_id = request.rollout_id
    rollout_ctx = RolloutContext(
        chat_completions_url=request.chat_completions_url,
        api_key=request.llm_api_key,
        rollout_id=rollout_id,
    )
    outcome: ExecutionOutcome | None = None
    try:
        with rollout_ctx:
            outcome = await backend.execute(
                ExecutionRequest(
                    id=rollout_id,
                    prompt=request.initial_messages,
                    label=request.label,
                    metadata=request.metadata,
                    agent_timeout_sec=request.agent_timeout_sec,
                    grader_timeout_sec=request.grader_timeout_sec,
                    grade=request.grade,
                )
            )
        result = _terminal_result(outcome)
        logger.info("Rollout %s finished with status %s", rollout_id, result.status)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("Rollout %s failed: %s", rollout_id, traceback.format_exc())
        result = ExecutionResult(
            status=RolloutStatus.FAILURE,
            err_message=str(exc) or "Internal server error",
            err_category=categorize_exception(exc),
        )
    finally:
        if outcome is not None:
            result_to_save = outcome.result_to_save
            await save_trajectory(
                rollout_id=rollout_id,
                result=result_to_save,
                request_label=request.label,
                request_metadata=request.metadata,
                request_extra_fields=request.extra_fields,
                diagnostics=result_to_save.extra_fields,
            )

    wire_sample = result.sample.json_safe_copy() if result.sample is not None else None
    if wire_sample is not None and (
        result.status is not RolloutStatus.SUCCESS or wire_sample.remove_sample
    ):
        wire_sample = wire_sample.model_copy(update={"reward": None})
    return RolloutResultResponse(
        rollout_id=rollout_id,
        status=result.status,
        sample=wire_sample,
        err_message=result.err_message,
        err_category=result.err_category,
    )


def _terminal_result(outcome: ExecutionOutcome) -> ExecutionResult:
    if (
        outcome.grader is not None
        and outcome.grader.status is not RolloutStatus.SUCCESS
    ):
        return outcome.grader
    if outcome.workflow.status is not RolloutStatus.SUCCESS:
        return outcome.workflow
    return outcome.result


__all__ = [
    "DEFAULT_POLLING_LEASE_TIMEOUT_SEC",
    "DEFAULT_RESULT_RETENTION_SEC",
    "DEFAULT_RESULT_WAIT_TIMEOUT_SEC",
    "POLLING_LEASE_HEADER",
    "create_rollout_server",
]
