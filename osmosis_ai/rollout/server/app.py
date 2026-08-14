import asyncio
import hashlib
import json
import logging
import os
import traceback
import uuid
from functools import partial
from typing import Any

from fastapi import FastAPI, HTTPException

from osmosis_ai.rollout.backend.base import ExecutionBackend
from osmosis_ai.rollout.context import RolloutContext
from osmosis_ai.rollout.server.auth import ControllerAuth
from osmosis_ai.rollout.trajectory import (
    TrajectoryReport,
    report_from_response,
    save_trajectory,
)
from osmosis_ai.rollout.types import (
    CancelRolloutsRequest,
    CancelRolloutsResponse,
    ExecutionRequest,
    ExecutionResult,
    GraderCompleteRequest,
    GraderStatus,
    RolloutCompleteRequest,
    RolloutInitRequest,
    RolloutInitResponse,
    RolloutSample,
    RolloutStatus,
    RolloutStatusResponse,
)
from osmosis_ai.rollout.utils.http import post_json_with_retry
from osmosis_ai.rollout.utils.ttl_cache import TtlCache

logger: logging.Logger = logging.getLogger(__name__)

# Credentials never enter the idempotency digest.
_IDEMPOTENCY_EXCLUDED_FIELDS = {"controller_api_key", "llm_api_key"}

# Completed idempotency digests are retained for this window so a duplicate
# retry of a finished rollout dedupes instead of re-executing. 15 minutes
# comfortably covers the driver's retry/status-recovery horizon while keeping
# memory bounded on long-lived training servers (same window as the Harbor
# backend's STATUS_RETENTION_SEC). Active entries never expire.
_COMPLETED_DIGEST_TTL_SEC = 900.0


def _canonical_request_digest(request: RolloutInitRequest) -> str:
    """Stable digest of one init body, excluding callback/LLM secrets."""
    payload = request.model_dump(mode="json", exclude=_IDEMPOTENCY_EXCLUDED_FIELDS)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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
    # Idempotency: rollout_id -> canonical request digest. Active entries
    # cover scheduled/running executions and never expire; when execution
    # finishes, only id + digest move into bounded recent retention. Digests
    # only; raw payloads and credentials are not retained. /status stays
    # backend-authoritative.
    active_digests: dict[str, str] = {}
    completed_digests: TtlCache[str, str] = TtlCache(_COMPLETED_DIGEST_TTL_SEC)
    # Strong references so eagerly scheduled rollout tasks are never GC'd.
    scheduled_tasks: set[asyncio.Task[bool]] = set()
    # The instance id env var is immutable for the process lifetime.
    instance_id = os.environ.get("_OSMOSIS_ROLLOUT_INSTANCE_ID")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        payload = dict(backend.health())
        if instance_id:
            payload["instance_id"] = instance_id
        return payload

    def _finish_rollout_task(rollout_id: str | None, task: asyncio.Task[bool]) -> None:
        scheduled_tasks.discard(task)
        exc = None if task.cancelled() else task.exception()
        if exc is not None:
            logger.error("Rollout task for %s crashed", rollout_id, exc_info=exc)
        if rollout_id is None:
            return
        digest = active_digests.pop(rollout_id, None)
        # Retain the digest only when every required terminal callback was
        # delivered. A crashed, cancelled, or delivery-failed task left the
        # controller without its terminal callback; a duplicate retry deduped
        # against it would get a 202 and wait forever for a callback that is
        # never coming. Dropping the digest lets the retry re-execute.
        delivered = not task.cancelled() and exc is None and task.result()
        if digest is not None and delivered:
            completed_digests.set(rollout_id, digest)

    # 202: the rollout is scheduled before this response is sent, so a
    # failed/disconnected response cannot record a digest for an execution
    # that never ran (the retry would dedupe against it and hang forever).
    @app.post("/rollout", status_code=202)
    async def rollout(request: RolloutInitRequest) -> RolloutInitResponse:
        rollout_id = request.rollout_id or None
        digest: str | None = None
        if rollout_id is not None:
            digest = _canonical_request_digest(request)
            existing = active_digests.get(rollout_id)
            if existing is None:
                existing = completed_digests.get(rollout_id)
            if existing is not None:
                if existing == digest:
                    return RolloutInitResponse()
                raise HTTPException(
                    status_code=409,
                    detail="conflicting duplicate rollout_id",
                )
        if not backend.has_capacity():
            raise HTTPException(
                status_code=429,
                detail="rollout queue is full; retry later",
                headers={"Retry-After": "5"},
            )
        try:
            task = asyncio.create_task(_handle_rollout(backend, request))
        except Exception as e:
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e)) from e
        scheduled_tasks.add(task)
        if rollout_id is not None and digest is not None:
            active_digests[rollout_id] = digest
        task.add_done_callback(partial(_finish_rollout_task, rollout_id))
        return RolloutInitResponse()

    @app.get("/rollout/{rollout_id}/status")
    async def rollout_status(rollout_id: str) -> RolloutStatusResponse:
        state = backend.rollout_status(rollout_id)
        if state is not None:
            return RolloutStatusResponse(rollout_id=rollout_id, **state)
        return RolloutStatusResponse(
            rollout_id=rollout_id, status=RolloutStatus.UNKNOWN
        )

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
            sum(1 for d in dispositions.values() if d.startswith("cancelled")),
            "all" if request.all else request.prefix or f"{len(request.ids or [])} ids",
        )
        return CancelRolloutsResponse(dispositions=dispositions)

    return app


async def _handle_rollout(
    backend: ExecutionBackend, request: RolloutInitRequest
) -> bool:
    """Execute one rollout; return whether every required terminal callback
    was delivered (the completion callback, plus the grader callback when
    ``grader_callback_url`` is set). The admission layer retains the
    idempotency digest only for delivered rollouts."""
    # Routing identity is in the URLs; ``rollout_id`` in the body is debug
    # metadata. We prefer the caller's id (so logs/cache rows correlate
    # across systems) and synthesize one only if the caller omits it.
    rollout_id = request.rollout_id or uuid.uuid4().hex
    auth = ControllerAuth(api_key=request.controller_api_key)

    llm_api_key = (
        request.controller_api_key
        if request.llm_api_key is None
        else request.llm_api_key
    )
    rollout_ctx = RolloutContext(
        chat_completions_url=request.chat_completions_url,
        api_key=llm_api_key,
        rollout_id=rollout_id,
    )

    # Two retention slots: the best sample-bearing result for the archive, and
    # the latest diagnostics so a sample-less failure still leaves a record.
    result_to_save: ExecutionResult | None = None
    last_diagnostics: dict[str, Any] | None = None
    # Latest metrics from callback acks.
    report: TrajectoryReport | None = None
    # A terminal callback is a promise the controller acts on: it closes out
    # ``rollout_id``. Posting a second one contradicts the first, so track what
    # has already been delivered and let the error path below skip it.
    completion_posted = False
    grader_posted = False

    def record_result_to_save(result: ExecutionResult) -> None:
        nonlocal result_to_save, last_diagnostics
        if result.extra_fields is not None:
            last_diagnostics = result.extra_fields
        if result_to_save is None or result.sample is not None:
            result_to_save = result

    async def on_workflow_complete(result: ExecutionResult) -> None:
        nonlocal report, completion_posted
        record_result_to_save(result)
        resp = await post_json_with_retry(
            url=request.completion_callback_url,
            payload=RolloutCompleteRequest(
                status=result.status,
                rollout_id=rollout_id,
                err_message=result.err_message,
                err_category=result.err_category,
            ).model_dump(),
            headers=auth.as_bearer_headers(),
        )
        completion_posted = True
        report = report_from_response(resp) or report

    async def on_grader_complete(result: ExecutionResult) -> None:
        nonlocal report, grader_posted
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
        status = (
            GraderStatus.SUCCESS
            if result.status == RolloutStatus.SUCCESS
            else GraderStatus.FAILURE
        )
        wire_sample = result.sample
        if wire_sample is not None:
            # Callers mutate ``metrics`` in place, past every validator;
            # this is the last point before the payload has to be JSON.
            wire_sample = wire_sample.json_safe_copy()
            # Consumers attach any numeric reward they see to eval metrics and
            # training trajectories, checking neither status nor remove_sample.
            # So the wire reward has to stand for exactly one thing: a kept,
            # successfully-graded sample. A failed grade or a sample the grader
            # marked for removal must not carry one. The archived trajectory
            # keeps the original sample, reward included.
            reward_is_trainable = (
                status is GraderStatus.SUCCESS and not wire_sample.remove_sample
            )
            if not reward_is_trainable and wire_sample.reward is not None:
                wire_sample = wire_sample.model_copy(update={"reward": None})

        def grader_payload(sample: RolloutSample | None) -> dict[str, Any]:
            return GraderCompleteRequest(
                rollout_id=rollout_id,
                status=status,
                sample=sample,
                err_message=result.err_message,
                err_category=result.err_category,
            ).model_dump(exclude={"sample": {"trajectory_messages"}})

        try:
            resp = await post_json_with_retry(
                url=request.grader_callback_url,
                payload=grader_payload(wire_sample),
                headers=auth.as_bearer_headers(),
            )
        except (TypeError, ValueError):
            # Encoding failed despite sanitization. Telemetry is optional;
            # the reward is not — retry with a minimal sample so an earned
            # reward still arrives instead of a fabricated grader failure.
            logger.exception(
                "Grader callback payload for %s is not JSON-encodable; "
                "retrying without telemetry",
                rollout_id,
            )
            minimal = (
                RolloutSample(
                    messages=[],
                    trajectory_messages=None,
                    label=wire_sample.label,
                    reward=wire_sample.reward,
                    remove_sample=wire_sample.remove_sample,
                )
                if wire_sample is not None
                else None
            )
            resp = await post_json_with_retry(
                url=request.grader_callback_url,
                payload=grader_payload(minimal),
                headers=auth.as_bearer_headers(),
            )
        grader_posted = True
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
        if completion_posted:
            # The workflow already reported a terminal status; whatever failed
            # afterwards (grading, callback encoding) belongs to the grader
            # callback below, not to a second, contradicting completion.
            logger.info(
                "Rollout %s already reported completion; not posting a second one",
                rollout_id,
            )
        else:
            try:
                resp = await post_json_with_retry(
                    url=request.completion_callback_url,
                    payload=RolloutCompleteRequest(
                        status=RolloutStatus.FAILURE,
                        rollout_id=rollout_id,
                        err_message="Internal server error",
                    ).model_dump(),
                    headers=auth.as_bearer_headers(),
                )
                completion_posted = True
                report = report_from_response(resp) or report
            except Exception:
                logger.error(
                    "Failed to post error callback: %s", traceback.format_exc()
                )
        if request.grader_callback_url and not grader_posted:
            try:
                await on_grader_complete(ExecutionResult(status=RolloutStatus.FAILURE))
            except Exception:
                logger.error(
                    "Failed to post grader error callback: %s",
                    traceback.format_exc(),
                )
    finally:
        # Best-effort archive once execute() has finished.
        if result_to_save is not None or last_diagnostics is not None:
            await save_trajectory(
                rollout_id=rollout_id,
                result=result_to_save or ExecutionResult(status=RolloutStatus.FAILURE),
                request_label=request.label,
                request_metadata=request.metadata,
                request_extra_fields=request.extra_fields,
                report=report,
                diagnostics=last_diagnostics,
            )
    return completion_posted and (not request.grader_callback_url or grader_posted)
