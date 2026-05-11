from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from osmosis_ai.rollout.driver import RolloutOutcome
from osmosis_ai.rollout.types import (
    GraderCompleteRequest,
    GraderStatus,
    RolloutCompleteRequest,
    RolloutSample,
    RolloutStatus,
)


def _callback_diag(
    obj: RolloutCompleteRequest | GraderCompleteRequest | None,
    *,
    created_sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    if obj is None:
        return {}
    diag = {
        "status": str(obj.status),
        "err_message": obj.err_message,
        "err_category": str(obj.err_category) if obj.err_category else None,
    }
    if isinstance(obj, GraderCompleteRequest):
        diag["available_sample_ids"] = sorted(obj.samples)
        if created_sample_ids is not None:
            callback_ids = set(obj.samples)
            diag["unknown_sample_ids"] = sorted(callback_ids - created_sample_ids)
            diag["missing_sample_ids"] = sorted(created_sample_ids - callback_ids)
    return diag


@dataclass
class ControllerRolloutState:
    rollout_id: str
    _rollout_future: asyncio.Future[RolloutCompleteRequest] | None = field(
        default=None, init=False, repr=False
    )
    _grader_future: asyncio.Future[GraderCompleteRequest] | None = field(
        default=None, init=False, repr=False
    )
    controller_created_sample_ids: set[str] = field(default_factory=set)
    completed_sample_ids: set[str] = field(default_factory=set)
    completion_counts: dict[str, int] = field(default_factory=dict)
    latest_messages: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    latest_multi_turn_mode: dict[str, str] = field(default_factory=dict)
    first_created_tools: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    current_request_tools_by_completion: dict[str, list[list[dict[str, Any]]]] = field(
        default_factory=dict
    )
    total_tokens: int = 0
    systemic_error: str | None = None
    primary_error: str | None = None
    rollout_callback: RolloutCompleteRequest | None = None
    grader_callback: GraderCompleteRequest | None = None
    stale_callbacks: list[str] = field(default_factory=list)

    @property
    def rollout_future(self) -> asyncio.Future[RolloutCompleteRequest]:
        if self._rollout_future is None:
            self._rollout_future = asyncio.get_running_loop().create_future()
            if self.rollout_callback is not None:
                self._rollout_future.set_result(self.rollout_callback)
        return self._rollout_future

    @property
    def grader_future(self) -> asyncio.Future[GraderCompleteRequest]:
        if self._grader_future is None:
            self._grader_future = asyncio.get_running_loop().create_future()
            if self.grader_callback is not None:
                self._grader_future.set_result(self.grader_callback)
        return self._grader_future

    def _normalized_tools(self, tools: Any) -> list[dict[str, Any]]:
        if isinstance(tools, list):
            return [t for t in tools if isinstance(t, dict)]
        return []

    def register_chat_create_branch(
        self,
        sample_id: str,
        mode: str,
        messages: list[dict[str, Any]],
        tools: Any,
    ) -> list[dict[str, Any]]:
        current_tools = self._normalized_tools(tools)
        if mode == "single_sample" and sample_id in self.completed_sample_ids:
            current_tools = self.first_created_tools.get(sample_id, current_tools)
        self.controller_created_sample_ids.add(sample_id)
        self.latest_multi_turn_mode[sample_id] = mode
        self.latest_messages[sample_id] = messages
        self.first_created_tools.setdefault(sample_id, current_tools)
        self.current_request_tools_by_completion.setdefault(sample_id, []).append(
            current_tools
        )
        return current_tools

    def register_chat_reuse_branch(
        self,
        sample_id: str,
        mode: str,
        messages: list[dict[str, Any]],
        tools: Any,
    ) -> list[dict[str, Any]]:
        self.latest_multi_turn_mode[sample_id] = mode
        self.latest_messages[sample_id] = messages
        if mode == "single_sample" and sample_id in self.completed_sample_ids:
            return self.first_created_tools.get(sample_id, [])
        return self.register_chat_create_branch(sample_id, mode, messages, tools)

    def mark_chat_completion(self, sample_id: str, tokens: int = 0) -> None:
        self.completed_sample_ids.add(sample_id)
        self.completion_counts[sample_id] = self.completion_counts.get(sample_id, 0) + 1
        self.total_tokens += max(tokens, 0)

    def mark_systemic_error(self, message: str) -> None:
        self.systemic_error = message
        self.primary_error = self.primary_error or message

    def mark_controller_error(self, message: str) -> None:
        self.mark_systemic_error(message)
        if self.rollout_callback is None:
            self.mark_rollout_completed(
                RolloutCompleteRequest(
                    rollout_id=self.rollout_id,
                    status=RolloutStatus.FAILURE,
                    err_message=message,
                )
            )
            return

        if self.grader_callback is None:
            self.mark_grader_completed(
                GraderCompleteRequest(
                    rollout_id=self.rollout_id,
                    status=GraderStatus.FAILURE,
                    samples={},
                    err_message=message,
                )
            )

    def mark_rollout_completed(self, request: RolloutCompleteRequest) -> None:
        if self.rollout_callback is not None:
            self.stale_callbacks.append("rollout")
            return
        self.rollout_callback = request
        if self._rollout_future is not None and not self._rollout_future.done():
            self._rollout_future.set_result(request)

    def mark_grader_completed(self, request: GraderCompleteRequest) -> None:
        if self.grader_callback is not None:
            self.stale_callbacks.append("grader")
            return
        self.grader_callback = request
        if self._grader_future is not None and not self._grader_future.done():
            self._grader_future.set_result(request)

    def cancel_pending(self) -> None:
        for future in (self._rollout_future, self._grader_future):
            if future is None:
                continue
            if not future.done():
                future.cancel()

    def _validate_rewards(self, samples: dict[str, RolloutSample]) -> str | None:
        created = set(self.controller_created_sample_ids)
        callback_ids = set(samples)
        unknown = sorted(callback_ids - created)
        if unknown:
            return f"Grader returned rewards for sample IDs not created by the controller: {unknown}"
        if not created:
            return "No samples returned from rollout. The rollout must call /chat/completions with x-sample-id before grading."
        missing = sorted(created - callback_ids)
        if missing:
            return (
                f"Missing grader rewards for controller-created sample IDs: {missing}"
            )
        none_rewards = sorted(
            sid
            for sid, sample in samples.items()
            if not sample.remove_sample and sample.reward is None
        )
        if none_rewards:
            return f"Grader returned None as reward for sample IDs: {none_rewards}"
        return None

    def to_outcome(self, duration_ms: float) -> RolloutOutcome:
        samples = (
            self.grader_callback.samples if self.grader_callback is not None else {}
        )
        reward_error = self._validate_rewards(samples)
        if self.primary_error or reward_error:
            return RolloutOutcome(
                status=RolloutStatus.FAILURE,
                rollout_id=self.rollout_id,
                samples=samples,
                error=self.primary_error or reward_error,
                duration_ms=duration_ms,
                tokens=self.total_tokens,
                systemic_error=self.systemic_error,
                controller_created_sample_ids=sorted(
                    self.controller_created_sample_ids
                ),
                completion_counts=dict(self.completion_counts),
                full_callback_sample_ids=sorted(samples),
                callback_diagnostics={
                    "rollout": _callback_diag(self.rollout_callback),
                    "grader": _callback_diag(
                        self.grader_callback,
                        created_sample_ids=self.controller_created_sample_ids,
                    ),
                    "stale_callbacks": list(self.stale_callbacks),
                },
            )

        scored = sorted(
            sid for sid, sample in samples.items() if not sample.remove_sample
        )
        skipped = sorted(sid for sid, sample in samples.items() if sample.remove_sample)
        return RolloutOutcome(
            status=RolloutStatus.SUCCESS,
            rollout_id=self.rollout_id,
            samples=samples,
            duration_ms=duration_ms,
            tokens=self.total_tokens,
            controller_created_sample_ids=sorted(self.controller_created_sample_ids),
            completion_counts=dict(self.completion_counts),
            full_callback_sample_ids=sorted(samples),
            scored_sample_ids=scored,
            skipped_sample_ids=skipped,
            callback_diagnostics={
                "rollout": _callback_diag(self.rollout_callback),
                "grader": _callback_diag(
                    self.grader_callback,
                    created_sample_ids=self.controller_created_sample_ids,
                ),
                "stale_callbacks": list(self.stale_callbacks),
            },
            skipped=bool(samples) and not scored,
        )
