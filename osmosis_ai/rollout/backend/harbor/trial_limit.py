"""How many Harbor trials one rollout server runs at once."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from harbor.trial.queue import TrialQueue

PLATFORM_TRIAL_LIMIT_ENV = "_OSMOSIS_ROLLOUT_CONCURRENCY"
LOCAL_TRIAL_LIMIT = 8


@dataclass(frozen=True)
class TrialLimit:
    value: int
    log_level: int
    message: str


def resolve_trial_limit(orchestrator: TrialQueue | None) -> TrialLimit:
    raw = os.environ.get(PLATFORM_TRIAL_LIMIT_ENV, "").strip()
    platform = int(raw) if raw else None
    if platform is not None and platform < 1:
        raise ValueError(
            f"{PLATFORM_TRIAL_LIMIT_ENV} must be at least 1, got {platform}."
        )
    # Harbor exposes no public accessor for the queue's limit.
    own = None if orchestrator is None else orchestrator._n_concurrent
    if own is not None and own < 1:
        raise ValueError(
            f"Trial limit must be at least 1, got {own} ({queue_label(own)})."
        )
    if platform is None:
        return local_trial_limit(own)
    return platform_trial_limit(platform, own)


def local_trial_limit(own: int | None) -> TrialLimit:
    if own is None:
        return TrialLimit(
            LOCAL_TRIAL_LIMIT,
            logging.INFO,
            f"Trial limit: {LOCAL_TRIAL_LIMIT} at once (SDK default for local runs; "
            "eval max_concurrent_attempts is capped to match).",
        )
    return TrialLimit(
        own,
        logging.INFO,
        f"Trial limit: {own} at once, set by this rollout's code ({queue_label(own)}).",
    )


def platform_trial_limit(platform: int, own: int | None) -> TrialLimit:
    if own is None or own == platform:
        return TrialLimit(
            platform,
            logging.INFO,
            f"Trial limit: {platform} at once (set by the platform for this run).",
        )
    if own < platform:
        return TrialLimit(
            own,
            logging.WARNING,
            f"Trial limit: {own} at once, set by this rollout's code "
            f"({queue_label(own)}). The platform allows {platform} for this run, "
            f"so it will run {own} at a time instead of {platform}. To use the "
            "platform's number, remove the orchestrator= argument from HarborBackend.",
        )
    return TrialLimit(
        platform,
        logging.INFO,
        f"Trial limit: {platform} at once. This rollout's code asks for {own}; "
        f"the platform allows {platform} for this run.",
    )


def queue_label(n_concurrent: int) -> str:
    return f"TrialQueue(n_concurrent={n_concurrent})"
