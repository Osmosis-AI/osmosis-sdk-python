"""Observability utilities for Osmosis rollout SDK.

This module provides logging and tracing capabilities with graceful
degradation when optional dependencies are not available.

Submodules:
    - logging: Structured logging with structlog
    - tracing: Distributed tracing with OpenTelemetry

Example:
    from osmosis_ai.rollout.observability import get_logger, span

    logger = get_logger(__name__)
    logger.info("Starting rollout", rollout_id=rid)

    with span("process_request") as s:
        s.set_attribute("request_id", req_id)
        result = await process()
"""

from osmosis_ai.rollout.observability.logging import (
    clear_context,
    configure_logging,
    get_logger,
    get_rollout_id,
    set_rollout_id,
)
from osmosis_ai.rollout.observability.tracing import (
    SpanNames,
    configure_tracing,
    get_tracer,
    reset_tracing,
    span,
    trace_async,
)

__all__ = [
    # Logging
    "get_logger",
    "configure_logging",
    "get_rollout_id",
    "set_rollout_id",
    "clear_context",
    # Tracing
    "get_tracer",
    "configure_tracing",
    "reset_tracing",
    "span",
    "trace_async",
    "SpanNames",
]
