"""HTTP middleware for observability integration.

This module provides middleware for integrating logging, tracing,
and metrics into the FastAPI application.

Example:
    from fastapi import FastAPI
    from osmosis_ai.rollout.server.middleware import add_observability_middleware

    app = FastAPI()
    add_observability_middleware(app)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

from osmosis_ai.rollout.observability.logging import get_logger, set_context, clear_context
from osmosis_ai.rollout.observability.metrics import get_metrics
from osmosis_ai.rollout.observability.tracing import span, SpanNames

logger = get_logger(__name__)


async def observability_middleware(
    request: "Request",
    call_next: Callable[..., Any],
) -> "Response":
    """Unified observability middleware.

    Integrates logging, tracing, and metrics collection for all HTTP requests.

    Features:
        - Automatic request ID generation
        - Request/response logging
        - Prometheus metrics (request count, duration)
        - Distributed tracing (when enabled)

    Args:
        request: The incoming request.
        call_next: The next middleware/handler in the chain.

    Returns:
        The response from the handler.
    """
    start_time = time.perf_counter()
    metrics = get_metrics()

    # Generate or extract request ID
    request_id = request.headers.get("X-Request-ID", str(uuid4()))

    # Extract endpoint info
    endpoint = request.url.path
    method = request.method

    # Set logging context
    set_context(request_id=request_id, endpoint=endpoint, method=method)

    try:
        with span(
            SpanNames.HTTP_REQUEST,
            attributes={
                "http.method": method,
                "http.url": str(request.url),
                "http.route": endpoint,
                "request_id": request_id,
            },
        ) as s:
            try:
                response = await call_next(request)
                status_code = response.status_code

                # Record metrics
                metrics.http_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status_code=str(status_code),
                ).inc()

                # Set span attributes
                s.set_attribute("http.status_code", status_code)

                # Add request ID to response headers
                response.headers["X-Request-ID"] = request_id

                return response

            except Exception as e:
                # Record error metrics
                metrics.http_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status_code="500",
                ).inc()

                s.set_attribute("http.status_code", 500)
                s.set_attribute("error", str(e))
                raise

    finally:
        # Calculate and record duration
        duration = time.perf_counter() - start_time
        metrics.http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint,
        ).observe(duration)

        # Log request completion
        logger.info(
            "http_request_completed",
            method=method,
            endpoint=endpoint,
            duration_ms=duration * 1000,
            request_id=request_id,
        )

        # Clear logging context
        clear_context()


def add_observability_middleware(app: Any) -> None:
    """Add observability middleware to a FastAPI application.

    Args:
        app: The FastAPI application instance.

    Example:
        from fastapi import FastAPI
        from osmosis_ai.rollout.server.middleware import add_observability_middleware

        app = FastAPI()
        add_observability_middleware(app)
    """
    app.middleware("http")(observability_middleware)
