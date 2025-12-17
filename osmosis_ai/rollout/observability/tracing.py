"""Distributed tracing for Osmosis rollout SDK.

This module provides OpenTelemetry-based distributed tracing with automatic
span creation and context propagation. Falls back to no-op implementations
when OpenTelemetry is not available.

Example:
    from osmosis_ai.rollout.observability.tracing import (
        span, trace_async, SpanNames
    )

    # Using context manager
    with span("process_rollout", attributes={"rollout_id": rid}) as s:
        s.add_event("started")
        result = await do_work()
        s.set_attribute("status", result.status)

    # Using decorator
    @trace_async("llm_completion")
    async def call_llm(messages):
        return await client.chat(messages)
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional, TypeVar

from osmosis_ai.rollout._compat import (
    OTEL_AVAILABLE,
    OTEL_SDK_AVAILABLE,
    OTLP_EXPORTER_AVAILABLE,
)
from osmosis_ai.rollout.config.settings import TracingSettings, get_settings

# Type variable for decorators
F = TypeVar("F", bound=Callable[..., Any])

# Module state
_tracer: Optional[Any] = None
_initialized: bool = False


class SpanNames:
    """Standardized span names for consistent tracing.

    Use these constants instead of string literals to ensure
    consistent naming across the codebase.
    """

    # Server spans
    ROLLOUT_INIT = "rollout.init"
    ROLLOUT_RUN = "rollout.run"
    ROLLOUT_COMPLETE = "rollout.complete"
    ROLLOUT_ERROR = "rollout.error"

    # Client spans
    LLM_CHAT_COMPLETIONS = "llm.chat_completions"
    LLM_COMPLETE_ROLLOUT = "llm.complete_rollout"

    # Tool spans
    TOOL_EXECUTE = "tool.execute"
    TOOL_EXECUTE_BATCH = "tool.execute_batch"
    TOOL_PARSE_ARGUMENTS = "tool.parse_arguments"

    # Internal spans
    MESSAGE_PROCESS = "message.process"
    REGISTRY_LOOKUP = "registry.lookup"
    HTTP_REQUEST = "http.request"


class NoopSpan:
    """No-op span implementation for when tracing is disabled.

    Provides the same interface as OpenTelemetry Span but does nothing.
    """

    def set_attribute(self, key: str, value: Any) -> "NoopSpan":
        """Set a span attribute (no-op)."""
        return self

    def set_attributes(self, attributes: Dict[str, Any]) -> "NoopSpan":
        """Set multiple span attributes (no-op)."""
        return self

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        timestamp: Optional[int] = None,
    ) -> "NoopSpan":
        """Add an event to the span (no-op)."""
        return self

    def record_exception(
        self,
        exception: BaseException,
        attributes: Optional[Dict[str, Any]] = None,
        timestamp: Optional[int] = None,
        escaped: bool = False,
    ) -> None:
        """Record an exception (no-op)."""
        pass

    def set_status(self, status: Any, description: Optional[str] = None) -> None:
        """Set the span status (no-op)."""
        pass

    def update_name(self, name: str) -> None:
        """Update the span name (no-op)."""
        pass

    def is_recording(self) -> bool:
        """Check if span is recording (always False for no-op)."""
        return False

    def end(self, end_time: Optional[int] = None) -> None:
        """End the span (no-op)."""
        pass

    def __enter__(self) -> "NoopSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    @property
    def context(self) -> "NoopSpanContext":
        """Get span context (no-op)."""
        return NoopSpanContext()


class NoopSpanContext:
    """No-op span context."""

    trace_id: int = 0
    span_id: int = 0
    is_valid: bool = False
    is_remote: bool = False
    trace_flags: int = 0
    trace_state: Any = None


class NoopTracer:
    """No-op tracer implementation for when tracing is disabled.

    Provides the same interface as OpenTelemetry Tracer but does nothing.
    """

    def start_span(
        self,
        name: str,
        context: Any = None,
        kind: Any = None,
        attributes: Optional[Dict[str, Any]] = None,
        links: Any = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> NoopSpan:
        """Start a new span (returns no-op span)."""
        return NoopSpan()

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        context: Any = None,
        kind: Any = None,
        attributes: Optional[Dict[str, Any]] = None,
        links: Any = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> Generator[NoopSpan, None, None]:
        """Start a span as current (yields no-op span)."""
        yield NoopSpan()


def configure_tracing(settings: Optional[TracingSettings] = None) -> None:
    """Configure OpenTelemetry tracing.

    Sets up the tracer provider, exporter, and propagators based on settings.
    When tracing is disabled or OpenTelemetry is not available, configures
    a no-op tracer.

    Args:
        settings: Tracing configuration. Defaults to global settings.

    Example:
        from osmosis_ai.rollout.observability.tracing import configure_tracing
        from osmosis_ai.rollout.config import TracingSettings

        configure_tracing(TracingSettings(
            enabled=True,
            exporter="otlp",
            endpoint="http://jaeger:4317",
        ))
    """
    global _tracer, _initialized

    if settings is None:
        settings = get_settings().tracing

    if not settings.enabled or not OTEL_AVAILABLE:
        _tracer = NoopTracer()
        _initialized = True
        return

    # Import OpenTelemetry components
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    # Get version
    try:
        from osmosis_ai.consts import PACKAGE_VERSION
    except ImportError:
        PACKAGE_VERSION = "unknown"

    # Create resource with service info
    resource = Resource.create(
        {
            SERVICE_NAME: settings.service_name,
            SERVICE_VERSION: PACKAGE_VERSION,
        }
    )

    # Configure exporter first (before creating provider)
    exporter = None
    if settings.exporter == "console":
        exporter = ConsoleSpanExporter()
    elif settings.exporter == "otlp" and OTLP_EXPORTER_AVAILABLE:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        endpoint = settings.endpoint or "http://localhost:4317"
        exporter = OTLPSpanExporter(endpoint=endpoint)
    elif settings.exporter == "jaeger":
        # Jaeger uses OTLP in newer versions
        if OTLP_EXPORTER_AVAILABLE:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            endpoint = settings.endpoint or "http://localhost:4317"
            exporter = OTLPSpanExporter(endpoint=endpoint)
    elif settings.exporter == "zipkin":
        try:
            from opentelemetry.exporter.zipkin.json import ZipkinExporter

            endpoint = settings.endpoint or "http://localhost:9411/api/v2/spans"
            exporter = ZipkinExporter(endpoint=endpoint)
        except ImportError:
            pass

    # Create tracer provider with optional sampler
    sampler = None
    if settings.sample_rate < 1.0:
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

        sampler = TraceIdRatioBased(settings.sample_rate)

    provider = TracerProvider(resource=resource, sampler=sampler)

    # Add exporter to provider
    if exporter:
        provider.add_span_processor(BatchSpanProcessor(exporter))

    # Set global tracer provider
    trace.set_tracer_provider(provider)

    # Configure propagators
    _configure_propagators(settings.propagators)

    # Get tracer
    _tracer = trace.get_tracer("osmosis_ai.rollout", PACKAGE_VERSION)
    _initialized = True


def _configure_propagators(propagators_str: str) -> None:
    """Configure trace context propagators."""
    try:
        from opentelemetry.propagate import set_global_textmap
        from opentelemetry.propagators.composite import CompositePropagator

        propagators = []
        propagator_names = [p.strip().lower() for p in propagators_str.split(",")]

        for name in propagator_names:
            if name == "tracecontext":
                from opentelemetry.trace.propagation.tracecontext import (
                    TraceContextTextMapPropagator,
                )

                propagators.append(TraceContextTextMapPropagator())
            elif name == "b3":
                try:
                    from opentelemetry.propagators.b3 import B3MultiFormat

                    propagators.append(B3MultiFormat())
                except ImportError:
                    pass
            elif name == "b3multi":
                try:
                    from opentelemetry.propagators.b3 import B3MultiFormat

                    propagators.append(B3MultiFormat())
                except ImportError:
                    pass
            elif name == "baggage":
                from opentelemetry.baggage.propagation import W3CBaggagePropagator

                propagators.append(W3CBaggagePropagator())

        if propagators:
            set_global_textmap(CompositePropagator(propagators))
    except ImportError:
        pass


def reset_tracing() -> None:
    """Reset tracing state.

    Primarily used for testing to ensure clean state between tests.
    """
    global _tracer, _initialized
    _tracer = None
    _initialized = False


def get_tracer() -> Any:
    """Get the configured tracer.

    Returns the OpenTelemetry tracer if configured and available,
    otherwise returns a no-op tracer.

    Returns:
        A tracer instance (OpenTelemetry or no-op).

    Example:
        tracer = get_tracer()
        with tracer.start_as_current_span("my_operation") as span:
            span.set_attribute("key", "value")
    """
    global _tracer, _initialized

    if not _initialized:
        configure_tracing()

    return _tracer or NoopTracer()


@contextmanager
def span(
    name: str,
    kind: Optional[Any] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Generator[Any, None, None]:
    """Create a tracing span as a context manager.

    Provides a convenient way to create spans without directly
    interacting with the tracer.

    Args:
        name: Span name. Use SpanNames constants for consistency.
        kind: Span kind (SpanKind.CLIENT, SpanKind.SERVER, etc.).
        attributes: Initial span attributes.

    Yields:
        The current span (OpenTelemetry Span or NoopSpan).

    Example:
        with span("process_request", attributes={"request_id": "123"}) as s:
            s.add_event("started")
            result = await process()
            s.set_attribute("status", "success")
    """
    tracer = get_tracer()

    if isinstance(tracer, NoopTracer):
        yield NoopSpan()
    else:
        # Build kwargs for start_as_current_span
        kwargs: Dict[str, Any] = {}
        if kind is not None:
            kwargs["kind"] = kind
        if attributes:
            kwargs["attributes"] = attributes

        with tracer.start_as_current_span(name, **kwargs) as s:
            yield s


def trace_async(
    name: Optional[str] = None,
    kind: Optional[Any] = None,
    record_exception: bool = True,
) -> Callable[[F], F]:
    """Decorator for tracing async functions.

    Automatically creates a span around the decorated function,
    recording exceptions and setting error status on failure.

    Args:
        name: Span name. Defaults to function name.
        kind: Span kind.
        record_exception: Whether to record exceptions in the span.

    Returns:
        Decorated function.

    Example:
        @trace_async("llm_completion", kind=SpanKind.CLIENT)
        async def call_llm(messages):
            return await client.chat(messages)
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()

            if isinstance(tracer, NoopTracer):
                return await func(*args, **kwargs)

            span_kwargs: Dict[str, Any] = {}
            if kind is not None:
                span_kwargs["kind"] = kind
            span_kwargs["record_exception"] = record_exception
            span_kwargs["set_status_on_exception"] = record_exception

            with tracer.start_as_current_span(span_name, **span_kwargs):
                return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def trace_sync(
    name: Optional[str] = None,
    kind: Optional[Any] = None,
    record_exception: bool = True,
) -> Callable[[F], F]:
    """Decorator for tracing sync functions.

    Automatically creates a span around the decorated function.

    Args:
        name: Span name. Defaults to function name.
        kind: Span kind.
        record_exception: Whether to record exceptions in the span.

    Returns:
        Decorated function.

    Example:
        @trace_sync("parse_arguments")
        def parse_args(data):
            return json.loads(data)
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()

            if isinstance(tracer, NoopTracer):
                return func(*args, **kwargs)

            span_kwargs: Dict[str, Any] = {}
            if kind is not None:
                span_kwargs["kind"] = kind
            span_kwargs["record_exception"] = record_exception
            span_kwargs["set_status_on_exception"] = record_exception

            with tracer.start_as_current_span(span_name, **span_kwargs):
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def get_current_span() -> Any:
    """Get the current active span.

    Returns:
        The current span, or a NoopSpan if no span is active.
    """
    if not OTEL_AVAILABLE:
        return NoopSpan()

    from opentelemetry import trace

    span = trace.get_current_span()
    if span is None:
        return NoopSpan()
    return span


def inject_trace_context(carrier: Dict[str, str]) -> None:
    """Inject trace context into a carrier dict (e.g., HTTP headers).

    Args:
        carrier: Dictionary to inject trace context into.

    Example:
        headers = {}
        inject_trace_context(headers)
        # headers now contains traceparent, tracestate, etc.
    """
    if not OTEL_AVAILABLE:
        return

    from opentelemetry.propagate import inject

    inject(carrier)


def extract_trace_context(carrier: Dict[str, str]) -> Any:
    """Extract trace context from a carrier dict.

    Args:
        carrier: Dictionary containing trace context.

    Returns:
        Context object to use when creating spans.

    Example:
        context = extract_trace_context(request.headers)
        with tracer.start_as_current_span("handle", context=context):
            ...
    """
    if not OTEL_AVAILABLE:
        return None

    from opentelemetry.propagate import extract

    return extract(carrier)
