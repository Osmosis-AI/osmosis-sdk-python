"""Prometheus metrics for Osmosis rollout SDK.

This module provides Prometheus metrics collection with automatic
fallback to no-op implementations when prometheus_client is not available.

Metrics follow Prometheus naming conventions:
- Counter names end with _total
- Histogram names describe the unit (e.g., _seconds, _bytes)
- Labels are lowercase with underscores

Example:
    from osmosis_ai.rollout.observability.metrics import get_metrics

    metrics = get_metrics()

    # Increment counters
    metrics.rollouts_started.labels(agent_loop="my_agent").inc()
    metrics.rollouts_completed.labels(
        agent_loop="my_agent",
        status="COMPLETED",
        finish_reason="stop"
    ).inc()

    # Observe histograms
    metrics.rollout_duration_seconds.labels(
        agent_loop="my_agent",
        status="COMPLETED"
    ).observe(1.5)
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional, Tuple

from osmosis_ai.rollout._compat import PROMETHEUS_AVAILABLE, prometheus_client
from osmosis_ai.rollout.config.settings import MetricsSettings, get_settings

# Module state
_metrics: Optional["RolloutMetrics"] = None


class NoopMetric:
    """No-op metric implementation for when Prometheus is not available.

    Provides the same interface as Prometheus metrics but does nothing.
    """

    def inc(self, amount: float = 1) -> None:
        """Increment counter (no-op)."""
        pass

    def dec(self, amount: float = 1) -> None:
        """Decrement gauge (no-op)."""
        pass

    def set(self, value: float) -> None:
        """Set gauge value (no-op)."""
        pass

    def observe(self, amount: float) -> None:
        """Observe histogram value (no-op)."""
        pass

    def time(self) -> "NoopTimer":
        """Return a timer context manager (no-op)."""
        return NoopTimer()

    def labels(self, *args: Any, **kwargs: Any) -> "NoopMetric":
        """Return labeled metric (returns self for no-op)."""
        return self

    def info(self, val: Dict[str, str]) -> None:
        """Set info metric (no-op)."""
        pass

    def remove(self, *args: Any) -> None:
        """Remove labeled metric (no-op)."""
        pass


class NoopTimer:
    """No-op timer for histogram time() method."""

    def __enter__(self) -> "NoopTimer":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


def _get_python_version() -> str:
    """Get Python version string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


class RolloutMetrics:
    """Prometheus metrics collection for Osmosis rollout SDK.

    Provides counters, gauges, and histograms for monitoring rollout
    operations, LLM calls, tool executions, and HTTP requests.

    Attributes:
        rollouts_started: Counter for rollouts started.
        rollouts_completed: Counter for rollouts completed.
        rollouts_failed: Counter for rollouts failed.
        rollouts_active: Gauge for currently active rollouts.
        rollout_duration_seconds: Histogram for rollout duration.
        llm_requests_total: Counter for LLM requests.
        llm_request_duration_seconds: Histogram for LLM request duration.
        llm_tokens_total: Counter for tokens processed.
        tool_calls_total: Counter for tool calls.
        tool_call_duration_seconds: Histogram for tool call duration.
        tool_errors_total: Counter for tool errors.
        http_requests_total: Counter for HTTP requests.
        http_request_duration_seconds: Histogram for HTTP request duration.
        service_info: Info metric with service details.

    Example:
        metrics = RolloutMetrics()
        metrics.rollouts_started.labels(agent_loop="my_agent").inc()
    """

    def __init__(
        self,
        settings: Optional[MetricsSettings] = None,
        registry: Optional[Any] = None,
    ):
        """Initialize metrics collection.

        Args:
            settings: Metrics configuration. Defaults to global settings.
            registry: Custom Prometheus registry. Uses default if None.
        """
        if settings is None:
            settings = get_settings().metrics

        self._enabled = settings.enabled and PROMETHEUS_AVAILABLE
        self._prefix = settings.prefix
        self._registry = registry

        if not self._enabled:
            self._create_noop_metrics()
        else:
            self._create_prometheus_metrics(registry)

    def _metric_name(self, name: str) -> str:
        """Generate full metric name with prefix."""
        return f"{self._prefix}_{name}"

    def _create_noop_metrics(self) -> None:
        """Create no-op metric instances."""
        # Rollout metrics
        self.rollouts_started: Any = NoopMetric()
        self.rollouts_completed: Any = NoopMetric()
        self.rollouts_failed: Any = NoopMetric()
        self.rollouts_active: Any = NoopMetric()
        self.rollout_duration_seconds: Any = NoopMetric()

        # LLM metrics
        self.llm_requests_total: Any = NoopMetric()
        self.llm_request_duration_seconds: Any = NoopMetric()
        self.llm_tokens_total: Any = NoopMetric()

        # Tool metrics
        self.tool_calls_total: Any = NoopMetric()
        self.tool_call_duration_seconds: Any = NoopMetric()
        self.tool_errors_total: Any = NoopMetric()

        # HTTP metrics
        self.http_requests_total: Any = NoopMetric()
        self.http_request_duration_seconds: Any = NoopMetric()

        # Service info
        self.service_info: Any = NoopMetric()

    def _create_prometheus_metrics(self, registry: Optional[Any]) -> None:
        """Create Prometheus metric instances."""
        from prometheus_client import Counter, Gauge, Histogram, Info

        kwargs: Dict[str, Any] = {}
        if registry is not None:
            kwargs["registry"] = registry

        # === Rollout Metrics ===

        self.rollouts_started = Counter(
            self._metric_name("rollouts_started_total"),
            "Total number of rollouts started",
            ["agent_loop"],
            **kwargs,
        )

        self.rollouts_completed = Counter(
            self._metric_name("rollouts_completed_total"),
            "Total number of rollouts completed successfully",
            ["agent_loop", "status", "finish_reason"],
            **kwargs,
        )

        self.rollouts_failed = Counter(
            self._metric_name("rollouts_failed_total"),
            "Total number of rollouts that failed",
            ["agent_loop", "error_type"],
            **kwargs,
        )

        self.rollouts_active = Gauge(
            self._metric_name("rollouts_active"),
            "Number of currently active rollouts",
            ["agent_loop"],
            **kwargs,
        )

        self.rollout_duration_seconds = Histogram(
            self._metric_name("rollout_duration_seconds"),
            "Rollout execution duration in seconds",
            ["agent_loop", "status"],
            buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600),
            **kwargs,
        )

        # === LLM Metrics ===

        self.llm_requests_total = Counter(
            self._metric_name("llm_requests_total"),
            "Total number of LLM completion requests",
            ["status"],  # success, error, timeout
            **kwargs,
        )

        self.llm_request_duration_seconds = Histogram(
            self._metric_name("llm_request_duration_seconds"),
            "LLM request duration in seconds",
            ["status"],
            buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
            **kwargs,
        )

        self.llm_tokens_total = Counter(
            self._metric_name("llm_tokens_total"),
            "Total number of tokens processed",
            ["type"],  # prompt, completion
            **kwargs,
        )

        # === Tool Metrics ===

        self.tool_calls_total = Counter(
            self._metric_name("tool_calls_total"),
            "Total number of tool calls",
            ["tool_name", "status"],  # success, error
            **kwargs,
        )

        self.tool_call_duration_seconds = Histogram(
            self._metric_name("tool_call_duration_seconds"),
            "Tool call execution duration in seconds",
            ["tool_name"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
            **kwargs,
        )

        self.tool_errors_total = Counter(
            self._metric_name("tool_errors_total"),
            "Total number of tool execution errors",
            ["tool_name", "error_type"],
            **kwargs,
        )

        # === HTTP Metrics ===

        self.http_requests_total = Counter(
            self._metric_name("http_requests_total"),
            "Total number of HTTP requests",
            ["method", "endpoint", "status_code"],
            **kwargs,
        )

        self.http_request_duration_seconds = Histogram(
            self._metric_name("http_request_duration_seconds"),
            "HTTP request duration in seconds",
            ["method", "endpoint"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
            **kwargs,
        )

        # === Service Info ===

        self.service_info = Info(
            self._metric_name("service"),
            "Service information",
            **kwargs,
        )

        # Set service info
        try:
            from osmosis_ai.consts import PACKAGE_VERSION
        except ImportError:
            PACKAGE_VERSION = "unknown"

        self.service_info.info(
            {
                "version": PACKAGE_VERSION,
                "python_version": _get_python_version(),
            }
        )


def get_metrics() -> RolloutMetrics:
    """Get the global metrics instance.

    Returns a configured RolloutMetrics instance. Creates one with
    default settings if not already configured.

    Returns:
        The global RolloutMetrics instance.

    Example:
        metrics = get_metrics()
        metrics.rollouts_started.labels(agent_loop="my_agent").inc()
    """
    global _metrics
    if _metrics is None:
        _metrics = RolloutMetrics()
    return _metrics


def configure_metrics(
    settings: Optional[MetricsSettings] = None,
    registry: Optional[Any] = None,
) -> RolloutMetrics:
    """Configure and return the global metrics instance.

    Creates a new RolloutMetrics instance with the given settings,
    replacing any existing instance.

    Args:
        settings: Metrics configuration. Defaults to global settings.
        registry: Custom Prometheus registry.

    Returns:
        The configured RolloutMetrics instance.

    Example:
        from osmosis_ai.rollout.config import MetricsSettings

        configure_metrics(MetricsSettings(
            enabled=True,
            prefix="my_agent",
        ))
    """
    global _metrics
    _metrics = RolloutMetrics(settings=settings, registry=registry)
    return _metrics


def reset_metrics() -> None:
    """Reset the global metrics instance.

    Primarily used for testing to ensure clean state between tests.
    """
    global _metrics
    _metrics = None


def get_metrics_content() -> Tuple[bytes, str]:
    """Get Prometheus metrics endpoint content.

    Returns the metrics in Prometheus exposition format, suitable
    for serving on a /metrics endpoint.

    Returns:
        Tuple of (content_bytes, content_type).

    Example:
        @app.get("/metrics")
        async def metrics_endpoint():
            content, content_type = get_metrics_content()
            return Response(content=content, media_type=content_type)
    """
    if not PROMETHEUS_AVAILABLE:
        return b"# Prometheus client not available\n", "text/plain"

    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return generate_latest(), CONTENT_TYPE_LATEST
