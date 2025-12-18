"""Structured logging for Osmosis rollout SDK.

This module provides structured logging with automatic context injection
(rollout_id, trace_id, span_id). Uses structlog when available, falls back
to standard library logging otherwise.

Example:
    from osmosis_ai.rollout.observability.logging import (
        get_logger, set_rollout_id, clear_context
    )

    logger = get_logger(__name__)

    # Bind rollout context
    set_rollout_id("rollout-123")

    # Logs will automatically include rollout_id
    logger.info("Processing request", user_id="u456")
    # Output: {"event": "Processing request", "rollout_id": "rollout-123", "user_id": "u456"}

    # Clear context when done
    clear_context()
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

from osmosis_ai.rollout._compat import STRUCTLOG_AVAILABLE, structlog
from osmosis_ai.rollout.config.settings import LoggingSettings, get_settings

# Context variables for correlation IDs
_rollout_id_var: ContextVar[Optional[str]] = ContextVar("rollout_id", default=None)
_trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
_span_id_var: ContextVar[Optional[str]] = ContextVar("span_id", default=None)
_extra_context_var: ContextVar[Dict[str, Any]] = ContextVar(
    "extra_context", default={}
)

# Module state
_configured: bool = False


def get_rollout_id() -> Optional[str]:
    """Get the current context's rollout_id."""
    return _rollout_id_var.get()


def set_rollout_id(rollout_id: str) -> None:
    """Set the current context's rollout_id."""
    _rollout_id_var.set(rollout_id)


def get_trace_id() -> Optional[str]:
    """Get the current context's trace_id."""
    return _trace_id_var.get()


def set_trace_id(trace_id: str) -> None:
    """Set the current context's trace_id."""
    _trace_id_var.set(trace_id)


def get_span_id() -> Optional[str]:
    """Get the current context's span_id."""
    return _span_id_var.get()


def set_span_id(span_id: str) -> None:
    """Set the current context's span_id."""
    _span_id_var.set(span_id)


def set_context(**kwargs: Any) -> None:
    """Set additional context variables."""
    current = _extra_context_var.get()
    # Create a new dict to avoid mutating the default empty dict
    _extra_context_var.set({**current, **kwargs})


def clear_context() -> None:
    """Clear all context variables."""
    _rollout_id_var.set(None)
    _trace_id_var.set(None)
    _span_id_var.set(None)
    _extra_context_var.set({})


class RolloutLoggerAdapter(logging.LoggerAdapter):
    """Standard library logger adapter with automatic context injection.

    Used when structlog is not available. Supports structlog-style keyword
    arguments by converting them to the 'extra' dict.
    """

    def process(
        self, msg: str, kwargs: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        extra = kwargs.pop("extra", {})

        # Move any non-standard kwargs to extra (structlog-style)
        standard_kwargs = {"exc_info", "stack_info", "stacklevel"}
        extra_keys = [k for k in kwargs if k not in standard_kwargs]
        for key in extra_keys:
            extra[key] = kwargs.pop(key)

        # Inject context variables
        if rollout_id := _rollout_id_var.get():
            extra["rollout_id"] = rollout_id
        if trace_id := _trace_id_var.get():
            extra["trace_id"] = trace_id
        if span_id := _span_id_var.get():
            extra["span_id"] = span_id

        # Add extra context
        extra.update(_extra_context_var.get())

        kwargs["extra"] = extra
        return msg, kwargs


class JsonFormatter(logging.Formatter):
    """JSON formatter for standard library logging."""

    def __init__(self, include_timestamp: bool = True, include_caller: bool = False):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_caller = include_caller

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }

        if self.include_timestamp:
            # Use datetime for microsecond precision (logging.Formatter.formatTime doesn't support %f)
            log_data["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        if self.include_caller:
            log_data["caller"] = f"{record.filename}:{record.lineno}"

        # Add extra fields
        for key in ["rollout_id", "trace_id", "span_id"]:
            if hasattr(record, key) and getattr(record, key):
                log_data[key] = getattr(record, key)

        # Add any other extra fields
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in [
                    "name",
                    "msg",
                    "args",
                    "created",
                    "filename",
                    "funcName",
                    "levelname",
                    "levelno",
                    "lineno",
                    "module",
                    "msecs",
                    "pathname",
                    "process",
                    "processName",
                    "relativeCreated",
                    "stack_info",
                    "exc_info",
                    "exc_text",
                    "thread",
                    "threadName",
                    "message",
                    "rollout_id",
                    "trace_id",
                    "span_id",
                ]:
                    if not key.startswith("_"):
                        log_data[key] = value

        return json.dumps(log_data, default=str)


def _add_rollout_context(
    logger: Any, method_name: str, event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Structlog processor: add rollout context variables."""
    if rollout_id := _rollout_id_var.get():
        event_dict["rollout_id"] = rollout_id
    if trace_id := _trace_id_var.get():
        event_dict["trace_id"] = trace_id
    if span_id := _span_id_var.get():
        event_dict["span_id"] = span_id

    # Add extra context
    event_dict.update(_extra_context_var.get())

    return event_dict


def _add_service_info(
    logger: Any, method_name: str, event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Structlog processor: add service information."""
    try:
        from osmosis_ai.consts import PACKAGE_VERSION

        event_dict.setdefault("service", "osmosis-rollout")
        event_dict.setdefault("version", PACKAGE_VERSION)
    except ImportError:
        pass
    return event_dict


def configure_logging(settings: Optional[LoggingSettings] = None) -> None:
    """Configure the logging system.

    Configures structlog or standard library logging based on availability.

    Args:
        settings: Logging configuration. Defaults to global settings.

    Example:
        from osmosis_ai.rollout.observability.logging import configure_logging
        from osmosis_ai.rollout.config import LoggingSettings

        configure_logging(LoggingSettings(level="DEBUG", format="console"))
    """
    global _configured

    if settings is None:
        settings = get_settings().logging

    level = getattr(logging, settings.level)

    if STRUCTLOG_AVAILABLE:
        _configure_structlog(settings, level)
    else:
        _configure_stdlib_logging(settings, level)

    _configured = True


def _configure_structlog(settings: LoggingSettings, level: int) -> None:
    """Configure structlog."""
    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _add_rollout_context,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.include_timestamp:
        processors.insert(0, structlog.processors.TimeStamper(fmt="iso"))

    if settings.include_caller:
        processors.append(
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            )
        )

    # Select renderer based on format
    if settings.format == "json":
        processors.append(structlog.processors.JSONRenderer())
    elif settings.format == "console":
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:  # plain
        processors.append(structlog.processors.KeyValueRenderer())

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure root logger
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
        force=True,
    )


def _configure_stdlib_logging(settings: LoggingSettings, level: int) -> None:
    """Configure standard library logging."""
    if settings.format == "json":
        formatter = JsonFormatter(
            include_timestamp=settings.include_timestamp,
            include_caller=settings.include_caller,
        )
    else:
        format_parts = []
        if settings.include_timestamp:
            format_parts.append("%(asctime)s")
        format_parts.extend(["%(levelname)s", "%(name)s", "%(message)s"])
        if settings.include_caller:
            format_parts.append("(%(filename)s:%(lineno)d)")
        format_str = " - ".join(format_parts)
        formatter = logging.Formatter(format_str)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logging.root.handlers = [handler]
    logging.root.setLevel(level)


def get_logger(name: str) -> Any:
    """Get a logger instance.

    Returns a structlog logger if available, otherwise a standard library
    logger with automatic context injection.

    Args:
        name: Logger name, typically __name__.

    Returns:
        A configured logger instance.

    Example:
        logger = get_logger(__name__)
        logger.info("Processing", user_id="u123")
    """
    global _configured

    # Auto-configure on first use if not already configured
    if not _configured:
        configure_logging()

    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    else:
        logger = logging.getLogger(name)
        return RolloutLoggerAdapter(logger, {})


# Convenience module-level logger
def _get_module_logger() -> Any:
    """Get the module-level logger (lazy initialization)."""
    return get_logger("osmosis_ai.rollout")


class _LazyLogger:
    """Lazy logger that defers initialization until first use."""

    _instance: Optional[Any] = None

    def __getattr__(self, name: str) -> Any:
        if self._instance is None:
            self._instance = _get_module_logger()
        return getattr(self._instance, name)


logger = _LazyLogger()


__all__ = [
    # Context management
    "get_rollout_id",
    "set_rollout_id",
    "get_trace_id",
    "set_trace_id",
    "get_span_id",
    "set_span_id",
    "set_context",
    "clear_context",
    # Configuration
    "configure_logging",
    # Logger
    "get_logger",
    "logger",
]
