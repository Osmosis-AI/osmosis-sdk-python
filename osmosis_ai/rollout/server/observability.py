"""Bounded, asynchronous ownership logs; never export rollout inputs or secrets."""

from __future__ import annotations

import logging
import math
import os
import socket
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from osmosis_ai.rollout.types import RolloutErrorCategory, RolloutStatus

if TYPE_CHECKING:
    from opentelemetry._logs import Logger
    from opentelemetry.sdk._logs import LoggerProvider

    from osmosis_ai.rollout.context import RolloutProgress

logger: logging.Logger = logging.getLogger(__name__)
METADATA_KEY = "osmosis_observability"


def _text(value: object) -> str | None:
    # Explicit text fields only, with bounded size and no control characters.
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 512
        or not value.isprintable()
    ):
        return None
    return value


def _seconds(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    try:
        return round(max(0.0, (end - start).total_seconds()), 3)
    except TypeError:  # A backend reported a naive timestamp.
        return None


def stage_durations(progress: RolloutProgress) -> dict[str, float]:
    """Seconds from admission to now, split at agent start and grading start."""
    finished = datetime.now(UTC)
    agent, grading = progress.agent_started_at, progress.grading_started_at
    durations = {
        "total_sec": _seconds(progress.created_at, finished),
        "startup_sec": _seconds(progress.created_at, agent),
        "agent_sec": _seconds(agent, grading or finished),
        "grading_sec": _seconds(grading, finished),
    }
    for name, value in progress.phase_timings.items():
        if (
            isinstance(name, str)
            and name.isidentifier()
            and len(name) <= 64
            and isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(value)
        ):
            durations[f"phase_{name}_sec"] = round(max(0.0, value), 3)
    return {key: value for key, value in durations.items() if value is not None}


class RolloutObservability:
    def __init__(self) -> None:
        self.provider: LoggerProvider | None = None
        self.log: Logger | None = None
        self._sequence: int = 0
        owner = {
            "server_id": os.environ.get("_OSMOSIS_ROLLOUT_SERVER_ID")
            or os.environ.get("OSMOSIS_ROLLOUT_SERVER_ID")
            or os.environ.get("_OSMOSIS_ROLLOUT_INSTANCE_ID")
            or socket.gethostname(),
            "server_name": os.environ.get("_OSMOSIS_ROLLOUT_NAME", ""),
            "namespace": os.environ.get("_OSMOSIS_ROLLOUT_NAMESPACE")
            or os.environ.get("OSMOSIS_ROLLOUT_NAMESPACE", ""),
        }
        self.owner: dict[str, str] = {
            key: value for key, value in owner.items() if _text(value) is not None
        }

    def start(self) -> None:
        endpoint = (
            os.environ.get("_OSMOSIS_ROLLOUT_OTLP_ENDPOINT")
            or os.environ.get("OSMOSIS_ROLLOUT_OTLP_ENDPOINT", "")
        ).rstrip("/")
        if not endpoint:
            return
        # Own a provider rather than replacing an application's global provider.
        from opentelemetry.exporter.otlp.proto.http._log_exporter import (
            OTLPLogExporter,
        )
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.resources import Resource

        self.provider = LoggerProvider(
            resource=Resource({"service.name": "osmosis-rollout-server"})
        )
        self.provider.add_log_record_processor(
            BatchLogRecordProcessor(
                OTLPLogExporter(endpoint=f"{endpoint}/v1/logs", timeout=2),
                max_queue_size=1024,
                max_export_batch_size=128,
                schedule_delay_millis=1000,
                export_timeout_millis=3000,
            )
        )
        self.log = self.provider.get_logger(__name__)

    def fields(
        self, rollout_id: str, metadata: Mapping[str, object] | None
    ) -> dict[str, str] | None:
        # Invalid required identifiers cannot be truncated or omitted without
        # merging unrelated owners in the dashboard's deduplication key.
        if _text(rollout_id) is None or "server_id" not in self.owner:
            return None
        fields = {"event": "rollout.ownership", "rollout_id": rollout_id, **self.owner}
        supplied = (metadata or {}).get(METADATA_KEY)
        if isinstance(supplied, dict):
            for key in ("run_id", "run_name"):
                if value := _text(supplied.get(key)):
                    fields[key] = value
        return fields

    def record(
        self,
        fields: Mapping[str, str] | None,
        status: RolloutStatus,
        *,
        progress: RolloutProgress | None = None,
        err_category: RolloutErrorCategory | None = None,
    ) -> None:
        """Emit one ownership event; pass ``progress`` only for the terminal one."""
        if self.log is None or fields is None:
            return
        try:
            self._sequence += 1
            attributes: dict[str, str | int | float] = {
                **fields,
                "status": status.value,
                "event_sequence": self._sequence,
            }
            if err_category is not None:
                attributes["err_category"] = err_category.value
            if progress is not None:
                attributes.update(stage_durations(progress))
            self.log.emit(
                timestamp=time.time_ns(),
                severity_text="INFO",
                body="Rollout ownership",
                attributes=attributes,
            )
        except Exception:
            # Observability failures must never affect admission, leases or results.
            logger.warning("Could not record rollout ownership", exc_info=False)

    def close(self) -> None:
        if self.provider is not None:
            self.provider.force_flush(timeout_millis=3000)
            self.provider.shutdown()
