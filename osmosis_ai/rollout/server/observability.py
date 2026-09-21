"""Bounded, asynchronous ownership logs; never export rollout inputs or secrets."""

from __future__ import annotations

import logging
import os
import socket
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING

from osmosis_ai.rollout.types import RolloutStatus

if TYPE_CHECKING:
    from opentelemetry._logs import Logger
    from opentelemetry.sdk._logs import LoggerProvider

logger = logging.getLogger(__name__)
METADATA_KEY = "osmosis_observability"


def _text(value: object) -> str | None:
    # Explicit text fields only, with bounded size and no control characters.
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        return None
    return "".join(c for c in value.strip() if c.isprintable()) or None


class RolloutObservability:
    def __init__(self) -> None:
        self.provider: LoggerProvider | None = None
        self.log: Logger | None = None
        self.owner = {
            "server_id": os.environ.get("OSMOSIS_ROLLOUT_SERVER_ID")
            or os.environ.get("_OSMOSIS_ROLLOUT_INSTANCE_ID")
            or socket.gethostname(),
            "server_name": os.environ.get("_OSMOSIS_ROLLOUT_NAME", ""),
            "namespace": os.environ.get("OSMOSIS_ROLLOUT_NAMESPACE", ""),
        }

    def start(self) -> None:
        endpoint = os.environ.get("OSMOSIS_ROLLOUT_OTLP_ENDPOINT", "").rstrip("/")
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
    ) -> dict[str, str]:
        fields = {"event": "rollout.ownership", "rollout_id": rollout_id, **self.owner}
        supplied = (metadata or {}).get(METADATA_KEY)
        if isinstance(supplied, dict):
            for key in ("run_id", "run_name"):
                if value := _text(supplied.get(key)):
                    fields[key] = value
        return fields

    def record(self, fields: Mapping[str, str], status: RolloutStatus) -> None:
        if self.log is None:
            return
        try:
            self.log.emit(
                timestamp=time.time_ns(),
                severity_text="INFO",
                body="Rollout ownership",
                attributes={**fields, "status": status.value},
            )
        except Exception:
            # Observability failures must never affect admission, leases or results.
            logger.warning("Could not record rollout ownership", exc_info=False)

    def close(self) -> None:
        if self.provider is not None:
            self.provider.force_flush(timeout_millis=3000)
            self.provider.shutdown()
