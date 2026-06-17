"""Sanitize rollout artifacts at the grader-callback wire boundary."""

from __future__ import annotations

import json
import logging
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)

# Hard size cap on the UTF-8 JSON serialization of an artifacts object.
MAX_ARTIFACTS_BYTES: int = 64 * 1024


def sanitize_artifacts(artifacts: Any) -> dict[str, Any] | None:
    """Validate and size-bound artifacts; never block reward delivery.

    Returns None when absent. A non-dict, non-serializable, or oversized
    payload is replaced with a small ``_error`` object instead of raising.
    """
    if artifacts is None:
        return None

    if not isinstance(artifacts, dict):
        logger.warning(
            "Dropping non-object artifacts of type %s", type(artifacts).__name__
        )
        return {
            "_error": {
                "code": "artifacts_invalid_type",
                "type": type(artifacts).__name__,
            }
        }

    try:
        encoded = json.dumps(artifacts).encode("utf-8")
    except (TypeError, ValueError) as exc:
        logger.warning("Dropping non-serializable artifacts: %s", exc)
        return {
            "_error": {
                "code": "artifacts_not_serializable",
                "detail": str(exc),
            }
        }

    size_bytes = len(encoded)
    if size_bytes > MAX_ARTIFACTS_BYTES:
        logger.warning(
            "Dropping oversized artifacts: %d bytes exceeds %d byte cap",
            size_bytes,
            MAX_ARTIFACTS_BYTES,
        )
        return {
            "_error": {
                "code": "artifacts_too_large",
                "size_bytes": size_bytes,
                "max_size_bytes": MAX_ARTIFACTS_BYTES,
            }
        }

    return artifacts
