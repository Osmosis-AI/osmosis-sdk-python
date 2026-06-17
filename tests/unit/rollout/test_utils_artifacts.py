"""Tests for osmosis_ai.rollout.utils.artifacts."""

import json

from osmosis_ai.rollout.utils.artifacts import (
    MAX_ARTIFACTS_BYTES,
    sanitize_artifacts,
)


class TestSanitizeArtifacts:
    def test_none_stays_none(self):
        assert sanitize_artifacts(None) is None

    def test_valid_object_passes_through_unchanged(self):
        artifacts = {
            "judge": {"explanation": "ok", "rubric_scores": {"correctness": 0.8}}
        }
        assert sanitize_artifacts(artifacts) is artifacts

    def test_empty_object_passes_through(self):
        artifacts: dict = {}
        assert sanitize_artifacts(artifacts) is artifacts

    def test_non_dict_downgraded_to_error(self):
        result = sanitize_artifacts(["not", "an", "object"])
        assert result == {"_error": {"code": "artifacts_invalid_type", "type": "list"}}

    def test_non_serializable_downgraded_to_error(self):
        result = sanitize_artifacts({"bad": {1, 2, 3}})
        assert result is not None
        assert result["_error"]["code"] == "artifacts_not_serializable"
        assert "detail" in result["_error"]

    def test_nan_downgraded_to_error(self):
        # NaN passes stdlib json.dumps defaults but breaks httpx's allow_nan=False
        # wire encoding, so it must be caught here, not at the callback POST.
        result = sanitize_artifacts({"score": float("nan")})
        assert result is not None
        assert result["_error"]["code"] == "artifacts_not_serializable"

    def test_infinity_downgraded_to_error(self):
        result = sanitize_artifacts({"score": float("inf")})
        assert result is not None
        assert result["_error"]["code"] == "artifacts_not_serializable"

    def test_sanitized_output_is_wire_serializable(self):
        # Whatever sanitize returns must survive httpx's strict encoding.
        for payload in ({"score": float("nan")}, {"bad": {1, 2, 3}}, ["not", "dict"]):
            result = sanitize_artifacts(payload)
            json.dumps(result, allow_nan=False, separators=(",", ":"))

    def test_oversized_downgraded_to_error(self):
        artifacts = {"blob": "x" * (MAX_ARTIFACTS_BYTES + 1)}
        result = sanitize_artifacts(artifacts)
        assert result is not None
        err = result["_error"]
        assert err["code"] == "artifacts_too_large"
        assert err["max_size_bytes"] == MAX_ARTIFACTS_BYTES
        assert err["size_bytes"] > MAX_ARTIFACTS_BYTES

    def test_at_cap_passes_through(self):
        # Build an object whose serialization is exactly at the cap.
        base = {"blob": ""}
        overhead = len(json.dumps(base).encode("utf-8"))
        artifacts = {"blob": "x" * (MAX_ARTIFACTS_BYTES - overhead)}
        assert len(json.dumps(artifacts).encode("utf-8")) == MAX_ARTIFACTS_BYTES
        assert sanitize_artifacts(artifacts) is artifacts
