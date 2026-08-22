"""Tests for cloud-style local evaluation run names."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from osmosis_ai.eval.local.run_name import generate_run_name


@patch("osmosis_ai.eval.local.run_name.secrets.randbelow", return_value=42)
@patch(
    "osmosis_ai.eval.local.run_name.secrets.choice",
    side_effect=["brave", "falcon"],
)
def test_generated_name_matches_cloud_eval(_choice: Any, _randbelow: Any) -> None:
    assert generate_run_name() == "brave-falcon-42"
