"""Strands integration for Osmosis rollouts.

Install ``osmosis-ai[strands]`` before importing this module.
"""

from osmosis_ai.rollout.integrations.agents.strands import (
    OsmosisRolloutModel,
    OsmosisStrandsAgent,
)

__all__ = [
    "OsmosisRolloutModel",
    "OsmosisStrandsAgent",
]
