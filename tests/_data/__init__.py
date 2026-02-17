"""Test fixture data directory.

Usage:
    from tests._data import DATA_DIR

    path = DATA_DIR / "sample_rollout_request.json"
"""

from pathlib import Path

DATA_DIR = Path(__file__).parent
