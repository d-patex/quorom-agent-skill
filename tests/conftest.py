"""Shared pytest fixtures for the Quorum agent test suite.

Two jobs:
1. Make the repo-root modules (tools, agent, schemas, seed_data) importable
   from tests/, since pytest's default import mode only adds the directory
   containing the test file to sys.path, not the project root.
2. Reset trip.db to the known sample state before every test, so tests never
   depend on what a previous test (or a manual `python agent.py` session)
   left behind.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import seed_data


@pytest.fixture(autouse=True)
def fresh_trip_db():
    """Rebuild trip.db from seed_data before every test function."""
    seed_data.create_database()
    yield