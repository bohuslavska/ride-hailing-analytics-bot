"""
Shared fixtures.

Tests split into two groups: those that need only Python, and those that need a
loaded database. The second group is skipped rather than failed when Postgres is
absent, so that `pytest` is still useful on a fresh checkout before `make reset`
has been run.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from src.database.connection import get_engine


def _database_available() -> bool:
    try:
        with get_engine(readonly=True).connect() as connection:
            count = connection.execute(text("SELECT COUNT(*) FROM rides")).scalar_one()
        return bool(count)
    except Exception:
        return False


DATABASE_READY = _database_available()

needs_database = pytest.mark.skipif(
    not DATABASE_READY,
    reason="Postgres is not reachable or has no data; run `make db-up && make reset`.",
)
