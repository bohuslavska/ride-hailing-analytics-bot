"""
The guardrails on model-generated SQL.

These are the tests worth having: everything else in the project produces a wrong
number when it breaks, whereas this produces a modified database. The read-only
role is the actual security boundary, so the point of these tests is that the
cheap layer in front of it gives the model a correctable error instead of a
confusing one, and that it cannot be talked out of rejecting a write.
"""

from __future__ import annotations

import pytest

from src.database.safe_sql import (
    UnsafeQueryError,
    apply_row_limit,
    strip_sql_literals,
    validate_select_statement,
)

WRITES = [
    "DROP TABLE rides",
    "DELETE FROM rides WHERE 1=1",
    "UPDATE rides SET accepted = true",
    "INSERT INTO rides (ride_id) VALUES ('x')",
    "TRUNCATE rides",
    "ALTER TABLE rides ADD COLUMN x int",
    "GRANT ALL ON rides TO PUBLIC",
    "CREATE TABLE evil (id int)",
    "REFRESH MATERIALIZED VIEW rides_enriched",
    "VACUUM FULL",
    "DO $$ BEGIN PERFORM 1; END $$",
    "CALL some_procedure()",
]


@pytest.mark.parametrize("statement", WRITES)
def test_write_statements_are_rejected(statement: str) -> None:
    with pytest.raises(UnsafeQueryError):
        validate_select_statement(statement)


SMUGGLING = [
    # A second statement hidden behind a semicolon.
    "SELECT 1; DROP TABLE rides",
    # ... and behind a trailing comment, which does not neutralise it.
    "SELECT 1; DELETE FROM rides -- oops",
    # A write dressed up as a subquery.
    "SELECT * FROM (DELETE FROM rides RETURNING *) t",
    # Leading parenthesis, which the shape check has to look past.
    "(DELETE FROM rides)",
    # CTE that performs a write, which PostgreSQL genuinely allows.
    "WITH gone AS (DELETE FROM rides RETURNING 1) SELECT * FROM gone",
]


@pytest.mark.parametrize("statement", SMUGGLING)
def test_statements_cannot_smuggle_a_write(statement: str) -> None:
    with pytest.raises(UnsafeQueryError):
        validate_select_statement(statement)


READS = [
    "SELECT 1",
    "select count(*) from rides",
    "  SELECT * FROM rides LIMIT 10  ",
    "SELECT * FROM rides;",
    "WITH per_hour AS (SELECT hour FROM rides) SELECT * FROM per_hour",
    "SELECT AVG(surge_multiplier)::numeric FROM zone_state",
    "(SELECT 1) UNION ALL (SELECT 2)",
]


@pytest.mark.parametrize("statement", READS)
def test_reads_are_accepted(statement: str) -> None:
    assert validate_select_statement(statement)


def test_empty_query_is_rejected() -> None:
    with pytest.raises(UnsafeQueryError, match="empty"):
        validate_select_statement("   ")


def test_keyword_inside_a_string_literal_does_not_trip_the_filter() -> None:
    """A zone legitimately named 'Update Square' must not read as a write."""
    statement = "SELECT * FROM zones WHERE zone_name = 'Update Square'"
    assert validate_select_statement(statement) == statement


def test_keyword_inside_a_comment_does_not_trip_the_filter() -> None:
    statement = "SELECT 1 -- we never DROP anything here"
    assert validate_select_statement(statement)


def test_keyword_inside_a_quoted_identifier_does_not_trip_the_filter() -> None:
    statement = 'SELECT "delete" FROM zones'
    assert validate_select_statement(statement)


def test_strip_literals_removes_comments_and_strings() -> None:
    stripped = strip_sql_literals(
        "SELECT 'drop me' /* drop */ FROM t -- drop\nWHERE x = 1"
    )
    assert "drop" not in stripped.lower()
    assert "select" in stripped.lower()
    assert "where" in stripped.lower()


def test_escaped_quotes_do_not_break_literal_stripping() -> None:
    statement = "SELECT * FROM zones WHERE zone_name = 'O''Brien delete'"
    assert validate_select_statement(statement) == statement


class TestRowLimit:
    def test_unlimited_query_is_wrapped(self) -> None:
        wrapped, was_wrapped = apply_row_limit("SELECT * FROM rides", 100)
        assert was_wrapped
        assert "LIMIT 101" in wrapped

    def test_a_smaller_existing_limit_is_left_alone(self) -> None:
        statement = "SELECT * FROM rides LIMIT 10"
        result, was_wrapped = apply_row_limit(statement, 100)
        assert not was_wrapped
        assert result == statement

    def test_a_larger_existing_limit_is_overridden(self) -> None:
        wrapped, was_wrapped = apply_row_limit("SELECT * FROM rides LIMIT 9999", 100)
        assert was_wrapped
        assert "LIMIT 101" in wrapped

    def test_set_operations_survive_wrapping(self) -> None:
        """Appending a LIMIT to a UNION would bind to the wrong branch."""
        wrapped, _ = apply_row_limit("SELECT 1 UNION ALL SELECT 2", 100)
        assert wrapped.strip().startswith("SELECT * FROM (")
