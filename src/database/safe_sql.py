"""
Guarded execution of model-generated SQL.

Four layers, ordered from weakest to strongest:

1.  Shape checks -- the statement must be a single SELECT or WITH. Cheap, and
    gives the model a readable error it can correct itself from.
2.  A read-only transaction, so even a statement that slipped past the parser
    cannot commit a change.
3.  A statement timeout, so a careless cross join cannot occupy the database.
4.  A database role holding nothing but SELECT.

Only the fourth layer is load-bearing. The first exists to produce good error
messages, not to be the security boundary -- string inspection of SQL is
guessable, and treating it as the defence would be a mistake.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text

from src.config import settings
from src.database.connection import get_engine
from src.observability import SQL_DURATION, SQL_QUERIES

FORBIDDEN_KEYWORDS = {
    "insert",
    "update",
    "delete",
    "drop",
    "truncate",
    "alter",
    "create",
    "grant",
    "revoke",
    "copy",
    "vacuum",
    "reindex",
    "call",
    "do",
    "merge",
    "refresh",
}

ALLOWED_LEADING_KEYWORDS = ("select", "with")


class UnsafeQueryError(ValueError):
    """Raised when a generated statement is rejected before it reaches the database."""


@dataclass
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    elapsed_ms: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sql": self.sql,
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "elapsed_ms": self.elapsed_ms,
            "notes": self.notes,
        }


def strip_sql_literals(statement: str) -> str:
    """
    Blank out string literals, quoted identifiers and comments.

    Keyword checks run against this stripped form so that a zone called
    'Update Square' or a comment mentioning DROP cannot trip the filter, and so
    that a keyword hidden inside a literal cannot smuggle past it either.
    """
    without_block_comments = re.sub(r"/\*.*?\*/", " ", statement, flags=re.DOTALL)
    without_line_comments = re.sub(r"--[^\n]*", " ", without_block_comments)
    without_single_quotes = re.sub(r"'(?:[^']|'')*'", "''", without_line_comments)
    without_double_quotes = re.sub(
        r'"(?:[^"]|"")*"', '""', without_single_quotes
    )

    return without_double_quotes


def validate_select_statement(statement: str) -> str:
    """Return the normalised statement, or raise UnsafeQueryError."""
    if not statement or not statement.strip():
        raise UnsafeQueryError("The query is empty.")

    normalised = statement.strip().rstrip(";").strip()
    inspectable = strip_sql_literals(normalised)

    if ";" in inspectable:
        raise UnsafeQueryError(
            "Only one statement per query is allowed. Remove the ';' and send a "
            "single SELECT."
        )

    lowered = inspectable.lower().lstrip("( \n\t")
    if not lowered.startswith(ALLOWED_LEADING_KEYWORDS):
        raise UnsafeQueryError(
            "Only SELECT queries are allowed. Start the statement with SELECT or WITH."
        )

    words = set(re.findall(r"\b[a-z_]+\b", inspectable.lower()))
    forbidden = sorted(words & FORBIDDEN_KEYWORDS)
    if forbidden:
        raise UnsafeQueryError(
            f"This query uses a write or DDL keyword ({', '.join(forbidden)}). "
            "The dataset is read-only, so use SELECT only."
        )

    return normalised


def apply_row_limit(statement: str, max_rows: int) -> tuple[str, bool]:
    """
    Wrap the statement in an outer LIMIT unless it already has a smaller one.

    Wrapping rather than appending keeps the original query intact when it ends
    in something that LIMIT cannot simply follow, such as a set operation.
    """
    inspectable = strip_sql_literals(statement).lower()
    existing = re.search(r"\blimit\s+(\d+)\s*$", inspectable)

    if existing and int(existing.group(1)) <= max_rows:
        return statement, False

    return f"SELECT * FROM (\n{statement}\n) AS limited_query LIMIT {max_rows + 1}", True


def run_readonly_query(
    statement: str, max_rows: int | None = None
) -> QueryResult:
    """Validate, execute and truncate a model-generated SELECT."""
    max_rows = max_rows or settings.sql_max_returned_rows

    try:
        validated = validate_select_statement(statement)
    except UnsafeQueryError:
        SQL_QUERIES.labels(outcome="rejected").inc()
        raise

    executable, was_wrapped = apply_row_limit(validated, max_rows)

    engine = get_engine(readonly=True)
    started_at = time.perf_counter()

    try:
        with engine.connect() as connection:
            # Belt and braces: the connection already sets this, but an explicit
            # read-only transaction makes the intent visible at the call site.
            connection.execute(text("SET TRANSACTION READ ONLY"))
            # exec_driver_sql bypasses SQLAlchemy's ':name' bind-parameter parsing,
            # which would otherwise choke on PostgreSQL '::type' casts.
            cursor = connection.exec_driver_sql(executable)
            columns = list(cursor.keys())
            fetched = cursor.fetchall()
    except Exception:
        # Valid SQL that the database still refused: a bad column name, a type
        # mismatch, or the statement timeout firing. Counted separately from a
        # guardrail rejection because the two call for different responses.
        SQL_QUERIES.labels(outcome="failed").inc()
        raise

    elapsed_seconds = time.perf_counter() - started_at
    elapsed_ms = int(elapsed_seconds * 1000)

    SQL_QUERIES.labels(outcome="executed").inc()
    SQL_DURATION.observe(elapsed_seconds)

    truncated = was_wrapped and len(fetched) > max_rows
    rows = [list(row) for row in fetched[:max_rows]]

    notes: list[str] = []
    if truncated:
        notes.append(
            f"Result truncated to the first {max_rows} rows. Aggregate in SQL if "
            "you need a summary of everything."
        )

    return QueryResult(
        sql=validated,
        columns=columns,
        rows=rows,
        row_count=len(rows),
        truncated=truncated,
        elapsed_ms=elapsed_ms,
        notes=notes,
    )
