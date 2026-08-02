"""
Load the generated Parquet tables into PostgreSQL.

    python -m scripts.load_to_postgres

Rows go in via COPY rather than INSERT, which matters at three quarters of a
million rides: COPY finishes in seconds where row-by-row inserts would take
minutes. Indexes and foreign keys are created by sql/schema.sql before the
load, so the constraints are checked against the data as it arrives.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

import pandas as pd
import psycopg
from psycopg import sql

# Run directly as `python scripts/load_to_postgres.py` without needing PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, settings
from src.data_generation.generate_users import PUBLIC_USER_COLUMNS
from src.database.connection import to_psycopg_dsn

# Load order respects the foreign keys.
TABLE_LOAD_ORDER = ["zones", "users", "zone_state", "rides"]


def _prepare_table(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    if name == "users":
        # Latent behavioural traits stay out of the analytical database.
        return frame[PUBLIC_USER_COLUMNS]

    if name == "zone_state":
        # `timestamp` is a type name in SQL; `ts` avoids forcing every generated
        # query to quote it.
        return frame.rename(columns={"timestamp": "ts"})

    return frame


def copy_frame(connection: psycopg.Connection, table: str, frame: pd.DataFrame) -> None:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, header=False, na_rep="")
    buffer.seek(0)

    column_list = ", ".join(f'"{column}"' for column in frame.columns)
    statement = (
        f"COPY {table} ({column_list}) FROM STDIN WITH (FORMAT csv, NULL '')"
    )

    with connection.cursor().copy(statement) as copy:
        while chunk := buffer.read(1024 * 1024):
            copy.write(chunk)


def ensure_readonly_role(
    connection: psycopg.Connection, role_name: str, password: str
) -> None:
    """
    Create or refresh the least-privilege role used for agent SQL.

    The guardrail that actually matters is this one. Statement parsing can be
    fooled; a role with no INSERT, UPDATE, DELETE or DDL grant cannot modify
    anything regardless of what the model writes.
    """
    role_exists = connection.execute(
        "SELECT 1 FROM pg_roles WHERE rolname = %s", (role_name,)
    ).fetchone()

    # CREATE ROLE is a utility statement, so PostgreSQL will not accept bind
    # parameters for it. The password has to be composed into the statement as
    # a properly quoted literal instead.
    role = sql.Identifier(role_name)
    secret = sql.Literal(password)
    action = sql.SQL("ALTER ROLE") if role_exists else sql.SQL("CREATE ROLE")

    connection.execute(
        sql.SQL("{action} {role} WITH LOGIN PASSWORD {secret}").format(
            action=action, role=role, secret=secret
        )
    )

    database_name = connection.execute("SELECT current_database()").fetchone()[0]

    for statement in (
        sql.SQL("GRANT CONNECT ON DATABASE {database} TO {role}").format(
            database=sql.Identifier(database_name), role=role
        ),
        sql.SQL("GRANT USAGE ON SCHEMA public TO {role}").format(role=role),
        sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {role}").format(
            role=role
        ),
        # Covers the reporting view, which is created after this runs.
        sql.SQL(
            "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO {role}"
        ).format(role=role),
        # Withhold everything else, including the ability to create objects.
        sql.SQL("REVOKE CREATE ON SCHEMA public FROM {role}").format(role=role),
    ):
        connection.execute(statement)


def load(
    data_dir: Path, database_url: str, schema_path: Path, indexes_path: Path
) -> None:
    missing = [
        name
        for name in TABLE_LOAD_ORDER
        if not (data_dir / f"{name}.parquet").exists()
    ]
    if missing:
        raise SystemExit(
            f"missing Parquet files for {', '.join(missing)} in {data_dir}. "
            "Run `python -m src.data_generation.build_all` first."
        )

    started_at = time.perf_counter()

    with psycopg.connect(to_psycopg_dsn(database_url), autocommit=False) as connection:
        print("applying schema...")
        connection.execute(schema_path.read_text())

        for name in TABLE_LOAD_ORDER:
            frame = _prepare_table(name, pd.read_parquet(data_dir / f"{name}.parquet"))

            table_started_at = time.perf_counter()
            copy_frame(connection, name, frame)
            elapsed = time.perf_counter() - table_started_at

            print(f"  {name:<12} {len(frame):>10,} rows in {elapsed:5.1f}s")

        print("building indexes and the reporting view...")
        index_started_at = time.perf_counter()
        connection.execute(indexes_path.read_text())
        print(f"  done in {time.perf_counter() - index_started_at:.1f}s")

        readonly_role = os.getenv("READONLY_DB_USER", "ride_hailing_readonly")
        readonly_password = os.getenv("READONLY_DB_PASSWORD")

        if readonly_password:
            ensure_readonly_role(connection, readonly_role, readonly_password)
            print(f"  read-only role '{readonly_role}' granted SELECT only")
        else:
            print(
                "  READONLY_DB_PASSWORD not set - skipping read-only role. "
                "Agent SQL will fall back to a read-only transaction."
            )

        connection.commit()

        print("analyzing...")
        connection.execute("ANALYZE")
        connection.commit()

    print(f"\nloaded in {time.perf_counter() - started_at:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--database-url", default=settings.database_url)
    parser.add_argument(
        "--schema", type=Path, default=PROJECT_ROOT / "sql" / "schema.sql"
    )
    parser.add_argument(
        "--indexes", type=Path, default=PROJECT_ROOT / "sql" / "indexes.sql"
    )
    arguments = parser.parse_args()

    load(
        arguments.data_dir,
        arguments.database_url,
        arguments.schema,
        arguments.indexes,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
