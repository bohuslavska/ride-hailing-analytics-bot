"""Database connection helpers shared by the loader, the API and the tools."""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine, text

from src.config import settings


def to_psycopg_dsn(database_url: str) -> str:
    """
    Strip the SQLAlchemy driver suffix so the URL can be handed to psycopg.

    `postgresql+psycopg://...` is what SQLAlchemy wants; psycopg itself only
    understands `postgresql://...`.
    """
    return database_url.replace("postgresql+psycopg://", "postgresql://").replace(
        "postgres+psycopg://", "postgresql://"
    )


def to_sqlalchemy_url(database_url: str) -> str:
    """Force the psycopg3 driver, whichever style the environment supplied."""
    if "+psycopg" in database_url:
        return database_url

    return database_url.replace("postgresql://", "postgresql+psycopg://").replace(
        "postgres://", "postgresql+psycopg://"
    )


@lru_cache(maxsize=2)
def get_engine(readonly: bool = False) -> Engine:
    """
    Engine for the requested privilege level.

    Read-only queries get their own engine so they can carry a statement
    timeout and, when configured, connect as a role that has no write grants at
    all. Cached because engines own a connection pool.
    """
    url = settings.query_database_url if readonly else settings.database_url

    connect_arguments: dict[str, str] = {}
    if readonly:
        connect_arguments["options"] = (
            f"-c statement_timeout={settings.sql_statement_timeout_ms} "
            f"-c default_transaction_read_only=on"
        )

    return create_engine(
        to_sqlalchemy_url(url),
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        connect_args=connect_arguments,
    )


def database_is_ready() -> tuple[bool, str]:
    """Cheap health probe: can we connect and is the rides table populated?"""
    try:
        with get_engine().connect() as connection:
            # Exact COUNT(*) over ~700k rows can exceed the Fly health-check
            # timeout on a small managed Postgres; planner stats are enough here.
            ride_count = connection.execute(
                text(
                    """
                    SELECT COALESCE(
                        (SELECT reltuples::bigint
                         FROM pg_class
                         WHERE oid = 'public.rides'::regclass),
                        0
                    )
                    """
                )
            ).scalar_one()
            if ride_count <= 0:
                has_row = connection.execute(
                    text("SELECT EXISTS (SELECT 1 FROM rides LIMIT 1)")
                ).scalar_one()
                if not has_row:
                    return False, "rides table is empty"
                return True, "rides table has rows"

        return True, f"~{ride_count:,} rides"

    except Exception as error:  # noqa: BLE001 - surfaced verbatim in /health
        return False, str(error)
