"""
Configuration handling, chiefly the database URL rewriting.

Fly and Heroku both hand out `postgres://`, which SQLAlchemy rejects, and a bare
`postgresql://` selects psycopg2, which is not installed. Both fail at the first
query rather than at startup, so they are worth a test.
"""

from __future__ import annotations

import pytest

from src.config import Settings, normalise_database_url

CREDENTIALS = "u:p@host:5432/db"


@pytest.mark.parametrize(
    "given",
    [f"postgres://{CREDENTIALS}", f"postgresql://{CREDENTIALS}"],
)
def test_provider_schemes_are_rewritten_onto_psycopg(given: str) -> None:
    assert normalise_database_url(given) == f"postgresql+psycopg://{CREDENTIALS}"


@pytest.mark.parametrize(
    "given",
    [f"postgresql+psycopg://{CREDENTIALS}", f"postgresql+psycopg2://{CREDENTIALS}"],
)
def test_an_explicit_driver_is_left_alone(given: str) -> None:
    assert normalise_database_url(given) == given


def test_none_is_preserved() -> None:
    """readonly_database_url is optional and must stay unset when absent."""
    assert normalise_database_url(None) is None


def test_credentials_and_query_parameters_survive_rewriting() -> None:
    given = "postgres://user:pa%40ss@db.internal:5432/rides?sslmode=require"
    rewritten = normalise_database_url(given)

    assert rewritten.startswith("postgresql+psycopg://")
    assert rewritten.endswith("user:pa%40ss@db.internal:5432/rides?sslmode=require")


def test_settings_apply_the_rewrite_to_both_urls() -> None:
    settings = Settings(
        database_url=f"postgres://{CREDENTIALS}",
        readonly_database_url="postgresql://readonly:p@host:5432/db",
    )

    assert settings.database_url.startswith("postgresql+psycopg://")
    assert settings.readonly_database_url.startswith("postgresql+psycopg://")


def test_query_url_prefers_the_readonly_role() -> None:
    settings = Settings(
        database_url="postgres://owner:p@host:5432/db",
        readonly_database_url="postgres://readonly:p@host:5432/db",
    )

    assert "readonly" in settings.query_database_url


def test_query_url_falls_back_to_the_main_url() -> None:
    settings = Settings(
        database_url="postgres://owner:p@host:5432/db",
        readonly_database_url=None,
    )

    assert settings.query_database_url == settings.database_url


def test_llm_is_disabled_without_a_key() -> None:
    """The analytics endpoints must keep working when only the chat is unusable."""
    assert Settings(openrouter_api_key=None).llm_enabled is False
    assert Settings(openrouter_api_key="").llm_enabled is False
    assert Settings(openrouter_api_key="sk-or-v1-x").llm_enabled is True
