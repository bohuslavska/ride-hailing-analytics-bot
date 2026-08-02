from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def normalise_database_url(url: str | None) -> str | None:
    """
    Force a connection string onto the psycopg 3 driver.

    Managed Postgres providers hand out URLs in their own dialect: Fly and Heroku
    both use the bare `postgres://` scheme, which SQLAlchemy rejects outright,
    and a plain `postgresql://` silently selects psycopg2, which is not
    installed. Both cases fail at the first connection rather than at startup, so
    they are cheaper to rewrite here than to debug in a deployed container.
    """
    if not url:
        return url

    for prefix in ("postgresql+psycopg://", "postgresql+psycopg2://"):
        if url.startswith(prefix):
            return url

    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]

    return url


class Settings(BaseSettings):
    """
    Runtime configuration for the API and the bot.

    Values are resolved from environment variables, then a local .env file,
    then the defaults below. Simulation parameters are deliberately not here:
    they live in configs/*.yaml so that the dataset stays reproducible
    independently of the deployment environment.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # PostgreSQL. Port 5434 matches docker-compose.yml, which avoids the 5432
    # and 5433 that other local projects tend to occupy.
    database_url: str = (
        "postgresql+psycopg://ride_hailing:ride_hailing@localhost:5434/ride_hailing"
    )
    # Optional least-privilege role used for every agent-generated query. When
    # unset, the agent falls back to database_url inside a read-only transaction.
    readonly_database_url: str | None = None

    # Guardrails applied to agent-generated SQL.
    sql_statement_timeout_ms: int = 15_000
    sql_max_returned_rows: int = 500

    # LLM (OpenRouter is OpenAI-compatible).
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "anthropic/claude-sonnet-5"
    llm_temperature: float = 0.0
    agent_max_iterations: int = 8

    # Optional Redis: analytics results + exact-match chat reuse. When unset,
    # every call goes straight to Postgres / the LLM; Redis is not required.
    redis_url: str | None = None
    redis_ttl_seconds: int = 3_600

    @field_validator("database_url", "readonly_database_url")
    @classmethod
    def _coerce_driver(cls, value: str | None) -> str | None:
        return normalise_database_url(value)

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def configs_dir(self) -> Path:
        return PROJECT_ROOT / "configs"

    @property
    def query_database_url(self) -> str:
        """Connection string used for agent-generated SQL."""
        return self.readonly_database_url or self.database_url

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openrouter_api_key)


settings = Settings()
