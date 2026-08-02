"""
Optional Redis cache for deterministic analytics.

The dataset does not change while the process is up, so the same funnel /
clustering / conversion call always produces the same payload. Caching it
avoids repeating expensive SQL and sklearn work every time the agent (or a
dashboard) asks the same question.

Redis is optional by design. When REDIS_URL is unset, or the server is down,
every call falls through to the underlying function. The assistant must keep
answering when the cache is missing; the cache is an optimisation, not a
dependency.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from contextlib import suppress
from typing import Any, TypeVar

from src.config import settings
from src.observability import CACHE_REQUESTS
from src.serialization import to_jsonable

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Bump when a cached payload shape changes incompatibly.
_KEY_PREFIX = "rha:v1:"

_client: Any = None
_client_failed = False


def reset_client() -> None:
    """Drop the cached connection. Used by tests and after a failed ping."""
    global _client, _client_failed
    if _client is not None:
        with suppress(Exception):
            _client.close()
    _client = None
    _client_failed = False


def configure_client(client: Any) -> None:
    """Inject a client (real or fake). Intended for tests."""
    global _client, _client_failed
    _client = client
    _client_failed = False


def _get_client() -> Any | None:
    """
    Lazy Redis connection.

    Importing redis at module load would make an optional dependency feel
    required, and connecting at import would fail the whole process when Redis
    is briefly unreachable at startup. Both are worse than connecting on the
    first cache lookup.
    """
    global _client, _client_failed

    if _client is not None:
        return _client
    if _client_failed or not settings.redis_url:
        return None

    try:
        import redis  # imported lazily so the package is only needed when used

        candidate = redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
        )
        candidate.ping()
        _client = candidate
        return _client
    except Exception as error:  # noqa: BLE001
        _client_failed = True
        logger.warning("Redis unavailable; analytics cache disabled: %s", error)
        return None


def redis_status() -> bool | None:
    """
    Health probe.

    Returns True when Redis answers PING, False when configured but unreachable,
    and None when caching is not configured at all.
    """
    if not settings.redis_url:
        return None
    client = _get_client()
    if client is None:
        return False
    try:
        client.ping()
        return True
    except Exception:  # noqa: BLE001
        reset_client()
        return False


def cache_key(name: str, params: dict[str, Any]) -> str:
    """Stable key: function name plus a hash of the normalised parameters."""
    payload = json.dumps(params, sort_keys=True, default=str, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{_KEY_PREFIX}{name}:{digest}"


def remember(
    name: str,
    params: dict[str, Any],
    compute: Callable[[], T],
    *,
    ttl_seconds: int | None = None,
) -> T:
    """
    Return a cached JSON-friendly result, or compute and store one.

    The value is passed through `to_jsonable` before storage so a cache hit and a
    cache miss hand the caller the same Python types (no leftover Decimal /
    numpy scalars).
    """
    client = _get_client()
    if client is None:
        CACHE_REQUESTS.labels(outcome="bypass").inc()
        return to_jsonable(compute())  # type: ignore[return-value]

    key = cache_key(name, params)
    try:
        cached = client.get(key)
        if cached is not None:
            CACHE_REQUESTS.labels(outcome="hit").inc()
            return json.loads(cached)
    except Exception as error:  # noqa: BLE001
        CACHE_REQUESTS.labels(outcome="error").inc()
        logger.warning("Redis GET failed for %s: %s", name, error)
        return to_jsonable(compute())  # type: ignore[return-value]

    value = to_jsonable(compute())
    try:
        client.setex(
            key,
            ttl_seconds if ttl_seconds is not None else settings.redis_ttl_seconds,
            json.dumps(value, ensure_ascii=False, default=str),
        )
        CACHE_REQUESTS.labels(outcome="miss").inc()
    except Exception as error:  # noqa: BLE001
        CACHE_REQUESTS.labels(outcome="error").inc()
        logger.warning("Redis SET failed for %s: %s", name, error)

    return value  # type: ignore[return-value]
