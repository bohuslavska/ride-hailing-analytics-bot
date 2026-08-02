"""
Optional Redis cache for deterministic analytics and exact-match chat reuse.

Analytics: the dataset does not change while the process is up, so the same
funnel / clustering / conversion call always produces the same payload.

Chat: a previous answer is reused only when the question string matches
exactly (byte-for-byte after the API's strip). History is ignored for the
cache key: the same wording reuses the stored answer even mid-conversation.

Redis is optional by design. When REDIS_URL is unset, or the server is down,
every call falls through. The assistant must keep answering when the cache is
missing; the cache is an optimisation, not a dependency.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
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
_CHAT_PREFIX = "rha:chat:v2:"
_LOG_PREVIEW_CHARS = 280
# Upstash over the Fly private network is usually fast, but the first connect
# after a deploy can exceed a sub-second budget; keep retries cheap either way.
_SOCKET_TIMEOUT_SECONDS = 3.0
_RETRY_AFTER_SECONDS = 30.0

_client: Any = None
_client_failed_at: float | None = None


def preview_for_log(text: str, limit: int = _LOG_PREVIEW_CHARS) -> str:
    """Short single-line snippet for request/response logs."""
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "…"


def reset_client() -> None:
    """Drop the cached connection. Used by tests and after a failed ping."""
    global _client, _client_failed_at
    if _client is not None:
        with suppress(Exception):
            _client.close()
    _client = None
    _client_failed_at = None


def configure_client(client: Any) -> None:
    """Inject a client (real or fake). Intended for tests."""
    global _client, _client_failed_at
    _client = client
    _client_failed_at = None


def _get_client() -> Any | None:
    """
    Lazy Redis connection.

    Importing redis at module load would make an optional dependency feel
    required, and connecting at import would fail the whole process when Redis
    is briefly unreachable at startup. Both are worse than connecting on the
    first cache lookup.
    """
    global _client, _client_failed_at

    if _client is not None:
        return _client
    if not settings.redis_url:
        return None
    if _client_failed_at is not None:
        if time.monotonic() - _client_failed_at < _RETRY_AFTER_SECONDS:
            return None
        _client_failed_at = None

    try:
        import redis  # imported lazily so the package is only needed when used

        candidate = redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=_SOCKET_TIMEOUT_SECONDS,
            socket_timeout=_SOCKET_TIMEOUT_SECONDS,
        )
        candidate.ping()
        _client = candidate
        return _client
    except Exception as error:  # noqa: BLE001
        _client_failed_at = time.monotonic()
        logger.warning("Redis unavailable; cache temporarily disabled: %s", error)
        return None


def redis_status() -> bool | None:
    """
    Health probe.

    Returns True when Redis answers PING, False when configured but unreachable,
    and None when caching is not configured at all.
    """
    if not settings.redis_url:
        return None
    # Health should always attempt a fresh probe rather than honouring the
    # temporary failure cooldown used by request-path lookups.
    global _client_failed_at
    _client_failed_at = None
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
        logger.info("analytics cache bypass name=%s", name)
        return to_jsonable(compute())  # type: ignore[return-value]

    key = cache_key(name, params)
    try:
        cached = client.get(key)
        if cached is not None:
            CACHE_REQUESTS.labels(outcome="hit").inc()
            logger.info("analytics cache hit name=%s key=%s", name, key)
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
        logger.info("analytics cache miss name=%s key=%s stored=true", name, key)
    except Exception as error:  # noqa: BLE001
        CACHE_REQUESTS.labels(outcome="error").inc()
        logger.warning("Redis SET failed for %s: %s", name, error)

    return value  # type: ignore[return-value]


def chat_cache_key(question: str) -> str:
    """
    Exact-match key for a chat question.

    Hashed as sent (no lowercasing or fuzzy normalisation). History is not
    part of the key: identical wording reuses the same answer.
    """
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
    return f"{_CHAT_PREFIX}{digest}"


def get_cached_chat(question: str) -> dict[str, Any] | None:
    """Return a previously stored chat payload, or None on miss / bypass / error."""
    client = _get_client()
    if client is None:
        CACHE_REQUESTS.labels(outcome="bypass").inc()
        return None

    key = chat_cache_key(question)
    try:
        cached = client.get(key)
    except Exception as error:  # noqa: BLE001
        CACHE_REQUESTS.labels(outcome="error").inc()
        logger.warning("Redis chat GET failed: %s", error)
        return None

    if cached is None:
        CACHE_REQUESTS.labels(outcome="miss").inc()
        return None

    try:
        payload = json.loads(cached)
    except json.JSONDecodeError:
        CACHE_REQUESTS.labels(outcome="error").inc()
        logger.warning("Redis chat payload was not valid JSON for key=%s", key)
        return None

    if not isinstance(payload, dict) or not payload.get("answer"):
        CACHE_REQUESTS.labels(outcome="error").inc()
        return None

    CACHE_REQUESTS.labels(outcome="hit").inc()
    logger.info(
        "chat cache hit key=%s question=%r answer=%r",
        key,
        preview_for_log(question),
        preview_for_log(str(payload.get("answer", ""))),
    )
    return payload


def store_cached_chat(
    question: str,
    payload: dict[str, Any],
    *,
    ttl_seconds: int | None = None,
) -> None:
    """Store a successful chat answer for exact reuse later."""
    client = _get_client()
    if client is None:
        return

    answer = (payload.get("answer") or "").strip()
    if not answer:
        return

    key = chat_cache_key(question)
    body = to_jsonable(
        {
            "answer": answer,
            "artifacts": payload.get("artifacts") or [],
            "tool_calls": payload.get("tool_calls") or [],
        }
    )
    try:
        client.setex(
            key,
            ttl_seconds if ttl_seconds is not None else settings.redis_ttl_seconds,
            json.dumps(body, ensure_ascii=False, default=str),
        )
        logger.info(
            "chat cache store key=%s question=%r answer=%r artifacts=%d",
            key,
            preview_for_log(question),
            preview_for_log(answer),
            len(body.get("artifacts") or []),
        )
    except Exception as error:  # noqa: BLE001
        CACHE_REQUESTS.labels(outcome="error").inc()
        logger.warning("Redis chat SET failed: %s", error)


def log_chat_exchange(
    *,
    question: str,
    history: list[dict[str, str]] | None,
    answer: str,
    source: str,
    error: str | None = None,
) -> None:
    """Structured request/response log line for every completed chat turn."""
    logger.info(
        "chat exchange source=%s history_turns=%d question=%r answer=%r error=%r",
        source,
        len(history or []),
        preview_for_log(question),
        preview_for_log(answer) if answer else "",
        preview_for_log(error) if error else None,
    )
