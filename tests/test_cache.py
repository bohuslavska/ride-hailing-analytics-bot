"""
The analytics cache.

These tests use an in-process fake rather than a live Redis, so they pin the
contract (hit / miss / bypass / error) without requiring another service in CI.
"""

from __future__ import annotations

from typing import Any

import pytest

from src import cache as cache_module
from src.cache import (
    cache_key,
    chat_cache_key,
    configure_client,
    get_cached_chat,
    redis_status,
    remember,
    reset_client,
    store_cached_chat,
)
from src.observability import CACHE_REQUESTS


class FakeRedis:
    def __init__(self, *, fail_get: bool = False, fail_set: bool = False) -> None:
        self.store: dict[str, str] = {}
        self.fail_get = fail_get
        self.fail_set = fail_set
        self.closed = False

    def ping(self) -> bool:
        return True

    def get(self, key: str) -> str | None:
        if self.fail_get:
            raise ConnectionError("get failed")
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: str) -> bool:
        if self.fail_set:
            raise ConnectionError("set failed")
        assert ttl > 0
        self.store[key] = value
        return True

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _isolate_cache(monkeypatch: pytest.MonkeyPatch) -> Any:
    reset_client()
    # Force a configured URL so redis_status / remember do not short-circuit on
    # the unset default; the injected FakeRedis is what actually answers.
    monkeypatch.setattr(cache_module.settings, "redis_url", "redis://localhost:6379/0")
    monkeypatch.setattr(cache_module.settings, "redis_ttl_seconds", 60)
    yield
    reset_client()


def _delta(outcome: str) -> float:
    return CACHE_REQUESTS.labels(outcome=outcome)._value.get()


def test_cache_key_is_stable_under_key_reordering() -> None:
    assert cache_key("funnel", {"b": 1, "a": 2}) == cache_key(
        "funnel", {"a": 2, "b": 1}
    )


def test_remember_computes_once_then_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeRedis()
    configure_client(fake)
    calls = {"n": 0}

    def compute() -> dict[str, int]:
        calls["n"] += 1
        return {"value": calls["n"]}

    hits_before = _delta("hit")
    misses_before = _delta("miss")

    first = remember("demo", {"x": 1}, compute)
    second = remember("demo", {"x": 1}, compute)

    assert first == {"value": 1}
    assert second == {"value": 1}
    assert calls["n"] == 1
    assert _delta("miss") == misses_before + 1
    assert _delta("hit") == hits_before + 1


def test_remember_bypasses_when_redis_is_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cache_module.settings, "redis_url", None)
    reset_client()
    bypass_before = _delta("bypass")

    assert remember("demo", {}, lambda: {"ok": True}) == {"ok": True}
    assert _delta("bypass") == bypass_before + 1


def test_remember_falls_through_on_redis_errors() -> None:
    configure_client(FakeRedis(fail_get=True))
    error_before = _delta("error")

    assert remember("demo", {"y": 2}, lambda: {"ok": True}) == {"ok": True}
    assert _delta("error") == error_before + 1


def test_chat_cache_requires_exact_question_only() -> None:
    fake = FakeRedis()
    configure_client(fake)

    store_cached_chat(
        "What is surge?",
        {"answer": "Surge tracks shortage.", "artifacts": [], "tool_calls": []},
    )

    assert get_cached_chat("What is surge?")["answer"] == "Surge tracks shortage."
    # Different casing / wording is not a hit.
    assert get_cached_chat("what is surge?") is None
    # History is not part of the key: identical wording still hits.


def test_chat_cache_key_is_exact_question_hash() -> None:
    assert chat_cache_key("Q") == chat_cache_key("Q")
    assert chat_cache_key("Q") != chat_cache_key("Q ")
    assert chat_cache_key("Q") != chat_cache_key("q")


def test_redis_status_reports_three_states(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_module.settings, "redis_url", None)
    reset_client()
    assert redis_status() is None

    monkeypatch.setattr(cache_module.settings, "redis_url", "redis://localhost:6379/0")
    configure_client(FakeRedis())
    assert redis_status() is True

    reset_client()
    # With a URL but a broken injected client, status is False rather than None.
    class BrokenRedis(FakeRedis):
        def ping(self) -> bool:
            raise ConnectionError("down")

    configure_client(BrokenRedis())
    assert redis_status() is False
