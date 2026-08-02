"""
Prometheus metrics for the API and the agent.

What is worth measuring here is not CPU and memory -- the platform reports those
already -- but the things that are specific to an LLM application and invisible
without instrumentation:

* which tools the agent actually reaches for, and whether it falls back to raw
  SQL instead of the purpose-built analyses;
* how many tokens a question costs, which is the real unit price of an answer;
* how long generated SQL runs, and how often the guardrails reject it;
* how often a conversation ends in an error frame rather than an answer.

All metrics live in the default registry. The app runs a single uvicorn worker
per machine (see fly.toml), so there is no multiprocess collector to configure;
if that ever changes, this is the file that has to change with it.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import Counter, Gauge, Histogram

# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

# `endpoint` is the route template ("/api/metrics/funnel"), never the raw path.
# Labelling by raw path would let a crawler hitting /a, /b, /c create unbounded
# time series, which is the standard way to take down a Prometheus instance.
HTTP_REQUESTS = Counter(
    "http_requests_total",
    "HTTP requests handled, by route template and response status.",
    ["method", "endpoint", "status"],
)

HTTP_DURATION = Histogram(
    "http_request_duration_seconds",
    "Wall-clock time to handle an HTTP request.",
    ["method", "endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------

CHAT_REQUESTS = Counter(
    "chat_requests_total",
    "Completed assistant runs, by outcome.",
    ["outcome"],  # answered | failed | unavailable (no API key configured)
)

CHAT_DURATION = Histogram(
    "chat_duration_seconds",
    "Time from question received to final answer, including all tool calls.",
    # An answer that chains three tools and a clustering fit is a slow request by
    # HTTP standards but a normal one here, so the buckets run out to two minutes.
    buckets=(1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0),
)

TOOL_CALLS = Counter(
    "agent_tool_calls_total",
    "Tool invocations by the agent, by tool name.",
    ["tool"],
)

TOKENS = Counter(
    "agent_tokens_total",
    "Tokens billed by the model provider, by direction.",
    ["kind"],  # input | output
)

# --------------------------------------------------------------------------
# Generated SQL
# --------------------------------------------------------------------------

SQL_QUERIES = Counter(
    "agent_sql_queries_total",
    "Model-generated SQL statements, by outcome.",
    ["outcome"],  # executed | rejected | failed
)

SQL_DURATION = Histogram(
    "agent_sql_query_duration_seconds",
    "Execution time of model-generated SQL that passed validation.",
    buckets=(0.005, 0.025, 0.1, 0.5, 1.0, 2.5, 5.0, 15.0),
)

# --------------------------------------------------------------------------
# Analytics cache (Redis)
# --------------------------------------------------------------------------

CACHE_REQUESTS = Counter(
    "analytics_cache_requests_total",
    "Lookups against the analytics result cache, by outcome.",
    ["outcome"],  # hit | miss | bypass | error
)

# --------------------------------------------------------------------------
# Build information
# --------------------------------------------------------------------------

APP_INFO = Gauge(
    "app_info",
    "Build and configuration of the running instance; the value is always 1.",
    ["version", "model"],
)


def record_app_info(version: str, model: str) -> None:
    """Publish static configuration as a labelled gauge, the usual Prometheus idiom."""
    APP_INFO.labels(version=version, model=model).set(1)


def record_token_usage(usage: object) -> None:
    """
    Record token counts from a LangChain usage-metadata mapping.

    Providers disagree on the key names and some omit usage entirely, so this
    accepts anything dict-like and silently records what it recognises. Metrics
    must never be the reason a request fails.
    """
    if not isinstance(usage, dict):
        return

    for kind, keys in (
        ("input", ("input_tokens", "prompt_tokens")),
        ("output", ("output_tokens", "completion_tokens")),
    ):
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)) and value > 0:
                TOKENS.labels(kind=kind).inc(value)
                break


@contextmanager
def observe_duration(histogram: Histogram) -> Iterator[None]:
    """Time a block and record it, whether or not the block raises."""
    started_at = time.perf_counter()
    try:
        yield
    finally:
        histogram.observe(time.perf_counter() - started_at)
