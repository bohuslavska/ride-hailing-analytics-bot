"""
FastAPI application: streaming chat plus the analytics endpoints behind it.

Every analysis the agent can run is also reachable as a plain REST call. That
keeps the interesting work testable without an LLM in the loop, and means the
dashboard does not have to ask a language model for a number it could fetch
directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text

from src.analytics.conversion import SUPPORTED_DRIVERS, SUPPORTED_OUTCOMES
from src.analytics.metrics import FUNNEL_DIMENSIONS
from src.analytics.service import (
    cached_acceptance_confounding,
    cached_conversion,
    cached_funnel,
    cached_marketplace_profile,
    cached_rider_clusters,
    cached_schema,
    cached_zone_clusters,
    cached_zone_supply_demand,
)
from src.api.schemas import ChatRequest, ChatResponse, HealthResponse
from src.bot.agent import answer_question, stream_answer
from src.bot.artifacts import to_jsonable
from src.cache import redis_status, reset_client
from src.config import PROJECT_ROOT, settings
from src.database.connection import get_engine
from src.observability import HTTP_DURATION, HTTP_REQUESTS, record_app_info

logger = logging.getLogger(__name__)

STATIC_DIR = PROJECT_ROOT / "static"

# Long enough to matter on a slow analysis, short enough to stay inside the
# idle timeout of any proxy between here and the browser.
KEEPALIVE_SECONDS = 15.0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm the schema description so the first question is not slower than the rest."""
    record_app_info(version=app.version, model=settings.openrouter_model)
    try:
        await asyncio.to_thread(cached_schema)
    except Exception as error:  # noqa: BLE001 - the app must still start
        logger.warning("Could not warm the schema description: %s", error)
    yield
    reset_client()


app = FastAPI(
    title="Ride-hailing analytics assistant",
    description=(
        "Synthetic ride-hailing marketplace with a tool-calling analytics "
        "assistant over it."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def record_request_metrics(request: Request, call_next: Callable) -> Any:
    """
    Count and time every request, labelled by route template.

    The template comes from the matched route rather than `request.url.path`, so
    `/api/metrics/zones?limit=5` and `?limit=50` share one time series and an
    unmatched path collapses to a single `unmatched` label instead of creating a
    new series per URL a crawler invents.
    """
    started_at = time.perf_counter()

    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        # An unhandled exception still becomes a 500 for the client, so it has to
        # be counted as one here rather than vanishing from the metrics.
        _observe_request(request, "500", time.perf_counter() - started_at)
        raise

    _observe_request(request, str(status), time.perf_counter() - started_at)
    return response


def _observe_request(request: Request, status: str, elapsed_seconds: float) -> None:
    route = request.scope.get("route")
    endpoint = getattr(route, "path", None) or "unmatched"

    HTTP_REQUESTS.labels(
        method=request.method, endpoint=endpoint, status=status
    ).inc()
    HTTP_DURATION.labels(method=request.method, endpoint=endpoint).observe(
        elapsed_seconds
    )


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Prometheus scrape endpoint, in the text exposition format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _run(analysis: Callable[[], Any]) -> Any:
    """
    Execute a blocking analysis off the event loop, converting failures to 4xx/5xx.

    The analytics functions are synchronous and database-bound, so running them
    inline would block every other request for the duration of a clustering
    fit.
    """
    try:
        return analysis()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:  # noqa: BLE001
        logger.exception("Analysis failed")
        raise HTTPException(
            status_code=503,
            detail=f"The analysis could not be completed: {error}",
        ) from error


async def _run_async(analysis: Callable[[], Any]) -> Any:
    return await asyncio.to_thread(_run, analysis)


# --------------------------------------------------------------------------
# health and schema
# --------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness plus a real query, so a running process with a dead database fails."""

    def probe() -> int:
        # Keep this under the Fly HTTP check timeout (10s). Exact COUNT(*) on
        # the full rides table is too slow on the managed Postgres plan.
        with get_engine(readonly=True).connect() as connection:
            connection.execute(text("SELECT 1")).scalar_one()
            estimate = connection.execute(
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
            if estimate and estimate > 0:
                return int(estimate)
            has_row = connection.execute(
                text("SELECT EXISTS (SELECT 1 FROM rides LIMIT 1)")
            ).scalar_one()
            return 1 if has_row else 0

    cache = await asyncio.to_thread(redis_status)

    try:
        rides = await asyncio.to_thread(probe)
    except Exception as error:  # noqa: BLE001
        return HealthResponse(
            status="degraded",
            database=False,
            redis=cache,
            llm_configured=settings.llm_enabled,
            detail=f"Database unreachable: {error}",
        )

    # Redis is optional: a miss only means slower analytics, not a dead service.
    # Report its state, but do not flip status to degraded when it is down.
    detail = None
    if not settings.llm_enabled:
        detail = "OPENROUTER_API_KEY is not set; chat is disabled."
    elif cache is False:
        detail = "Redis unreachable; analytics cache bypassed."

    return HealthResponse(
        status="ok",
        database=True,
        redis=cache,
        llm_configured=settings.llm_enabled,
        calculated_rides=rides,
        detail=detail,
    )


@app.get("/api/schema")
async def schema() -> dict[str, Any]:
    """Tables, columns, metric definitions and caveats, as given to the model."""
    return to_jsonable(await _run_async(cached_schema))


# --------------------------------------------------------------------------
# analytics
# --------------------------------------------------------------------------


@app.get("/api/metrics/funnel")
async def funnel(
    dimension: str | None = Query(
        default=None,
        description=f"One of: {', '.join(sorted(FUNNEL_DIMENSIONS))}",
    ),
) -> dict[str, Any]:
    """Conversion counts and rates, optionally split by one dimension."""
    if dimension is not None and dimension not in FUNNEL_DIMENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown dimension '{dimension}'. "
                f"Available: {', '.join(sorted(FUNNEL_DIMENSIONS))}."
            ),
        )
    return to_jsonable(await _run_async(lambda: cached_funnel(dimension=dimension)))


@app.get("/api/metrics/profile")
async def profile(
    dimension: str = Query(default="hour", description="hour or day_of_week"),
    zone_type: str | None = Query(default=None),
) -> dict[str, Any]:
    """Demand, driver supply, surge and ETA across the day or the week."""
    return to_jsonable(
        await _run_async(
            lambda: cached_marketplace_profile(
                dimension=dimension, zone_type=zone_type
            )
        )
    )


@app.get("/api/metrics/zones")
async def zones(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
    """Per-zone demand, driver availability, supply gap, surge and ETA."""
    return to_jsonable(
        await _run_async(lambda: cached_zone_supply_demand(limit=limit))
    )


@app.get("/api/conversion")
async def conversion(
    driver: str = Query(default="eta_minutes"),
    outcome: str = Query(default="placed"),
) -> dict[str, Any]:
    """Bucketed rates, raw correlation and a controlled model for one driver."""
    if driver not in SUPPORTED_DRIVERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown driver '{driver}'. Available: {', '.join(sorted(SUPPORTED_DRIVERS))}.",
        )
    if outcome not in SUPPORTED_OUTCOMES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown outcome '{outcome}'. Available: {', '.join(sorted(SUPPORTED_OUTCOMES))}.",
        )
    return to_jsonable(
        await _run_async(
            lambda: cached_conversion(driver=driver, outcome=outcome)
        )
    )


@app.get("/api/conversion/confounding")
async def confounding() -> dict[str, Any]:
    """Whether surge helps acceptance, before and after controlling for ETA."""
    return to_jsonable(await _run_async(cached_acceptance_confounding))


@app.get("/api/clusters/zones")
async def zone_clusters(
    number_of_clusters: int | None = Query(default=None, ge=2, le=8),
) -> dict[str, Any]:
    """Behavioural clustering of the 20 zones."""
    return to_jsonable(
        await _run_async(
            lambda: cached_zone_clusters(number_of_clusters=number_of_clusters)
        )
    )


@app.get("/api/clusters/riders")
async def rider_clusters(
    number_of_clusters: int | None = Query(default=None, ge=2, le=10),
    minimum_rides: int = Query(default=30, ge=5, le=500),
) -> dict[str, Any]:
    """Behavioural clustering of riders with enough history."""
    return to_jsonable(
        await _run_async(
            lambda: cached_rider_clusters(
                number_of_clusters=number_of_clusters, minimum_rides=minimum_rides
            )
        )
    )


# --------------------------------------------------------------------------
# chat
# --------------------------------------------------------------------------


def _sse(event: dict[str, Any]) -> str:
    """Encode one event as an SSE frame."""
    return f"data: {json.dumps(to_jsonable(event), ensure_ascii=False)}\n\n"


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest, http_request: Request) -> StreamingResponse:
    """
    Stream the assistant's answer as Server-Sent Events.

    Frames carry a `type`: `status`, `tool`, `artifact`, `token`, `error` or
    `done`. Charts arrive as `artifact` frames as soon as the tool that produced
    them returns, so they render while the model is still writing about them.
    """

    async def publish() -> AsyncIterator[str]:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def produce() -> None:
            try:
                async for event in stream_answer(
                    request.question, request.history_as_dicts()
                ):
                    await queue.put(event)
            except Exception:  # noqa: BLE001
                # Logged in full, reported generically. The exception text can
                # carry a connection string, a provider payload or the model's
                # own SQL, none of which belongs in a browser.
                logger.exception("Chat stream failed")
                await queue.put(
                    {
                        "type": "error",
                        "message": (
                            "The assistant could not complete this request. "
                            "Please try again, or rephrase the question."
                        ),
                    }
                )
            finally:
                await queue.put(None)

        producer = asyncio.create_task(produce())

        # A comment frame, not a status event: this exists only to flush the
        # response headers so the browser opens the stream immediately. Emitting a
        # real event here would duplicate the agent's own opening status and put
        # a frame on the wire that no tool or model produced.
        yield ": open\n\n"

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_SECONDS)
                except TimeoutError:
                    # A comment frame keeps intermediaries from closing an idle
                    # connection during a slow analysis.
                    yield ": keepalive\n\n"
                    continue

                if event is None:
                    break

                yield _sse(event)

                if await http_request.is_disconnected():
                    break
        finally:
            producer.cancel()

    return StreamingResponse(
        publish(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Tells nginx-style proxies not to buffer, which would otherwise
            # hold the whole answer back and defeat streaming entirely.
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Non-streaming equivalent of /api/chat/stream, for scripts and tests."""
    result = await answer_question(request.question, request.history_as_dicts())

    if result["error"] and not result["answer"]:
        return JSONResponse(status_code=503, content=to_jsonable(result))

    return ChatResponse(**result)


# --------------------------------------------------------------------------
# static UI, mounted last so it cannot shadow the API routes
# --------------------------------------------------------------------------

if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
