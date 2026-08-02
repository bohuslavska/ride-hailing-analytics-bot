"""
The HTTP surface.

The chat endpoints are tested with a stubbed agent rather than a live model: the
thing worth pinning down is the SSE framing and the failure behaviour, and a real
model would make the test slow, costly and non-deterministic without checking
anything extra.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from tests.conftest import needs_database


@pytest.fixture
def client() -> AsyncIterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def sse_events(payload: str) -> list[dict[str, Any]]:
    """Parse an SSE body into its JSON frames, ignoring keepalive comments."""
    events = []
    for frame in payload.split("\n\n"):
        for line in frame.split("\n"):
            if line.startswith("data:"):
                events.append(json.loads(line[5:].strip()))
    return events


class TestStaticFiles:
    def test_the_ui_is_served_at_the_root(self, client: TestClient) -> None:
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Аналітика райд-хейлінгу" in response.text
        assert 'lang="uk"' in response.text

    @pytest.mark.parametrize("path", ["/styles.css", "/app.js"])
    def test_assets_are_served(self, client: TestClient, path: str) -> None:
        assert client.get(path).status_code == 200


@needs_database
class TestHealth:
    def test_health_reports_a_real_row_count(self, client: TestClient) -> None:
        """
        A health check that only proves the process is up would keep a machine in
        rotation while the database was unreachable, so this one runs a query.
        """
        body = client.get("/api/health").json()

        assert body["status"] == "ok"
        assert body["database"] is True
        assert body["calculated_rides"] > 0

    def test_health_reports_whether_chat_is_available(self, client: TestClient) -> None:
        assert isinstance(client.get("/api/health").json()["llm_configured"], bool)


@needs_database
class TestAnalyticsEndpoints:
    @pytest.mark.parametrize(
        "path",
        [
            "/api/schema",
            "/api/metrics/funnel",
            "/api/metrics/funnel?dimension=hour",
            "/api/metrics/profile",
            "/api/metrics/profile?dimension=day_of_week",
            "/api/metrics/profile?zone_type=entertainment",
            "/api/metrics/zones",
            "/api/conversion",
            "/api/conversion/confounding",
            "/api/clusters/zones",
        ],
    )
    def test_endpoints_return_json(self, client: TestClient, path: str) -> None:
        response = client.get(path)

        assert response.status_code == 200
        assert response.json()

    @pytest.mark.parametrize(
        "path",
        [
            "/api/metrics/funnel?dimension=not_a_column",
            "/api/metrics/profile?dimension=zone_id",
            "/api/metrics/profile?zone_type=airport",
        ],
    )
    def test_unknown_parameters_are_rejected_not_interpolated(
        self, client: TestClient, path: str
    ) -> None:
        assert client.get(path).status_code == 400

    def test_responses_are_json_serialisable_end_to_end(
        self, client: TestClient
    ) -> None:
        """The confounding heatmap contains NaN cells, which must arrive as null."""
        body = client.get("/api/conversion/confounding").json()
        flattened = [cell for row in body["chart"]["z"] for cell in row]

        assert any(cell is None for cell in flattened)
        assert all(cell is None or isinstance(cell, float) for cell in flattened)


class TestChatValidation:
    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"question": ""},
            {"question": "   "},
            {"question": "x" * 5000},
            {"question": "ok", "history": [{"role": "wizard", "content": "hi"}]},
        ],
    )
    def test_malformed_requests_are_rejected(
        self, client: TestClient, payload: dict
    ) -> None:
        assert client.post("/api/chat", json=payload).status_code == 422


class TestChatStreaming:
    """The SSE contract, with the agent stubbed out."""

    def test_events_are_framed_and_ordered(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_stream(question: str, history: Any = None):
            yield {"type": "status", "message": "thinking"}
            yield {"type": "tool", "tool": "funnel_metrics", "arguments": {}}
            yield {"type": "artifact", "artifact": {"kind": "chart", "title": "t"}}
            yield {"type": "token", "text": "51.7%"}
            yield {"type": "done", "answer": "51.7%"}

        monkeypatch.setattr("src.api.app.stream_answer", fake_stream)

        with client.stream(
            "POST", "/api/chat/stream", json={"question": "rate?"}
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            # Buffering a stream would defeat the point of streaming it.
            assert response.headers.get("x-accel-buffering") == "no"
            assert response.headers.get("cache-control") == "no-cache"

            events = sse_events(response.read().decode())

        assert [event["type"] for event in events] == [
            "status",
            "tool",
            "artifact",
            "token",
            "done",
        ]
        assert events[-1]["answer"] == "51.7%"

    def test_an_agent_failure_becomes_an_error_frame(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        The response has already begun, so the status code cannot change. The
        browser has to learn about the failure from the stream itself.
        """

        async def failing_stream(question: str, history: Any = None):
            yield {"type": "status", "message": "thinking"}
            raise RuntimeError("upstream exploded")

        monkeypatch.setattr("src.api.app.stream_answer", failing_stream)

        with client.stream(
            "POST", "/api/chat/stream", json={"question": "rate?"}
        ) as response:
            events = sse_events(response.read().decode())

        assert events[-1]["type"] == "error"
        assert events[-1]["message"]
        # The internal exception text should not be forwarded verbatim.
        assert "upstream exploded" not in events[-1]["message"]


class TestMetrics:
    """
    The Prometheus endpoint.

    The label values matter more than the counts. A route template keeps one time
    series per endpoint; the raw path would create one per distinct query string,
    which is the usual way an exposition endpoint turns into an outage.
    """

    def test_metrics_are_exposed_in_the_prometheus_text_format(
        self, client: TestClient
    ) -> None:
        response = client.get("/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "# HELP http_requests_total" in response.text
        assert "# TYPE http_requests_total counter" in response.text

    def test_requests_are_labelled_by_route_template_not_raw_path(
        self, client: TestClient
    ) -> None:
        client.get("/api/health?ignored=1")
        client.get("/api/health?ignored=2")

        body = client.get("/metrics").text
        series = [
            line
            for line in body.splitlines()
            if line.startswith("http_requests_total{") and "/api/health" in line
        ]

        assert len(series) == 1, f"expected one series for /api/health, got {series}"
        assert "ignored" not in body

    def test_unmatched_paths_collapse_into_a_single_series(
        self, client: TestClient
    ) -> None:
        for path in ("/nope-one", "/nope-two", "/nope-three"):
            client.get(path)

        body = client.get("/metrics").text

        assert 'endpoint="unmatched"' in body
        for path in ("nope-one", "nope-two", "nope-three"):
            assert path not in body

    def test_the_running_configuration_is_published(self, client: TestClient) -> None:
        body = client.get("/metrics").text

        assert "app_info{" in body
