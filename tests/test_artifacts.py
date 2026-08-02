"""
The artifact side channel and its JSON coercion.

Everything here exists because Postgres and numpy hand back types that
json.dumps refuses, and an SSE stream that raises mid-serialisation produces a
half-written frame the browser cannot recover from.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from decimal import Decimal

import numpy as np
import pytest

from src.bot.artifacts import ArtifactCollector, to_jsonable


class TestToJsonable:
    def test_passes_through_plain_types(self) -> None:
        assert to_jsonable("text") == "text"
        assert to_jsonable(7) == 7
        assert to_jsonable(1.5) == 1.5
        assert to_jsonable(True) is True
        assert to_jsonable(None) is None

    def test_decimal_becomes_float(self) -> None:
        """Postgres returns Decimal for numeric columns, including every AVG."""
        assert to_jsonable(Decimal("1.25")) == 1.25
        assert isinstance(to_jsonable(Decimal("3")), float)

    @pytest.mark.parametrize(
        "value",
        [float("nan"), float("inf"), float("-inf"), Decimal("NaN"), Decimal("Infinity")],
    )
    def test_non_finite_numbers_become_null(self, value: object) -> None:
        """A heatmap cell with no observations is NaN, which is not valid JSON."""
        assert to_jsonable(value) is None

    def test_temporal_types_become_iso_strings(self) -> None:
        assert to_jsonable(dt.date(2026, 7, 30)) == "2026-07-30"
        assert to_jsonable(dt.datetime(2026, 7, 30, 18, 5)) == "2026-07-30T18:05:00"
        assert to_jsonable(dt.time(23, 15)) == "23:15:00"
        assert to_jsonable(dt.timedelta(minutes=2)) == 120.0

    def test_numpy_scalars_are_unwrapped(self) -> None:
        assert to_jsonable(np.int64(4)) == 4
        assert to_jsonable(np.float32(0.5)) == 0.5
        assert to_jsonable(np.bool_(True)) is True
        assert to_jsonable(np.float64("nan")) is None

    def test_nested_structures_are_converted_throughout(self) -> None:
        converted = to_jsonable(
            {
                "z": [[Decimal("0.9"), float("nan")], [np.float64(0.5), None]],
                "when": dt.date(2026, 1, 1),
                1: "integer keys become strings",
            }
        )

        assert converted == {
            "z": [[0.9, None], [0.5, None]],
            "when": "2026-01-01",
            "1": "integer keys become strings",
        }
        # The point of all of the above.
        json.dumps(converted)

    def test_unknown_objects_degrade_to_their_string_form(self) -> None:
        class Opaque:
            def __str__(self) -> str:
                return "opaque"

        assert to_jsonable(Opaque()) == "opaque"

    def test_a_real_heatmap_row_survives(self) -> None:
        """The confounding chart genuinely contains NaN for empty ETA/surge cells."""
        spec = {"z": [[0.88, 0.89, math.nan], [Decimal("0.9"), None, np.float64(0.8)]]}
        assert json.loads(json.dumps(to_jsonable(spec)))["z"][0][2] is None


class TestArtifactCollector:
    def test_artifacts_are_numbered_in_order(self) -> None:
        collector = ArtifactCollector()
        collector.add_chart("first", {"kind": "heatmap"}, source="tool")
        collector.add_table("second", ["a"], [[1]], source="tool")

        assert [a["id"] for a in collector.artifacts] == ["artifact-1", "artifact-2"]

    def test_empty_charts_and_records_are_not_collected(self) -> None:
        """A tool that found nothing should not produce a blank panel in the UI."""
        collector = ArtifactCollector()
        collector.add_chart("nothing", {}, source="tool")
        collector.add_records("nothing", [], source="tool")

        assert collector.artifacts == []

    def test_records_keep_first_seen_column_order(self) -> None:
        collector = ArtifactCollector()
        collector.add_records(
            "rows",
            [{"hour": 0, "surge": 1.0}, {"hour": 1, "surge": 1.1}],
            source="tool",
        )

        assert collector.artifacts[0]["columns"] == ["hour", "surge"]
        assert collector.artifacts[0]["rows"] == [[0, 1.0], [1, 1.1]]

    def test_records_with_ragged_keys_are_padded_with_null(self) -> None:
        collector = ArtifactCollector()
        collector.add_records(
            "rows",
            [{"a": 1}, {"a": 2, "b": 3}],
            source="tool",
        )

        assert collector.artifacts[0]["columns"] == ["a", "b"]
        assert collector.artifacts[0]["rows"] == [[1, None], [2, 3]]

    def test_draining_empties_the_collector(self) -> None:
        collector = ArtifactCollector()
        collector.add_table("t", ["a"], [[1]], source="tool")

        assert len(collector.drain()) == 1
        assert collector.drain() == []

    def test_collected_artifacts_are_json_serialisable(self) -> None:
        collector = ArtifactCollector()
        collector.add_records(
            "mixed",
            [{"value": Decimal("1.5"), "when": dt.date(2026, 1, 1), "bad": math.nan}],
            source="tool",
        )

        json.dumps(collector.drain())
