"""
Side channel for the things a tool produces that the model should not read.

A chart specification is several kilobytes of coordinates. Feeding it back
through the context window costs tokens, invites the model to recite numbers it
has already been given in tabular form, and crowds out the conversation. But the
browser does need it.

So every tool returns a compact summary to the model and pushes the full result
here. The API layer drains the collector and forwards the artifacts to the UI as
separate SSE events.

The collector is passed explicitly to the tool factory rather than held in a
context variable, because tools may run in a worker thread and context
propagation across that boundary is not something worth depending on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.serialization import to_jsonable

__all__ = ["ArtifactCollector", "to_jsonable"]


@dataclass
class ArtifactCollector:
    """Ordered artifacts produced during a single agent run."""

    artifacts: list[dict[str, Any]] = field(default_factory=list)

    def _add(self, artifact: dict[str, Any]) -> None:
        artifact["id"] = f"artifact-{len(self.artifacts) + 1}"
        self.artifacts.append(to_jsonable(artifact))

    def add_chart(self, title: str, spec: dict[str, Any], source: str) -> None:
        if not spec:
            return
        self._add({"kind": "chart", "title": title, "source": source, "spec": spec})

    def add_table(
        self,
        title: str,
        columns: list[str],
        rows: list[list[Any]],
        source: str,
        note: str | None = None,
    ) -> None:
        self._add(
            {
                "kind": "table",
                "title": title,
                "source": source,
                "columns": columns,
                "rows": rows,
                "note": note,
            }
        )

    def add_records(
        self,
        title: str,
        records: list[dict[str, Any]],
        source: str,
        note: str | None = None,
    ) -> None:
        """Add a table from row dictionaries, preserving first-seen column order."""
        if not records:
            return

        columns: list[str] = []
        for record in records:
            for key in record:
                if key not in columns:
                    columns.append(key)

        self.add_table(
            title=title,
            columns=columns,
            rows=[[record.get(column) for column in columns] for record in records],
            source=source,
            note=note,
        )

    def drain(self) -> list[dict[str, Any]]:
        """Return artifacts collected since the last drain, and clear them."""
        pending, self.artifacts = self.artifacts, []
        return pending
