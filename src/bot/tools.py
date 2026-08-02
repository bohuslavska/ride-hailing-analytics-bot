"""
The agent's tools.

Each tool does the same three things: run an analytics function, push the chart
and the full table to the artifact collector for the browser, and return a
compact JSON summary to the model.

The split matters. Handing the model a 24-row table with nine columns per row
invites it to transcribe the table into prose, which is slower, more expensive
and less readable than the chart the user is already looking at. Trimming the
payload is what makes the model summarise instead of recite.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from src.analytics.conversion import SUPPORTED_DRIVERS, SUPPORTED_OUTCOMES
from src.analytics.metrics import FUNNEL_DIMENSIONS
from src.analytics.service import (
    cached_acceptance_confounding,
    cached_conversion,
    cached_funnel,
    cached_marketplace_profile,
    cached_rider_clusters,
    cached_zone_clusters,
    cached_zone_supply_demand,
)
from src.bot.artifacts import ArtifactCollector, to_jsonable
from src.database.safe_sql import UnsafeQueryError, run_readonly_query

# How much of a result the model sees. The browser gets everything.
MODEL_ROW_LIMIT = 30
MODEL_SQL_ROW_LIMIT = 40


def _dumps(payload: Any) -> str:
    return json.dumps(to_jsonable(payload), ensure_ascii=False, default=str)


def _trim(records: list[dict[str, Any]], limit: int) -> tuple[list[dict], str | None]:
    """Cap the rows shown to the model, with a note explaining the cap."""
    if len(records) <= limit:
        return records, None

    note = (
        f"Showing {limit} of {len(records)} rows. The chart and full table have "
        "already been sent to the user, so summarise rather than listing rows. "
        "Use run_sql with an explicit ORDER BY and LIMIT if you need a specific "
        "part of the result."
    )
    return records[:limit], note


def build_tools(collector: ArtifactCollector) -> list[BaseTool]:
    """
    Construct the tool set bound to one run's artifact collector.

    Tools are built per request rather than once at import, so that each run
    writes its artifacts to its own collector without any shared mutable state
    between concurrent requests.
    """

    def run_sql(sql: str) -> str:
        """
        Run a read-only SQL SELECT against the dataset and return the rows.

        Use for anything the purpose-built tools do not cover: specific filters,
        time ranges, rankings, or combinations of dimensions. Start from
        rides_enriched, which is pre-joined to zone names and types and has
        hour, ride_date and day_of_week derived.

        Only a single SELECT or WITH statement is allowed. Aggregate in SQL
        rather than pulling rows and summarising them yourself.

        Args:
            sql: A single PostgreSQL SELECT statement, without a trailing
                semicolon.
        """
        try:
            result = run_readonly_query(sql)
        except UnsafeQueryError as error:
            return _dumps({"error": str(error), "rejected_sql": sql})
        except Exception as error:  # noqa: BLE001 - surfaced to the model to retry
            return _dumps(
                {
                    "error": f"{type(error).__name__}: {error}",
                    "rejected_sql": sql,
                    "hint": (
                        "Check column names against describe_schema. Remember "
                        "the timestamp column on zone_state is called 'ts'."
                    ),
                }
            )

        collector.add_table(
            title="Query result",
            columns=result.columns,
            rows=result.rows,
            source="run_sql",
            note="; ".join(result.notes) or None,
        )

        rows = result.rows[:MODEL_SQL_ROW_LIMIT]
        payload: dict[str, Any] = {
            "columns": result.columns,
            "rows": rows,
            "row_count": result.row_count,
            "elapsed_ms": result.elapsed_ms,
        }

        notes = list(result.notes)
        if result.row_count > MODEL_SQL_ROW_LIMIT:
            notes.append(
                f"Showing {MODEL_SQL_ROW_LIMIT} of {result.row_count} rows; the "
                "user already has the full table."
            )
        if notes:
            payload["notes"] = notes

        return _dumps(payload)

    def funnel_metrics(dimension: str | None = None) -> str:
        """
        Conversion counts and rates for the funnel
        calculated -> placed -> (accepted | churned_to_competitor).

        Rates are computed from raw counts rather than averaged across groups,
        which is the usual way these numbers go wrong.

        'calculated' is the demand volume; the conversion columns are rates.
        competitor_churn_rate is placed orders lost after a long search
        (~2-5 min patience), not a separate top-of-funnel drop. A question
        about how busy a period is, is answered by 'calculated', not by
        place_conversion. Splitting by a time dimension adds an is_curfew_hour
        flag; read it before drawing conclusions about quiet periods.

        Args:
            dimension: Optional split. One of hour, day_of_week, is_weekend,
                is_peak_hour, curfew, air_raid_alert, weather, origin_zone,
                origin_zone_type, destination_zone, destination_zone_type,
                ride_date, special_event. Omit for dataset totals.
        """
        if dimension is not None and dimension not in FUNNEL_DIMENSIONS:
            return _dumps(
                {
                    "error": f"Unknown dimension '{dimension}'.",
                    "available": sorted(FUNNEL_DIMENSIONS),
                }
            )

        result = cached_funnel(dimension=dimension)
        table = result["table"]

        if dimension is not None:
            label = dimension.replace("_", " ")
            collector.add_chart(
                title=f"Funnel by {label}",
                spec=result["chart"],
                source="funnel_metrics",
            )
            collector.add_records(
                title=f"Funnel by {label}",
                records=table,
                source="funnel_metrics",
            )

        rows, note = _trim(table, MODEL_ROW_LIMIT)
        payload = {
            "dimension": dimension,
            "metric_definitions": result["metric_definitions"],
            "table": rows,
        }
        if "curfew_warning" in result:
            payload["curfew_warning"] = result["curfew_warning"]
        if note:
            payload["note"] = note

        return _dumps(payload)

    def marketplace_profile(
        dimension: str = "hour", zone_type: str | None = None
    ) -> str:
        """
        Demand, driver supply, surge and ETA across the day or the week.

        Use for any "how does X vary by time of day" question, and for anything
        about the balance between the two sides of the market. Sourced from
        zone_state, so it sees idle drivers, which the ride table cannot.

        Args:
            dimension: 'hour' for the daily profile or 'day_of_week' for the
                weekly one.
            zone_type: Optional restriction to one of city_center, business,
                residential, entertainment, suburban, railway_station.
        """
        try:
            result = cached_marketplace_profile(
                dimension=dimension, zone_type=zone_type
            )
        except ValueError as error:
            return _dumps({"error": str(error)})

        label = "hour of day" if dimension == "hour" else "day of week"
        scope = f" ({zone_type.replace('_', ' ')} zones)" if zone_type else ""

        collector.add_chart(
            title=f"Demand, supply and surge by {label}{scope}",
            spec=result["chart"],
            source="marketplace_profile",
        )
        collector.add_records(
            title=f"Marketplace profile by {label}{scope}",
            records=result["table"],
            source="marketplace_profile",
        )

        payload: dict[str, Any] = {
            "dimension": dimension,
            "zone_type": zone_type,
            "table": result["table"],
        }
        if "curfew_warning" in result:
            payload["curfew_warning"] = result["curfew_warning"]

        return _dumps(payload)

    def zone_supply_demand(limit: int = 20) -> str:
        """
        Per-zone demand, driver availability, supply gap, surge and ETA.

        Sourced from zone_state rather than rides, because rides only exist
        where demand happened and idle supply is invisible in the ride table.
        Ordered by average supply gap, so the most under-served zones come
        first.

        Args:
            limit: Number of zones to return. There are 20 in total.
        """
        result = cached_zone_supply_demand(limit=limit)

        collector.add_chart(
            title="Demand vs available drivers by zone",
            spec=result["chart"],
            source="zone_supply_demand",
        )
        collector.add_records(
            title="Supply and demand by zone",
            records=result["table"],
            source="zone_supply_demand",
        )

        rows, note = _trim(result["table"], MODEL_ROW_LIMIT)
        payload: dict[str, Any] = {"table": rows}
        if note:
            payload["note"] = note

        return _dumps(payload)

    def conversion_analysis(
        driver: str = "eta_minutes", outcome: str = "placed"
    ) -> str:
        """
        Relate one driver to conversion, both raw and with controls applied.

        Returns bucketed rates, the raw correlation, and a logistic model that
        holds the other conditions fixed. Report the controlled estimate when
        the question is causal, and say so.

        Args:
            driver: One of eta_minutes, surge_multiplier, final_price,
                distance_km.
            outcome: 'placed' (over all calculated rides) or 'accepted' (over
                placed orders).
        """
        if driver not in SUPPORTED_DRIVERS:
            return _dumps(
                {
                    "error": f"Unknown driver '{driver}'.",
                    "available": sorted(SUPPORTED_DRIVERS),
                }
            )
        if outcome not in SUPPORTED_OUTCOMES:
            return _dumps(
                {
                    "error": f"Unknown outcome '{outcome}'.",
                    "available": sorted(SUPPORTED_OUTCOMES),
                }
            )

        result = cached_conversion(driver=driver, outcome=outcome)
        label = f"{result['outcome_label']} by {result['driver_label']}"

        collector.add_chart(title=label, spec=result["chart"], source="conversion_analysis")
        collector.add_records(
            title=label, records=result["buckets"], source="conversion_analysis"
        )

        return _dumps(
            {
                "driver": result["driver_label"],
                "outcome": result["outcome_label"],
                "population": result["population"],
                "total_observations": result["total_observations"],
                "buckets": result["buckets"],
                "raw_association": result["raw_association"],
                "controlled_model": result["controlled_model"],
                "interpretation": result["interpretation"],
            }
        )

    def acceptance_confounding() -> str:
        """
        Test whether surge pricing actually improves driver acceptance.

        The marginal comparison and the conditional one disagree, because surge
        and ETA both rise with the same driver shortage. Use this tool for that
        question instead of reasoning from a cross-tab, and report both views.
        """
        result = cached_acceptance_confounding()

        collector.add_chart(
            title="Acceptance by surge, before and after controlling for ETA",
            spec=result["chart"],
            source="acceptance_confounding",
        )

        return _dumps(
            {
                "question": result["question"],
                "marginal_view": result["marginal_view"],
                "conditional_view": result["conditional_view"],
                "surge_eta_correlation": result["surge_eta_correlation"],
                "interpretation": result["interpretation"],
            }
        )

    def segment_zones(number_of_clusters: int | None = None) -> str:
        """
        Cluster the 20 zones by marketplace behaviour.

        Zone type is not an input, so any alignment between the clusters and the
        zone types is a finding rather than an assumption.

        Args:
            number_of_clusters: Optional override. Chosen automatically when
                omitted.
        """
        result = cached_zone_clusters(number_of_clusters=number_of_clusters)
        return _dumps(_cluster_payload(result, collector, "segment_zones"))

    def segment_riders(
        number_of_clusters: int | None = None, minimum_rides: int = 30
    ) -> str:
        """
        Cluster riders by observed behaviour.

        Only riders with enough history are included, so these clusters describe
        the active base rather than every user. Rider segments overlap much more
        than zone types do; treat them as a description, not a clean partition.

        Args:
            number_of_clusters: Optional override. Chosen automatically when
                omitted.
            minimum_rides: Minimum calculated rides for a rider to be included.
        """
        result = cached_rider_clusters(
            number_of_clusters=number_of_clusters, minimum_rides=minimum_rides
        )
        return _dumps(_cluster_payload(result, collector, "segment_riders"))

    functions: list[Callable[..., str]] = [
        run_sql,
        funnel_metrics,
        marketplace_profile,
        zone_supply_demand,
        conversion_analysis,
        acceptance_confounding,
        segment_zones,
        segment_riders,
    ]

    return [
        StructuredTool.from_function(
            func=function,
            name=function.__name__,
            description=(function.__doc__ or "").strip(),
            parse_docstring=True,
        )
        for function in functions
    ]


def _cluster_payload(
    result: dict[str, Any], collector: ArtifactCollector, source: str
) -> dict[str, Any]:
    """
    Reduce a clustering result to the part worth reasoning about.

    The per-entity assignments and the standardised profile matrix are large and
    are better seen than read, so they go to the browser. The model keeps the
    cluster profiles, the descriptions and the caveats.
    """
    entity = result["entity"]

    collector.add_chart(
        title=f"{entity} clusters", spec=result["chart"], source=source
    )
    collector.add_chart(
        title=f"{entity} cluster profiles (z-scores)",
        spec=result["profile_chart"],
        source=source,
    )
    collector.add_records(
        title=f"{entity} cluster assignments",
        records=result["assignments"],
        source=source,
    )

    payload = {
        "entity": entity,
        "members": result["members"],
        "number_of_clusters": result["number_of_clusters"],
        "cluster_count_selection": result["cluster_count_selection"],
        "silhouette_score": result["silhouette_score"],
        "silhouette_by_k": result["silhouette_by_k"],
        "features_used": result["features_used"],
        "cluster_profiles": result["cluster_profiles"],
        "caveats": result["caveats"],
    }

    # Which entities landed in which cluster. Withholding this forced the model
    # to describe a cluster as "probably the station zones" when the membership
    # was known, and naming the members is most of what makes a segment
    # actionable. Only included when the list is short enough to be worth the
    # tokens: rider assignments are already aggregated to one row per cluster,
    # whereas an unaggregated per-entity list would be thousands of rows.
    assignments = result["assignments"]
    if len(assignments) <= 40:
        payload["cluster_members"] = assignments

    return payload
