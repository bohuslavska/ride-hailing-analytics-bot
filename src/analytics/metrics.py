"""Funnel metrics with the denominators fixed in one place."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import text

from src.database.connection import get_engine

# Dimensions the funnel can be broken down by, mapped to the SQL expression
# that produces them. Restricting to a known set means this tool never has to
# interpolate model-supplied text into a query.
FUNNEL_DIMENSIONS = {
    "hour": "hour",
    "day_of_week": "day_of_week",
    "is_weekend": "is_weekend",
    "is_peak_hour": "is_peak_hour",
    "curfew": "curfew",
    "air_raid_alert": "air_raid_alert",
    "weather": "weather",
    "origin_zone": "origin_zone_name",
    "origin_zone_type": "origin_zone_type",
    "destination_zone": "destination_zone_name",
    "destination_zone_type": "destination_zone_type",
    "ride_date": "ride_date",
    "special_event": "special_event",
}

FUNNEL_SELECT = """
    COUNT(*)                                                        AS calculated,
    SUM(placed)                                                     AS placed,
    SUM(accepted)                                                   AS accepted,
    SUM(churned_to_competitor)                                      AS churned_to_competitor,
    AVG(placed)::double precision                                   AS place_conversion,
    (SUM(accepted)::double precision / NULLIF(SUM(placed), 0))      AS acceptance_rate,
    (SUM(churned_to_competitor)::double precision
        / NULLIF(SUM(placed), 0))                                   AS competitor_churn_rate,
    AVG(accepted)::double precision                                 AS end_to_end_conversion,
    AVG(eta_minutes)                                                AS average_eta_minutes,
    AVG(search_wait_minutes) FILTER (WHERE placed = 1)              AS average_search_wait_minutes,
    AVG(surge_multiplier)                                           AS average_surge,
    AVG(final_price)                                                AS average_final_price
"""


def calculate_funnel(dimension: str | None = None) -> dict[str, Any]:
    """
    Funnel counts and rates, optionally split by one dimension.

    Rates are computed in SQL from the raw counts rather than averaged over
    groups, which is the usual way these numbers go wrong: the mean of per-hour
    conversion rates is not the overall conversion rate.
    """
    if dimension is not None and dimension not in FUNNEL_DIMENSIONS:
        raise ValueError(
            f"Unknown dimension '{dimension}'. "
            f"Available: {', '.join(sorted(FUNNEL_DIMENSIONS))}."
        )

    # Splitting by a time dimension carries the curfew flag alongside it. A
    # reader looking at 24 rows of hourly counts cannot otherwise tell that the
    # overnight rows are a legal restriction rather than quiet demand, and
    # putting it in the data works far better than saying so in prose.
    marks_curfew = dimension in {"hour", "day_of_week", "ride_date"}
    curfew_column = "BOOL_OR(curfew) AS is_curfew_hour," if marks_curfew else ""

    if dimension is None:
        query = f"SELECT {FUNNEL_SELECT} FROM rides_enriched"
    else:
        column = FUNNEL_DIMENSIONS[dimension]
        query = f"""
            SELECT {column} AS {dimension}, {curfew_column} {FUNNEL_SELECT}
            FROM rides_enriched
            GROUP BY {column}
            ORDER BY {column}
        """

    with get_engine(readonly=True).connect() as connection:
        frame = pd.read_sql(text(query), connection)

    numeric_columns = [
        "place_conversion",
        "acceptance_rate",
        "competitor_churn_rate",
        "end_to_end_conversion",
        "average_eta_minutes",
        "average_search_wait_minutes",
        "average_surge",
        "average_final_price",
    ]
    frame[numeric_columns] = frame[numeric_columns].astype(float).round(4)

    result: dict[str, Any] = {
        "dimension": dimension,
        "metric_definitions": {
            "place_conversion": "placed / calculated",
            "acceptance_rate": "accepted / placed",
            "competitor_churn_rate": (
                "churned_to_competitor / placed — share of placed orders lost "
                "when search outran rider patience (~2-5 min), standing in for "
                "the real-world price-slider / switch-to-Bolt exit"
            ),
            "end_to_end_conversion": "accepted / calculated",
            "average_search_wait_minutes": (
                "mean search duration over placed orders (match time if "
                "accepted, patience if churned)"
            ),
        },
        "table": frame.to_dict(orient="records"),
    }

    if marks_curfew and bool(frame["is_curfew_hour"].any()):
        result["curfew_warning"] = (
            "Rows flagged is_curfew_hour cover 00:00-05:00, when civilian "
            "movement is prohibited. Their volume is a legal restriction, not "
            "weak demand, and no promotion or driver incentive can change it. "
            "Exclude them from any 'quietest period' answer and never "
            "recommend stimulating demand during them."
        )

    if dimension is not None:
        result["chart"] = {
            "kind": "funnel_by_dimension",
            "title": f"Funnel by {dimension.replace('_', ' ')}",
            "x": frame[dimension].astype(str).tolist(),
            "x_title": dimension.replace("_", " "),
            "series": [
                {
                    "name": "calculated rides",
                    "type": "bar",
                    "axis": "y2",
                    "values": frame["calculated"].tolist(),
                },
                {
                    "name": "place conversion",
                    "type": "line",
                    "axis": "y",
                    "values": frame["place_conversion"].tolist(),
                },
                {
                    "name": "acceptance rate",
                    "type": "line",
                    "axis": "y",
                    "values": frame["acceptance_rate"].tolist(),
                },
            ],
            "y_title": "rate",
            "y2_title": "calculated rides",
        }

    return result


# Time dimensions available on zone_state, and the zone types the profile can be
# restricted to. Both are closed sets, so no model-supplied text is ever
# interpolated into the query.
PROFILE_DIMENSIONS = {"hour": "hour", "day_of_week": "day_of_week"}

PROFILE_ZONE_TYPES = {
    "city_center",
    "business",
    "residential",
    "entertainment",
    "suburban",
    "railway_station",
}


def marketplace_profile(
    dimension: str = "hour", zone_type: str | None = None
) -> dict[str, Any]:
    """
    Demand, driver supply, surge and ETA across the day or the week.

    Read from zone_state rather than rides, because the question this answers is
    usually about the balance between the two sides of the market, and idle
    drivers do not appear in the ride table at all.
    """
    if dimension not in PROFILE_DIMENSIONS:
        raise ValueError(
            f"Unknown dimension '{dimension}'. "
            f"Available: {', '.join(sorted(PROFILE_DIMENSIONS))}."
        )
    if zone_type is not None and zone_type not in PROFILE_ZONE_TYPES:
        raise ValueError(
            f"Unknown zone_type '{zone_type}'. "
            f"Available: {', '.join(sorted(PROFILE_ZONE_TYPES))}."
        )

    column = PROFILE_DIMENSIONS[dimension]
    zone_filter = "WHERE z.zone_type = :zone_type" if zone_type else ""

    query = text(
        f"""
        SELECT
            s.{column}                    AS {dimension},
            BOOL_OR(s.curfew)             AS is_curfew_hour,
            AVG(s.demand_count)           AS avg_demand,
            AVG(s.available_drivers)      AS avg_available_drivers,
            AVG(s.demand_supply_ratio)    AS avg_demand_supply_ratio,
            AVG(s.surge_multiplier)       AS avg_surge,
            AVG(s.average_eta_minutes)    AS avg_eta_minutes,
            SUM(s.demand_count)           AS total_demand
        FROM zone_state s
        JOIN zones z ON z.zone_id = s.zone_id
        {zone_filter}
        GROUP BY s.{column}
        ORDER BY s.{column}
        """
    )

    parameters = {"zone_type": zone_type} if zone_type else {}

    with get_engine(readonly=True).connect() as connection:
        frame = pd.read_sql(query, connection, params=parameters)

    numeric_columns = [
        "avg_demand",
        "avg_available_drivers",
        "avg_demand_supply_ratio",
        "avg_surge",
        "avg_eta_minutes",
    ]
    frame[numeric_columns] = frame[numeric_columns].astype(float).round(3)

    scope = f"{zone_type.replace('_', ' ')} zones" if zone_type else "all zones"
    label = "hour of day" if dimension == "hour" else "day of week"

    result: dict[str, Any] = {
        "dimension": dimension,
        "zone_type": zone_type,
        "table": frame.to_dict(orient="records"),
        "chart": {
            "kind": "funnel_by_dimension",
            "title": f"Demand, supply and surge by {label} — {scope}",
            "x": frame[dimension].astype(str).tolist(),
            "x_title": label,
            "series": [
                {
                    "name": "avg demand",
                    "type": "bar",
                    "axis": "y2",
                    "values": frame["avg_demand"].tolist(),
                },
                {
                    "name": "avg available drivers",
                    "type": "bar",
                    "axis": "y2",
                    "values": frame["avg_available_drivers"].tolist(),
                },
                {
                    "name": "avg surge",
                    "type": "line",
                    "axis": "y",
                    "values": frame["avg_surge"].tolist(),
                },
            ],
            "y_title": "surge multiplier",
            "y2_title": "count per 15-minute interval",
        },
    }

    if bool(frame["is_curfew_hour"].any()):
        result["curfew_warning"] = (
            "Rows flagged is_curfew_hour cover 00:00-05:00, when civilian "
            "movement is prohibited. Their near-zero demand is a legal "
            "restriction, not weak demand, and surge there is meaningless "
            "because almost nothing is being requested."
        )

    return result


def zone_supply_demand_summary(limit: int = 20) -> dict[str, Any]:
    """
    Per-zone supply and demand picture from zone_state.

    Uses zone_state rather than rides because rides only exist where demand
    happened; idle supply is invisible in the ride table.
    """
    query = text(
        """
        SELECT
            z.zone_id,
            z.zone_name,
            z.zone_type,
            AVG(s.demand_count)        AS avg_demand,
            AVG(s.available_drivers)   AS avg_available_drivers,
            AVG(s.demand_supply_ratio) AS avg_demand_supply_ratio,
            AVG(s.supply_gap)          AS avg_supply_gap,
            AVG(s.surge_multiplier)    AS avg_surge,
            AVG(s.average_eta_minutes) AS avg_eta_minutes,
            SUM(s.demand_count)        AS total_demand
        FROM zone_state s
        JOIN zones z ON z.zone_id = s.zone_id
        GROUP BY z.zone_id, z.zone_name, z.zone_type
        ORDER BY avg_supply_gap DESC
        LIMIT :limit
        """
    )

    with get_engine(readonly=True).connect() as connection:
        frame = pd.read_sql(query, connection, params={"limit": limit})

    numeric_columns = frame.select_dtypes("number").columns
    frame[numeric_columns] = frame[numeric_columns].astype(float).round(3)

    return {
        "table": frame.to_dict(orient="records"),
        "chart": {
            "kind": "supply_demand_by_zone",
            "title": "Average demand vs available drivers by zone",
            "x": frame["zone_name"].tolist(),
            "x_title": "zone",
            "series": [
                {
                    "name": "avg demand",
                    "type": "bar",
                    "axis": "y",
                    "values": frame["avg_demand"].tolist(),
                },
                {
                    "name": "avg available drivers",
                    "type": "bar",
                    "axis": "y",
                    "values": frame["avg_available_drivers"].tolist(),
                },
                {
                    "name": "avg surge",
                    "type": "line",
                    "axis": "y2",
                    "values": frame["avg_surge"].tolist(),
                },
            ],
            "y_title": "count per 15-minute interval",
            "y2_title": "surge multiplier",
        },
    }
