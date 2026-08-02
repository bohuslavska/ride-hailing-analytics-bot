"""
The dataset description handed to the model before it writes any SQL.

Column names and types are introspected from PostgreSQL rather than hard-coded,
so this can never drift out of step with the actual schema. The prose around
them is curated, because the two things a model gets wrong unaided are not
column names but *semantics*: which denominator a conversion metric uses, and
which apparent relationships in this data are confounded.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from sqlalchemy import text

from src.database.connection import get_engine

RELATION_PURPOSE = {
    "rides_enriched": (
        "Start here for almost every question. One row per price calculation, "
        "already joined to origin and destination zone names and types, with "
        "hour / ride_date / day_of_week derived."
    ),
    "rides": (
        "The underlying fact table behind rides_enriched. Prefer rides_enriched "
        "unless you specifically need the un-joined table."
    ),
    "zone_state": (
        "Marketplace state per zone per 15-minute interval: demand_count, "
        "available_drivers, surge_multiplier, average_eta_minutes. Use this for "
        "supply and demand questions, since rides only exist where demand did."
    ),
    "zones": "Static description of the 20 zones: type, coordinates, structural levels.",
    "users": (
        "One row per rider: user_id, home_zone_id, signup_date. Behavioural "
        "traits are deliberately NOT stored, so user segments must be derived "
        "from ride behaviour."
    ),
}

COLUMN_NOTES = {
    "final_status": (
        "One of 'calculated', 'accepted', 'churned_to_competitor'. "
        "After place the order either matches or the rider leaves for a competitor."
    ),
    "churned_to_competitor": (
        "1 if the rider placed an order and abandoned search for a competitor "
        "after waiting past their patience (~2-5 minutes). Mutually exclusive "
        "with accepted; always implies placed = 1."
    ),
    "search_wait_minutes": (
        "Minutes spent searching after place. Equal to match time when accepted, "
        "or to rider patience when churned. Null when placed = 0."
    ),
    "eta_minutes": (
        "Passenger-facing pickup ETA (how long until the car arrives once "
        "matched). Typically under ~10 minutes; longer values are the shortage "
        "tail. Not the driver's individual routing time from their GPS."
    ),
    "placed": "1 if the rider placed the order after seeing the quote, else 0.",
    "accepted": (
        "1 if a driver accepted before the rider gave up searching. "
        "Always 0 when placed = 0; mutually exclusive with churned_to_competitor."
    ),
    "estimated_duration_minutes": "Predicted length of the trip itself. Not the ETA.",
    "surge_multiplier": "Price multiplier, 1.0 = no surge, capped at 2.5.",
    "demand_supply_ratio": "demand_count / max(available_drivers, 1) for that zone-interval. Above 1.0 means drivers are scarce.",
    "final_price": "base_price * surge_multiplier, in UAH.",
    "is_peak_hour": "True on weekdays at 07:00-09:59 and 17:00-19:59.",
    "special_event": "True when a concert, match or conference was inflating demand in that zone.",
    "demand_count": "Calculated rides requested in that zone-interval.",
    "available_drivers": "Drivers available in that zone-interval.",
    "curfew": (
        "True between 00:00 and 05:00, when civilian movement is prohibited. "
        "Demand falls by roughly 96%, so overnight rows are a handful of "
        "exceptional trips rather than a normal quiet period."
    ),
    "air_raid_alert": (
        "True during a city-wide air raid alert. Demand dips only mildly "
        "(ride-hailing partly substitutes for stopped surface transit), while "
        "driver availability falls further, so alerts still raise surge and ETA."
    ),
}

METRIC_DEFINITIONS = {
    "place conversion": {
        "formula": "SUM(placed) / COUNT(*)",
        "meaning": "Share of price calculations that became orders.",
        "denominator": "all calculated rides",
    },
    "acceptance rate": {
        "formula": "SUM(accepted) / NULLIF(SUM(placed), 0)",
        "meaning": "Share of placed orders a driver accepted before the rider gave up.",
        "denominator": "placed rides only, never all rides",
    },
    "competitor churn rate": {
        "formula": "SUM(churned_to_competitor) / NULLIF(SUM(placed), 0)",
        "meaning": (
            "Share of placed orders lost when search outran rider patience "
            "(~2-5 min). Proxy for switching to a competitor / abandoning "
            "instead of raising the price slider."
        ),
        "denominator": "placed rides only",
    },
    "end-to-end conversion": {
        "formula": "SUM(accepted) / COUNT(*)",
        "meaning": "Share of price calculations that ended with an accepted order.",
        "denominator": "all calculated rides",
    },
    "supply gap": {
        "formula": "demand_count - available_drivers",
        "meaning": "Absolute shortage of drivers in a zone-interval.",
        "denominator": "n/a",
    },
}

DOMAIN_CONTEXT = """
This market operates under wartime conditions, which dominate the daily
profile and must be accounted for in any answer about time of day:

  - A curfew runs 00:00-05:00. Demand collapses to about 4% of normal rather
    than merely dipping. Averaging "by hour" without excluding or flagging
    curfew hours will understate every overnight metric and make 05:00 look
    like an implausible spike. When asked about quiet or busy hours, say
    explicitly that the overnight trough is a curfew, not consumer behaviour.
  - The 1.5 hours before curfew carry a rush of riders getting home in time,
    while drivers are already going offline. This is when surge peaks for the
    day, at around 23:00 - not during the evening commute, which is the
    intuitive but wrong answer.
  - Air raid alerts occur several times a week and cover the whole city.
    Surface public transport stops; ride-hailing demand therefore dips only
    mildly (substitution). Modelled available driver supply falls further
    (app pause / fewer accepts / harder routing — not "everyone in a shelter").
    Alerts remain a major source of short-term surge, ETA and search-churn.
  - After place, riders search for a car. If search exceeds ~2-5 minutes they
    churn to a competitor (churned_to_competitor). The passenger price slider
    is not modelled; churn stands in for that exit.
  - Civil aviation is suspended, so there are no airport zones. Railway
    stations are the intercity gateways and behave the way airports would in a
    peacetime city.
"""

ANALYSIS_CAVEATS = [
    "surge_multiplier and eta_minutes are both consequences of the same driver "
    "shortage and correlate at about r = 0.82. Comparing acceptance across surge "
    "buckets without controlling for ETA is therefore misleading: the raw "
    "comparison looks flat, while the effect of surge on acceptance is positive "
    "once ETA is held fixed. Use the eta_conversion_analysis tool for this rather "
    "than a GROUP BY.",
    "Every relationship in this dataset is an association measured on simulated "
    "observational data. None of it identifies a causal effect.",
    "rides rows only exist where demand occurred, so questions about driver "
    "availability or idle supply must use zone_state, not rides.",
]

EXAMPLE_QUERIES = [
    {
        "question": "What is the average ETA?",
        "sql": "SELECT ROUND(AVG(eta_minutes)::numeric, 2) AS average_eta_minutes\nFROM rides_enriched",
    },
    {
        "question": "Which zones have the highest ETA?",
        "sql": (
            "SELECT origin_zone_name,\n"
            "       ROUND(AVG(eta_minutes)::numeric, 2) AS average_eta_minutes,\n"
            "       COUNT(*) AS calculated_rides\n"
            "FROM rides_enriched\n"
            "GROUP BY origin_zone_name\n"
            "HAVING COUNT(*) >= 100\n"
            "ORDER BY average_eta_minutes DESC\n"
            "LIMIT 10"
        ),
    },
    {
        "question": "What is the funnel by hour of day?",
        "sql": (
            "-- curfew is carried through so the overnight collapse is labelled\n"
            "-- rather than presented as ordinary low demand\n"
            "SELECT hour,\n"
            "       BOOL_OR(curfew) AS is_curfew_hour,\n"
            "       COUNT(*) AS calculated,\n"
            "       SUM(placed) AS placed,\n"
            "       SUM(accepted) AS accepted,\n"
            "       ROUND(AVG(placed)::numeric, 4) AS place_conversion,\n"
            "       ROUND((SUM(accepted)::numeric / NULLIF(SUM(placed), 0)), 4) AS acceptance_rate\n"
            "FROM rides_enriched\n"
            "GROUP BY hour\n"
            "ORDER BY hour"
        ),
    },
    {
        "question": "What do air raid alerts do to the marketplace?",
        "sql": (
            "SELECT air_raid_alert,\n"
            "       COUNT(*) AS zone_intervals,\n"
            "       ROUND(AVG(demand_count)::numeric, 2) AS avg_demand,\n"
            "       ROUND(AVG(available_drivers)::numeric, 2) AS avg_drivers,\n"
            "       ROUND(AVG(surge_multiplier)::numeric, 3) AS avg_surge,\n"
            "       ROUND(AVG(average_eta_minutes)::numeric, 2) AS avg_eta\n"
            "FROM zone_state\n"
            "WHERE NOT curfew\n"
            "GROUP BY air_raid_alert"
        ),
    },
    {
        "question": "When does surge peak during the day?",
        "sql": (
            "SELECT hour,\n"
            "       BOOL_OR(curfew) AS is_curfew_hour,\n"
            "       ROUND(AVG(surge_multiplier)::numeric, 3) AS avg_surge,\n"
            "       ROUND(AVG(demand_count)::numeric, 2) AS avg_demand,\n"
            "       ROUND(AVG(available_drivers)::numeric, 2) AS avg_drivers\n"
            "FROM zone_state\n"
            "GROUP BY hour\n"
            "ORDER BY avg_surge DESC"
        ),
    },
    {
        "question": "Where is the driver shortage worst?",
        "sql": (
            "SELECT z.zone_name,\n"
            "       ROUND(AVG(s.demand_supply_ratio)::numeric, 3) AS avg_demand_supply_ratio,\n"
            "       ROUND(AVG(s.supply_gap)::numeric, 2) AS avg_supply_gap,\n"
            "       ROUND(AVG(s.surge_multiplier)::numeric, 3) AS avg_surge\n"
            "FROM zone_state s\n"
            "JOIN zones z ON z.zone_id = s.zone_id\n"
            "GROUP BY z.zone_name\n"
            "ORDER BY avg_supply_gap DESC\n"
            "LIMIT 10"
        ),
    },
]


@lru_cache(maxsize=1)
def introspect_columns() -> dict[str, list[dict[str, str]]]:
    """Live column listing per relation, straight from the catalog."""
    query = text(
        """
        SELECT table_name, column_name, data_type, ordinal_position
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name IN ('rides_enriched', 'rides', 'zone_state', 'zones', 'users')
        ORDER BY table_name, ordinal_position
        """
    )

    with get_engine(readonly=True).connect() as connection:
        records = connection.execute(query).fetchall()

    relations: dict[str, list[dict[str, str]]] = {}
    for table_name, column_name, data_type, _ in records:
        relations.setdefault(table_name, []).append(
            {
                "name": column_name,
                "type": data_type,
                "note": COLUMN_NOTES.get(column_name, ""),
            }
        )

    return relations


@lru_cache(maxsize=1)
def dataset_totals() -> dict[str, Any]:
    """Row counts and the covered date range, so the model can size its answers."""
    query = text(
        """
        SELECT
            (SELECT COUNT(*) FROM rides)                  AS calculated_rides,
            (SELECT SUM(placed)   FROM rides)             AS placed_rides,
            (SELECT SUM(accepted) FROM rides)             AS accepted_rides,
            (SELECT COUNT(*) FROM zone_state)             AS zone_intervals,
            (SELECT COUNT(*) FROM zones)                  AS zones,
            (SELECT COUNT(*) FROM users)                  AS users,
            (SELECT MIN(calculated_at) FROM rides)        AS first_ride_at,
            (SELECT MAX(calculated_at) FROM rides)        AS last_ride_at
        """
    )

    with get_engine(readonly=True).connect() as connection:
        row = connection.execute(query).mappings().one()

    totals = dict(row)
    totals["first_ride_at"] = str(totals["first_ride_at"])
    totals["last_ride_at"] = str(totals["last_ride_at"])

    return totals


def get_schema_description() -> dict[str, Any]:
    relations = introspect_columns()

    return {
        "totals": dataset_totals(),
        "domain_context": DOMAIN_CONTEXT.strip(),
        "relations": [
            {
                "name": name,
                "purpose": RELATION_PURPOSE.get(name, ""),
                "columns": columns,
            }
            for name, columns in sorted(
                relations.items(),
                key=lambda item: list(RELATION_PURPOSE).index(item[0])
                if item[0] in RELATION_PURPOSE
                else 99,
            )
        ],
        "metric_definitions": METRIC_DEFINITIONS,
        "caveats": ANALYSIS_CAVEATS,
        "example_queries": EXAMPLE_QUERIES,
    }


def render_schema_for_prompt() -> str:
    """Compact text rendering, which tokenises better than nested JSON."""
    description = get_schema_description()
    totals = description["totals"]

    lines = [
        "DATASET: synthetic ride-hailing marketplace (PostgreSQL)",
        f"  {totals['calculated_rides']:,} calculated rides, "
        f"{totals['placed_rides']:,} placed, {totals['accepted_rides']:,} accepted",
        f"  {totals['zones']} zones, {totals['users']:,} users, "
        f"{totals['zone_intervals']:,} zone-intervals",
        f"  covering {totals['first_ride_at']} to {totals['last_ride_at']}",
        "",
        "OPERATING CONTEXT",
        DOMAIN_CONTEXT.strip(),
        "",
        "RELATIONS",
    ]

    for relation in description["relations"]:
        lines.append(f"\n{relation['name']}")
        if relation["purpose"]:
            lines.append(f"  {relation['purpose']}")
        for column in relation["columns"]:
            note = f"  -- {column['note']}" if column["note"] else ""
            lines.append(f"    {column['name']:<28} {column['type']}{note}")

    lines.append("\nMETRIC DEFINITIONS")
    for name, definition in METRIC_DEFINITIONS.items():
        lines.append(f"  {name}: {definition['formula']}")
        lines.append(
            f"    {definition['meaning']} Denominator: {definition['denominator']}."
        )

    lines.append("\nANALYSIS CAVEATS")
    for caveat in ANALYSIS_CAVEATS:
        lines.append(f"  - {caveat}")

    lines.append("\nEXAMPLE QUERIES")
    for example in EXAMPLE_QUERIES:
        lines.append(f"\n  -- {example['question']}")
        for sql_line in example["sql"].splitlines():
            lines.append(f"  {sql_line}")

    return "\n".join(lines)
