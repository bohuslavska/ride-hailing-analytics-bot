"""
Cached analytics entry points shared by the REST API and the agent's tools.

Keeping one path means a question answered through chat warms the cache for the
same question asked as a REST call, and the other way around. `run_sql` is
deliberately absent: arbitrary SQL has an unbounded key space and is already
bounded by statement timeouts.
"""

from __future__ import annotations

from typing import Any

from src.analytics.clustering import cluster_users, cluster_zones
from src.analytics.conversion import analyze_conversion, compare_acceptance_confounding
from src.analytics.metrics import (
    calculate_funnel,
    marketplace_profile,
    zone_supply_demand_summary,
)
from src.analytics.schema_description import get_schema_description
from src.cache import remember


def cached_schema() -> dict[str, Any]:
    return remember("schema", {}, get_schema_description)


def cached_funnel(*, dimension: str | None = None) -> dict[str, Any]:
    return remember(
        "funnel",
        {"dimension": dimension},
        lambda: calculate_funnel(dimension=dimension),
    )


def cached_marketplace_profile(
    *, dimension: str = "hour", zone_type: str | None = None
) -> dict[str, Any]:
    return remember(
        "marketplace_profile",
        {"dimension": dimension, "zone_type": zone_type},
        lambda: marketplace_profile(dimension=dimension, zone_type=zone_type),
    )


def cached_zone_supply_demand(*, limit: int = 20) -> dict[str, Any]:
    return remember(
        "zone_supply_demand",
        {"limit": limit},
        lambda: zone_supply_demand_summary(limit=limit),
    )


def cached_conversion(
    *, driver: str = "eta_minutes", outcome: str = "placed"
) -> dict[str, Any]:
    return remember(
        "conversion",
        {"driver": driver, "outcome": outcome},
        lambda: analyze_conversion(driver=driver, outcome=outcome),
    )


def cached_acceptance_confounding() -> dict[str, Any]:
    return remember("acceptance_confounding", {}, compare_acceptance_confounding)


def cached_zone_clusters(*, number_of_clusters: int | None = None) -> dict[str, Any]:
    return remember(
        "zone_clusters",
        {"number_of_clusters": number_of_clusters},
        lambda: cluster_zones(number_of_clusters=number_of_clusters),
    )


def cached_rider_clusters(
    *, number_of_clusters: int | None = None, minimum_rides: int = 30
) -> dict[str, Any]:
    return remember(
        "rider_clusters",
        {
            "number_of_clusters": number_of_clusters,
            "minimum_rides": minimum_rides,
        },
        lambda: cluster_users(
            number_of_clusters=number_of_clusters, minimum_rides=minimum_rides
        ),
    )
