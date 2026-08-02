"""
Clustering of zones and of users.

Two design decisions worth stating:

*   `k` is chosen by silhouette score over a small range rather than fixed,
    because "KMeans produced four clusters" is not an insight if four was
    hard-coded.
*   Clusters are described in domain terms derived from their standardised
    profile, so the answer reads as "underserved outer zones" rather than
    "cluster 2". The description is generated from the feature z-scores, not
    written by the model, so it cannot drift from the numbers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from src.analytics.numerics import quiet_linear_algebra
from src.database.connection import get_engine

RANDOM_STATE = 42

ZONE_FEATURE_QUERY = """
    WITH zone_rides AS (
        SELECT
            origin_zone_id                                            AS zone_id,
            AVG(placed::double precision)                             AS place_conversion,
            SUM(accepted)::double precision / NULLIF(SUM(placed), 0)  AS acceptance_rate,
            AVG(final_price)                                          AS average_price,
            AVG(distance_km)                                          AS average_distance_km,
            AVG(CASE WHEN is_peak_hour THEN 1.0 ELSE 0.0 END)         AS peak_hour_share,
            AVG(CASE WHEN is_weekend  THEN 1.0 ELSE 0.0 END)          AS weekend_share
        FROM rides_enriched
        GROUP BY origin_zone_id
    ),
    zone_market AS (
        SELECT
            zone_id,
            AVG(demand_count)        AS average_demand,
            AVG(available_drivers)   AS average_available_drivers,
            AVG(demand_supply_ratio) AS average_demand_supply_ratio,
            AVG(surge_multiplier)    AS average_surge,
            AVG(average_eta_minutes) AS average_eta_minutes
        FROM zone_state
        GROUP BY zone_id
    )
    SELECT
        z.zone_id,
        z.zone_name,
        z.zone_type,
        z.distance_from_center_km,
        m.average_demand,
        m.average_available_drivers,
        m.average_demand_supply_ratio,
        m.average_surge,
        m.average_eta_minutes,
        r.place_conversion,
        r.acceptance_rate,
        r.average_price,
        r.average_distance_km,
        r.peak_hour_share,
        r.weekend_share
    FROM zones z
    JOIN zone_market m ON m.zone_id = z.zone_id
    JOIN zone_rides  r ON r.zone_id = z.zone_id
"""

ZONE_FEATURES = [
    "average_demand",
    "average_available_drivers",
    "average_demand_supply_ratio",
    "average_surge",
    "average_eta_minutes",
    "place_conversion",
    "acceptance_rate",
    "average_price",
    "average_distance_km",
    "peak_hour_share",
    "weekend_share",
]

# Two of these features are behavioural *responses* rather than averages:
# surge_response and eta_response measure how a rider's own ordering rate shifts
# between cheap and expensive conditions. They are what separates a price
# sensitive rider from one who simply happens to travel at busy times, which
# averages alone cannot distinguish.
USER_FEATURE_QUERY = """
    SELECT
        user_id,
        COUNT(*)                                                   AS calculated_rides,
        LN(COUNT(*)::double precision)                             AS log_calculated_rides,
        AVG(placed::double precision)                              AS place_conversion,
        AVG(final_price)                                           AS average_price,
        AVG(distance_km)                                           AS average_distance_km,
        AVG(surge_multiplier)                                      AS average_surge_seen,
        AVG(eta_minutes)                                           AS average_eta_seen,
        AVG(CASE WHEN hour BETWEEN 6 AND 10 THEN 1.0 ELSE 0.0 END) AS morning_share,
        AVG(CASE WHEN hour >= 21 OR hour <= 2 THEN 1.0 ELSE 0.0 END) AS night_share,
        AVG(CASE WHEN is_weekend THEN 1.0 ELSE 0.0 END)            AS weekend_share,
        AVG(CASE WHEN destination_zone_type = 'railway_station'
                   OR origin_zone_type = 'railway_station'
                 THEN 1.0 ELSE 0.0 END)                            AS rail_trip_share,
        -- Where a rider starts their trips is one of the strongest available
        -- signals, and unlike the response features it is measured from every
        -- ride rather than a subset, so it carries far less sampling noise.
        AVG(CASE WHEN origin_zone_type = 'city_center'   THEN 1.0 ELSE 0.0 END) AS origin_city_center_share,
        AVG(CASE WHEN origin_zone_type = 'residential'   THEN 1.0 ELSE 0.0 END) AS origin_residential_share,
        AVG(CASE WHEN origin_zone_type = 'business'      THEN 1.0 ELSE 0.0 END) AS origin_business_share,
        AVG(CASE WHEN origin_zone_type = 'entertainment' THEN 1.0 ELSE 0.0 END) AS origin_entertainment_share,
        AVG(CASE WHEN origin_zone_type = 'suburban'      THEN 1.0 ELSE 0.0 END) AS origin_suburban_share,
        AVG(placed::double precision) FILTER (WHERE surge_multiplier >= 1.2)
            - AVG(placed::double precision) FILTER (WHERE surge_multiplier < 1.2)
                                                                   AS surge_response,
        AVG(placed::double precision) FILTER (WHERE eta_minutes >= 10)
            - AVG(placed::double precision) FILTER (WHERE eta_minutes < 10)
                                                                   AS eta_response
    FROM rides_enriched
    GROUP BY user_id
    HAVING COUNT(*) >= :minimum_rides
       -- Both response features need enough rides on each side of the split to
       -- be anything other than noise.
       AND COUNT(*) FILTER (WHERE surge_multiplier >= 1.2) >= 5
       AND COUNT(*) FILTER (WHERE surge_multiplier <  1.2) >= 5
       AND COUNT(*) FILTER (WHERE eta_minutes >= 10) >= 5
       AND COUNT(*) FILTER (WHERE eta_minutes <  10) >= 5
"""

USER_FEATURES = [
    "log_calculated_rides",
    "place_conversion",
    "average_price",
    "average_distance_km",
    "average_surge_seen",
    "average_eta_seen",
    "morning_share",
    "night_share",
    "weekend_share",
    "rail_trip_share",
    "origin_city_center_share",
    "origin_residential_share",
    "origin_business_share",
    "origin_entertainment_share",
    "origin_suburban_share",
    "surge_response",
    "eta_response",
]

# Wording used when a cluster sits far from the overall mean on a feature.
FEATURE_PHRASES = {
    "average_demand": ("high demand", "low demand"),
    "average_available_drivers": ("many drivers", "few drivers"),
    "average_demand_supply_ratio": ("driver shortage", "driver surplus"),
    "average_surge": ("high surge", "little surge"),
    "average_eta_minutes": ("long waits", "short waits"),
    "place_conversion": ("strong place conversion", "weak place conversion"),
    "acceptance_rate": ("high acceptance", "low acceptance"),
    "average_price": ("expensive trips", "cheap trips"),
    "average_distance_km": ("long trips", "short trips"),
    "peak_hour_share": ("commute-heavy", "little commute traffic"),
    "weekend_share": ("weekend-heavy", "weekday-heavy"),
    "calculated_rides": ("very active", "occasional"),
    "log_calculated_rides": ("very active", "occasional"),
    "average_surge_seen": ("travels when surge is on", "travels in quiet conditions"),
    "average_eta_seen": ("faces long waits", "faces short waits"),
    "morning_share": ("morning traveller", "rarely travels in the morning"),
    "night_share": ("night traveller", "rarely travels at night"),
    "rail_trip_share": ("frequent station trips", "rarely travels via a station"),
    "surge_response": ("keeps ordering when surge rises", "price sensitive"),
    "eta_response": ("orders despite long waits", "impatient about waits"),
    "origin_city_center_share": ("starts trips downtown", "rarely starts downtown"),
    "origin_residential_share": ("starts trips at home", "rarely starts in housing areas"),
    "origin_business_share": ("starts trips in business districts", "avoids business districts"),
    "origin_entertainment_share": ("starts trips in nightlife areas", "avoids nightlife areas"),
    "origin_suburban_share": ("suburban rider", "not a suburban rider"),
}


def _choose_cluster_count(
    scaled_features: np.ndarray,
    candidate_range: range,
    maximum_cluster_share: float = 0.45,
) -> tuple[int, list[dict[str, float]]]:
    """
    Pick k by silhouette score, rejecting degenerate splits.

    Silhouette on its own is unusable here. Rider behaviour is one dense blob
    with segments layered inside it, so the metric is maximised by k=2, which
    shaves off a small extreme group and leaves 88% of riders in a single
    cluster whose profile is, by construction, the population average. That
    scores well geometrically and is worthless commercially: nobody can act on
    a segment that contains almost everyone.

    So a solution is only eligible if no cluster holds more than
    `maximum_cluster_share` of the population, and the best silhouette among
    the eligible solutions wins. The full score table is returned either way,
    including the rejected candidates, so the choice can be audited.
    """
    scores: list[dict[str, float]] = []

    for k in candidate_range:
        if k >= len(scaled_features):
            break

        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)

        with quiet_linear_algebra():
            labels = model.fit_predict(scaled_features)
            score = float(silhouette_score(scaled_features, labels))

        largest_share = float(np.bincount(labels).max()) / len(labels)

        scores.append(
            {
                "k": k,
                "silhouette_score": round(score, 4),
                "largest_cluster_share": round(largest_share, 4),
                "eligible": largest_share <= maximum_cluster_share,
            }
        )

    eligible = [row for row in scores if row["eligible"]]
    ranked = eligible or scores

    best_k = int(max(ranked, key=lambda row: row["silhouette_score"])["k"])

    return best_k, scores


def _describe_clusters(
    frame: pd.DataFrame,
    feature_names: list[str],
    threshold: float = 0.7,
) -> dict[int, str]:
    """
    Turn each cluster's standardised profile into a phrase.

    Features are ranked by how far the cluster sits from the overall mean, and
    the top few above the threshold become the description.
    """
    standardised = (
        frame[feature_names] - frame[feature_names].mean()
    ) / frame[feature_names].std(ddof=0).replace(0, np.nan)
    standardised["cluster"] = frame["cluster"].to_numpy()

    profiles = standardised.groupby("cluster")[feature_names].mean()
    descriptions: dict[int, str] = {}

    for cluster, row in profiles.iterrows():
        ranked = row.reindex(row.abs().sort_values(ascending=False).index)
        phrases = []

        for feature, z_score in ranked.items():
            if abs(z_score) < threshold or len(phrases) >= 3:
                continue
            if feature not in FEATURE_PHRASES:
                continue
            high_phrase, low_phrase = FEATURE_PHRASES[feature]
            phrases.append(high_phrase if z_score > 0 else low_phrase)

        descriptions[int(cluster)] = (
            ", ".join(phrases) if phrases else "close to the average on every feature"
        )

    return descriptions


def _run_kmeans(
    frame: pd.DataFrame,
    feature_names: list[str],
    number_of_clusters: int | None,
    candidate_range: range,
) -> dict[str, Any]:
    features = frame[feature_names].astype(float).fillna(0.0).to_numpy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    silhouette_table: list[dict[str, float]] = []
    if number_of_clusters is None:
        number_of_clusters, silhouette_table = _choose_cluster_count(
            scaled, candidate_range
        )
        selection = "chosen automatically by silhouette score"
    else:
        selection = "supplied by the caller"

    model = KMeans(n_clusters=number_of_clusters, random_state=RANDOM_STATE, n_init=10)
    frame = frame.copy()
    projection = PCA(n_components=2, random_state=RANDOM_STATE)

    with quiet_linear_algebra():
        frame["cluster"] = model.fit_predict(scaled)
        final_silhouette = float(silhouette_score(scaled, frame["cluster"].to_numpy()))
        coordinates = projection.fit_transform(scaled)
    frame["pca_x"] = coordinates[:, 0]
    frame["pca_y"] = coordinates[:, 1]

    descriptions = _describe_clusters(frame, feature_names)
    frame["cluster_description"] = frame["cluster"].map(descriptions)

    profile = (
        frame.groupby("cluster")[feature_names].mean().round(3).reset_index()
    )
    profile["members"] = (
        frame.groupby("cluster").size().reindex(profile["cluster"]).to_numpy()
    )
    profile["cluster_description"] = profile["cluster"].map(descriptions)

    # Heatmap of standardised profiles, so features on different scales are
    # comparable in one picture.
    standardised_profile = (
        (
            frame.groupby("cluster")[feature_names].mean()
            - frame[feature_names].mean()
        )
        / frame[feature_names].std(ddof=0).replace(0, np.nan)
    ).round(3)

    return {
        "number_of_clusters": int(number_of_clusters),
        "cluster_count_selection": selection,
        "silhouette_score": round(final_silhouette, 4),
        "silhouette_by_k": silhouette_table,
        "explained_variance_ratio": [
            round(float(value), 4) for value in projection.explained_variance_ratio_
        ],
        "features_used": feature_names,
        "cluster_profiles": profile.to_dict(orient="records"),
        "standardised_profile": {
            "clusters": [int(value) for value in standardised_profile.index],
            "features": feature_names,
            "z_scores": standardised_profile.fillna(0.0).values.tolist(),
        },
        "frame": frame,
    }


def cluster_zones(number_of_clusters: int | None = None) -> dict[str, Any]:
    """Segment the 20 zones by their marketplace and funnel behaviour."""
    with get_engine(readonly=True).connect() as connection:
        frame = pd.read_sql(text(ZONE_FEATURE_QUERY), connection)

    result = _run_kmeans(frame, ZONE_FEATURES, number_of_clusters, range(2, 7))
    clustered = result.pop("frame")

    assignments = clustered[
        ["zone_id", "zone_name", "zone_type", "cluster", "cluster_description"]
    ].sort_values(["cluster", "zone_name"])

    return {
        "entity": "zones",
        "members": int(len(clustered)),
        **result,
        "assignments": assignments.to_dict(orient="records"),
        "chart": {
            "kind": "cluster_scatter",
            "title": "Zones projected onto their first two principal components",
            "x": clustered["pca_x"].round(3).tolist(),
            "y": clustered["pca_y"].round(3).tolist(),
            "labels": clustered["zone_name"].tolist(),
            "clusters": clustered["cluster"].astype(int).tolist(),
            "cluster_descriptions": {
                str(cluster): description
                for cluster, description in zip(
                    clustered["cluster"],
                    clustered["cluster_description"],
                    strict=True,
                )
            },
            "x_title": (
                f"PC1 ({result['explained_variance_ratio'][0]:.0%} of variance)"
            ),
            "y_title": (
                f"PC2 ({result['explained_variance_ratio'][1]:.0%} of variance)"
            ),
        },
        "profile_chart": {
            "kind": "heatmap",
            "title": "Cluster profiles (standard deviations from the overall mean)",
            "x": result["standardised_profile"]["features"],
            "y": [
                f"cluster {cluster}"
                for cluster in result["standardised_profile"]["clusters"]
            ],
            "z": result["standardised_profile"]["z_scores"],
            "x_title": "feature",
            "y_title": "cluster",
            "color_title": "z-score",
        },
        "caveats": [
            "KMeans on standardised features. k is chosen by silhouette score "
            "among solutions where no cluster holds more than 45% of the "
            "population, which rules out splits that leave most entities in a "
            "single average cluster.",
            "Zone type was not used as an input, so any alignment between the "
            "clusters and the zone types is a result rather than an assumption.",
        ],
    }


def cluster_users(
    number_of_clusters: int | None = None,
    minimum_rides: int = 30,
    sample_size: int = 6000,
) -> dict[str, Any]:
    """
    Segment riders by observed behaviour.

    The latent traits that generated these users are not in the database, so
    this is a genuine recovery problem rather than a lookup.
    """
    with get_engine(readonly=True).connect() as connection:
        frame = pd.read_sql(
            text(USER_FEATURE_QUERY),
            connection,
            params={"minimum_rides": minimum_rides},
        )

    if frame.empty:
        raise ValueError(
            f"No users have at least {minimum_rides} calculated rides. "
            "Lower minimum_rides."
        )

    population_size = len(frame)
    if population_size > sample_size:
        frame = frame.sample(sample_size, random_state=RANDOM_STATE).reset_index(
            drop=True
        )

    result = _run_kmeans(frame, USER_FEATURES, number_of_clusters, range(2, 9))
    clustered = result.pop("frame")

    size_table = (
        clustered.groupby(["cluster", "cluster_description"])
        .size()
        .reset_index(name="users")
        .sort_values("cluster")
    )
    size_table["share_of_clustered_users"] = (
        size_table["users"] / len(clustered)
    ).round(4)

    return {
        "entity": "users",
        "members": int(len(clustered)),
        "population_matching_filter": int(population_size),
        "minimum_rides": minimum_rides,
        **result,
        "assignments": size_table.to_dict(orient="records"),
        "chart": {
            "kind": "cluster_scatter",
            "title": "Riders projected onto their first two principal components",
            "x": clustered["pca_x"].round(2).tolist(),
            "y": clustered["pca_y"].round(2).tolist(),
            # No per-point labels: a rider id means nothing on hover, and
            # thousands of them tripled the size of this response.
            "labels": None,
            "clusters": clustered["cluster"].astype(int).tolist(),
            "cluster_descriptions": {
                str(cluster): description
                for cluster, description in zip(
                    clustered["cluster"],
                    clustered["cluster_description"],
                    strict=True,
                )
            },
            "x_title": (
                f"PC1 ({result['explained_variance_ratio'][0]:.0%} of variance)"
            ),
            "y_title": (
                f"PC2 ({result['explained_variance_ratio'][1]:.0%} of variance)"
            ),
        },
        "profile_chart": {
            "kind": "heatmap",
            "title": "Rider cluster profiles (standard deviations from the mean)",
            "x": result["standardised_profile"]["features"],
            "y": [
                f"cluster {cluster}"
                for cluster in result["standardised_profile"]["clusters"]
            ],
            "z": result["standardised_profile"]["z_scores"],
            "x_title": "feature",
            "y_title": "cluster",
            "color_title": "z-score",
        },
        "caveats": [
            f"Only riders with at least {minimum_rides} calculated rides are "
            "included, so these are not the whole user base.",
            "Behavioural features only. The simulation's latent user traits are "
            "not stored in the database and were not used.",
            "k is chosen by silhouette score among solutions where no cluster "
            "holds more than 45% of riders. Silhouette alone prefers k=2 here, "
            "which is geometrically clean but commercially useless: it leaves "
            "88% of riders in one cluster sitting at the population average.",
            "Rider segments overlap far more than zone types do. Treat these "
            "clusters as a description of behaviour, not as sharply separated "
            "groups.",
        ],
    }
