"""
Expand the marketplace state into individual ride records and simulate the
calculated -> placed -> (accepted | churned_to_competitor) funnel.

Every row is one price calculation. The funnel is generated sequentially rather
than by assigning a random status: a rider decides whether to place the order
given what they were quoted; the app then searches for a car; if the search
outruns the rider's patience they leave for a competitor, otherwise a willing
driver may accept. That ordering is what makes the funnel inequalities hold by
construction, and it is why surge can push place and accept in opposite
directions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_generation.generate_zones import build_distance_matrix
from src.data_generation.simulation_config import (
    SimulationConfig,
    build_hour_band_lookup,
)

RIDE_COLUMNS = [
    "ride_id",
    "user_id",
    "calculated_at",
    "origin_zone_id",
    "destination_zone_id",
    "distance_km",
    "estimated_duration_minutes",
    "eta_minutes",
    "demand_count",
    "available_drivers",
    "demand_supply_ratio",
    "traffic_index",
    "weather",
    "is_peak_hour",
    "is_weekend",
    "curfew",
    "air_raid_alert",
    "special_event",
    "base_price",
    "surge_multiplier",
    "final_price",
    "placed",
    "accepted",
    "churned_to_competitor",
    "search_wait_minutes",
    "final_status",
    "placed_at",
    "accepted_at",
]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _sample_from_weights(
    cumulative_weights: np.ndarray, rng: np.random.Generator, size: int
) -> np.ndarray:
    """
    Draw indices proportional to weights via inverse-CDF sampling.

    `rng.choice(..., p=...)` rebuilds its lookup on every call, which is far too
    slow for hundreds of thousands of draws.
    """
    return np.searchsorted(cumulative_weights, rng.random(size), side="right")


def _build_destination_probabilities(
    config: SimulationConfig,
    zones: pd.DataFrame,
    distance_matrix: np.ndarray,
) -> np.ndarray:
    """
    Gravity model of shape (origins, hour_bands, destinations), stored as a
    cumulative distribution along the destination axis.

    Pull towards a destination grows with how attractive that zone is at that
    time of day and decays with distance. The decay exponent is per zone type,
    which is what keeps the railway stations reachable from the whole city
    while a suburb only draws from its neighbours.
    """
    zone_type_names = zones["zone_type"].to_numpy()
    band_names = list(config.hour_bands)
    number_of_zones = len(zones)

    attractiveness = np.array(
        [
            [
                config.zone_types[zone_type]["destination_attractiveness"][band]
                for band in band_names
            ]
            for zone_type in zone_type_names
        ],
        dtype=float,
    )  # (destinations, bands)

    decay_exponent = np.array(
        [
            config.zone_types[zone_type]["distance_decay_exponent"]
            for zone_type in zone_type_names
        ],
        dtype=float,
    )  # (destinations,)

    distance_penalty = (1.0 + distance_matrix) ** decay_exponent[None, :]

    weights = (
        attractiveness.T[None, :, :] / distance_penalty[:, None, :]
    )  # (origins, bands, destinations)

    # Trips that start and end in the same zone are real but less common than
    # the raw gravity weight would suggest.
    diagonal = np.arange(number_of_zones)
    weights[diagonal, :, diagonal] *= 0.35

    weights /= weights.sum(axis=2, keepdims=True)

    return np.cumsum(weights, axis=2)


def _sample_users(
    config: SimulationConfig,
    users: pd.DataFrame,
    zones: pd.DataFrame,
    origin_zone_position: np.ndarray,
    band_index: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Pick the requesting user for every ride.

    Three things decide who requests a ride: how active they are, whether they
    live in the origin zone, and whether the ride falls in their preferred time
    band. The last of these is what makes rider segments visible in behaviour --
    without it a "night owl" would be indistinguishable from a commuter in the
    data, since both would appear uniformly across the clock.

    Rides are grouped by (zone, time band) and sampled group by group, so the
    weights only have to be built once per group rather than once per ride.
    """
    user_settings = config.simulation["users"]
    home_zone_share = float(user_settings["home_zone_share"])
    preferred_multiplier = float(user_settings["preferred_band_multiplier"])
    other_multiplier = float(user_settings["other_band_multiplier"])

    activity_weight = users["activity_weight"].to_numpy()
    band_names = list(config.hour_bands)
    number_of_bands = len(band_names)

    band_name_to_index = {name: index for index, name in enumerate(band_names)}
    preferred_band = (
        users["preferred_hour_band"].map(band_name_to_index).to_numpy()
    )

    zone_id_to_position = {
        zone_id: position for position, zone_id in enumerate(zones["zone_id"])
    }
    home_zone_position = users["home_zone_id"].map(zone_id_to_position).to_numpy()

    number_of_rides = origin_zone_position.size
    number_of_zones = len(zones)
    user_position = np.empty(number_of_rides, dtype=np.int64)
    use_home_zone = rng.random(number_of_rides) < home_zone_share

    # Per-band weights, and the residents of each zone, precomputed once.
    band_weights = [
        activity_weight
        * np.where(preferred_band == band, preferred_multiplier, other_multiplier)
        for band in range(number_of_bands)
    ]
    band_cumulative = [
        np.cumsum(weights / weights.sum()) for weights in band_weights
    ]
    residents_by_zone = [
        np.flatnonzero(home_zone_position == position)
        for position in range(number_of_zones)
    ]

    # Group rides by (zone, band) with one sort rather than a mask per group.
    group_key = origin_zone_position * number_of_bands + band_index
    order = np.argsort(group_key, kind="stable")
    group_starts = np.searchsorted(
        group_key[order], np.arange(number_of_zones * number_of_bands + 1)
    )

    for group in range(number_of_zones * number_of_bands):
        rides_here = order[group_starts[group] : group_starts[group + 1]]

        if rides_here.size == 0:
            continue

        zone = group // number_of_bands
        band = group % number_of_bands
        residents = residents_by_zone[zone]

        local_rides = rides_here[use_home_zone[rides_here]]
        visitor_rides = rides_here[~use_home_zone[rides_here]]

        if residents.size > 0 and local_rides.size > 0:
            resident_weights = band_weights[band][residents]
            resident_cumulative = np.cumsum(
                resident_weights / resident_weights.sum()
            )
            user_position[local_rides] = residents[
                _sample_from_weights(resident_cumulative, rng, local_rides.size)
            ]
        else:
            visitor_rides = rides_here

        if visitor_rides.size > 0:
            user_position[visitor_rides] = _sample_from_weights(
                band_cumulative[band], rng, visitor_rides.size
            )

    return np.clip(user_position, 0, len(users) - 1)


def generate_rides(
    config: SimulationConfig,
    zones: pd.DataFrame,
    users: pd.DataFrame,
    zone_state: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    zone_id_to_position = {
        zone_id: position for position, zone_id in enumerate(zones["zone_id"])
    }

    distance_matrix = build_distance_matrix(
        zones, float(config.simulation["city"]["km_per_degree_latitude"])
    )
    destination_cumulative = _build_destination_probabilities(
        config, zones, distance_matrix
    )
    hour_band_lookup = build_hour_band_lookup(config.hour_bands)

    # One ride row per calculated request.
    demand_count = zone_state["demand_count"].to_numpy()
    repeat_index = np.repeat(np.arange(len(zone_state)), demand_count)
    number_of_rides = repeat_index.size

    state = zone_state.iloc[repeat_index].reset_index(drop=True)

    origin_zone_position = (
        state["zone_id"].map(zone_id_to_position).to_numpy().astype(np.int64)
    )
    hour = state["hour"].to_numpy()
    band_index = hour_band_lookup[hour]

    # Destination via the gravity model for this origin and time of day.
    # Each ride needs inverse-CDF sampling against its own row of the
    # (origin, band) table, done in chunks so the intermediate stays small.
    destination_draw = rng.random(number_of_rides)
    destination_zone_position = np.empty(number_of_rides, dtype=np.int64)
    chunk_size = 200_000

    for start in range(0, number_of_rides, chunk_size):
        stop = min(start + chunk_size, number_of_rides)
        chunk_cumulative = destination_cumulative[
            origin_zone_position[start:stop], band_index[start:stop]
        ]
        destination_zone_position[start:stop] = (
            chunk_cumulative > destination_draw[start:stop, None]
        ).argmax(axis=1)

    user_position = _sample_users(
        config, users, zones, origin_zone_position, band_index, rng
    )

    # Frequent intercity travellers pull a slice of their trips towards the
    # railway stations regardless of where they happen to be standing, which is
    # what makes that segment visible in the ride data.
    station_positions = np.flatnonzero(
        (zones["zone_type"] == "railway_station").to_numpy()
    )
    if station_positions.size > 0:
        rail_affinity = users["rail_affinity"].to_numpy()[user_position]
        redirect_probability = np.clip(0.03 * rail_affinity, 0.0, 0.15)
        redirect = rng.random(number_of_rides) < redirect_probability
        station_choice = rng.integers(
            0, station_positions.size, size=int(redirect.sum())
        )
        destination_zone_position[redirect] = station_positions[station_choice]

    is_same_zone = origin_zone_position == destination_zone_position
    straight_line_km = distance_matrix[
        origin_zone_position, destination_zone_position
    ]

    # Road distance exceeds straight-line distance; intra-zone trips get a
    # short random length instead.
    road_distance = straight_line_km * rng.uniform(
        1.15, 1.45, size=number_of_rides
    ) + rng.uniform(0.2, 1.0, size=number_of_rides)
    distance_km = np.where(
        is_same_zone,
        rng.uniform(0.8, 3.0, size=number_of_rides),
        road_distance,
    )
    distance_km = np.maximum(distance_km, 0.8)

    speed_config = config.simulation["speed"]
    traffic_index = state["traffic_index"].to_numpy()
    speed_kmh = speed_config["free_flow_kmh"] - (
        speed_config["free_flow_kmh"] - speed_config["congested_kmh"]
    ) * traffic_index
    estimated_duration_minutes = distance_km / speed_kmh * 60.0

    eta_config = config.simulation["eta"]
    eta_minutes = np.clip(
        state["average_eta_minutes"].to_numpy()
        + rng.normal(
            0.0,
            eta_config["ride_level_noise_standard_deviation"],
            size=number_of_rides,
        ),
        eta_config["minimum_minutes"],
        eta_config["maximum_minutes"],
    )

    pricing = config.simulation["pricing"]
    base_price = np.maximum(
        pricing["fixed_fee"]
        + pricing["price_per_km"] * distance_km
        + pricing["price_per_minute"] * estimated_duration_minutes,
        pricing["minimum_fare"],
    )
    surge_multiplier = state["surge_multiplier"].to_numpy()
    final_price = base_price * surge_multiplier

    # ---- Stage 1: does the rider place the order? ----
    place_config = config.simulation["conversion"]["place"]

    price_sensitivity = users["price_sensitivity"].to_numpy()[user_position]
    eta_sensitivity = users["eta_sensitivity"].to_numpy()[user_position]
    base_propensity = users["base_propensity"].to_numpy()[user_position]

    destination_zone_type = zones["zone_type"].to_numpy()[destination_zone_position]
    origin_zone_type = zones["zone_type"].to_numpy()[origin_zone_position]
    # Catching a train is not a discretionary trip, so these convert better
    # than an equivalent journey elsewhere.
    is_rail_trip = (destination_zone_type == "railway_station") | (
        origin_zone_type == "railway_station"
    )

    place_logit = (
        place_config["intercept"]
        + base_propensity
        + place_config["eta_coefficient"]
        * eta_minutes
        * (1.0 + place_config["eta_sensitivity_gain"] * eta_sensitivity)
        + place_config["surge_coefficient"]
        * (surge_multiplier - 1.0)
        * (1.0 + place_config["price_sensitivity_gain"] * price_sensitivity)
        + place_config["log_price_coefficient"]
        * np.log(final_price / place_config["reference_price"])
        + place_config["peak_hour_coefficient"] * state["is_peak_hour"].to_numpy()
        + place_config["bad_weather_coefficient"]
        * np.isin(state["weather"].to_numpy(), ["rain", "snow"])
        + place_config["rail_trip_coefficient"] * is_rail_trip
        + place_config["curfew_coefficient"] * state["curfew"].to_numpy()
        + place_config["air_raid_alert_coefficient"]
        * state["air_raid_alert"].to_numpy()
    )

    placed = rng.binomial(1, _sigmoid(place_logit)).astype(np.int8)

    # ---- Stage 2: does a driver accept it? ----
    accept_config = config.simulation["conversion"]["accept"]
    band_names = list(config.hour_bands)

    destination_attractiveness = np.array(
        [
            [
                config.zone_types[zone_type]["destination_attractiveness"][band]
                for band in band_names
            ]
            for zone_type in zones["zone_type"].to_numpy()
        ],
        dtype=float,
    )[destination_zone_position, band_index]
    # Centred so the coefficient describes deviation from an average destination.
    destination_attractiveness = destination_attractiveness - 1.0

    accept_logit = (
        accept_config["intercept"]
        + accept_config["surge_coefficient"] * (surge_multiplier - 1.0)
        + accept_config["eta_coefficient"] * eta_minutes
        + accept_config["suburban_origin_coefficient"]
        * (origin_zone_type == "suburban")
        + accept_config["attractive_destination_coefficient"]
        * destination_attractiveness
        + accept_config["short_trip_coefficient"]
        * (distance_km < accept_config["short_trip_threshold_km"])
        # Drivers are scarcer and warier during an alert, so the orders that do
        # come in are less likely to find one.
        + accept_config["air_raid_alert_coefficient"]
        * state["air_raid_alert"].to_numpy()
    )

    driver_willing = rng.binomial(1, _sigmoid(accept_logit)).astype(np.int8)

    # ---- Stage 3: search wait vs rider patience (competitor churn) ----
    search_config = config.simulation["search"]
    ratio = state["demand_supply_ratio"].to_numpy()
    tightness = np.maximum(ratio - 1.0, 0.0)
    match_time_minutes = np.clip(
        search_config["base_match_minutes"]
        + search_config["ratio_coefficient"] * tightness
        + search_config["alert_extra_minutes"] * state["air_raid_alert"].to_numpy()
        + rng.normal(
            0.0,
            search_config["noise_standard_deviation"],
            size=number_of_rides,
        ),
        search_config["minimum_match_minutes"],
        search_config["maximum_match_minutes"],
    )
    patience_minutes = rng.uniform(
        search_config["patience_minutes_min"],
        search_config["patience_minutes_max"],
        size=number_of_rides,
    )

    # Acceptance needs both a willing driver and a search that finishes before
    # the rider's patience runs out. Everything else that was placed ends as
    # competitor churn (timeout, or no willing driver found in time) — we do
    # not leave dangling "still searching" terminals.
    timed_out = match_time_minutes > patience_minutes
    accepted = (
        placed * driver_willing * (~timed_out).astype(np.int8)
    ).astype(np.int8)
    churned_to_competitor = (placed * (1 - accepted)).astype(np.int8)
    # Accepted / refused-after-match: record the match clock. True timeouts:
    # record patience (how long the rider waited before leaving).
    search_wait_minutes = np.where(
        accepted == 1,
        match_time_minutes,
        np.where(
            churned_to_competitor == 1,
            np.where(timed_out, patience_minutes, match_time_minutes),
            np.nan,
        ),
    )

    final_status = np.select(
        [accepted == 1, churned_to_competitor == 1],
        ["accepted", "churned_to_competitor"],
        default="calculated",
    )

    # ---- Timestamps ----
    timing = config.simulation["timing"]
    interval_seconds = config.frequency_minutes * 60

    calculated_at = state["timestamp"].to_numpy() + (
        rng.integers(0, interval_seconds, size=number_of_rides)
        * np.timedelta64(1, "s")
    )

    place_delay_seconds = rng.lognormal(
        mean=np.log(timing["place_delay_seconds_median"]),
        sigma=timing["place_delay_seconds_sigma"],
        size=number_of_rides,
    )
    search_delay_seconds = np.nan_to_num(search_wait_minutes, nan=0.0) * 60.0

    placed_at = calculated_at + np.round(place_delay_seconds).astype(
        "int64"
    ).astype("timedelta64[s]")
    accepted_at = placed_at + np.round(search_delay_seconds).astype(
        "int64"
    ).astype("timedelta64[s]")

    not_a_time = np.datetime64("NaT", "ns")
    placed_at = np.where(placed == 1, placed_at.astype("datetime64[ns]"), not_a_time)
    accepted_at = np.where(
        accepted == 1, accepted_at.astype("datetime64[ns]"), not_a_time
    )

    rides = pd.DataFrame(
        {
            "ride_id": np.arange(1, number_of_rides + 1, dtype=np.int64),
            "user_id": users["user_id"].to_numpy()[user_position],
            "calculated_at": calculated_at,
            "origin_zone_id": zones["zone_id"].to_numpy()[origin_zone_position],
            "destination_zone_id": zones["zone_id"].to_numpy()[
                destination_zone_position
            ],
            "distance_km": np.round(distance_km, 3),
            "estimated_duration_minutes": np.round(estimated_duration_minutes, 2),
            "eta_minutes": np.round(eta_minutes, 2),
            "demand_count": state["demand_count"].to_numpy(),
            "available_drivers": state["available_drivers"].to_numpy(),
            "demand_supply_ratio": state["demand_supply_ratio"].to_numpy(),
            "traffic_index": traffic_index,
            "weather": state["weather"].to_numpy(),
            "is_peak_hour": state["is_peak_hour"].to_numpy(),
            "is_weekend": state["is_weekend"].to_numpy(),
            "curfew": state["curfew"].to_numpy(),
            "air_raid_alert": state["air_raid_alert"].to_numpy(),
            "special_event": state["special_event"].to_numpy(),
            "base_price": np.round(base_price, 2),
            "surge_multiplier": surge_multiplier,
            "final_price": np.round(final_price, 2),
            "placed": placed,
            "accepted": accepted,
            "churned_to_competitor": churned_to_competitor,
            "search_wait_minutes": np.round(search_wait_minutes, 2),
            "final_status": final_status,
            "placed_at": placed_at,
            "accepted_at": accepted_at,
        }
    )

    return rides[RIDE_COLUMNS]
