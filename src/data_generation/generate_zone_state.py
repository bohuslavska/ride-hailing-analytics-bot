"""
Build the per-zone, per-interval state of the marketplace.

This is where the causal structure of the dataset lives. The chain is:

    time / zone / weather / events  ->  demand
    shift pattern + lagged surge    ->  supply
    demand vs supply                ->  surge and ETA

Supply is computed inside a sequential loop rather than vectorised over the
whole horizon, because drivers react to the surge they saw one or two intervals
ago. That lag is what stops the relationship from being circular: surge does not
conjure drivers instantly, it pulls them in over the next half hour, which then
relieves the surge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_generation.simulation_config import (
    SimulationConfig,
    build_hourly_profile,
)

ZONE_STATE_COLUMNS = [
    "timestamp",
    "zone_id",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_peak_hour",
    "curfew",
    "air_raid_alert",
    "weather",
    "special_event",
    "traffic_index",
    "demand_count",
    "available_drivers",
    "demand_supply_ratio",
    "supply_gap",
    "surge_multiplier",
    "average_eta_minutes",
]

MORNING_PEAK_HOURS = {7, 8, 9}
EVENING_PEAK_HOURS = {17, 18, 19}


def build_timestamps(config: SimulationConfig) -> pd.DatetimeIndex:
    horizon = config.simulation["horizon"]

    return pd.date_range(
        start=horizon["start_date"],
        periods=config.number_of_intervals,
        freq=f"{config.frequency_minutes}min",
    )


def simulate_weather(
    config: SimulationConfig,
    number_of_intervals: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    City-wide weather as a Markov chain over 15-minute steps.

    A chain rather than independent draws, so that rain lasts for hours and
    actually shows up as a sustained demand and ETA effect.
    """
    weather = config.simulation["weather"]
    states = weather["states"]
    transition_matrix = np.array(weather["transition_matrix"], dtype=float)
    transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

    state_index = np.empty(number_of_intervals, dtype=int)
    state_index[0] = rng.choice(
        len(states), p=np.array(weather["initial_distribution"], dtype=float)
    )

    for interval in range(1, number_of_intervals):
        state_index[interval] = rng.choice(
            len(states), p=transition_matrix[state_index[interval - 1]]
        )

    return np.array(states, dtype=object)[state_index]


def simulate_curfew(
    config: SimulationConfig, timestamps: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Curfew mask plus the demand and supply multipliers it implies.

    Returns (in_curfew, demand_multiplier, supply_multiplier), each of length
    `intervals`. The hour and a half before curfew carries a rush rather than a
    taper, because riders are trying to be home before movement is prohibited.
    """
    curfew = config.simulation["curfew"]
    number_of_intervals = len(timestamps)

    in_curfew = np.zeros(number_of_intervals, dtype=bool)
    demand_multiplier = np.ones(number_of_intervals)
    supply_multiplier = np.ones(number_of_intervals)

    if not curfew.get("enabled", False):
        return in_curfew, demand_multiplier, supply_multiplier

    hour_of_day = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60.0
    start_hour = float(curfew["start_hour"])
    end_hour = float(curfew["end_hour"])

    def window(start: float, end: float) -> np.ndarray:
        if start < end:
            return (hour_of_day >= start) & (hour_of_day < end)
        # The window wraps past midnight.
        return (hour_of_day >= start) | (hour_of_day < end)

    in_curfew = window(start_hour, end_hour)
    demand_multiplier[in_curfew] = float(curfew["demand_multiplier"])
    supply_multiplier[in_curfew] = float(curfew["supply_multiplier"])

    rush_start = (start_hour - float(curfew["pre_curfew_rush_hours"])) % 24.0
    rush = window(rush_start, start_hour)
    demand_multiplier[rush] = float(curfew["pre_curfew_rush_multiplier"])

    return in_curfew, demand_multiplier, supply_multiplier


def simulate_air_raid_alerts(
    config: SimulationConfig,
    timestamps: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    City-wide air raid alerts and their effect on demand and supply.

    An alert is not a simple lull. Demand spikes in the interval it begins as
    people head home or change plans, then only dips mildly while it runs
    (ride-hailing partly substitutes for stopped surface transit), and rebounds
    afterwards as deferred trips are taken. Available driver supply is assumed
    to fall further than demand (app pause / fewer accepts / harder routing —
    not literal sheltering), which is what turns an alert into a surge event.

    Alert windows are merged before the multipliers are applied, so two
    overlapping alerts behave like one longer alert rather than producing a
    spurious second onset spike in the middle.
    """
    settings = config.simulation["air_raid_alerts"]
    number_of_intervals = len(timestamps)

    is_alert = np.zeros(number_of_intervals, dtype=bool)
    demand_multiplier = np.ones(number_of_intervals)
    supply_multiplier = np.ones(number_of_intervals)

    if not settings.get("enabled", False):
        return is_alert, demand_multiplier, supply_multiplier

    number_of_days = int(config.simulation["horizon"]["number_of_days"])
    expected_alerts = float(settings["alerts_per_week"]) * number_of_days / 7.0
    number_of_alerts = int(rng.poisson(expected_alerts))

    for _ in range(number_of_alerts):
        start = int(rng.integers(0, number_of_intervals))
        duration = int(
            rng.integers(
                settings["minimum_duration_intervals"],
                settings["maximum_duration_intervals"] + 1,
            )
        )
        is_alert[start : min(start + duration, number_of_intervals)] = True

    previous = np.concatenate(([False], is_alert[:-1]))
    following = np.concatenate((is_alert[1:], [False]))

    onset = is_alert & ~previous
    ongoing = is_alert & ~onset

    demand_multiplier[ongoing] = float(settings["during_demand_multiplier"])
    demand_multiplier[onset] = float(settings["onset_demand_multiplier"])
    supply_multiplier[is_alert] = float(settings["during_supply_multiplier"])

    recovery_intervals = int(settings["recovery_intervals"])
    recovery_multiplier = float(settings["recovery_demand_multiplier"])

    for end_index in np.flatnonzero(is_alert & ~following) + 1:
        stop = min(end_index + recovery_intervals, number_of_intervals)
        window = slice(end_index, stop)
        not_in_alert = ~is_alert[window]
        demand_multiplier[window] = np.where(
            not_in_alert, recovery_multiplier, demand_multiplier[window]
        )

    return is_alert, demand_multiplier, supply_multiplier


def simulate_special_events(
    config: SimulationConfig,
    zones: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Demand multiplier matrix of shape (intervals, zones) for concerts, matches
    and conferences. Values are 1.0 outside of an event window.
    """
    events_config = config.simulation["special_events"]
    number_of_intervals = len(timestamps)
    number_of_zones = len(zones)

    multiplier = np.ones((number_of_intervals, number_of_zones), dtype=float)

    eligible_zone_positions = np.flatnonzero(
        zones["zone_type"].isin(events_config["eligible_zone_types"]).to_numpy()
    )

    if eligible_zone_positions.size == 0:
        return multiplier

    number_of_days = int(config.simulation["horizon"]["number_of_days"])
    expected_events = events_config["events_per_week"] * number_of_days / 7.0
    number_of_events = int(rng.poisson(expected_events))

    for _ in range(number_of_events):
        zone_position = int(rng.choice(eligible_zone_positions))
        duration = int(
            rng.integers(
                events_config["minimum_duration_intervals"],
                events_config["maximum_duration_intervals"] + 1,
            )
        )

        # Events start in the late afternoon or evening.
        start_day = int(rng.integers(0, number_of_days))
        start_hour = float(rng.uniform(17.0, 22.0))
        start_interval = int(
            start_day * config.intervals_per_day
            + start_hour * 60 / config.frequency_minutes
        )
        end_interval = min(start_interval + duration, number_of_intervals)

        if start_interval >= number_of_intervals:
            continue

        event_multiplier = float(
            rng.uniform(
                events_config["demand_multiplier_minimum"],
                events_config["demand_multiplier_maximum"],
            )
        )

        multiplier[start_interval:end_interval, zone_position] = np.maximum(
            multiplier[start_interval:end_interval, zone_position],
            event_multiplier,
        )

    return multiplier


def _build_traffic_index(
    config: SimulationConfig,
    zones: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    weather_states: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Congestion in [0.05, 1.0] of shape (intervals, zones)."""
    traffic_config = config.simulation["traffic"]

    hour_of_day = (
        timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60.0
    )

    def rush_component(peak_hour: float) -> np.ndarray:
        distance = np.abs(hour_of_day - peak_hour)
        circular_distance = np.minimum(distance, 24.0 - distance)
        return np.exp(
            -0.5 * (circular_distance / traffic_config["peak_width_hours"]) ** 2
        )

    rush_shape = traffic_config["peak_amplitude"] * (
        rush_component(traffic_config["morning_peak_hour"])
        + rush_component(traffic_config["evening_peak_hour"])
    )

    is_weekend = timestamps.dayofweek.to_numpy() >= 5
    rush_shape = np.where(
        is_weekend, rush_shape * traffic_config["weekend_multiplier"], rush_shape
    )

    weather_increment = np.array(
        [traffic_config["weather_increment"][state] for state in weather_states],
        dtype=float,
    )

    base_traffic = zones["base_traffic"].to_numpy()

    traffic = (
        base_traffic[None, :] * (1.0 + rush_shape[:, None])
        + weather_increment[:, None]
        + rng.normal(0.0, 0.03, size=(len(timestamps), len(zones)))
    )

    return np.clip(traffic, 0.05, 1.0)


def generate_zone_state(
    config: SimulationConfig,
    zones: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    timestamps = build_timestamps(config)
    number_of_intervals = len(timestamps)
    number_of_zones = len(zones)

    zone_type_names = zones["zone_type"].to_numpy()

    demand_profiles = {
        name: build_hourly_profile(profile["demand_profile"])
        for name, profile in config.zone_types.items()
    }
    supply_profiles = {
        name: build_hourly_profile(profile["supply_profile"])
        for name, profile in config.zone_types.items()
    }

    # (zones, 24) multiplier lookups.
    demand_shape = np.array([demand_profiles[name] for name in zone_type_names])
    supply_shape = np.array([supply_profiles[name] for name in zone_type_names])

    weekend_demand_multiplier = np.array(
        [
            config.zone_types[name]["weekend_demand_multiplier"]
            for name in zone_type_names
        ],
        dtype=float,
    )
    is_entertainment_zone = zone_type_names == "entertainment"

    weather_states = simulate_weather(config, number_of_intervals, rng)
    event_multiplier = simulate_special_events(config, zones, timestamps, rng)
    traffic_index = _build_traffic_index(
        config, zones, timestamps, weather_states, rng
    )

    in_curfew, curfew_demand, curfew_supply = simulate_curfew(config, timestamps)
    is_alert, alert_demand, alert_supply = simulate_air_raid_alerts(
        config, timestamps, rng
    )
    alert_eta_increment = float(
        config.simulation["air_raid_alerts"]["eta_increment_minutes"]
    )

    demand_config = config.simulation["demand"]
    supply_config = config.simulation["supply"]
    surge_config = config.simulation["surge"]
    eta_config = config.simulation["eta"]

    demand_weather_multiplier = np.array(
        [demand_config["weather_multiplier"][state] for state in weather_states],
        dtype=float,
    )
    is_bad_weather = np.isin(weather_states, ["rain", "snow"])

    hours = timestamps.hour.to_numpy()
    days_of_week = timestamps.dayofweek.to_numpy()
    is_weekend = days_of_week >= 5

    base_demand = zones["base_demand"].to_numpy()
    base_supply = zones["base_supply"].to_numpy()
    base_eta = zones["base_eta_minutes"].to_numpy()

    surge_lag = int(supply_config["surge_response_lag_intervals"])
    surge_gain = float(supply_config["surge_response_gain"])
    conserve_city_pool = bool(supply_config["conserve_city_pool"])
    weekend_supply_multiplier = float(supply_config["weekend_multiplier"])

    demand_count = np.zeros((number_of_intervals, number_of_zones), dtype=np.int32)
    available_drivers = np.zeros(
        (number_of_intervals, number_of_zones), dtype=np.int32
    )
    surge_multiplier = np.ones((number_of_intervals, number_of_zones), dtype=float)
    average_eta = np.zeros((number_of_intervals, number_of_zones), dtype=float)

    surge_noise = rng.normal(
        0.0,
        surge_config["noise_standard_deviation"],
        size=(number_of_intervals, number_of_zones),
    )
    eta_noise = rng.normal(
        0.0,
        eta_config["noise_standard_deviation"],
        size=(number_of_intervals, number_of_zones),
    )

    smoothing_weight = float(surge_config["smoothing_weight"])

    for interval in range(number_of_intervals):
        hour = hours[interval]
        weekend = is_weekend[interval]

        expected_demand = (
            base_demand
            * demand_shape[:, hour]
            * demand_weather_multiplier[interval]
            * event_multiplier[interval]
            * curfew_demand[interval]
            * alert_demand[interval]
        )

        if weekend:
            expected_demand = expected_demand * weekend_demand_multiplier

        # Friday after 21:00 behaves like a weekend night for nightlife zones.
        if days_of_week[interval] == 4 and hour >= 21:
            expected_demand = np.where(
                is_entertainment_zone,
                expected_demand * demand_config["friday_night_entertainment_boost"],
                expected_demand,
            )

        interval_demand = rng.poisson(np.maximum(expected_demand, 1e-6))

        # Drivers follow their shift pattern, then reallocate towards whichever
        # zones were surging `surge_lag` intervals ago.
        shift_supply = (
            base_supply
            * supply_shape[:, hour]
            * curfew_supply[interval]
            * alert_supply[interval]
        )

        if weekend:
            shift_supply = shift_supply * weekend_supply_multiplier

        if interval >= surge_lag:
            lagged_surge = surge_multiplier[interval - surge_lag]
        else:
            lagged_surge = np.ones(number_of_zones)

        attraction = shift_supply * (1.0 + surge_gain * (lagged_surge - 1.0))

        if conserve_city_pool:
            # Surge moves drivers between zones; it does not create them.
            expected_supply = (
                shift_supply.sum() * attraction / max(attraction.sum(), 1e-9)
            )
        else:
            expected_supply = attraction

        interval_supply = rng.poisson(np.maximum(expected_supply, 1e-6))

        ratio = interval_demand / np.maximum(interval_supply, 1)
        deficit = np.maximum(ratio - 1.0, 0.0)

        raw_surge = (
            1.0
            + surge_config["deficit_coefficient"] * deficit
            + surge_config["event_bonus"] * (event_multiplier[interval] > 1.0)
            + surge_noise[interval]
        )

        if interval == 0:
            smoothed_surge = raw_surge
        else:
            smoothed_surge = (
                smoothing_weight * surge_multiplier[interval - 1]
                + (1.0 - smoothing_weight) * raw_surge
            )

        interval_surge = np.clip(
            smoothed_surge, surge_config["minimum"], surge_config["maximum"]
        )

        interval_eta = np.clip(
            base_eta
            + eta_config["deficit_coefficient"] * deficit
            + eta_config["traffic_coefficient"] * traffic_index[interval]
            + eta_config["bad_weather_increment"] * is_bad_weather[interval]
            + alert_eta_increment * is_alert[interval]
            + eta_noise[interval],
            eta_config["minimum_minutes"],
            eta_config["maximum_minutes"],
        )

        demand_count[interval] = interval_demand
        available_drivers[interval] = interval_supply
        surge_multiplier[interval] = interval_surge
        average_eta[interval] = interval_eta

    is_peak_hour = np.isin(hours, sorted(MORNING_PEAK_HOURS | EVENING_PEAK_HOURS)) & (
        ~is_weekend
    )

    repeated_zone_ids = np.tile(zones["zone_id"].to_numpy(), number_of_intervals)
    repeated_timestamps = np.repeat(timestamps.to_numpy(), number_of_zones)

    flat_demand = demand_count.reshape(-1)
    flat_supply = available_drivers.reshape(-1)

    zone_state = pd.DataFrame(
        {
            "timestamp": repeated_timestamps,
            "zone_id": repeated_zone_ids,
            "hour": np.repeat(hours, number_of_zones),
            "day_of_week": np.repeat(days_of_week, number_of_zones),
            "is_weekend": np.repeat(is_weekend, number_of_zones),
            "is_peak_hour": np.repeat(is_peak_hour, number_of_zones),
            "curfew": np.repeat(in_curfew, number_of_zones),
            "air_raid_alert": np.repeat(is_alert, number_of_zones),
            "weather": np.repeat(weather_states, number_of_zones),
            "special_event": (event_multiplier > 1.0).reshape(-1),
            "traffic_index": np.round(traffic_index.reshape(-1), 4),
            "demand_count": flat_demand,
            "available_drivers": flat_supply,
            "demand_supply_ratio": np.round(
                flat_demand / np.maximum(flat_supply, 1), 4
            ),
            "supply_gap": flat_demand - flat_supply,
            "surge_multiplier": np.round(surge_multiplier.reshape(-1), 4),
            "average_eta_minutes": np.round(average_eta.reshape(-1), 4),
        }
    )

    return zone_state[ZONE_STATE_COLUMNS]
