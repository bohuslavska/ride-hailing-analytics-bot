"""
Build the user population.

Each user carries latent behavioural traits that steer the simulation: how
often they request a ride, how strongly price and ETA put them off, and when
they tend to travel. These traits are the *cause* of the observable funnel, so
they are deliberately kept out of the analytical database (see
scripts/load_to_postgres.py). Any clustering the bot performs therefore has to
recover segments from observed behaviour rather than read the answer key.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_generation.simulation_config import SimulationConfig

# Latent segments and the trait distributions they imply. `share` values are
# normalised, so they can be read as approximate population proportions.
USER_SEGMENTS = {
    "commuter": {
        "share": 0.35,
        "activity_log_mean": 0.35,
        "price_sensitivity_mean": 0.10,
        "eta_sensitivity_mean": 0.45,
        "propensity_mean": 0.15,
        "rail_affinity": 0.4,
        "preferred_band_weights": {
            "night": 0.02,
            "morning": 0.42,
            "midday": 0.10,
            "evening": 0.42,
            "late": 0.04,
        },
        "home_zone_type_weights": {
            "residential": 3.0,
            "suburban": 1.6,
            "city_center": 0.8,
            "business": 0.3,
            "railway_station": 0.4,
            "entertainment": 0.3,
        },
    },
    "casual": {
        "share": 0.30,
        "activity_log_mean": -0.10,
        "price_sensitivity_mean": 0.00,
        "eta_sensitivity_mean": 0.00,
        "propensity_mean": 0.00,
        "rail_affinity": 0.6,
        "preferred_band_weights": {
            "night": 0.05,
            "morning": 0.15,
            "midday": 0.35,
            "evening": 0.33,
            "late": 0.12,
        },
        "home_zone_type_weights": {
            "residential": 2.5,
            "suburban": 1.0,
            "city_center": 1.8,
            "business": 0.6,
            "railway_station": 0.7,
            "entertainment": 0.8,
        },
    },
    # Named for the late evening rather than the night: the curfew means there
    # is no post-midnight social travel to be had, so this segment's activity
    # concentrates in the 21:00-23:00 window before movement is prohibited.
    "nightlife": {
        "share": 0.12,
        "activity_log_mean": -0.05,
        "price_sensitivity_mean": -0.55,
        "eta_sensitivity_mean": -0.35,
        "propensity_mean": 0.20,
        "rail_affinity": 0.3,
        "preferred_band_weights": {
            "night": 0.05,
            "morning": 0.03,
            "midday": 0.09,
            "evening": 0.31,
            "late": 0.52,
        },
        "home_zone_type_weights": {
            "residential": 1.6,
            "suburban": 0.5,
            "city_center": 2.6,
            "business": 0.3,
            "railway_station": 0.6,
            "entertainment": 2.2,
        },
    },
    "business_traveller": {
        "share": 0.10,
        "activity_log_mean": 0.10,
        "price_sensitivity_mean": -0.95,
        "eta_sensitivity_mean": 0.60,
        "propensity_mean": 0.45,
        # With civil aviation closed, the frequent business traveller is a rail
        # traveller: this is the propensity to start or end a trip at a station.
        "rail_affinity": 2.8,
        "preferred_band_weights": {
            "night": 0.06,
            "morning": 0.34,
            "midday": 0.24,
            "evening": 0.26,
            "late": 0.10,
        },
        "home_zone_type_weights": {
            "residential": 1.2,
            "suburban": 0.4,
            "city_center": 2.4,
            "business": 2.0,
            "railway_station": 0.8,
            "entertainment": 0.5,
        },
    },
    "budget": {
        "share": 0.13,
        "activity_log_mean": -0.55,
        "price_sensitivity_mean": 1.15,
        "eta_sensitivity_mean": -0.10,
        "propensity_mean": -0.40,
        "rail_affinity": 0.2,
        "preferred_band_weights": {
            "night": 0.06,
            "morning": 0.28,
            "midday": 0.24,
            "evening": 0.32,
            "late": 0.10,
        },
        "home_zone_type_weights": {
            "residential": 2.8,
            "suburban": 2.4,
            "city_center": 0.7,
            "business": 0.2,
            "railway_station": 0.6,
            "entertainment": 0.4,
        },
    },
}

# Columns safe to expose in the analytical database.
PUBLIC_USER_COLUMNS = ["user_id", "home_zone_id", "signup_date"]


def generate_users(
    config: SimulationConfig,
    zones: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    number_of_users = int(config.simulation["users"]["number_of_users"])
    band_names = list(config.hour_bands)

    segment_names = list(USER_SEGMENTS)
    segment_shares = np.array(
        [USER_SEGMENTS[name]["share"] for name in segment_names], dtype=float
    )
    segment_shares /= segment_shares.sum()

    segment_index = rng.choice(
        len(segment_names), size=number_of_users, p=segment_shares
    )

    activity_weight = np.empty(number_of_users)
    price_sensitivity = np.empty(number_of_users)
    eta_sensitivity = np.empty(number_of_users)
    propensity = np.empty(number_of_users)
    rail_affinity = np.empty(number_of_users)
    preferred_band = np.empty(number_of_users, dtype=int)
    home_zone_index = np.empty(number_of_users, dtype=int)

    zone_type_values = zones["zone_type"].to_numpy()
    zone_base_demand = zones["base_demand"].to_numpy()

    for index, segment_name in enumerate(segment_names):
        segment = USER_SEGMENTS[segment_name]
        members = np.flatnonzero(segment_index == index)

        if members.size == 0:
            continue

        activity_weight[members] = rng.lognormal(
            mean=segment["activity_log_mean"], sigma=0.7, size=members.size
        )
        price_sensitivity[members] = rng.normal(
            segment["price_sensitivity_mean"], 0.45, size=members.size
        )
        eta_sensitivity[members] = rng.normal(
            segment["eta_sensitivity_mean"], 0.45, size=members.size
        )
        propensity[members] = rng.normal(
            segment["propensity_mean"], 0.35, size=members.size
        )
        rail_affinity[members] = segment["rail_affinity"] * rng.uniform(
            0.7, 1.3, size=members.size
        )

        band_weights = np.array(
            [segment["preferred_band_weights"][name] for name in band_names],
            dtype=float,
        )
        band_weights /= band_weights.sum()
        preferred_band[members] = rng.choice(
            len(band_names), size=members.size, p=band_weights
        )

        # People live where housing is, weighted by how much traffic the zone
        # generates in the first place.
        zone_weights = np.array(
            [
                segment["home_zone_type_weights"].get(zone_type, 0.5)
                for zone_type in zone_type_values
            ],
            dtype=float,
        )
        zone_weights = zone_weights * zone_base_demand
        zone_weights /= zone_weights.sum()
        home_zone_index[members] = rng.choice(
            len(zones), size=members.size, p=zone_weights
        )

    start_date = pd.Timestamp(config.simulation["horizon"]["start_date"])
    tenure_days = rng.integers(0, 730, size=number_of_users)
    signup_date = start_date - pd.to_timedelta(tenure_days, unit="D")

    return pd.DataFrame(
        {
            "user_id": [f"U{index:06d}" for index in range(number_of_users)],
            "home_zone_id": zones["zone_id"].to_numpy()[home_zone_index],
            "signup_date": signup_date.date,
            "latent_segment": [segment_names[index] for index in segment_index],
            "activity_weight": np.round(activity_weight, 5),
            "price_sensitivity": np.round(price_sensitivity, 5),
            "eta_sensitivity": np.round(eta_sensitivity, 5),
            "base_propensity": np.round(propensity, 5),
            "rail_affinity": np.round(rail_affinity, 5),
            "preferred_hour_band": [band_names[index] for index in preferred_band],
        }
    )
