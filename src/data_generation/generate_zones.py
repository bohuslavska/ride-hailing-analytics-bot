"""Build the static zone table: geography and structural characteristics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_generation.simulation_config import SimulationConfig

ZONE_COLUMNS = [
    "zone_id",
    "zone_name",
    "zone_type",
    "latitude",
    "longitude",
    "distance_from_center_km",
    "base_demand",
    "base_supply",
    "base_traffic",
    "base_eta_minutes",
]


def generate_zones(config: SimulationConfig, rng: np.random.Generator) -> pd.DataFrame:
    city = config.simulation["city"]
    center_latitude = float(city["center_latitude"])
    center_longitude = float(city["center_longitude"])
    km_per_degree_latitude = float(city["km_per_degree_latitude"])

    # Applied to demand and supply together so that resizing the marketplace
    # does not silently make it structurally under- or over-supplied.
    volume_scale = float(config.simulation["volume"]["scale"])

    records = []

    for zone in config.zones:
        zone_type = config.zone_types[zone["zone_type"]]

        distance_km = float(zone["distance_km"])
        bearing_radians = np.deg2rad(float(zone["bearing_degrees"]))

        # Bearing is measured clockwise from north, so north maps to the
        # latitude axis and east to the longitude axis.
        latitude = center_latitude + (
            distance_km * np.cos(bearing_radians) / km_per_degree_latitude
        )
        longitude = center_longitude + (
            distance_km
            * np.sin(bearing_radians)
            / (km_per_degree_latitude * np.cos(np.deg2rad(center_latitude)))
        )

        base_demand = float(zone["base_demand"]) * volume_scale

        # Zones of the same type should not be carbon copies of each other, so
        # the structural levels get a small multiplicative jitter.
        base_supply = (
            base_demand
            * float(zone_type["supply_per_demand"])
            * rng.uniform(0.92, 1.08)
        )
        base_traffic = float(zone_type["base_traffic"]) * rng.uniform(0.90, 1.10)
        base_eta_minutes = float(zone_type["base_eta_minutes"]) * rng.uniform(0.90, 1.10)

        records.append(
            {
                "zone_id": zone["zone_id"],
                "zone_name": zone["zone_name"],
                "zone_type": zone["zone_type"],
                "latitude": round(float(latitude), 6),
                "longitude": round(float(longitude), 6),
                "distance_from_center_km": distance_km,
                "base_demand": base_demand,
                "base_supply": round(base_supply, 4),
                "base_traffic": round(float(np.clip(base_traffic, 0.05, 0.95)), 4),
                "base_eta_minutes": round(base_eta_minutes, 4),
            }
        )

    return pd.DataFrame.from_records(records, columns=ZONE_COLUMNS)


def build_distance_matrix(
    zones: pd.DataFrame, km_per_degree_latitude: float
) -> np.ndarray:
    """
    Straight-line distances between zone centroids, in kilometres.

    An equirectangular approximation is accurate enough at city scale and keeps
    the matrix cheap to compute.
    """
    latitudes = zones["latitude"].to_numpy()
    longitudes = zones["longitude"].to_numpy()
    mean_latitude_radians = np.deg2rad(latitudes.mean())

    latitude_km = latitudes * km_per_degree_latitude
    longitude_km = longitudes * km_per_degree_latitude * np.cos(mean_latitude_radians)

    latitude_difference = latitude_km[:, None] - latitude_km[None, :]
    longitude_difference = longitude_km[:, None] - longitude_km[None, :]

    return np.sqrt(latitude_difference**2 + longitude_difference**2)
