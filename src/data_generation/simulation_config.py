"""Typed access to the YAML simulation parameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.config import PROJECT_ROOT

HOURS_IN_DAY = 24


@dataclass(frozen=True)
class SimulationConfig:
    simulation: dict[str, Any]
    zone_types: dict[str, dict[str, Any]]
    hour_bands: dict[str, list[int]]
    zones: list[dict[str, Any]]

    @property
    def seed(self) -> int:
        return int(self.simulation["seed"])

    @property
    def frequency_minutes(self) -> int:
        return int(self.simulation["horizon"]["frequency_minutes"])

    @property
    def intervals_per_day(self) -> int:
        return (24 * 60) // self.frequency_minutes

    @property
    def number_of_intervals(self) -> int:
        return self.intervals_per_day * int(self.simulation["horizon"]["number_of_days"])

    def zone_type_names(self) -> list[str]:
        return sorted(self.zone_types)


def load_simulation_config(configs_dir: Path | None = None) -> SimulationConfig:
    configs_dir = configs_dir or (PROJECT_ROOT / "configs")

    simulation = yaml.safe_load((configs_dir / "simulation.yaml").read_text())
    zones_document = yaml.safe_load((configs_dir / "zones.yaml").read_text())

    return SimulationConfig(
        simulation=simulation,
        zone_types=zones_document["zone_types"],
        hour_bands=zones_document["hour_bands"],
        zones=zones_document["zones"],
    )


def build_hourly_profile(profile: dict[str, Any]) -> np.ndarray:
    """
    Turn a baseline-plus-Gaussian-peaks description into a 24-value multiplier.

    Peaks wrap around midnight, so a peak at 23:30 still lifts 00:30. The result
    is normalised to a 24-hour mean of 1.0 so that the zone's `base_demand`
    stays readable as an average rather than an arbitrary scale factor.
    """
    hours = np.arange(HOURS_IN_DAY, dtype=float)
    values = np.full(HOURS_IN_DAY, float(profile["baseline"]))

    for peak in profile.get("peaks", []):
        distance = np.abs(hours - float(peak["hour"]))
        circular_distance = np.minimum(distance, HOURS_IN_DAY - distance)
        values += float(peak["amplitude"]) * np.exp(
            -0.5 * (circular_distance / float(peak["width"])) ** 2
        )

    return values / values.mean()


def build_hour_band_lookup(hour_bands: dict[str, list[int]]) -> np.ndarray:
    """Map each hour of the day to the index of its band, ordered by band name."""
    band_names = list(hour_bands)
    lookup = np.zeros(HOURS_IN_DAY, dtype=int)

    for band_index, band_name in enumerate(band_names):
        for hour in hour_bands[band_name]:
            lookup[hour] = band_index

    return lookup
