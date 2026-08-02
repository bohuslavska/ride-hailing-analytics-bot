"""
Entry point for regenerating the whole synthetic dataset.

    python -m src.data_generation.build_all

A single seeded generator is threaded through every stage, so the same config
plus the same seed always produces byte-identical Parquet files.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.data_generation.generate_rides import generate_rides
from src.data_generation.generate_users import generate_users
from src.data_generation.generate_zone_state import generate_zone_state
from src.data_generation.generate_zones import generate_zones
from src.data_generation.simulation_config import load_simulation_config


def build_dataset(
    output_dir: Path,
    configs_dir: Path | None = None,
    seed: int | None = None,
) -> dict[str, pd.DataFrame]:
    config = load_simulation_config(configs_dir)
    rng = np.random.default_rng(config.seed if seed is None else seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()

    zones = generate_zones(config, rng)
    print(f"zones        {len(zones):>10,} rows")

    users = generate_users(config, zones, rng)
    print(f"users        {len(users):>10,} rows")

    zone_state = generate_zone_state(config, zones, rng)
    print(f"zone_state   {len(zone_state):>10,} rows")

    rides = generate_rides(config, zones, users, zone_state, rng)
    print(f"rides        {len(rides):>10,} rows")

    tables = {
        "zones": zones,
        "users": users,
        "zone_state": zone_state,
        "rides": rides,
    }

    for name, table in tables.items():
        destination = output_dir / f"{name}.parquet"
        table.to_parquet(destination, index=False, compression="zstd")
        size_mb = destination.stat().st_size / 1024 / 1024
        print(f"wrote {destination.name:<20} {size_mb:>6.1f} MB")

    elapsed = time.perf_counter() - started_at
    print(f"\ngenerated in {elapsed:.1f}s")

    placed = int(rides["placed"].sum())
    accepted = int(rides["accepted"].sum())
    churned = int(rides["churned_to_competitor"].sum())
    placed_rate = rides["placed"].mean()
    accepted_rate = accepted / placed if placed else 0.0
    churn_rate = churned / placed if placed else 0.0
    print(
        f"funnel: {len(rides):,} calculated -> "
        f"{placed:,} placed ({placed_rate:.1%}) -> "
        f"{accepted:,} accepted ({accepted_rate:.1%} of placed), "
        f"{churned:,} churned to competitor ({churn_rate:.1%} of placed)"
    )

    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory to write the Parquet tables into.",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=None,
        help="Directory holding simulation.yaml and zones.yaml.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed from simulation.yaml.",
    )
    arguments = parser.parse_args()

    build_dataset(
        output_dir=arguments.output_dir,
        configs_dir=arguments.configs_dir,
        seed=arguments.seed,
    )


if __name__ == "__main__":
    main()
