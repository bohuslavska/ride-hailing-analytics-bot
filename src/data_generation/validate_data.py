"""
Validate the generated dataset.

Two families of checks:

*   Technical invariants -- things that must hold for the data to be internally
    consistent at all (a ride cannot be accepted without being placed).
*   Behavioural invariants -- things that must hold for the data to be a
    *designed* simulation rather than noise (surge has to rise when drivers are
    scarce, place conversion has to fall as ETA grows).

The second family is the point. Anyone can emit a CSV of random numbers that
passes a null check; these assertions are what claim the dependencies are real.

    python -m src.data_generation.validate_data
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    category: str


class Validator:
    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def check(self, name: str, passed: bool, detail: str, category: str) -> None:
        self.results.append(
            CheckResult(
                name=name, passed=bool(passed), detail=detail, category=category
            )
        )

    @property
    def failures(self) -> list[CheckResult]:
        return [result for result in self.results if not result.passed]


def _monotonically_decreasing(values: list[float], tolerance: float = 0.0) -> bool:
    # Ragged by design: pairing a sequence with its own tail is how consecutive
    # pairs are formed, so the shorter side is meant to end first.
    return all(
        later <= earlier + tolerance
        for earlier, later in zip(values, values[1:], strict=False)
    )


def _monotonically_increasing(values: list[float], tolerance: float = 0.0) -> bool:
    return all(
        later >= earlier - tolerance
        for earlier, later in zip(values, values[1:], strict=False)
    )


def _fit_acceptance_model(
    placed_rides: pd.DataFrame, sample_size: int = 100_000
) -> dict[str, float]:
    """Logistic fit of accepted ~ surge + ETA, returning the coefficients."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sample = (
        placed_rides.sample(sample_size, random_state=0)
        if len(placed_rides) > sample_size
        else placed_rides
    )

    feature_names = ["surge_multiplier", "eta_minutes"]
    features = sample[feature_names].to_numpy(dtype=float)
    target = sample["accepted"].to_numpy()

    scaler = StandardScaler()
    model = LogisticRegression(max_iter=1000)
    model.fit(scaler.fit_transform(features), target)

    # Undo the scaling so the coefficients are per natural unit.
    coefficients = model.coef_[0] / scaler.scale_

    return dict(zip(feature_names, coefficients, strict=True))


def validate_technical_invariants(
    validator: Validator,
    zones: pd.DataFrame,
    users: pd.DataFrame,
    zone_state: pd.DataFrame,
    rides: pd.DataFrame,
) -> None:
    category = "technical"

    validator.check(
        "ride_id is unique",
        rides["ride_id"].is_unique,
        f"{rides['ride_id'].nunique():,} distinct ids over {len(rides):,} rows",
        category,
    )

    validator.check(
        "accepted implies placed",
        bool((rides["accepted"] <= rides["placed"]).all()),
        "no ride is accepted without having been placed",
        category,
    )

    validator.check(
        "churn implies placed and excludes accepted",
        bool(
            (rides["churned_to_competitor"] <= rides["placed"]).all()
            and ((rides["accepted"] + rides["churned_to_competitor"]) <= 1).all()
            and (
                rides.loc[rides["placed"] == 1, "accepted"]
                + rides.loc[rides["placed"] == 1, "churned_to_competitor"]
                == 1
            ).all()
        ),
        "every placed order ends as accepted or churned_to_competitor",
        category,
    )

    validator.check(
        "final_status agrees with the flags",
        bool(
            (
                rides["final_status"]
                == np.select(
                    [
                        rides["accepted"] == 1,
                        rides["churned_to_competitor"] == 1,
                    ],
                    ["accepted", "churned_to_competitor"],
                    default="calculated",
                )
            ).all()
        ),
        "status column is derivable from accepted/churned",
        category,
    )

    validator.check(
        "search_wait present exactly on placed orders",
        bool(
            (
                rides["search_wait_minutes"].notna() == (rides["placed"] == 1)
            ).all()
        ),
        "search_wait_minutes is null iff the order was never placed",
        category,
    )

    validator.check(
        "funnel timestamps are ordered",
        bool(
            (
                rides.loc[rides["placed"] == 1, "placed_at"]
                >= rides.loc[rides["placed"] == 1, "calculated_at"]
            ).all()
            and (
                rides.loc[rides["accepted"] == 1, "accepted_at"]
                >= rides.loc[rides["accepted"] == 1, "placed_at"]
            ).all()
        ),
        "calculated_at <= placed_at <= accepted_at",
        category,
    )

    validator.check(
        "stage timestamps are null exactly when the stage did not happen",
        bool(
            (rides["placed_at"].isna() == (rides["placed"] == 0)).all()
            and (rides["accepted_at"].isna() == (rides["accepted"] == 0)).all()
        ),
        "no orphaned or missing stage timestamps",
        category,
    )

    validator.check(
        "ETA is positive and bounded",
        bool(
            (rides["eta_minutes"] > 0).all()
            and (rides["eta_minutes"] <= 18.0).all()
        ),
        f"range {rides['eta_minutes'].min():.2f}-{rides['eta_minutes'].max():.2f} min",
        category,
    )

    accepted_eta = rides.loc[rides["accepted"] == 1, "eta_minutes"]
    typical_eta_share = float((accepted_eta <= 10.0).mean()) if len(accepted_eta) else 0.0
    validator.check(
        "most accepted pickups arrive within about 10 minutes",
        typical_eta_share >= 0.80,
        f"{typical_eta_share:.1%} of accepted rides have ETA ≤ 10 min "
        f"(p90={accepted_eta.quantile(0.9):.1f})",
        category,
    )

    validator.check(
        "surge stays inside its configured band",
        bool(
            (rides["surge_multiplier"] >= 1.0).all()
            and (rides["surge_multiplier"] <= 2.5).all()
        ),
        f"range {rides['surge_multiplier'].min():.3f}-"
        f"{rides['surge_multiplier'].max():.3f}",
        category,
    )

    validator.check(
        "final_price never undercuts base_price",
        bool((rides["final_price"] >= rides["base_price"] - 0.01).all()),
        "surge is applied multiplicatively and is never below 1.0",
        category,
    )

    validator.check(
        "counts are non-negative",
        bool(
            (zone_state["demand_count"] >= 0).all()
            and (zone_state["available_drivers"] >= 0).all()
        ),
        "demand_count and available_drivers are counts",
        category,
    )

    known_zone_ids = set(zones["zone_id"])
    validator.check(
        "zone references resolve",
        bool(
            set(rides["origin_zone_id"]).issubset(known_zone_ids)
            and set(rides["destination_zone_id"]).issubset(known_zone_ids)
            and set(zone_state["zone_id"]).issubset(known_zone_ids)
        ),
        f"{len(known_zone_ids)} zones referenced consistently",
        category,
    )

    validator.check(
        "user references resolve",
        bool(set(rides["user_id"]).issubset(set(users["user_id"]))),
        f"{rides['user_id'].nunique():,} of {len(users):,} users appear in rides",
        category,
    )

    required_columns = [
        "ride_id",
        "user_id",
        "calculated_at",
        "origin_zone_id",
        "destination_zone_id",
        "eta_minutes",
        "surge_multiplier",
        "final_price",
        "placed",
        "accepted",
        "final_status",
    ]
    validator.check(
        "no nulls in required ride columns",
        bool(rides[required_columns].notna().all().all()),
        f"checked {len(required_columns)} columns",
        category,
    )

    validator.check(
        "zone_state grid is complete",
        len(zone_state)
        == zone_state["timestamp"].nunique() * zone_state["zone_id"].nunique(),
        f"{zone_state['timestamp'].nunique():,} intervals x "
        f"{zone_state['zone_id'].nunique()} zones",
        category,
    )


def validate_behavioural_invariants(
    validator: Validator,
    zones: pd.DataFrame,
    zone_state: pd.DataFrame,
    rides: pd.DataFrame,
) -> None:
    category = "behavioural"

    scarce = zone_state.loc[zone_state["demand_supply_ratio"] > 1.5]
    abundant = zone_state.loc[zone_state["demand_supply_ratio"] < 0.8]

    validator.check(
        "surge rises when drivers are scarce",
        scarce["surge_multiplier"].mean() > abundant["surge_multiplier"].mean(),
        f"surge {scarce['surge_multiplier'].mean():.3f} when ratio>1.5 vs "
        f"{abundant['surge_multiplier'].mean():.3f} when ratio<0.8",
        category,
    )

    validator.check(
        "ETA rises when drivers are scarce",
        scarce["average_eta_minutes"].mean() > abundant["average_eta_minutes"].mean(),
        f"ETA {scarce['average_eta_minutes'].mean():.2f} min when ratio>1.5 vs "
        f"{abundant['average_eta_minutes'].mean():.2f} min when ratio<0.8",
        category,
    )

    eta_bins = pd.cut(rides["eta_minutes"], [0, 3, 5, 8, 12, 100])
    place_by_eta = (
        rides.groupby(eta_bins, observed=True)["placed"].mean().round(4).tolist()
    )
    validator.check(
        "place conversion falls as ETA rises",
        _monotonically_decreasing(place_by_eta, tolerance=0.005),
        " -> ".join(f"{value:.3f}" for value in place_by_eta),
        category,
    )

    surge_bins = pd.cut(rides["surge_multiplier"], [0.99, 1.05, 1.2, 1.5, 2.0, 2.6])
    place_by_surge = (
        rides.groupby(surge_bins, observed=True)["placed"].mean().round(4).tolist()
    )
    validator.check(
        "place conversion falls as surge rises",
        _monotonically_decreasing(place_by_surge, tolerance=0.005),
        " -> ".join(f"{value:.3f}" for value in place_by_surge),
        category,
    )

    # Marginally, acceptance looks flat in surge because surge and ETA share a
    # cause. Holding ETA roughly fixed, the positive incentive effect shows up.
    placed_rides = rides.loc[rides["placed"] == 1].copy()
    placed_rides["eta_bin"] = pd.cut(
        placed_rides["eta_minutes"], [0, 6, 8, 10, 13, 100]
    )
    placed_rides["surge_bin"] = pd.cut(
        placed_rides["surge_multiplier"], [0.99, 1.05, 1.2, 1.5, 2.6]
    )
    acceptance_grid = placed_rides.pivot_table(
        index="eta_bin",
        columns="surge_bin",
        values="accepted",
        aggfunc="mean",
        observed=True,
    )
    cell_counts = placed_rides.pivot_table(
        index="eta_bin",
        columns="surge_bin",
        values="accepted",
        aggfunc="size",
        observed=True,
    )

    # Thin cells are dominated by sampling noise -- a 200-ride cell has a
    # standard error of roughly 2 percentage points -- so they are excluded
    # rather than absorbed by a loose tolerance.
    minimum_cell_size = 500
    rows_increasing = []
    for eta_band in acceptance_grid.index:
        rates = acceptance_grid.loc[eta_band]
        counts = cell_counts.loc[eta_band]
        reliable = rates[(counts >= minimum_cell_size) & rates.notna()].tolist()

        if len(reliable) >= 2:
            # 1 pp tolerance: adjacent surge bins can wobble from sampling even
            # when the overall within-band gradient is clearly positive.
            rows_increasing.append(_monotonically_increasing(reliable, tolerance=0.01))

    validator.check(
        "acceptance rises with surge once ETA is held fixed",
        all(rows_increasing),
        f"{sum(rows_increasing)}/{len(rows_increasing)} ETA bands increase in surge "
        f"(cells with >= {minimum_cell_size} rides)",
        category,
    )

    # The grid is descriptive; this is the actual estimate. Fitting both terms
    # together recovers the positive incentive effect of surge on drivers that
    # the raw marginal comparison hides.
    coefficients = _fit_acceptance_model(placed_rides)
    validator.check(
        "modelled surge effect on acceptance is positive, ETA effect negative",
        coefficients["surge_multiplier"] > 0 and coefficients["eta_minutes"] < 0,
        f"logit coefficients: surge {coefficients['surge_multiplier']:+.3f}, "
        f"ETA {coefficients['eta_minutes']:+.3f} per minute",
        category,
    )

    surge_eta_correlation = rides["surge_multiplier"].corr(rides["eta_minutes"])
    validator.check(
        "surge and ETA are strongly correlated (the confounder is present)",
        surge_eta_correlation > 0.5,
        f"pearson r = {surge_eta_correlation:.3f}, both driven by driver deficit",
        category,
    )

    churn_by_surge = (
        placed_rides.groupby("surge_bin", observed=True)["churned_to_competitor"]
        .mean()
        .tolist()
    )
    validator.check(
        "competitor churn rises toward the high-surge tail",
        len(churn_by_surge) >= 2 and churn_by_surge[-1] > churn_by_surge[0],
        " -> ".join(f"{value:.3f}" for value in churn_by_surge),
        category,
    )

    peak = zone_state.loc[zone_state["is_peak_hour"]]
    off_peak = zone_state.loc[~zone_state["is_peak_hour"]]
    validator.check(
        "demand is higher during peak hours",
        peak["demand_count"].mean() > off_peak["demand_count"].mean(),
        f"{peak['demand_count'].mean():.2f} vs {off_peak['demand_count'].mean():.2f} "
        "requests per zone-interval",
        category,
    )

    demand_by_zone = zone_state.groupby("zone_id")["demand_count"].mean()
    eta_by_zone = zone_state.groupby("zone_id")["average_eta_minutes"].mean()
    validator.check(
        "zones have distinct profiles",
        demand_by_zone.std() > 0.5 and eta_by_zone.std() > 0.5,
        f"demand sd across zones {demand_by_zone.std():.2f}, "
        f"ETA sd {eta_by_zone.std():.2f} min",
        category,
    )

    zone_types = zones.set_index("zone_id")["zone_type"]
    state_with_type = zone_state.assign(
        zone_type=zone_state["zone_id"].map(zone_types)
    )
    business_zones = state_with_type.loc[state_with_type["zone_type"] == "business"]
    validator.check(
        "business districts empty out at the weekend",
        business_zones.loc[business_zones["is_weekend"], "demand_count"].mean()
        < business_zones.loc[~business_zones["is_weekend"], "demand_count"].mean(),
        f"weekend {business_zones.loc[business_zones['is_weekend'], 'demand_count'].mean():.2f}"
        f" vs weekday "
        f"{business_zones.loc[~business_zones['is_weekend'], 'demand_count'].mean():.2f}",
        category,
    )

    entertainment_zones = state_with_type.loc[
        state_with_type["zone_type"] == "entertainment"
    ]
    night_hours = entertainment_zones.loc[
        entertainment_zones["hour"].isin([21, 22, 23])
    ]
    day_hours = entertainment_zones.loc[
        entertainment_zones["hour"].isin([9, 10, 11, 12, 13])
    ]
    validator.check(
        "nightlife zones surge in the late evening, not at lunchtime",
        night_hours["surge_multiplier"].mean()
        > day_hours["surge_multiplier"].mean(),
        f"late-evening surge {night_hours['surge_multiplier'].mean():.3f} vs "
        f"daytime {day_hours['surge_multiplier'].mean():.3f}",
        category,
    )

    # Supply should follow surge with a lag: today's driver count responds to
    # the surge two intervals ago, not the other way round.
    sample_zone = zone_state["zone_id"].iloc[0]
    single_zone = zone_state.loc[zone_state["zone_id"] == sample_zone].sort_values(
        "timestamp"
    )
    lagged_surge = single_zone["surge_multiplier"].shift(2)
    supply_change = single_zone["available_drivers"].diff()
    lag_correlation = lagged_surge.corr(supply_change)
    validator.check(
        "driver supply responds to earlier surge",
        lag_correlation > 0,
        f"corr(surge[t-2], supply[t] - supply[t-1]) = {lag_correlation:.3f} "
        f"in zone {sample_zone}",
        category,
    )

    suburban_zones = state_with_type.loc[state_with_type["zone_type"] == "suburban"]
    central_zones = state_with_type.loc[state_with_type["zone_type"] == "city_center"]
    validator.check(
        "suburbs wait longer for a car than the centre",
        suburban_zones["average_eta_minutes"].mean()
        > central_zones["average_eta_minutes"].mean(),
        f"suburban {suburban_zones['average_eta_minutes'].mean():.2f} min vs "
        f"central {central_zones['average_eta_minutes'].mean():.2f} min",
        category,
    )

    bad_weather = rides.loc[rides["weather"].isin(["rain", "snow"])]
    good_weather = rides.loc[rides["weather"] == "clear"]
    validator.check(
        "bad weather lengthens ETA",
        bad_weather["eta_minutes"].mean() > good_weather["eta_minutes"].mean(),
        f"{bad_weather['eta_minutes'].mean():.2f} min in rain/snow vs "
        f"{good_weather['eta_minutes'].mean():.2f} min when clear",
        category,
    )

    # --- wartime operating conditions ---

    curfew_state = zone_state.loc[zone_state["curfew"]]
    open_state = zone_state.loc[~zone_state["curfew"]]
    curfew_ratio = curfew_state["demand_count"].mean() / max(
        open_state["demand_count"].mean(), 1e-9
    )
    validator.check(
        "demand collapses during curfew rather than merely dipping",
        curfew_ratio < 0.10,
        f"curfew demand is {curfew_ratio:.1%} of non-curfew demand "
        f"({curfew_state['demand_count'].mean():.2f} vs "
        f"{open_state['demand_count'].mean():.2f} per zone-interval)",
        category,
    )

    validator.check(
        "curfew covers exactly the configured hours",
        set(curfew_state["hour"].unique()) == {0, 1, 2, 3, 4},
        f"curfew hours present: {sorted(curfew_state['hour'].unique())}",
        category,
    )

    # The pre-curfew scramble is the day's surge peak, which is the
    # counter-intuitive part: not the evening commute.
    hourly_surge = zone_state.groupby("hour")["surge_multiplier"].mean()
    peak_surge_hour = int(hourly_surge.idxmax())
    validator.check(
        "surge peaks in the pre-curfew rush, not at the evening commute",
        peak_surge_hour in {22, 23},
        f"highest average surge at {peak_surge_hour:02d}:00 "
        f"({hourly_surge.max():.3f}) vs 18:00 ({hourly_surge.get(18, float('nan')):.3f})",
        category,
    )

    alert_state = zone_state.loc[zone_state["air_raid_alert"] & ~zone_state["curfew"]]
    calm_state = zone_state.loc[
        ~zone_state["air_raid_alert"] & ~zone_state["curfew"]
    ]
    validator.check(
        "air raid alerts cut driver supply harder than they cut demand",
        (
            alert_state["available_drivers"].mean()
            / calm_state["available_drivers"].mean()
        )
        < (alert_state["demand_count"].mean() / calm_state["demand_count"].mean()),
        f"drivers fall to "
        f"{alert_state['available_drivers'].mean() / calm_state['available_drivers'].mean():.1%} "
        f"of normal while demand falls only to "
        f"{alert_state['demand_count'].mean() / calm_state['demand_count'].mean():.1%}",
        category,
    )

    validator.check(
        "alerts raise surge and ETA",
        alert_state["surge_multiplier"].mean()
        > calm_state["surge_multiplier"].mean()
        and alert_state["average_eta_minutes"].mean()
        > calm_state["average_eta_minutes"].mean(),
        f"surge {alert_state['surge_multiplier'].mean():.3f} vs "
        f"{calm_state['surge_multiplier'].mean():.3f}, ETA "
        f"{alert_state['average_eta_minutes'].mean():.2f} vs "
        f"{calm_state['average_eta_minutes'].mean():.2f} min",
        category,
    )

    validator.check(
        "no airport zones remain",
        "airport" not in set(zones["zone_type"]),
        f"zone types present: {', '.join(sorted(zones['zone_type'].unique()))}",
        category,
    )

    entertainment_state = state_with_type.loc[
        state_with_type["zone_type"] == "entertainment"
    ]
    late_evening = entertainment_state.loc[
        entertainment_state["hour"].isin([21, 22, 23])
    ]
    after_midnight = entertainment_state.loc[
        entertainment_state["hour"].isin([0, 1, 2])
    ]
    validator.check(
        "nightlife demand happens before curfew, not after midnight",
        late_evening["demand_count"].mean()
        > 10 * after_midnight["demand_count"].mean(),
        f"{late_evening['demand_count'].mean():.2f} per zone-interval at 21:00-23:00 "
        f"vs {after_midnight['demand_count'].mean():.2f} at 00:00-02:00",
        category,
    )


def run_validation(data_dir: Path) -> Validator:
    zones = pd.read_parquet(data_dir / "zones.parquet")
    users = pd.read_parquet(data_dir / "users.parquet")
    zone_state = pd.read_parquet(data_dir / "zone_state.parquet")
    rides = pd.read_parquet(data_dir / "rides.parquet")

    validator = Validator()
    validate_technical_invariants(validator, zones, users, zone_state, rides)
    validate_behavioural_invariants(validator, zones, zone_state, rides)

    return validator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    arguments = parser.parse_args()

    validator = run_validation(arguments.data_dir)

    current_category = None
    for result in validator.results:
        if result.category != current_category:
            current_category = result.category
            print(f"\n{current_category.upper()} INVARIANTS")
            print("-" * 78)

        marker = "PASS" if result.passed else "FAIL"
        print(f"  [{marker}] {result.name}")
        print(f"         {result.detail}")

    print("-" * 78)
    passed = len(validator.results) - len(validator.failures)
    print(f"{passed}/{len(validator.results)} checks passed")

    return 1 if validator.failures else 0


if __name__ == "__main__":
    sys.exit(main())
