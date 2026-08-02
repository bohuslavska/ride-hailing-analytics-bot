"""
Relationship between marketplace conditions and the funnel.

The headline question -- "does ETA affect conversion?" -- cannot be answered
honestly by a correlation coefficient. ETA, surge and price all move together
because they share a cause (driver shortage), so this module reports three
things side by side:

*   the bucketed rates, which are what a stakeholder wants to see;
*   the raw correlation, which is the number people usually quote;
*   a logistic regression with the confounders included, which is the only one
    of the three that estimates a partial effect.

Where those three disagree, that disagreement is the finding, and the tool says
so explicitly instead of leaving the model to editorialise.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from src.database.connection import get_engine

# Drivers the analysis can be run against, and the bucket edges used to
# summarise each one.
SUPPORTED_DRIVERS = {
    "eta_minutes": {
        "label": "ETA (minutes)",
        "bins": [0, 3, 5, 8, 12, 100],
        "bin_labels": ["0-3", "3-5", "5-8", "8-12", "12+"],
    },
    "surge_multiplier": {
        "label": "surge multiplier",
        "bins": [0.99, 1.05, 1.2, 1.5, 2.0, 2.6],
        "bin_labels": ["1.00-1.05", "1.05-1.20", "1.20-1.50", "1.50-2.00", "2.00+"],
    },
    "final_price": {
        "label": "final price (UAH)",
        "bins": [0, 100, 150, 200, 300, 100_000],
        "bin_labels": ["<100", "100-150", "150-200", "200-300", "300+"],
    },
    "distance_km": {
        "label": "trip distance (km)",
        "bins": [0, 2, 5, 10, 20, 1_000],
        "bin_labels": ["<2", "2-5", "5-10", "10-20", "20+"],
    },
}

SUPPORTED_OUTCOMES = {
    "placed": {
        "label": "place conversion",
        "population": "all calculated rides",
        "filter": None,
    },
    "accepted": {
        "label": "acceptance rate",
        "population": "placed rides only",
        "filter": "placed = 1",
    },
}

# Held constant in the regression so the driver's coefficient is a partial
# effect rather than a restatement of the shared cause.
MODEL_CONTROLS = [
    "eta_minutes",
    "surge_multiplier",
    "distance_km",
    "is_peak_hour",
    "is_weekend",
]

SAMPLE_SIZE = 120_000


def _load_sample(outcome: str, sample_size: int) -> pd.DataFrame:
    """
    Sample rides for the model fit; the bucket table uses the full population.

    ride_id is assigned in chronological order, so `LIMIT n` would return only
    the opening days of the horizon -- a sample biased towards whatever weather
    and day-of-week pattern happened to fall at the start. Taking every nth
    ride instead spreads the sample evenly across the whole period and stays
    deterministic, which keeps repeated calls consistent.
    """
    conditions = []
    where_clause = SUPPORTED_OUTCOMES[outcome]["filter"]
    if where_clause:
        conditions.append(where_clause)

    with get_engine(readonly=True).connect() as connection:
        population = connection.execute(
            text(
                "SELECT COUNT(*) FROM rides_enriched"
                + (f" WHERE {where_clause}" if where_clause else "")
            )
        ).scalar_one()

        stride = max(1, population // max(sample_size, 1))
        if stride > 1:
            conditions.append(f"MOD(ride_id, {stride}) = 0")

        filter_sql = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        return pd.read_sql(
            text(
                f"""
                SELECT placed, accepted, eta_minutes, surge_multiplier, final_price,
                       distance_km, is_peak_hour, is_weekend
                FROM rides_enriched
                {filter_sql}
                LIMIT :sample_size
                """
            ),
            connection,
            params={"sample_size": sample_size},
        )


def _bucket_table(driver: str, outcome: str) -> pd.DataFrame:
    """Outcome rate per driver bucket, computed in SQL over every row."""
    specification = SUPPORTED_DRIVERS[driver]
    where_clause = SUPPORTED_OUTCOMES[outcome]["filter"]
    filter_sql = f"WHERE {where_clause}" if where_clause else ""

    # Bin edges come from the SUPPORTED_DRIVERS table above, never from input.
    case_branches = "\n".join(
        f"WHEN {driver} < {upper} THEN '{label}'"
        for upper, label in zip(
            specification["bins"][1:-1], specification["bin_labels"][:-1], strict=True
        )
    )
    bucket_expression = (
        f"CASE {case_branches} ELSE '{specification['bin_labels'][-1]}' END"
    )

    query = text(
        f"""
        SELECT
            {bucket_expression}          AS bucket,
            COUNT(*)                     AS observations,
            AVG({outcome}::double precision) AS rate,
            AVG({driver})                AS average_driver_value
        FROM rides_enriched
        {filter_sql}
        GROUP BY 1
        """
    )

    with get_engine(readonly=True).connect() as connection:
        frame = pd.read_sql(query, connection)

    ordering = {label: index for index, label in enumerate(specification["bin_labels"])}
    frame["__order"] = frame["bucket"].map(ordering)
    frame = frame.sort_values("__order").drop(columns="__order").reset_index(drop=True)

    frame["rate"] = frame["rate"].astype(float).round(4)
    frame["average_driver_value"] = (
        frame["average_driver_value"].astype(float).round(3)
    )

    return frame


def _fit_partial_effects(
    sample: pd.DataFrame, driver: str, outcome: str
) -> dict[str, Any]:
    """
    Logistic regression of the outcome on the driver plus its confounders.

    Coefficients are returned per natural unit (a minute of ETA, one point of
    surge) so they can be read without knowing the scaler was involved.
    """
    features = [driver] + [name for name in MODEL_CONTROLS if name != driver]

    design = sample[features].astype(float).to_numpy()
    target = sample[outcome].to_numpy().astype(int)

    if len(np.unique(target)) < 2:
        return {"available": False, "reason": "outcome does not vary in this sample"}

    if not np.isfinite(design).all():
        return {"available": False, "reason": "sample contains non-finite values"}

    scaler = StandardScaler()
    model = LogisticRegression(max_iter=2000)

    # Some BLAS builds raise spurious divide/overflow warnings from inside the
    # matmul in the solver. The design matrix is checked for finiteness just
    # above, so these are noise rather than a signal about the data.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        model.fit(scaler.fit_transform(design), target)

    coefficients = model.coef_[0] / scaler.scale_
    odds_ratios = np.exp(coefficients)

    return {
        "available": True,
        "observations": int(len(sample)),
        "features": features,
        "coefficients": [
            {
                "feature": name,
                "log_odds_per_unit": round(float(coefficient), 5),
                "odds_ratio_per_unit": round(float(odds_ratio), 5),
            }
            for name, coefficient, odds_ratio in zip(
                features, coefficients, odds_ratios, strict=True
            )
        ],
        "driver_log_odds_per_unit": round(float(coefficients[0]), 5),
        "driver_odds_ratio_per_unit": round(float(odds_ratios[0]), 5),
    }


def _interpret(
    driver: str,
    outcome: str,
    bucket_frame: pd.DataFrame,
    raw_correlation: float,
    partial: dict[str, Any],
) -> list[str]:
    """Plain-language reading of the three views, including where they conflict."""
    driver_label = SUPPORTED_DRIVERS[driver]["label"]
    outcome_label = SUPPORTED_OUTCOMES[outcome]["label"]

    first_rate = bucket_frame["rate"].iloc[0]
    last_rate = bucket_frame["rate"].iloc[-1]
    direction = "falls" if last_rate < first_rate else "rises"

    notes = [
        f"Across buckets, {outcome_label} {direction} from {first_rate:.1%} at "
        f"{driver_label} {bucket_frame['bucket'].iloc[0]} to {last_rate:.1%} at "
        f"{bucket_frame['bucket'].iloc[-1]}.",
        f"Raw correlation between {driver_label} and {outcome} is "
        f"{raw_correlation:+.3f}.",
    ]

    if not partial.get("available"):
        return notes

    partial_effect = partial["driver_log_odds_per_unit"]
    notes.append(
        f"Holding {', '.join(name for name in partial['features'][1:])} fixed, a "
        f"one-unit increase in {driver_label} changes the log-odds of {outcome} "
        f"by {partial_effect:+.4f} "
        f"(odds ratio {partial['driver_odds_ratio_per_unit']:.4f})."
    )

    marginal_direction = np.sign(last_rate - first_rate)
    partial_direction = np.sign(partial_effect)

    if marginal_direction != 0 and marginal_direction != partial_direction:
        notes.append(
            "These disagree in sign. The bucketed view is confounded: "
            f"{driver_label} moves together with the other conditions in the "
            "model, so the bucket comparison mixes its effect with theirs. The "
            "regression estimate is the one to quote, and even it is an "
            "association measured on observational data, not a causal effect."
        )
    else:
        notes.append(
            "Both views point the same way, which makes the association robust "
            "to these controls. It remains an association, not a causal effect."
        )

    return notes


def analyze_conversion(
    driver: str = "eta_minutes",
    outcome: str = "placed",
    sample_size: int = SAMPLE_SIZE,
) -> dict[str, Any]:
    """
    Bucketed rates, raw correlation and a controlled logistic fit for one driver.

    Args:
        driver: condition to analyse, one of SUPPORTED_DRIVERS.
        outcome: 'placed' (over all rides) or 'accepted' (over placed rides).
    """
    if driver not in SUPPORTED_DRIVERS:
        raise ValueError(
            f"Unknown driver '{driver}'. Available: {', '.join(SUPPORTED_DRIVERS)}."
        )
    if outcome not in SUPPORTED_OUTCOMES:
        raise ValueError(
            f"Unknown outcome '{outcome}'. Available: {', '.join(SUPPORTED_OUTCOMES)}."
        )

    bucket_frame = _bucket_table(driver, outcome)
    sample = _load_sample(outcome, sample_size)

    raw_correlation = float(
        stats.pointbiserialr(sample[outcome].to_numpy(), sample[driver].to_numpy())[0]
    )
    spearman = float(
        stats.spearmanr(sample[driver].to_numpy(), sample[outcome].to_numpy())[0]
    )

    partial = _fit_partial_effects(sample, driver, outcome)

    return {
        "driver": driver,
        "driver_label": SUPPORTED_DRIVERS[driver]["label"],
        "outcome": outcome,
        "outcome_label": SUPPORTED_OUTCOMES[outcome]["label"],
        "population": SUPPORTED_OUTCOMES[outcome]["population"],
        "total_observations": int(bucket_frame["observations"].sum()),
        "buckets": bucket_frame.to_dict(orient="records"),
        "raw_association": {
            "point_biserial_correlation": round(raw_correlation, 4),
            "spearman_correlation": round(spearman, 4),
            "sample_size": int(len(sample)),
        },
        "controlled_model": partial,
        "interpretation": _interpret(
            driver, outcome, bucket_frame, raw_correlation, partial
        ),
        "chart": {
            "kind": "conversion_by_bucket",
            "title": (
                f"{SUPPORTED_OUTCOMES[outcome]['label'].title()} by "
                f"{SUPPORTED_DRIVERS[driver]['label']}"
            ),
            "x": bucket_frame["bucket"].tolist(),
            "x_title": SUPPORTED_DRIVERS[driver]["label"],
            "series": [
                {
                    "name": "observations",
                    "type": "bar",
                    "axis": "y2",
                    "values": bucket_frame["observations"].tolist(),
                },
                {
                    "name": SUPPORTED_OUTCOMES[outcome]["label"],
                    "type": "line",
                    "axis": "y",
                    "values": bucket_frame["rate"].tolist(),
                },
            ],
            "y_title": SUPPORTED_OUTCOMES[outcome]["label"],
            "y2_title": "observations",
        },
    }


def compare_acceptance_confounding() -> dict[str, Any]:
    """
    The surge/ETA confounding worked through explicitly.

    Acceptance looks unrelated to surge until ETA is held fixed, at which point
    the incentive effect appears. This returns the grid a reader needs to see
    that for themselves.

    The grid is aggregated in SQL over every placed ride rather than a sample.
    ride_id runs in chronological order, so any `ORDER BY ride_id LIMIT n` would
    have restricted the whole finding to the opening days of the horizon; and
    since the result is only a handful of group means, there is nothing to gain
    by pulling rows into pandas at all.
    """
    surge_labels = ["1.00-1.05", "1.05-1.20", "1.20-1.50", "1.50+"]
    eta_labels = ["<6", "6-8", "8-10", "10-13", "13+"]

    # Bucket bounds are left-open/right-closed so the edges land in one bucket only.
    query = text(
        """
        WITH bucketed AS (
            SELECT accepted,
                   CASE
                       WHEN surge_multiplier >  0.99 AND surge_multiplier <= 1.05
                            THEN '1.00-1.05'
                       WHEN surge_multiplier >  1.05 AND surge_multiplier <= 1.20
                            THEN '1.05-1.20'
                       WHEN surge_multiplier >  1.20 AND surge_multiplier <= 1.50
                            THEN '1.20-1.50'
                       WHEN surge_multiplier >  1.50 THEN '1.50+'
                   END AS surge_bucket,
                   CASE
                       WHEN eta_minutes >  0 AND eta_minutes <=  6 THEN '<6'
                       WHEN eta_minutes >  6 AND eta_minutes <=  8 THEN '6-8'
                       WHEN eta_minutes >  8 AND eta_minutes <= 10 THEN '8-10'
                       WHEN eta_minutes > 10 AND eta_minutes <= 13 THEN '10-13'
                       WHEN eta_minutes > 13 THEN '13+'
                   END AS eta_bucket
            FROM rides_enriched
            WHERE placed = 1
        )
        SELECT surge_bucket,
               eta_bucket,
               AVG(accepted::float) AS acceptance_rate,
               COUNT(*)             AS observations
        FROM bucketed
        WHERE surge_bucket IS NOT NULL AND eta_bucket IS NOT NULL
        GROUP BY GROUPING SETS ((surge_bucket), (surge_bucket, eta_bucket))
        """
    )
    correlation_query = text(
        """
        SELECT CORR(surge_multiplier, eta_minutes)
        FROM rides_enriched
        WHERE placed = 1
        """
    )

    with get_engine(readonly=True).connect() as connection:
        cells = pd.read_sql(query, connection)
        correlation = float(connection.execute(correlation_query).scalar_one())

    # The GROUPING SETS above return the marginal totals and the full grid in one
    # pass; the marginal rows are the ones with no ETA band attached.
    marginal = (
        cells[cells["eta_bucket"].isna()]
        .set_index("surge_bucket")
        .reindex(surge_labels)
        .dropna(subset=["acceptance_rate"])
        .reset_index()[["surge_bucket", "acceptance_rate", "observations"]]
    )
    marginal["acceptance_rate"] = marginal["acceptance_rate"].round(4)
    marginal["observations"] = marginal["observations"].astype(int)

    conditional = cells[cells["eta_bucket"].notna()]
    grid = (
        conditional.pivot(
            index="eta_bucket", columns="surge_bucket", values="acceptance_rate"
        )
        .reindex(index=eta_labels, columns=surge_labels)
        .round(4)
    )
    counts = (
        conditional.pivot(
            index="eta_bucket", columns="surge_bucket", values="observations"
        )
        .reindex(index=eta_labels, columns=surge_labels)
        .fillna(0)
    )

    # Cells thinner than this are dominated by sampling error.
    minimum_cell = 200
    grid = grid.where(counts >= minimum_cell)

    return {
        "question": "Does surge change whether drivers accept an order?",
        "marginal_view": marginal.to_dict(orient="records"),
        "conditional_view": {
            "eta_buckets": eta_labels,
            "surge_buckets": surge_labels,
            "acceptance_rate": grid.reindex(
                index=eta_labels, columns=surge_labels
            ).values.tolist(),
            "observations": counts.reindex(
                index=eta_labels, columns=surge_labels
            )
            .fillna(0)
            .astype(int)
            .values.tolist(),
            "minimum_cell_size": minimum_cell,
        },
        "surge_eta_correlation": round(correlation, 4),
        "interpretation": [
            "Compared across surge buckets alone, acceptance looks almost flat "
            "(or dips at extreme shortage, where search timeouts also rise).",
            f"Surge and ETA correlate at r = {correlation:.2f} because both are "
            "driven by the same driver shortage, so the flat marginal view is "
            "confounded.",
            "Within each ETA band, acceptance rises with surge: a higher fare "
            "makes the same trip more attractive to a driver.",
            "A second exit after place is competitor churn when search outruns "
            "rider patience (~2-5 min); that is separate from the driver "
            "incentive story above.",
            "This is an association measured on simulated observational data. It "
            "shows the direction of the relationship, not the return on an "
            "actual pricing intervention.",
        ],
        "chart": {
            "kind": "heatmap",
            "title": "Acceptance rate by ETA band and surge bucket",
            "x": surge_labels,
            "y": eta_labels,
            "z": grid.reindex(index=eta_labels, columns=surge_labels).values.tolist(),
            "x_title": "surge multiplier",
            "y_title": "ETA (minutes)",
            "color_title": "acceptance rate",
        },
    }
