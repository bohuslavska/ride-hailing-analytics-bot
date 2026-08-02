"""
The analytics layer, against the loaded database.

Two kinds of assertion here. The mechanical ones check that rates are computed
from summed counts rather than averaged across groups, which is the usual way
funnel numbers go wrong. The substantive ones check the findings the assistant is
supposed to report: that the curfew is a collapse rather than a dip, that surge
peaks before the curfew rather than at the evening commute, and that controlling
for ETA reverses the apparent sign of the surge/acceptance relationship. If the
simulation is ever retuned and those stop holding, the answers become wrong while
every unit test still passes.
"""

from __future__ import annotations

import pytest

from src.analytics.conversion import analyze_conversion, compare_acceptance_confounding
from src.analytics.metrics import (
    calculate_funnel,
    marketplace_profile,
    zone_supply_demand_summary,
)
from tests.conftest import needs_database

pytestmark = needs_database


class TestFunnel:
    def test_overall_rates_are_consistent_with_the_counts(self) -> None:
        result = calculate_funnel()
        row = result["table"][0]

        assert row["place_conversion"] == pytest.approx(
            row["placed"] / row["calculated"], abs=1e-4
        )
        assert row["acceptance_rate"] == pytest.approx(
            row["accepted"] / row["placed"], abs=1e-4
        )
        assert row["end_to_end_conversion"] == pytest.approx(
            row["accepted"] / row["calculated"], abs=1e-4
        )

    def test_the_funnel_only_narrows(self) -> None:
        row = calculate_funnel()["table"][0]
        assert row["calculated"] >= row["placed"] >= row["accepted"] > 0

    def test_splitting_by_a_dimension_preserves_the_total(self) -> None:
        """A group-by that loses or double-counts rows would go unnoticed otherwise."""
        overall = calculate_funnel()["table"][0]
        by_hour = calculate_funnel(dimension="hour")["table"]

        assert sum(row["calculated"] for row in by_hour) == overall["calculated"]
        assert sum(row["accepted"] for row in by_hour) == overall["accepted"]

    def test_hourly_rows_carry_the_curfew_flag_and_a_warning(self) -> None:
        result = calculate_funnel(dimension="hour")

        assert "curfew_warning" in result
        flagged = {row["hour"] for row in result["table"] if row["is_curfew_hour"]}
        assert flagged == {0, 1, 2, 3, 4}

    def test_an_unknown_dimension_is_rejected_by_name(self) -> None:
        with pytest.raises(ValueError, match="Unknown dimension"):
            calculate_funnel(dimension="'; DROP TABLE rides --")


class TestMarketplaceProfile:
    def test_the_daily_profile_covers_every_hour(self) -> None:
        table = marketplace_profile(dimension="hour")["table"]
        assert [row["hour"] for row in table] == list(range(24))

    def test_demand_collapses_during_curfew_rather_than_dipping(self) -> None:
        """
        The single most distinctive feature of this market. A model that treats
        the overnight trough as ordinary quiet demand will recommend promotions
        during hours when movement is prohibited.
        """
        table = marketplace_profile(dimension="hour")["table"]
        curfew = [r["avg_demand"] for r in table if r["is_curfew_hour"]]
        awake = [r["avg_demand"] for r in table if not r["is_curfew_hour"]]

        assert max(curfew) < min(awake) / 10

    def test_surge_peaks_in_the_pre_curfew_rush_not_the_evening_commute(self) -> None:
        table = marketplace_profile(dimension="hour")["table"]
        awake = [row for row in table if not row["is_curfew_hour"]]
        peak = max(awake, key=lambda row: row["avg_surge"])

        assert peak["hour"] in {22, 23}

        evening_commute = max(
            (row for row in table if row["hour"] in {17, 18, 19}),
            key=lambda row: row["avg_surge"],
        )
        assert peak["avg_surge"] > evening_commute["avg_surge"]

    def test_the_surge_peak_is_a_supply_shortfall_not_a_demand_spike(self) -> None:
        """The mechanism behind the finding above, which the answer should explain."""
        table = {row["hour"]: row for row in marketplace_profile()["table"]}

        assert table[23]["avg_demand"] < table[19]["avg_demand"]
        assert table[23]["avg_available_drivers"] < table[19]["avg_available_drivers"]
        assert table[23]["avg_demand_supply_ratio"] > table[19]["avg_demand_supply_ratio"]

    def test_zone_type_filtering_changes_the_profile(self) -> None:
        entertainment = marketplace_profile(zone_type="entertainment")["table"]
        business = marketplace_profile(zone_type="business")["table"]

        busiest = lambda table: max(  # noqa: E731
            (r for r in table if not r["is_curfew_hour"]),
            key=lambda r: r["avg_demand"],
        )["hour"]

        # Nightlife peaks late, offices peak at commuting time.
        assert busiest(entertainment) >= 20
        assert busiest(business) < 20

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"dimension": "zone_id"}, "Unknown dimension"),
            ({"zone_type": "airport"}, "Unknown zone_type"),
        ],
    )
    def test_unknown_arguments_are_rejected(self, kwargs: dict, message: str) -> None:
        """Neither argument may be interpolated from model-supplied text."""
        with pytest.raises(ValueError, match=message):
            marketplace_profile(**kwargs)

    def test_no_airport_zone_type_survives_in_the_data(self) -> None:
        """Civil aviation is closed; rail is the intercity gateway instead."""
        zone_types = {row["zone_type"] for row in zone_supply_demand_summary()["table"]}

        assert "airport" not in zone_types
        assert "railway_station" in zone_types


class TestConversion:
    def test_longer_etas_convert_worse(self) -> None:
        buckets = analyze_conversion()["buckets"]
        assert buckets[0]["rate"] > buckets[-1]["rate"]

    def test_the_eta_effect_survives_controlling_for_the_other_drivers(self) -> None:
        result = analyze_conversion()
        model = result["controlled_model"]

        assert model["available"]
        # Waiting longer makes an order less likely, so the odds ratio per extra
        # minute has to sit below 1.
        assert model["driver_odds_ratio_per_unit"] < 1
        assert result["raw_association"]["point_biserial_correlation"] < 0

    def test_controlling_for_eta_reverses_the_apparent_effect_of_surge(self) -> None:
        """
        The trap this project is built around. Marginally, acceptance looks flat
        or slightly falling as surge rises, because surge and ETA both rise when
        drivers are scarce. Within a fixed ETA band the relationship turns
        positive.
        """
        result = compare_acceptance_confounding()

        assert result["surge_eta_correlation"] > 0.5

        marginal = result["marginal_view"]
        assert marginal[-1]["acceptance_rate"] <= marginal[0]["acceptance_rate"]

        # Within each ETA band that has enough observations, acceptance should
        # rise as surge rises -- the opposite of the marginal picture above.
        conditional = result["conditional_view"]
        rates = conditional["acceptance_rate"]
        counts = conditional["observations"]
        floor = conditional["minimum_cell_size"]

        improving_bands = 0
        comparable_bands = 0
        for band_rates, band_counts in zip(rates, counts, strict=True):
            usable = [
                rate
                for rate, count in zip(band_rates, band_counts, strict=True)
                if rate is not None and count >= floor
            ]
            if len(usable) >= 2:
                comparable_bands += 1
                improving_bands += usable[-1] > usable[0]

        # Sparse ETA×surge corners (e.g. very long ETA at low surge) are empty
        # by construction; only bands with enough cells to compare are scored.
        assert comparable_bands >= 3
        assert improving_bands == comparable_bands

    def test_confounding_grid_covers_every_placed_ride(self) -> None:
        """
        This analysis used to be computed from `ORDER BY ride_id LIMIT n`. Because
        ride_id runs in chronological order, that silently restricted the headline
        finding to the opening days of the horizon. The marginal buckets partition
        the placed population, so their counts have to add back up to the funnel.
        """
        result = compare_acceptance_confounding()
        counted = sum(row["observations"] for row in result["marginal_view"])

        placed = calculate_funnel()["table"][0]["placed"]
        assert counted == placed

    def test_the_confounding_result_explains_itself_in_words(self) -> None:
        """
        The interpretation is what the assistant paraphrases. If it stopped
        mentioning the confound, the model would lose its cue to report both
        views and would confidently give the marginal answer alone.
        """
        interpretation = " ".join(compare_acceptance_confounding()["interpretation"])
        assert "flat" in interpretation.lower()
        assert "eta" in interpretation.lower()
