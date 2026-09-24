"""Independent examples for fixed-cohort rankings and regional report aggregation."""

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from omnia_results.metrics import Results, SECTORS
from omnia_results.report_data import ReportData


class ReportDataTests(unittest.TestCase):
    def setUp(self):
        self.config = {"macroregions": {"Pair": ["A", "B"], "Other": ["C"]},
                       "producer_groups": {"A": ["A"], "B": ["B"], "C": ["C"]},
                       "ranking_year": 2050, "price_weight_year": 2019, "top_n": 2}

    def report(self, rows=None, config=None, policy="preserve"):
        if rows is None:
            rows = [["baseline_noce", r, "placeholder", "Mt/yr", 1, 1] for r in "ABC"]
        frame = pd.DataFrame(rows, columns=["Scenario", "Region", "Variable", "Unit", "2019", "2050"])
        with patch("omnia_results.metrics.pd.read_csv", return_value=frame):
            results = Results(Path("fixture.csv"), "baseline_noce", list("ABC"), [2019, 2050], policy)
        return ReportData(results, self.config if config is None else config)

    @staticmethod
    def rows(variable, unit, values):
        return [["baseline_noce", region, variable, unit, *years] for region, years in values.items()]

    def production(self, values, sector=SECTORS[0], route=""):
        return self.rows(f"Production|{sector.path}{route}", "Mt/yr", values)

    def test_both_group_definitions_must_partition_regions(self):
        for field in ("macroregions", "producer_groups"):
            for invalid in ({"one": ["A", "B", "C"], "duplicate": ["A"]},
                            {"one": ["A", "A", "B", "C"]},
                            {"one": ["A", "B"]},
                            {"one": ["A", "B", "C", "D"]}):
                with self.subTest(field=field, invalid=invalid):
                    config = deepcopy(self.config)
                    config[field] = invalid
                    with self.assertRaisesRegex(ValueError, "partition"):
                        self.report(config=config)

    def test_empty_groups_and_invalid_configuration_are_rejected(self):
        for field in ("macroregions", "producer_groups"):
            config = deepcopy(self.config)
            config[field]["empty"] = []
            with self.assertRaisesRegex(ValueError, "nonempty"):
                self.report(config=config)
        for value in (0, -1, 1.5, True, "12"):
            with self.subTest(top_n=value), self.assertRaisesRegex(ValueError, "positive integer"):
                self.report(config={**self.config, "top_n": value})
        for field in ("ranking_year", "price_weight_year"):
            with self.assertRaisesRegex(ValueError, field):
                self.report(config={**self.config, field: 2040})

    def test_carbon_prices_use_fixed_emissions_weights(self):
        rows = self.rows("Emissions|GHG|Industry", "MtCO2e/yr",
                         {"A": [1, 9], "B": [3, 1], "C": [2, 5]})
        rows += self.rows("Price|Carbon", "USD_2010/t CO2e",
                          {"A": [10, 20], "B": [30, 80], "C": [5, 10]})
        report = self.report(rows)
        self.assertEqual(report.carbon_prices().loc["Pair"].tolist(), [25, 65])
        self.assertEqual(report.carbon_prices().loc["Other"].tolist(), [5, 10])
        weights = report.carbon_price_weights()
        self.assertEqual(weights.Weight.tolist(), [0.25, 0.75, 1])
        self.assertEqual(weights.WeightYear.tolist(), [2019, 2019, 2019])
        self.assertEqual(weights.groupby("Group").Weight.sum().tolist(), [1, 1])
        # Time-varying weights would give 26 in 2050, not the fixed-weight 65.

    def test_zero_weight_missing_price_is_ignored_positive_weight_is_not(self):
        rows = self.rows("Emissions|GHG|Industry", "MtCO2e/yr",
                         {"A": [0, 5], "B": [4, 5], "C": [2, 2]})
        rows += self.rows("Price|Carbon", "USD_2010/t CO2e",
                          {"A": [None, None], "B": [10, None], "C": [5, 10]})
        prices = self.report(rows, policy="zero").carbon_prices()
        self.assertEqual(prices.loc["Pair", 2019], 10)
        self.assertTrue(np.isnan(prices.loc["Pair", 2050]))

    def test_missing_negative_or_zero_group_weight_is_rejected(self):
        for values in ({"A": [None, 1], "B": [1, 1], "C": [1, 1]},
                       {"A": [-1, 1], "B": [3, 1], "C": [1, 1]},
                       {"A": [0, 1], "B": [0, 1], "C": [1, 1]}):
            report = self.report(self.rows("Emissions|GHG|Industry", "MtCO2e/yr", values), policy="zero")
            with self.assertRaisesRegex(ValueError, "base emissions"):
                report.carbon_price_weights()

    def test_absent_base_emissions_row_is_not_zero_filled(self):
        rows = self.rows("Emissions|GHG|Industry", "MtCO2e/yr", {"A": [1, 1], "B": [1, 1]})
        rows += self.rows("placeholder", "Mt/yr", {"C": [1, 1]})
        with self.assertRaisesRegex(ValueError, "missing emissions are not zero-filled"):
            self.report(rows, policy="zero").carbon_price_weights()

    def test_production_excludes_clinker_and_sums_primary_secondary(self):
        rows = self.production({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        rows += self.rows("Production|Non-Metallic Minerals|Cement Clinker", "Mt/yr",
                          {"A": [100, 100], "B": [100, 100], "C": [100, 100]})
        for sector in SECTORS[1:]:
            rows += self.production({"A": [1, 2], "B": [3, 4], "C": [5, 6]}, sector, "|Primary")
            rows += self.production({"A": [2, 3], "B": [4, 5], "C": [6, 7]}, sector, "|Secondary")
        report = self.report(rows)
        self.assertEqual(report.production_by_region("cement").loc["A"].tolist(), [1, 2])
        for sector in SECTORS[1:]:
            self.assertEqual(report.production_by_region(sector).loc["A"].tolist(), [3, 5])

    def test_group_sum_preserves_incomplete_members(self):
        rows = self.production({"A": [1, 2], "B": [None, 4], "C": [5, 6]})
        config = {**self.config, "producer_groups": self.config["macroregions"]}
        grouped = self.report(rows, config=config).production_groups("cement")
        self.assertTrue(np.isnan(grouped.loc["Pair", 2019]))
        self.assertEqual(grouped.loc["Pair", 2050], 6)

    def test_ranked_cohort_is_fixed_and_ties_use_label_order(self):
        rows = self.production({"A": [1, 20], "B": [2, 20], "C": [100, 10]})
        config = {**self.config, "producer_groups": {"C": ["C"], "B": ["B"], "A": ["A"]}}
        report = self.report(rows, config=config)
        self.assertEqual(report.ranked_producers("cement"), ["A", "B"])
        np.testing.assert_allclose(report.producer_coverage("cement"), [300 / 103, 80])

    def test_ranking_excludes_zero_negative_and_missing_endpoints(self):
        for excluded in (0, -1, None):
            rows = self.production({"A": [1, 20], "B": [2, 10], "C": [100, excluded]})
            self.assertEqual(self.report(rows, config={**self.config, "top_n": 12})
                             .ranked_producers("cement"), ["A", "B"])

    def test_missing_global_denominator_keeps_coverage_undefined(self):
        rows = self.production({"A": [1, 20], "B": [2, 10], "C": [None, None]})
        self.assertTrue(self.report(rows).producer_coverage("cement").isna().all())

    def test_reported_intensity_is_production_weighted(self):
        rows = self.production({"A": [10, 10], "B": [30, 30], "C": [0, 0]})
        rows += self.rows("Energy Intensity|Industry|" + SECTORS[0].path, "EJ/Mt",
                          {"A": [0.1, 0.1], "B": [0.3, 0.3], "C": [None, None]})
        config = {**self.config, "producer_groups": self.config["macroregions"]}
        intensity = self.report(rows, config=config).reported_intensity_groups("cement")
        self.assertEqual(intensity.loc["Pair"].tolist(), [250, 250])
        self.assertTrue(intensity.loc["Other"].isna().all())
        # Ratio of implied energy to production is 10 / 40 * 1000 = 250 GJ/t;
        # an unweighted mean of source intensities would incorrectly give 200.

    def test_missing_intensity_for_positive_production_remains_missing(self):
        rows = self.production({"A": [10, 10], "B": [0, 30], "C": [1, 1]})
        rows += self.rows("Energy Intensity|Industry|" + SECTORS[0].path, "EJ/Mt",
                          {"A": [0.1, 0.1], "B": [None, None], "C": [0.1, 0.1]})
        config = {**self.config, "producer_groups": self.config["macroregions"]}
        intensity = self.report(rows, config=config, policy="zero").reported_intensity_groups("cement")
        self.assertEqual(intensity.loc["Pair", 2019], 100)
        self.assertTrue(np.isnan(intensity.loc["Pair", 2050]))

    def test_missing_production_cannot_be_excluded_from_intensity(self):
        rows = self.production({"A": [10, 10], "B": [None, 30], "C": [1, 1]})
        rows += self.rows("Energy Intensity|Industry|" + SECTORS[0].path, "EJ/Mt",
                          {"A": [0.1, 0.1], "B": [0.3, 0.3], "C": [0.1, 0.1]})
        config = {**self.config, "producer_groups": self.config["macroregions"]}
        intensity = self.report(rows, config=config).reported_intensity_groups("cement")
        self.assertTrue(np.isnan(intensity.loc["Pair", 2019]))

    def test_membership_and_rank_exports(self):
        rows = self.production({"A": [1, 20], "B": [2, 10], "C": [100, 5]})
        for sector in SECTORS[1:]:
            rows += self.production({"A": [1, 10], "B": [2, 5], "C": [100, 2.5]}, sector, "|Primary")
            rows += self.production({"A": [1, 10], "B": [2, 5], "C": [100, 2.5]}, sector, "|Secondary")
        report = self.report(rows)
        membership = report.membership()
        self.assertEqual(membership.groupby("GroupType").size().to_dict(), {"macroregion": 3, "producer": 3})
        ranking = report.producer_ranking_table()
        self.assertEqual(len(ranking), 6)
        self.assertEqual(ranking.Rank.tolist(), [1, 2, 1, 2, 1, 2])
        np.testing.assert_allclose(ranking.GlobalShare_percent, [2000 / 35, 1000 / 35] * 3)

    def test_emitter_ranking_uses_emissions_and_exact_unit_with_fixed_cohort(self):
        variable = f"Emissions|GHG|Industry|{SECTORS[0].path}"
        rows = self.rows(variable, "MtCO2e/yr", {"A": [1, 30], "B": [2, 20], "C": [100, 10]})
        rows += self.rows(variable, "MtCO2e/Mt", {"A": [999, 1], "B": [999, 2], "C": [999, 3]})
        rows += self.production({"A": [1, 1], "B": [2, 2], "C": [100, 100]})
        rows += self.rows("Emissions|GHG|Industry", "MtCO2e/yr",
                          {"A": [1000, 1000], "B": [1000, 1000], "C": [1000, 1000]})
        report = self.report(rows)
        self.assertEqual(report.emissions_by_region("cement").loc["A"].tolist(), [1, 30])
        self.assertEqual(report.ranked_emitters("cement"), ["A", "B"])
        self.assertEqual(report.ranked_producers("cement"), ["C", "B"])
        np.testing.assert_allclose(report.emitter_coverage("cement"), [300 / 103, 5000 / 60])

    def test_emitter_regions_are_summed_before_ranking_and_remainder_closes_total(self):
        variable = f"Emissions|GHG|Industry|{SECTORS[0].path}"
        rows = self.rows(variable, "MtCO2e/yr", {"A": [1, 12], "B": [2, 12], "C": [100, 20]})
        config = {**self.config, "producer_groups": {"Europe": ["A", "B"], "Other": ["C"]}, "top_n": 1}
        report = self.report(rows, config=config)
        groups = report.emissions_groups("cement")
        self.assertEqual(groups.loc["Europe"].tolist(), [3, 24])
        self.assertEqual(report.ranked_emitters("cement"), ["Europe"])
        np.testing.assert_allclose(report.emitter_coverage("cement"), [300 / 103, 2400 / 44])
        selected = groups.loc[report.ranked_emitters("cement")].sum(min_count=1)
        remainder = groups.drop(index=report.ranked_emitters("cement")).sum(min_count=1)
        np.testing.assert_allclose(selected + remainder, [103, 44])
        np.testing.assert_allclose(selected + remainder, report.results.emissions(SECTORS[0]))

    def test_missing_emissions_in_group_or_remainder_keep_coverage_undefined(self):
        variable = f"Emissions|GHG|Industry|{SECTORS[0].path}"
        rows = self.rows(variable, "MtCO2e/yr", {"A": [1, 12], "B": [None, 12], "C": [5, None]})
        config = {**self.config, "producer_groups": {"Europe": ["A", "B"], "Other": ["C"]}, "top_n": 1}
        report = self.report(rows, config=config)
        groups = report.emissions_groups("cement")
        self.assertTrue(np.isnan(groups.loc["Europe", 2019]))
        self.assertEqual(groups.loc["Europe", 2050], 24)
        self.assertTrue(np.isnan(groups.loc["Other", 2050]))
        self.assertEqual(report.ranked_emitters("cement"), ["Europe"])
        self.assertTrue(report.emitter_coverage("cement").isna().all())

    def test_emissions_respect_explicit_zero_policy_for_blanks_and_absent_regions(self):
        variable = f"Emissions|GHG|Industry|{SECTORS[0].path}"
        rows = self.rows(variable, "MtCO2e/yr", {"A": [None, 10], "B": [5, 20]})
        rows += self.rows("placeholder", "Mt/yr", {"C": [1, 1]})
        report = self.report(rows, policy="zero")
        self.assertEqual(report.emissions_by_region("cement").loc["A"].tolist(), [0, 10])
        self.assertEqual(report.emissions_by_region("cement").loc["C"].tolist(), [0, 0])
        self.assertEqual(report.ranked_emitters("cement"), ["B", "A"])
        np.testing.assert_allclose(report.emitter_coverage("cement"), [100, 100])

    def test_emitter_ties_are_deterministic_and_only_positive_totals_are_ranked(self):
        variable = f"Emissions|GHG|Industry|{SECTORS[0].path}"
        config = {**self.config, "producer_groups": {"C": ["C"], "B": ["B"], "A": ["A"]}, "top_n": 12}
        for excluded in (0, -1, None):
            rows = self.rows(variable, "MtCO2e/yr", {"A": [1, 20], "B": [2, 20], "C": [10, excluded]})
            report = self.report(rows, config=config)
            self.assertEqual(report.ranked_emitters("cement"), ["A", "B"])
            if excluded == -1:
                # Net negative emissions stay in global accounting, despite exclusion from the positive-emitter ranking.
                self.assertEqual(report.emissions_groups("cement").loc["C", 2050], -1)
                self.assertAlmostEqual(report.emitter_coverage("cement").loc[2050], 4000 / 39)

    def test_emitter_rank_export_covers_each_sector_and_preserves_unknown_global_share(self):
        rows = []
        for i, sector in enumerate(SECTORS):
            rows += self.rows(f"Emissions|GHG|Industry|{sector.path}", "MtCO2e/yr",
                              {"A": [1, 30], "B": [2, 20], "C": [3, None if i == 2 else 10]})
        table = self.report(rows).emitter_ranking_table()
        self.assertEqual(table.columns.tolist(), ["Sector", "Rank", "Group", "RankingYear",
                                                  "Emissions_MtCO2e_yr", "GlobalShare_percent"])
        self.assertEqual(table.Sector.tolist(), [sector.label for sector in SECTORS for _ in range(2)])
        self.assertEqual(table.Rank.tolist(), [1, 2] * 3)
        self.assertEqual(table.Group.tolist(), ["A", "B"] * 3)
        self.assertEqual(table.RankingYear.tolist(), [2050] * 6)
        self.assertEqual(table.Emissions_MtCO2e_yr.tolist(), [30, 20] * 3)
        np.testing.assert_allclose(table.GlobalShare_percent.iloc[:4], [50, 100 / 3] * 2)
        self.assertTrue(table.GlobalShare_percent.iloc[4:].isna().all())


if __name__ == "__main__":
    unittest.main()
