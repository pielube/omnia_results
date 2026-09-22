"""Calculation tests with independent, deliberately asymmetric examples."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from omnia_results.metrics import Results, SECTORS, ratio


class MetricTests(unittest.TestCase):
    def results(self, rows, policy="preserve"):
        frame = pd.DataFrame(rows, columns=["Scenario", "Region", "Variable", "Unit", "2019", "2050"])
        with patch("omnia_results.metrics.pd.read_csv", return_value=frame):
            return Results(Path("fixture.csv"), "baseline_noce", ["A", "B"], [2019, 2050], policy)

    def test_scenario_and_unit_selection_precede_aggregation(self):
        variable = "Emissions|GHG|Industry|Iron and Steel"
        rows = [["baseline_noce", r, variable, "MtCO2e/yr", x, x] for r, x in [("A", 2), ("B", 3)]]
        rows += [["baseline_noce", "A", variable, "MtCO2e/Mt", 100, 100],
                 ["ndc_noce", "A", variable, "MtCO2e/yr", 1000, 1000]]
        self.assertEqual(self.results(rows).emissions(SECTORS[1]).tolist(), [5, 5])

    def test_incomplete_global_sum_remains_missing(self):
        r = self.results([["baseline_noce", "A", "x", "EJ/yr", 2, 3],
                          ["baseline_noce", "B", "x", "EJ/yr", None, 4]])
        self.assertTrue(np.isnan(r.series("x", "EJ/yr").loc[2019]))
        self.assertEqual(r.series("x", "EJ/yr").loc[2050], 7)

    def test_explicit_zero_policy_covers_absent_rows_but_not_intensities(self):
        r = self.results([["baseline_noce", "A", "x", "EJ/yr", None, 3],
                          ["baseline_noce", "B", "other", "EJ/yr", 1, 1]], "zero")
        self.assertEqual(r.series("x", "EJ/yr").tolist(), [0, 3])
        self.assertTrue(r.series("x", "EJ/yr", activity=False).isna().all())
        self.assertEqual({x["Issue"] for x in r.coverage}, {"blank cell", "absent row"})

    def test_entirely_absent_variable_cannot_be_silently_zeroed(self):
        r = self.results([["baseline_noce", region, "x", "EJ/yr", 1, 1] for region in ("A", "B")], "zero")
        with self.assertRaisesRegex(ValueError, "Required series is absent"):
            r.series("typo", "EJ/yr")

    def test_carrier_parents_are_summed_once_and_intensity_is_ratio_of_sums(self):
        sector = SECTORS[0]
        rows = []
        for region, production, energy in [("A", 10, 1), ("B", 30, 9)]:
            for fuel in sector.fuels:
                rows.append(["baseline_noce", region, f"Final Energy|Industry|{sector.path}|{fuel}", "EJ/yr", energy / 4, energy / 4])
            rows.extend([
                ["baseline_noce", region, f"Final Energy|Industry|{sector.path}|Solids|Coal", "EJ/yr", 99, 99],
                ["baseline_noce", region, "Production|" + sector.path, "Mt/yr", production, production],
                ["baseline_noce", region, "Production|Non-Metallic Minerals|Cement Clinker", "Mt/yr", 1000, 1000],
            ])
        r = self.results(rows)
        self.assertEqual(r.energy(sector).tolist(), [10, 10])
        self.assertEqual(r.global_metric("energy_intensity", sector).tolist(), [250, 250])
        # An unweighted mean would be 200; adding clinker would also be wrong.

    def test_primary_plus_secondary_production(self):
        sector = SECTORS[1]
        rows = [["baseline_noce", region, f"Production|{sector.path}|{route}", "Mt/yr", value, value]
                for region in ("A", "B") for route, value in [("Primary", 2), ("Secondary", 3)]]
        self.assertEqual(self.results(rows).production_total(sector).tolist(), [10, 10])

    def test_emissions_share_uses_industry_total(self):
        rows = [["baseline_noce", region, variable, "MtCO2e/yr", value, value]
                for region in ("A", "B") for variable, value in
                [("Emissions|GHG|Industry", 100), ("Emissions|GHG|Industry|Iron and Steel", 12)]]
        self.assertEqual(self.results(rows).global_metric("emissions_share", SECTORS[1]).tolist(), [12, 12])

    def test_non_positive_denominators_are_undefined(self):
        result = ratio(pd.Series([10, 10, 10, 10]), pd.Series([2, 0, -1, np.nan]), 100)
        self.assertEqual(result.iloc[0], 500)
        self.assertTrue(result.iloc[1:].isna().all())

    def test_negative_values_are_preserved(self):
        rows = [["baseline_noce", "A", "Total Annualised Cost", "Millions USD_2010/yr", -20, 30],
                ["baseline_noce", "B", "Total Annualised Cost", "Millions USD_2010/yr", 5, 10]]
        self.assertEqual(self.results(rows).series("Total Annualised Cost", "Millions USD_2010/yr", activity=False).tolist(), [-15, 40])

    def test_duplicate_source_keys_are_rejected(self):
        row = ["baseline_noce", "A", "x", "EJ/yr", 1, 1]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.results([row, row, ["baseline_noce", "B", "x", "EJ/yr", 1, 1]])

    def test_unexpected_aggregate_region_is_rejected(self):
        rows = [["baseline_noce", region, "x", "EJ/yr", 1, 1] for region in ("A", "B", "World")]
        with self.assertRaisesRegex(ValueError, "regions must match"):
            self.results(rows)


if __name__ == "__main__":
    unittest.main()
