"""Asymmetric examples for climate-matched CE comparisons and common weights."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from omnia_results.comparison_data import ComparisonData, percent_change
from omnia_results.metrics import SECTORS


class ComparisonDataTests(unittest.TestCase):
    def setUp(self):
        self.regions = ["EUE", "EUW", "CHN", "AFE"]
        self.scenarios = [f"{climate}_{policy}" for climate in ComparisonData.climates
                          for policy in ComparisonData.policies]
        self.config = {"input": "fixture.csv", "policy_regions_csv": "policy.csv",
                       "regions": self.regions, "years": [2019, 2050],
                       "scenarios": self.scenarios, "comparison_year": 2050,
                       "price_weight_year": 2019, "price_weight_scenario": "baseline_noce",
                       "missing_activity": "preserve"}
        self.policy = pd.DataFrame({"Region": self.regions, "Description": ["a", "b", "c", "d"],
                                    "EU": [True, True, False, False],
                                    "EU_plus_partners": [True, True, True, False]})
        rows = []
        for climate_index, climate in enumerate(ComparisonData.climates):
            for policy_index, policy in enumerate(ComparisonData.policies):
                scenario = f"{climate}_{policy}"
                for region_index, region in enumerate(self.regions):
                    def add(variable, unit, start, end):
                        rows.append([scenario, region, variable, unit, start, end])
                    industry_base = ([1, 3, 2, 4] if policy_index == climate_index == 0
                                     else [3, 1, 2, 4])[region_index]
                    add("Emissions|GHG|Industry", "MtCO2e/yr", industry_base,
                        100 + 20 * climate_index + 5 * policy_index + region_index)
                    add("Price|Carbon", "USD_2010/t CO2e", [10, 30, 40, 50][region_index],
                        [20, 100, 60, 80][region_index] + policy_index)
                    add("Total Annualised Cost", "Millions USD_2010/yr", 1000,
                        1000 + 100 * climate_index - 10 * policy_index)
                    for sector_index, sector in enumerate(SECTORS):
                        start = 1 + region_index + sector_index
                        end = (10 + 2 * region_index + sector_index + 20 * climate_index
                               - policy_index * (region_index + 1))
                        add(f"Emissions|GHG|Industry|{sector.path}", "MtCO2e/yr", start, end)
                        for fuel_index, fuel in enumerate(sector.fuels):
                            add(f"Final Energy|Industry|{sector.path}|{fuel}", "EJ/yr",
                                fuel_index + region_index, fuel_index + 2 * region_index)
                        if sector.key == "cement":
                            add(f"Production|{sector.path}", "Mt/yr", 10, 20 + policy_index)
                            add("Production|Non-Metallic Minerals|Cement Clinker", "Mt/yr", 5, 8)
                        else:
                            add(f"Production|{sector.path}|Primary", "Mt/yr", 10, 15 - policy_index)
                            add(f"Production|{sector.path}|Secondary", "Mt/yr", 2, 5 + policy_index)
                    add(f"Final Energy|Industry|{SECTORS[0].path}|Gases|Gas", "EJ/yr", 999, 999)
                    add(f"Emissions|GHG|Industry|{SECTORS[0].path}", "MtCO2e/Mt", 999, 999)
        self.source = pd.DataFrame(rows, columns=["Scenario", "Region", "Variable", "Unit", "2019", "2050"])

    def data(self, config=None, source=None, policy=None):
        config = self.config if config is None else config
        source = self.source if source is None else source
        policy = self.policy if policy is None else policy
        def read(path):
            return (policy if Path(path).name == "policy.csv" else source).copy()
        with patch("omnia_results.comparison_data.pd.read_csv", side_effect=read):
            return ComparisonData(config, Path("."))

    def selected(self, scenario, variable, region=None):
        mask = (self.source.Scenario == scenario) & (self.source.Variable == variable)
        if region is not None:
            mask &= self.source.Region == region
        return mask

    def test_references_are_paired_by_climate_and_not_baseline_for_every_scenario(self):
        data = self.data()
        self.assertEqual(data.reference("baseline_highce_gbl"), "baseline_noce")
        self.assertEqual(data.reference("ndc_highce_gbl"), "ndc_noce")
        delta = data.emissions_delta("ndc_highce_gbl")
        self.assertEqual(delta.loc["Cement", 2050], -40)
        baseline_difference = data.target_emissions("ndc_highce_gbl") - data.target_emissions("baseline_noce")
        self.assertEqual(baseline_difference.loc[2050], 120)
        self.assertEqual(data.target_emissions("ndc_highce_gbl").loc[2050]
                         - data.target_emissions("ndc_noce").loc[2050], -120)

    def test_exact_region_partition_and_regional_delta_closure(self):
        data = self.data()
        self.assertEqual(data.groups, {"EU": ["EUE", "EUW"], "Partners": ["CHN"], "Rest": ["AFE"]})
        delta = data.regional_emissions_delta("baseline_highce_gbl")
        self.assertEqual(delta[2050].tolist(), [-36, -36, -48])
        expected = data.target_emissions("baseline_highce_gbl") - data.target_emissions("baseline_noce")
        pd.testing.assert_series_equal(delta.sum(axis=0, min_count=3), expected)
        self.assertEqual(data.membership().Region.tolist(), self.regions)
        self.assertEqual(data.membership().Description.tolist(), ["a", "b", "c", "d"])

    def test_prices_use_one_scenario_and_year_for_every_scenario(self):
        data = self.data()
        self.assertEqual(data.carbon_price("baseline_noce").tolist(), [25, 80])
        self.assertEqual(data.carbon_price("ndc_highce_gbl").tolist(), [25, 84])
        self.assertEqual(data.price_weights.loc[data.price_weights.Group == "EU", "Weight"].tolist(), [.25, .75])
        self.assertEqual(data.price_weights.WeightScenario.unique().tolist(), ["baseline_noce"])
        self.assertEqual(data.price_weights.WeightYear.unique().tolist(), [2019])
        # NDC's own 2019 weights are .75/.25 and would incorrectly give 44 in 2050.
        self.assertEqual(data.price_weights.groupby("Group").Weight.sum().tolist(), [1, 1, 1])

    def test_carriers_use_exact_parent_rows_and_emissions_use_exact_units(self):
        data = self.data()
        energy = data.energy_by_carrier("baseline_noce", SECTORS[0])
        self.assertEqual(energy.index.tolist(), list(SECTORS[0].fuels))
        self.assertEqual(energy.loc["Gases"].tolist(), [10, 16])
        self.assertEqual(data.emissions_components("baseline_noce").loc["Cement", 2050], 52)

    def test_production_preserves_component_definitions(self):
        data = self.data()
        cement = data.global_production("baseline_noce", SECTORS[0])
        self.assertEqual(cement["Cement"].tolist(), [40, 80])
        self.assertEqual(cement["Clinker"].tolist(), [20, 32])
        steel = data.global_production("baseline_highce_gbl", SECTORS[1])
        self.assertEqual(steel["Primary"].tolist(), [40, 44])
        self.assertEqual(steel["Secondary"].tolist(), [8, 36])
        self.assertEqual(steel["Total"].tolist(), [48, 80])

    def test_residual_is_complete_accounting_arithmetic_and_retains_negative_values(self):
        variable = "Emissions|GHG|Industry"
        self.source.loc[self.selected("baseline_highce_gbl", variable), "2050"] = 1
        data = self.data()
        components = data.emissions_components("baseline_highce_gbl")
        self.assertEqual(components.loc["Other industry", 2050], -44)
        pd.testing.assert_series_equal(components.sum(axis=0, min_count=4),
                                       data.industry_emissions("baseline_highce_gbl"))
        variable = f"Emissions|GHG|Industry|{SECTORS[0].path}"
        self.source.loc[self.selected("baseline_highce_gbl", variable, "AFE"), "2050"] = None
        components = self.data().emissions_components("baseline_highce_gbl")
        self.assertTrue(np.isnan(components.loc["Cement", 2050]))
        self.assertTrue(np.isnan(components.loc["Other industry", 2050]))
        self.assertTrue(np.isnan(self.data().target_emissions("baseline_highce_gbl").loc[2050]))

    def test_activity_zero_policy_is_explicit_and_costs_still_preserve_missingness(self):
        variable = f"Emissions|GHG|Industry|{SECTORS[0].path}"
        self.source.loc[self.selected("baseline_highce_gbl", variable, "AFE"), "2050"] = None
        self.source.loc[self.selected("baseline_highce_gbl", "Total Annualised Cost", "AFE"), "2050"] = None
        data = self.data(config={**self.config, "missing_activity": "zero"})
        self.assertEqual(data.emissions_components("baseline_highce_gbl").loc["Cement", 2050], 12)
        self.assertTrue(np.isnan(data.system_cost("baseline_highce_gbl").loc[2050]))
        self.assertEqual(data.system_cost("baseline_noce").loc[2050], 4)

    def test_weights_never_fill_missing_or_negative_base_emissions(self):
        for invalid in (None, -1):
            source = self.source.copy()
            source.loc[self.selected("baseline_noce", "Emissions|GHG|Industry", "AFE"), "2019"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                self.data(source=source, config={**self.config, "missing_activity": "zero"})
        source = self.source.loc[~self.selected("baseline_noce", "Emissions|GHG|Industry", "AFE")]
        with self.assertRaisesRegex(ValueError, "not zero-filled"):
            self.data(source=source, config={**self.config, "missing_activity": "zero"})

    def test_zero_weight_prices_can_be_missing_but_positive_weight_prices_cannot(self):
        self.source.loc[self.selected("baseline_noce", "Emissions|GHG|Industry", "EUE"), "2019"] = 0
        self.source.loc[self.selected("ndc_highce_gbl", "Price|Carbon", "EUE"), "2050"] = None
        data = self.data(config={**self.config, "missing_activity": "zero"})
        self.assertEqual(data.carbon_price("ndc_highce_gbl").loc[2050], 104)
        self.source.loc[self.selected("ndc_highce_gbl", "Price|Carbon", "EUW"), "2050"] = None
        self.assertTrue(np.isnan(self.data().carbon_price("ndc_highce_gbl").loc[2050]))

    def test_incomplete_duplicate_or_inconsistent_geography_is_rejected(self):
        for policy in (self.policy.iloc[:-1], pd.concat([self.policy, self.policy.iloc[[0]]]),
                       self.policy.assign(EU_plus_partners=[False, True, True, False])):
            with self.subTest(policy=policy.to_dict()), self.assertRaisesRegex(ValueError, "partition|subset"):
                self.data(policy=policy)
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.data(policy=self.policy.assign(EU=["yes", "true", "false", "false"]))

    def test_all_ten_scenarios_and_configured_years_are_required(self):
        for scenarios in (self.scenarios[:-1], self.scenarios[:-1] + [self.scenarios[0]]):
            with self.subTest(scenarios=scenarios), self.assertRaisesRegex(ValueError, "all ten"):
                self.data(config={**self.config, "scenarios": scenarios})
        extra = self.source.iloc[[0]].assign(Scenario="extra_noce")
        with self.assertRaisesRegex(ValueError, "exactly the ten"):
            self.data(source=pd.concat([self.source, extra]))
        for field in ("comparison_year", "price_weight_year"):
            with self.assertRaisesRegex(ValueError, field):
                self.data(config={**self.config, field: 2040})

    def test_percent_change_handles_zero_negative_missing_and_small_positive_references(self):
        self.assertEqual(percent_change(8, 10), -20)
        for reference in (0, -1, np.nan, np.inf):
            self.assertTrue(np.isnan(percent_change(8, reference)))
        self.assertAlmostEqual(percent_change(2e-12, 1e-12), 100)
        values = pd.Series([8, 0, 2e-12, 5], index=list("abcd"))
        reference = pd.Series([10, 0, 1e-12, -2], index=list("abcd"))
        result = percent_change(values, reference)
        self.assertEqual(result.loc["a"], -20)
        self.assertAlmostEqual(result.loc["c"], 100)
        self.assertTrue(result.loc[["b", "d"]].isna().all())


if __name__ == "__main__":
    unittest.main()
