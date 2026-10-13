"""Independent examples for regional, cross-scenario carbon-price indices."""

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from omnia_results.carbon_prices import CarbonPriceBuilder, CarbonPriceData


class CarbonPriceTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(".")
        self.scenarios = [
            f"{climate}_{policy}"
            for climate in ("ndc", "baseline")
            for policy in ("noce", "medce_eu", "highce_eu", "medce_gbl", "highce_gbl")
        ]
        self.config = {
            "suite": "carbon_prices", "input": "raw_prices.csv", "output": "figures",
            "scenarios": self.scenarios, "years": [2019, 2050], "regions": ["A", "B", "C"],
            "macroregions": {"Pair": ["A", "B"], "Other": ["C"]},
            "price_weight_scenario": "baseline_noce", "price_weight_year": 2019,
            "missing_activity": "preserve", "formats": ["png"], "dpi": 300,
        }
        rows = []
        for offset, scenario in enumerate(self.scenarios):
            for position, region in enumerate(self.config["regions"]):
                # Only the configured common reference has .25/.75 weights.
                # Other scenarios and future years reverse the two emissions.
                emissions = [1, 3, 2] if scenario == "baseline_noce" else [3, 1, 2]
                start, end = [10, 30, 5][position], [20, 80, 10][position]
                rows.extend([
                    [scenario, region, "Emissions|GHG|Industry", "MtCO2e/yr",
                     emissions[position], [9, 1, 2][position]],
                    [scenario, region, "Price|Carbon", "USD_2010/t CO2e",
                     start + offset, end + offset],
                    # Decoys make exact variable and unit selection observable.
                    [scenario, region, "Price|Carbon", "USD_2020/t CO2e", 99999, 99999],
                    [scenario, region, "Price|Carbon|Detail", "USD_2010/t CO2e", 99999, 99999],
                    [scenario, region, "Emissions|GHG|Industry", "MtCO2e/Mt", 99999, 99999],
                    [scenario, region, "Emissions|GHG|Industry|Detail", "MtCO2e/yr", 99999, 99999],
                ])
        self.source = pd.DataFrame(
            rows, columns=["Scenario", "Region", "Variable", "Unit", "2019", "2050"]
        )

    def data(self, *, source=None, config=None):
        source = self.source if source is None else source
        config = self.config if config is None else config
        # The same boundary is mocked in the report/comparison calculation
        # tests; Results still validates and selects the synthetic raw table.
        with patch("omnia_results.metrics.pd.read_csv", return_value=source.copy()):
            return CarbonPriceData(config, self.base)

    @staticmethod
    def selected(source, scenario, region, variable, unit):
        return ((source.Scenario == scenario) & (source.Region == region)
                & (source.Variable == variable) & (source.Unit == unit))

    @staticmethod
    def trajectory(frame, scenario, group):
        return frame.loc[(frame.Scenario == scenario) & (frame.Group == group)].sort_values("Year")

    def test_all_scenarios_use_the_same_fixed_reference_weights(self):
        data = self.data()
        frame = data.prices()
        self.assertEqual(data.groups, self.config["macroregions"])
        self.assertEqual(len(frame), 40)
        self.assertEqual(set(frame.Scenario), set(self.scenarios))
        self.assertFalse(frame.duplicated(["Scenario", "Group", "Year"]).any())
        for offset, scenario in enumerate(self.scenarios):
            with self.subTest(scenario=scenario):
                self.assertEqual(self.trajectory(frame, scenario, "Pair").Value.tolist(),
                                 [25 + offset, 65 + offset])
                self.assertEqual(self.trajectory(frame, scenario, "Other").Value.tolist(),
                                 [5 + offset, 10 + offset])
        weights = data.weights.set_index("Region")
        self.assertEqual(weights.loc[["A", "B", "C"], "Weight"].tolist(), [.25, .75, 1])
        self.assertEqual(weights.loc[["A", "B", "C"], "BaseEmissions_MtCO2e"].tolist(), [1, 3, 2])
        self.assertEqual(set(weights.WeightScenario), {"baseline_noce"})
        self.assertEqual(set(weights.WeightYear), {2019})
        np.testing.assert_allclose(data.weights.groupby("Group").Weight.sum(), 1)
        # Own-scenario weights would give 35 + offset, and future weights 26 + offset.

    def test_price_and_weight_inputs_use_exact_variables_and_units(self):
        frame = self.data().prices()
        self.assertEqual(set(frame.SourceVariable), {"Price|Carbon"})
        self.assertEqual(set(frame.SourceUnit), {"USD_2010/t CO2e"})
        self.assertLess(frame.Value.max(), 100)
        self.assertEqual(set(frame.WeightScenario), {"baseline_noce"})
        self.assertEqual(set(frame.WeightYear), {2019})
        for scenario in self.scenarios:
            rows = frame.loc[frame.Scenario == scenario]
            self.assertEqual(set(rows.Policy), {scenario.split("_", 1)[1]})
            self.assertEqual(set(rows.Climate), {"NDC" if scenario.startswith("ndc_") else "NDC+LTT"})

    def test_configured_weight_reference_changes_all_scenarios_together(self):
        config = {**self.config, "price_weight_scenario": "ndc_noce"}
        data = self.data(config=config)
        self.assertEqual(data.weights.set_index("Region").loc[["A", "B"], "Weight"].tolist(), [.75, .25])
        for offset, scenario in enumerate(self.scenarios):
            self.assertEqual(self.trajectory(data.prices(), scenario, "Pair").Value.tolist(),
                             [15 + offset, 35 + offset])

    def test_positive_weight_missing_price_remains_a_gap_for_both_activity_policies(self):
        source = self.source.copy()
        mask = self.selected(source, "ndc_noce", "B", "Price|Carbon", "USD_2010/t CO2e")
        source.loc[mask, "2050"] = np.nan
        for policy in ("zero", "preserve"):
            with self.subTest(policy=policy):
                data = self.data(source=source, config={**self.config, "missing_activity": policy})
                rows = self.trajectory(data.prices(), "ndc_noce", "Pair")
                self.assertEqual(rows.Value.iloc[0], 25)
                self.assertTrue(pd.isna(rows.Value.iloc[1]))
                components = data.components()
                missing = components.loc[(components.Scenario == "ndc_noce")
                                         & (components.Region == "B") & (components.Year == 2050)]
                self.assertTrue(missing.Price.isna().all())
                self.assertTrue(missing.WeightedPrice.isna().all())

    def test_absent_positive_weight_price_row_is_not_replaced_by_wrong_unit(self):
        source = self.source.loc[~self.selected(
            self.source, "ndc_noce", "B", "Price|Carbon", "USD_2010/t CO2e"
        )].copy()
        data = self.data(source=source, config={**self.config, "missing_activity": "zero"})
        self.assertTrue(self.trajectory(data.prices(), "ndc_noce", "Pair").Value.isna().all())
        self.assertTrue(self.trajectory(data.prices(), "baseline_noce", "Pair").Value.notna().all())

    def test_zero_weight_missing_price_has_zero_contribution(self):
        source = self.source.copy()
        mask = self.selected(source, "baseline_noce", "A", "Emissions|GHG|Industry", "MtCO2e/yr")
        source.loc[mask, "2019"] = 0
        mask = self.selected(source, "ndc_noce", "A", "Price|Carbon", "USD_2010/t CO2e")
        source.loc[mask, ["2019", "2050"]] = np.nan
        for policy in ("zero", "preserve"):
            with self.subTest(policy=policy):
                data = self.data(source=source, config={**self.config, "missing_activity": policy})
                self.assertEqual(self.trajectory(data.prices(), "ndc_noce", "Pair").Value.tolist(), [30, 80])
                components = data.components()
                ignored = components.loc[(components.Scenario == "ndc_noce") & (components.Region == "A")]
                self.assertEqual(len(ignored), 2)
                self.assertTrue(ignored.Price.isna().all())
                self.assertEqual(ignored.Weight.tolist(), [0, 0])
                self.assertEqual(ignored.WeightedPrice.tolist(), [0, 0])

    def test_components_reconstruct_every_available_index(self):
        data = self.data()
        components = data.components()
        self.assertEqual(len(components), 60)
        self.assertFalse(components.duplicated(["Scenario", "Group", "Region", "Year"]).any())
        np.testing.assert_allclose(components.WeightedPrice, components.Price * components.Weight)
        reconstructed = components.groupby(["Scenario", "Group", "Year"]).WeightedPrice.sum()
        prices = data.prices().set_index(["Scenario", "Group", "Year"]).Value.sort_index()
        np.testing.assert_allclose(reconstructed.sort_index(), prices)

    def test_missing_or_negative_base_emissions_are_rejected_even_under_zero_activity(self):
        for value in (np.nan, -1):
            with self.subTest(value=value):
                source = self.source.copy()
                mask = self.selected(source, "baseline_noce", "A", "Emissions|GHG|Industry", "MtCO2e/yr")
                source.loc[mask, "2019"] = value
                with self.assertRaises(ValueError):
                    data = self.data(source=source, config={**self.config, "missing_activity": "zero"})
                    data.prices()
        source = self.source.loc[~self.selected(
            self.source, "baseline_noce", "A", "Emissions|GHG|Industry", "MtCO2e/yr"
        )].copy()
        with self.assertRaises(ValueError):
            data = self.data(source=source, config={**self.config, "missing_activity": "zero"})
            data.prices()

    def test_zero_total_weight_group_is_rejected(self):
        source = self.source.copy()
        mask = ((source.Scenario == "baseline_noce") & source.Region.isin(["A", "B"])
                & (source.Variable == "Emissions|GHG|Industry") & (source.Unit == "MtCO2e/yr"))
        source.loc[mask, "2019"] = 0
        with self.assertRaises(ValueError):
            data = self.data(source=source)
            data.prices()

    def test_macroregions_must_partition_source_regions_exactly_once(self):
        invalid = [
            {"Pair": ["A", "B"], "Other": ["C", "A"]},
            {"Pair": ["A", "A", "B"], "Other": ["C"]},
            {"Pair": ["A", "B"]},
            {"Pair": ["A", "B"], "Other": ["C", "D"]},
            {"Pair": ["A", "B", "C"], "Empty": []},
        ]
        for groups in invalid:
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                self.data(config={**self.config, "macroregions": groups})

    def test_scenario_selection_requires_each_of_the_ten_scenarios_once(self):
        for selected in (self.scenarios[:-1], self.scenarios + [self.scenarios[0]],
                         self.scenarios[:-1] + [self.scenarios[0]]):
            config = deepcopy(self.config)
            config["scenarios"] = selected
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                self.data(config=config)

    def test_each_regional_figure_has_ten_distinct_scenarios_at_source_coordinates(self):
        data = self.data()
        figures = list(CarbonPriceBuilder(data, self.config).figures())
        self.addCleanup(lambda: [plt.close(spec.figure) for spec in figures])
        self.assertEqual(len(figures), 2)
        self.assertEqual(len({spec.slug for spec in figures}), 2)
        all_rows = []
        for spec in figures:
            self.assertEqual(spec.panel_count, 1)
            self.assertEqual(len(spec.figure.axes), 1)
            self.assertEqual(len(spec.data), 20)
            self.assertEqual(spec.data.Group.nunique(), 1)
            group = spec.data.Group.iloc[0]
            ax = spec.figure.axes[0]
            self.assertEqual(len(ax.lines), 10)
            self.assertEqual({line.get_label() for line in ax.lines}, set(self.scenarios))
            for line in ax.lines:
                scenario = line.get_label()
                expected = self.trajectory(data.prices(), scenario, group)
                np.testing.assert_array_equal(line.get_xdata(), [2019, 2050])
                np.testing.assert_allclose(line.get_ydata(), expected.Value)
                self.assertEqual(line.get_linestyle(), "-" if scenario.startswith("ndc_") else "--")
                exported = self.trajectory(spec.data, scenario, group)
                np.testing.assert_array_equal(exported.Year, expected.Year)
                np.testing.assert_allclose(exported.Value, expected.Value)
            all_rows.append(spec.data)
        exported = pd.concat(all_rows, ignore_index=True)
        self.assertFalse(exported.duplicated(["Scenario", "Group", "Year"]).any())
        self.assertEqual(set(exported.Group), {"Pair", "Other"})


if __name__ == "__main__":
    unittest.main()
