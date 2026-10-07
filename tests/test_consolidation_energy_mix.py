"""Asymmetric source examples for CE energy-carrier stacks."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from omnia_results.consolidation_data import ConsolidationData
from omnia_results.metrics import SECTORS


POLICIES = ("noce", "medce_eu", "highce_eu", "medce_gbl", "highce_gbl")
QUANTITY = "Final energy"
CHANGE = "Final energy change from no CE"
CARRIERS = ("Electricity", "Gases", "Hydrogen", "Liquids", "Solids")
REGIONAL_CHANGES = {"Electricity": (.5, .75), "Gases": (-.2, -.6),
                    "Hydrogen": (.5, .75), "Liquids": (-.1, -.8), "Solids": (-.3, -1)}


class ConsolidationEnergyMixTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "input": "raw_energy_mix.csv",
            "scenarios": [f"{climate}_{policy}" for climate in ("baseline", "ndc") for policy in POLICIES],
            "regions": ["EUE", "CHN"], "years": [2019, 2050], "missing_activity": "preserve",
            "historical_year": 2019, "comparison_year": 2050,
        }
        self.endpoints = {
            "ndc": {"cement": [[1, 2, 3, 4], [5, 6, 7, 8]],
                    "steel": [[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]],
                    "aluminium": [[.5, 1, 1.5, 2, 2.5], [3, 3.5, 4, 4.5, 5]]},
            "baseline": {"cement": [[2, 4, 6, 8], [10, 12, 14, 16]],
                         "steel": [[3, 5, 7, 9, 11], [13, 15, 17, 19, 21]],
                         "aluminium": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]},
        }
        rows = []
        for climate in ("baseline", "ndc"):
            factor = 1.5 if climate == "baseline" else 1
            for p, policy in enumerate(POLICIES):
                scenario = f"{climate}_{policy}"
                for sector in SECTORS:
                    for index, region in enumerate(self.config["regions"]):
                        for fuel_index, fuel in enumerate(sector.fuels):
                            endpoint = self.endpoints[climate][sector.key][index][fuel_index]
                            start = endpoint / (2 if index == 0 else 4) + p * 10000
                            final = endpoint + p * factor * REGIONAL_CHANGES[fuel][index]
                            variable = f"Final Energy|Industry|{sector.path}|{fuel}"
                            rows.append([scenario, region, variable, "EJ/yr", start, final])
                            rows.append([scenario, region, variable, "PJ/yr", 99999, 99999])
                            rows.append([scenario, region, variable + ("|Gas" if fuel == "Gases" else "|Detail"),
                                         "EJ/yr", 99999, 99999])
                        rows.append([scenario, region, f"Final Energy|Industry|{sector.path}", "EJ/yr", 99999, 99999])
                        if sector.key == "cement":
                            rows.append([scenario, region, f"Final Energy|Industry|{sector.path}|Hydrogen",
                                         "EJ/yr", 99999, 99999])
        self.source = pd.DataFrame(rows, columns=["Scenario", "Region", "Variable", "Unit", "2019", "2050"])

    def data(self, *, source=None, config=None):
        source = self.source if source is None else source
        config = self.config if config is None else config
        with patch("omnia_results.metrics.pd.read_csv", return_value=source.copy()) as read:
            data = ConsolidationData(config, Path("."))
        self.assertTrue(all(Path(call.args[0]).name == "raw_energy_mix.csv" for call in read.call_args_list))
        return data

    @staticmethod
    def select(frame, scenario, sector, series, metric=QUANTITY, year=2050):
        return frame[(frame.Scenario == scenario) & (frame.Sector == sector)
                     & (frame.Series == series) & (frame.Metric == metric) & (frame.Year == year)]

    def test_all_ten_scenarios_have_exact_global_carrier_values_and_complete_totals(self):
        frame = self.data().energy_mix()
        self.assertEqual(len(frame), 234)
        for climate in ("ndc", "baseline"):
            factor = 1.5 if climate == "baseline" else 1
            for p, policy in enumerate(POLICIES):
                scenario = f"{climate}_{policy}"
                for sector in SECTORS:
                    total = 0
                    for index, fuel in enumerate(sector.fuels):
                        base = sum(region[index] for region in self.endpoints[climate][sector.key])
                        expected = base + p * factor * sum(REGIONAL_CHANGES[fuel])
                        total += expected
                        row = self.select(frame, scenario, sector.label, fuel).iloc[0]
                        with self.subTest(scenario=scenario, sector=sector.key, fuel=fuel):
                            self.assertAlmostEqual(row.Value, expected)
                            self.assertEqual(row.Unit, "EJ/yr")
                            self.assertEqual(row.Role, "plotted")
                    row = self.select(frame, scenario, sector.label, "Total").iloc[0]
                    self.assertAlmostEqual(row.Value, total)
                    self.assertEqual(row.Role, "context")
        self.assertEqual(len(frame[(frame.Metric == QUANTITY) & (frame.Role == "plotted")]), 168)

    def test_exact_parent_units_exclude_nested_carriers_sector_aggregates_and_cement_hydrogen(self):
        frame = self.data().energy_mix()
        exact_variables = [f"Final Energy|Industry|{sector.path}|{fuel}"
                           for sector in SECTORS for fuel in sector.fuels]
        exact = self.source[(self.source.Unit == "EJ/yr") & self.source.Variable.isin(exact_variables)]
        pd.testing.assert_frame_equal(frame, self.data(source=exact).energy_mix())
        cement = frame[(frame.Sector == "Cement") & (frame.Role == "plotted")]
        self.assertEqual(set(cement.Series), {"Electricity", "Gases", "Liquids", "Solids"})
        for sector in SECTORS[1:]:
            plotted = frame[(frame.Sector == sector.label) & (frame.Role == "plotted")]
            self.assertEqual(set(plotted.Series), set(CARRIERS))

    def test_historical_bars_select_each_climates_no_ce_without_scenario_averaging(self):
        frame = self.data().energy_mix()
        expected = {("ndc", "Cement"): 11.5, ("ndc", "Iron and steel"): 21.25,
                    ("ndc", "Aluminium"): 8.75, ("baseline", "Cement"): 23,
                    ("baseline", "Iron and steel"): 38.75, ("baseline", "Aluminium"): 17.5}
        historical = frame[frame.BarType == "historical"]
        self.assertEqual(set(historical.Scenario), {"ndc_noce", "baseline_noce"})
        self.assertEqual(historical.Year.unique().tolist(), [2019])
        self.assertEqual(historical.BarOrder.unique().tolist(), [0])
        for (climate, sector), value in expected.items():
            self.assertAlmostEqual(self.select(frame, f"{climate}_noce", sector, "Total", year=2019).Value.iloc[0], value)

    def test_changes_use_own_climate_future_no_ce_rather_than_the_historical_anchor(self):
        frame = self.data().energy_mix()
        reference_totals = {("ndc", "Cement"): 36, ("ndc", "Iron and steel"): 65,
                            ("ndc", "Aluminium"): 27.5, ("baseline", "Cement"): 72,
                            ("baseline", "Iron and steel"): 120, ("baseline", "Aluminium"): 55}
        for climate in ("ndc", "baseline"):
            for policy in POLICIES:
                scenario = f"{climate}_{policy}"
                for sector in SECTORS:
                    row = self.select(frame, scenario, sector.label, "Total", CHANGE).iloc[0]
                    quantity = self.select(frame, scenario, sector.label, "Total").Value.iloc[0]
                    denominator = reference_totals[climate, sector.label]
                    self.assertAlmostEqual(row.Value, (quantity / denominator - 1) * 100)
                    self.assertAlmostEqual(row.Numerator, quantity)
                    self.assertEqual(row.Denominator, denominator)
                    self.assertEqual(row.NumeratorUnit, "EJ/yr")
                    self.assertEqual(row.DenominatorUnit, "EJ/yr")
                    self.assertEqual(row.ReferenceScenario, f"{climate}_noce")
                    self.assertEqual(row.ReferenceYear, 2050)
                    self.assertEqual(row.ConversionFactor, 100)
                    self.assertEqual(row.Offset, -100)
                    self.assertEqual(row.Unit, "%")
        self.assertAlmostEqual(self.select(frame, "ndc_medce_eu", "Cement", "Total", CHANGE).Value.iloc[0],
                               (34.25 / 36 - 1) * 100)
        self.assertFalse((frame[frame.Metric == CHANGE].Year == 2019).any())

    def test_quantities_retain_exact_source_variables_regions_and_unit_provenance(self):
        frame = self.data().energy_mix()
        for sector in SECTORS:
            variables = [f"Final Energy|Industry|{sector.path}|{fuel}" for fuel in sector.fuels]
            for observation in frame[frame.Sector == sector.label].itertuples():
                expected = (variables if observation.Series == "Total" else
                            [f"Final Energy|Industry|{sector.path}|{observation.Series}"])
                self.assertEqual(json.loads(observation.SourceVariables), expected)
                self.assertEqual(json.loads(observation.SourceRegions), ["EUE", "CHN"])
                if observation.Metric == QUANTITY:
                    self.assertAlmostEqual(observation.Numerator, observation.Value)
                    self.assertEqual(observation.NumeratorUnit, "EJ/yr")
                    self.assertEqual(observation.ConversionFactor, 1)
                    self.assertEqual(observation.Offset, 0)

    def test_fixed_row_and_bar_order_survive_reversed_scenario_configuration(self):
        frame = self.data().energy_mix()
        reversed_config = {**self.config, "scenarios": self.config["scenarios"][::-1]}
        pd.testing.assert_frame_equal(frame, self.data(config=reversed_config).energy_mix())
        for climate, panels, label in (("ndc", "abc", "NDC"), ("baseline", "def", "NDC+LTT")):
            selected = frame[frame.Panel.isin(list(panels))]
            self.assertTrue(selected.Scenario.str.startswith(climate + "_").all())
            self.assertEqual(selected.ClimatePathway.unique().tolist(), [label])
            for panel in panels:
                bars = selected[selected.Panel == panel][["BarOrder", "BarLabel", "Year"]].drop_duplicates().sort_values("BarOrder")
                self.assertEqual(bars.BarOrder.tolist(), list(range(6)))
                self.assertEqual(bars.BarLabel.tolist(), ["2019", "No CE", "M–EU", "H–EU", "M–EU+", "H–EU+"])
                self.assertEqual(bars.Year.tolist(), [2019, 2050, 2050, 2050, 2050, 2050])

    def test_missing_carrier_preserves_unknown_total_and_change_or_is_explicitly_zero_filled(self):
        source = self.source.copy()
        selected = ((source.Scenario == "ndc_highce_gbl") & (source.Region == "EUE")
                    & (source.Variable == "Final Energy|Industry|Non-Metallic Minerals|Cement|Gases")
                    & (source.Unit == "EJ/yr"))
        source.loc[selected, "2050"] = np.nan
        for policy in ("preserve", "zero"):
            with self.subTest(policy=policy):
                data = self.data(source=source, config={**self.config, "missing_activity": policy})
                frame = data.energy_mix()
                gas = self.select(frame, "ndc_highce_gbl", "Cement", "Gases").Value.iloc[0]
                total = self.select(frame, "ndc_highce_gbl", "Cement", "Total").Value.iloc[0]
                change = self.select(frame, "ndc_highce_gbl", "Cement", "Total", CHANGE).Value.iloc[0]
                if policy == "preserve":
                    self.assertTrue(np.isnan(gas))
                    self.assertTrue(np.isnan(total))
                    self.assertTrue(np.isnan(change))
                else:
                    self.assertAlmostEqual(gas, 3.6)
                    self.assertAlmostEqual(total, 27.8)
                    self.assertAlmostEqual(change, (27.8 / 36 - 1) * 100)
                self.assertAlmostEqual(self.select(frame, "ndc_highce_gbl", "Cement", "Electricity").Value.iloc[0], 11)
                self.assertEqual(data.results["ndc_highce_gbl"].coverage[0]["Treatment"], policy)

    def test_absent_regional_carrier_makes_the_reference_unknown_without_ignoring_the_region(self):
        selected = ((self.source.Scenario == "baseline_noce") & (self.source.Region == "CHN")
                    & (self.source.Variable == "Final Energy|Industry|Non-Ferrous Metals|Aluminum|Hydrogen")
                    & (self.source.Unit == "EJ/yr"))
        for policy in ("preserve", "zero"):
            with self.subTest(policy=policy):
                data = self.data(source=self.source.loc[~selected], config={**self.config, "missing_activity": policy})
                frame = data.energy_mix()
                changes = frame[(frame.Scenario.str.startswith("baseline_")) & (frame.Sector == "Aluminium")
                                & (frame.Metric == CHANGE)]
                if policy == "preserve":
                    self.assertTrue(changes.Value.isna().all())
                    self.assertTrue(changes.Denominator.isna().all())
                else:
                    np.testing.assert_allclose(changes.Denominator, 47)
                    self.assertAlmostEqual(self.select(frame, "baseline_noce", "Aluminium", "Total").Value.iloc[0], 47)
                    self.assertAlmostEqual(self.select(frame, "baseline_noce", "Aluminium", "Total", year=2019).Value.iloc[0], 15.5)
                audit = data.results["baseline_noce"].coverage
                self.assertEqual(len(audit), 2)
                self.assertTrue(all(record["Issue"] == "absent row" and record["Treatment"] == policy for record in audit))

    def test_nonpositive_reference_totals_are_undefined_and_all_scenarios_and_years_are_required(self):
        cement_variables = [f"Final Energy|Industry|Non-Metallic Minerals|Cement|{fuel}"
                            for fuel in ("Electricity", "Gases", "Liquids", "Solids")]
        for value in (0, -1):
            source = self.source.copy()
            selected = ((source.Scenario == "baseline_noce") & (source.Unit == "EJ/yr")
                        & source.Variable.isin(cement_variables))
            source.loc[selected, "2050"] = value
            frame = self.data(source=source).energy_mix()
            changes = frame[(frame.Scenario.str.startswith("baseline_")) & (frame.Sector == "Cement")
                            & (frame.Metric == CHANGE)]
            self.assertTrue(changes.Value.isna().all())
            np.testing.assert_allclose(changes.Denominator, value * 8)
        for config in ({**self.config, "scenarios": self.config["scenarios"][:-1]},
                       {**self.config, "comparison_year": 2040},
                       {**self.config, "historical_year": 2020}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.data(config=config).energy_mix()

    def test_rendered_carriers_match_source_heights_stack_baselines_and_climate_rows(self):
        from omnia_results.consolidation_energy_mix import EnergyMixFigures

        class Builder(EnergyMixFigures):
            def __init__(self, data, config):
                self.data, self.config = data, config

        spec = Builder(self.data(), self.config).energy_mix()
        try:
            self.assertEqual(len(spec.figure.axes), 6)
            for index, (ax, panel) in enumerate(zip(spec.figure.axes, "abcdef")):
                selected = spec.data[(spec.data.Panel == panel) & (spec.data.Metric == QUANTITY)
                                     & (spec.data.Role == "plotted")]
                actual = {container.get_label(): container for container in ax.containers}
                self.assertEqual(set(actual), set(selected.Series))
                baseline = np.zeros(6)
                for carrier in CARRIERS:
                    if carrier not in actual:
                        continue
                    bars = actual[carrier]
                    expected = selected[selected.Series == carrier].sort_values("BarOrder").Value.to_numpy()
                    np.testing.assert_allclose([bar.get_height() for bar in bars], expected)
                    np.testing.assert_allclose([bar.get_y() for bar in bars], baseline)
                    baseline += expected
                totals = spec.data[(spec.data.Panel == panel) & (spec.data.Metric == QUANTITY)
                                   & (spec.data.Series == "Total")].sort_values("BarOrder").Value.to_numpy()
                np.testing.assert_allclose(baseline, totals)
                if index < 3:
                    self.assertEqual(ax.get_ylim(), spec.figure.axes[index + 3].get_ylim())
        finally:
            plt.close(spec.figure)


if __name__ == "__main__":
    unittest.main()
