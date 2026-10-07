"""Asymmetric raw-cost examples for climate-, sector- and year-matched changes."""

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
COST_UNIT = "Millions USD_2010/yr"


class ConsolidationCostTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "input": "raw_costs.csv",
            "scenarios": [f"{climate}_{policy}" for climate in ("baseline", "ndc") for policy in POLICIES],
            "regions": ["EUE", "CHN"], "years": [2019, 2024, 2030, 2050],
            "missing_activity": "preserve", "main_cost_start_year": 2024,
        }
        self.reference = {
            "baseline": {
                "cement": [[3, 20, 30, 50], [17, 80, 170, 450]],
                "steel": [[10, 60, 100, 160], [40, 140, 300, 640]],
                "aluminium": [[1, 2, 4, 5], [3, 8, 16, 25]],
            },
            "ndc": {
                "cement": [[5, 25, 10, 20], [5, 25, 90, 230]],
                "steel": [[60, 300, 40, 200], [20, 100, 160, 800]],
                "aluminium": [[4, 3, 20, 5], [5, 12, 30, 20]],
            },
        }
        self.regional_changes = {
            "baseline": {
                "cement": [[3, 5, 20, -5], [-1, -25, -10, -20]],
                "steel": [[1, -15, 20, -40], [4, -5, 20, -120]],
                "aluminium": [[0, 1, 1, 1], [.4, -2, 3, -4]],
            },
            "ndc": {
                "cement": [[1, 10, 4, 2], [-2, -20, -14, -7]],
                "steel": [[-2, -20, 10, -30], [-6, -60, -40, -10]],
                "aluminium": [[-.4, -1, 6, -2], [-.5, -2, -1, -3]],
            },
        }
        rows = []
        for climate in ("baseline", "ndc"):
            for p, policy in enumerate(POLICIES):
                for index, region in enumerate(self.config["regions"]):
                    scenario = f"{climate}_{policy}"
                    for sector in SECTORS:
                        values = np.asarray(self.reference[climate][sector.key][index]) + p * np.asarray(self.regional_changes[climate][sector.key][index])
                        variable = f"Total Annualised Cost|Industry|{sector.path}"
                        rows.append([scenario, region, variable, COST_UNIT, *values])
                        rows.append([scenario, region, variable, "Billions USD_2010/yr", 99999, 99999, 99999, 99999])
                        rows.append([scenario, region, variable + "|Detail", COST_UNIT, 99999, 99999, 99999, 99999])
                    rows.append([scenario, region, "Total Annualised Cost", COST_UNIT, 99999, 99999, 99999, 99999])
                    rows.append([scenario, region, "Total Annualised Cost|Industry", COST_UNIT, 99999, 99999, 99999, 99999])
        self.source = pd.DataFrame(rows, columns=["Scenario", "Region", "Variable", "Unit", "2019", "2024", "2030", "2050"])

    def data(self, *, source=None, config=None):
        source = self.source if source is None else source
        config = self.config if config is None else config
        with patch("omnia_results.metrics.pd.read_csv", return_value=source.copy()) as read:
            data = ConsolidationData(config, Path("."))
        self.assertTrue(all(Path(call.args[0]).name == "raw_costs.csv" for call in read.call_args_list))
        return data

    @staticmethod
    def select(frame, scenario, sector):
        return frame[(frame.Scenario == scenario) & (frame.Sector == sector)]

    def test_every_scenario_uses_its_own_sector_climate_and_year_global_cost_reference(self):
        frame = self.data().sector_costs()
        self.assertEqual(len(frame), 120)
        for climate in ("ndc", "baseline"):
            for p, policy in enumerate(POLICIES):
                scenario = f"{climate}_{policy}"
                for sector in SECTORS:
                    reference = np.asarray(self.reference[climate][sector.key]).sum(axis=0)
                    changes = np.asarray(self.regional_changes[climate][sector.key]).sum(axis=0) * p
                    selected = self.select(frame, scenario, sector.label)
                    with self.subTest(scenario=scenario, sector=sector.key):
                        self.assertEqual(selected.Year.tolist(), self.config["years"])
                        np.testing.assert_allclose(selected.Value, changes / reference * 100, atol=1e-12)
                        np.testing.assert_allclose(selected.ScenarioCost, reference + changes)
                        np.testing.assert_allclose(selected.ReferenceCost, reference)
                        np.testing.assert_allclose(selected.Numerator, reference + changes)
                        np.testing.assert_allclose(selected.Denominator, reference)
                        self.assertEqual(selected.ReferenceScenario.unique().tolist(), [f"{climate}_noce"])
                        self.assertEqual(selected.ReferenceYear.tolist(), self.config["years"])

    def test_percentage_is_change_of_global_sums_not_mean_regional_changes(self):
        frame = self.data().sector_costs()
        baseline = self.select(frame, "baseline_medce_eu", "Cement").set_index("Year")
        ndc = self.select(frame, "ndc_medce_eu", "Cement").set_index("Year")
        self.assertAlmostEqual(baseline.loc[2024, "Value"], -20)
        self.assertNotAlmostEqual(baseline.loc[2024, "Value"], (5 / 20 - 25 / 80) / 2 * 100)
        self.assertAlmostEqual(baseline.loc[2030, "Value"], 5)
        self.assertAlmostEqual(ndc.loc[2030, "Value"], -10)
        self.assertAlmostEqual(ndc.loc[2050, "Value"], -2)
        self.assertEqual(ndc.loc[2050, "ReferenceCost"], 250)
        self.assertNotEqual(ndc.loc[2050, "ReferenceCost"], ndc.loc[2019, "ReferenceCost"])

    def test_exact_sector_parents_and_units_exclude_nested_and_system_costs(self):
        frame = self.data().sector_costs()
        variables = [f"Total Annualised Cost|Industry|{sector.path}" for sector in SECTORS]
        exact = self.source[(self.source.Unit == COST_UNIT) & self.source.Variable.isin(variables)]
        pd.testing.assert_frame_equal(frame, self.data(source=exact).sector_costs())
        for observation in frame.itertuples():
            sector = next(sector for sector in SECTORS if sector.label == observation.Sector)
            self.assertEqual(json.loads(observation.SourceVariables), [f"Total Annualised Cost|Industry|{sector.path}"])
            self.assertEqual(json.loads(observation.SourceRegions), ["EUE", "CHN"])
            self.assertEqual(observation.Unit, "%")
            self.assertEqual(observation.NumeratorUnit, COST_UNIT)
            self.assertEqual(observation.DenominatorUnit, COST_UNIT)
            self.assertEqual(observation.ConversionFactor, 100)
            self.assertEqual(observation.Offset, -100)
            self.assertEqual(observation.Metric, "Annualised sector-cost change")
            self.assertEqual(observation.Series, observation.Sector)

    def test_historical_context_is_retained_and_main_period_needs_no_interpolated_year(self):
        frame = self.data().sector_costs()
        context = frame[frame.Role == "context"]
        plotted = frame[frame.Role == "plotted"]
        self.assertEqual(len(context), 30)
        self.assertEqual(context.Year.unique().tolist(), [2019])
        self.assertEqual(sorted(plotted.Year.unique()), [2024, 2030, 2050])
        self.assertAlmostEqual(self.select(frame, "baseline_highce_gbl", "Cement").Value.iloc[0], 40)
        self.assertAlmostEqual(self.select(frame, "ndc_highce_gbl", "Cement").Value.iloc[0], -40)
        shifted = self.data(config={**self.config, "main_cost_start_year": 2025}).sector_costs()
        self.assertEqual(sorted(shifted.loc[shifted.Role == "context", "Year"].unique()), [2019, 2024])
        self.assertEqual(sorted(shifted.loc[shifted.Role == "plotted", "Year"].unique()), [2030, 2050])

    def test_missing_current_or_reference_cost_cells_are_preserved_under_both_activity_policies(self):
        variable = "Total Annualised Cost|Industry|Non-Metallic Minerals|Cement"
        for scenario in ("baseline_medce_eu", "baseline_noce"):
            source = self.source.copy()
            selected = ((source.Scenario == scenario) & (source.Region == "EUE")
                        & (source.Variable == variable) & (source.Unit == COST_UNIT))
            source.loc[selected, "2030"] = np.nan
            for policy in ("preserve", "zero"):
                with self.subTest(scenario=scenario, activity_policy=policy):
                    data = self.data(source=source, config={**self.config, "missing_activity": policy})
                    frame = data.sector_costs()
                    current = self.select(frame, scenario, "Cement").set_index("Year").loc[2030]
                    self.assertTrue(np.isnan(current.Value))
                    self.assertTrue(np.isnan(current.ScenarioCost))
                    if scenario == "baseline_noce":
                        affected = frame[(frame.Scenario.str.startswith("baseline_"))
                                         & (frame.Sector == "Cement") & (frame.Year == 2030)]
                        self.assertTrue(affected.Value.isna().all())
                        self.assertTrue(affected.ReferenceCost.isna().all())
                    else:
                        self.assertEqual(current.ReferenceCost, 200)
                    audit = data.results[scenario].coverage
                    self.assertEqual(len(audit), 1)
                    self.assertEqual(audit[0]["Treatment"], "preserve")

    def test_absent_regional_cost_rows_are_not_zero_filled_or_excluded_from_global_sums(self):
        selected = ((self.source.Scenario == "ndc_highce_eu") & (self.source.Region == "CHN")
                    & (self.source.Variable == "Total Annualised Cost|Industry|Iron and Steel")
                    & (self.source.Unit == COST_UNIT))
        for policy in ("preserve", "zero"):
            with self.subTest(activity_policy=policy):
                data = self.data(source=self.source.loc[~selected],
                                 config={**self.config, "missing_activity": policy})
                frame = data.sector_costs()
                rows = self.select(frame, "ndc_highce_eu", "Iron and steel")
                self.assertTrue(rows.Value.isna().all())
                self.assertTrue(rows.ScenarioCost.isna().all())
                np.testing.assert_allclose(rows.ReferenceCost, [80, 400, 200, 1000])
                audit = data.results["ndc_highce_eu"].coverage
                self.assertEqual(len(audit), 4)
                self.assertTrue(all(record["Issue"] == "absent row" and record["Treatment"] == "preserve" for record in audit))

    def test_nonpositive_no_ce_reference_is_undefined_and_negative_current_cost_is_valid(self):
        variable = "Total Annualised Cost|Industry|Non-Metallic Minerals|Cement"
        for value in (0, -1):
            source = self.source.copy()
            selected = ((source.Scenario == "ndc_noce") & (source.Variable == variable)
                        & (source.Unit == COST_UNIT))
            source.loc[selected, "2050"] = value
            frame = self.data(source=source).sector_costs()
            affected = frame[(frame.Scenario.str.startswith("ndc_")) & (frame.Sector == "Cement") & (frame.Year == 2050)]
            self.assertEqual(len(affected), 5)
            self.assertTrue(affected.Value.isna().all())
            np.testing.assert_allclose(affected.ReferenceCost, value * 2)
        source = self.source.copy()
        selected = ((source.Scenario == "ndc_medce_eu") & (source.Variable == variable)
                    & (source.Unit == COST_UNIT))
        source.loc[selected, "2050"] = [-300, 100]
        row = self.select(self.data(source=source).sector_costs(), "ndc_medce_eu", "Cement").set_index("Year").loc[2050]
        self.assertEqual(row.ScenarioCost, -200)
        self.assertEqual(row.ReferenceCost, 250)
        self.assertAlmostEqual(row.Value, -180)

    def test_climate_panel_order_is_stable_and_all_scenarios_and_a_main_period_are_required(self):
        frame = self.data().sector_costs()
        reversed_config = {**self.config, "scenarios": self.config["scenarios"][::-1]}
        pd.testing.assert_frame_equal(frame, self.data(config=reversed_config).sector_costs())
        for climate, panels, label in (("ndc", "abc", "NDC"), ("baseline", "def", "NDC+LTT")):
            selected = frame[frame.Panel.isin(list(panels))]
            self.assertTrue(selected.Scenario.str.startswith(climate + "_").all())
            self.assertEqual(selected.ClimatePathway.unique().tolist(), [label])
        for config in ({**self.config, "scenarios": self.config["scenarios"][:-1]},
                       {**self.config, "main_cost_start_year": 2051}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.data(config=config).sector_costs()

    def test_rendered_five_curves_use_source_values_only_in_main_period_and_correct_climate_rows(self):
        from omnia_results.consolidation_costs import SectorCostFigures

        class Builder(SectorCostFigures):
            def __init__(self, data, config):
                self.data, self.config = data, config

        spec = Builder(self.data(), self.config).sector_costs()
        try:
            self.assertEqual(len(spec.figure.axes), 6)
            for index, (ax, panel) in enumerate(zip(spec.figure.axes, "abcdef")):
                expected = spec.data[(spec.data.Panel == panel) & (spec.data.Role == "plotted")]
                actual = {line.get_label(): line for line in ax.lines if not line.get_label().startswith("_")}
                self.assertEqual(set(actual), set(expected.Scenario))
                self.assertEqual(len(actual), 5)
                for scenario, rows in expected.groupby("Scenario", sort=False):
                    np.testing.assert_array_equal(actual[scenario].get_xdata(), [2024, 2030, 2050])
                    np.testing.assert_allclose(actual[scenario].get_ydata(), rows.Value, atol=1e-12)
                self.assertTrue(ax.get_title(loc="left").endswith("NDC" if index < 3 else "NDC+LTT"))
                if index < 3:
                    self.assertEqual(ax.get_ylim(), spec.figure.axes[index + 3].get_ylim())
        finally:
            plt.close(spec.figure)


if __name__ == "__main__":
    unittest.main()
