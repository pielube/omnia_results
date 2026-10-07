"""Asymmetric raw-source examples for the refined production figure."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from omnia_results.consolidation import ConsolidationBuilder
from omnia_results.consolidation_data import ConsolidationData, climate_label, policy_label


CEMENT = "Production|Non-Metallic Minerals|Cement"
CLINKER = "Production|Non-Metallic Minerals|Cement Clinker"
STEEL = "Production|Iron and Steel"
ALUMINIUM = "Production|Non-Ferrous Metals|Aluminum"


class ConsolidationTests(unittest.TestCase):
    def setUp(self):
        self.config = {"input": "raw_results.csv", "scenarios": ["baseline_noce", "ndc_noce"],
                       "regions": ["EUE", "CHN"], "years": [2019, 2050],
                       "missing_activity": "preserve"}
        # Regional values differ by sector, year and climate. Aggregating the
        # wrong row, using exported figures or counting detail children fails.
        observations = {
            "baseline_noce": {
                CEMENT: [(10, 12), (20, 23)], CLINKER: [(2, 3), (4, 6)],
                STEEL + "|Primary": [(30, 40), (5, 8)],
                STEEL + "|Secondary": [(7, 11), (9, 13)],
                ALUMINIUM + "|Primary": [(3, 4), (2, 6)],
                ALUMINIUM + "|Secondary": [(1, 2), (4, 9)],
            },
            "ndc_noce": {
                CEMENT: [(90, 130), (25, 20)], CLINKER: [(31, 45), (8, 7)],
                STEEL + "|Primary": [(200, 150), (30, 20)],
                STEEL + "|Secondary": [(10, 35), (20, 40)],
                ALUMINIUM + "|Primary": [(30, 20), (25, 18)],
                ALUMINIUM + "|Secondary": [(2, 12), (5, 8)],
            },
        }
        rows = []
        for scenario, variables in observations.items():
            for variable, regional_values in variables.items():
                for region, values in zip(self.config["regions"], regional_values):
                    rows.append([scenario, region, variable, "Mt/yr", *values])
                    rows.append([scenario, region, variable, "kt/yr", 9999, 9999])
                    rows.append([scenario, region, variable + "|Detail", "Mt/yr", 9999, 9999])
        self.source = pd.DataFrame(rows, columns=["Scenario", "Region", "Variable", "Unit",
                                                 "2019", "2050"])

    def data(self, *, source=None, config=None):
        source = self.source if source is None else source
        config = self.config if config is None else config
        with patch("omnia_results.metrics.pd.read_csv", return_value=source.copy()) as read:
            data = ConsolidationData(config, Path("."))
        self.assertTrue(all(Path(call.args[0]).name == "raw_results.csv"
                            for call in read.call_args_list))
        return data

    @staticmethod
    def values(frame, scenario, sector, series):
        return frame.loc[(frame.Scenario == scenario) & (frame.Sector == sector)
                         & (frame.Series == series), "Value"].tolist()

    def test_each_climate_row_is_independently_extracted_from_the_raw_source(self):
        frame = self.data().production()
        self.assertEqual(len(frame), 32)
        expected = {
            "baseline_noce": {("Cement", "Cement"): [30, 35],
                              ("Iron and steel", "Total"): [51, 72],
                              ("Aluminium", "Total"): [10, 21]},
            "ndc_noce": {("Cement", "Cement"): [115, 150],
                         ("Iron and steel", "Total"): [260, 245],
                         ("Aluminium", "Total"): [62, 58]},
        }
        for scenario, series in expected.items():
            for (sector, route), values in series.items():
                with self.subTest(scenario=scenario, sector=sector):
                    self.assertEqual(self.values(frame, scenario, sector, route), values)
        self.assertEqual(frame.Unit.unique().tolist(), ["Mt/yr"])
        self.assertEqual(frame.Scope.unique().tolist(), ["Global"])

    def test_cement_and_clinker_are_separate_and_metals_have_exact_route_totals(self):
        frame = self.data().production()
        self.assertEqual(self.values(frame, "baseline_noce", "Cement", "Clinker"), [6, 9])
        self.assertEqual(self.values(frame, "ndc_noce", "Cement", "Clinker"), [39, 52])
        self.assertEqual(frame.loc[frame.Sector == "Cement", "Series"].unique().tolist(),
                         ["Cement", "Clinker"])
        for scenario in self.config["scenarios"]:
            for sector in ("Iron and steel", "Aluminium"):
                table = frame.loc[(frame.Scenario == scenario) & (frame.Sector == sector)]
                table = table.pivot(index="Year", columns="Series", values="Value")
                np.testing.assert_array_equal(table.Total, table.Primary + table.Secondary)
        totals = frame.loc[(frame.Sector == "Iron and steel") & (frame.Series == "Total")]
        self.assertTrue(all(json.loads(variables) == [STEEL + "|Primary", STEEL + "|Secondary"]
                            for variables in totals.SourceVariables))

    def test_wrong_units_and_detail_children_do_not_change_global_production(self):
        parent_rows = self.source[(self.source.Unit == "Mt/yr")
                                  & ~self.source.Variable.str.endswith("|Detail")]
        pd.testing.assert_frame_equal(self.data().production(),
                                      self.data(source=parent_rows).production())

    def test_scenario_selection_order_cannot_reverse_the_report_rows(self):
        data = self.data(config={**self.config, "scenarios": ["baseline_noce", "ndc_noce"]})
        frame = data.production()
        self.assertEqual(frame.Panel.unique().tolist(), list("abcdef"))
        self.assertEqual(frame.loc[frame.Panel.isin(list("abc")), "Scenario"].unique().tolist(),
                         ["ndc_noce"])
        self.assertEqual(frame.loc[frame.Panel.isin(list("def")), "Scenario"].unique().tolist(),
                         ["baseline_noce"])
        self.assertEqual(frame.ClimatePathway.unique().tolist(), ["NDC", "NDC+LTT"])
        self.assertEqual(frame.CE.unique().tolist(), ["No CE"])

    def test_climate_interpretation_applies_to_every_ce_setting(self):
        labels = {"noce": "No CE", "medce_eu": "Medium CE, EU",
                  "highce_eu": "High CE, EU", "medce_gbl": "Medium CE, EU + partners",
                  "highce_gbl": "High CE, EU + partners"}
        for prefix, climate in (("baseline", "NDC+LTT"), ("ndc", "NDC")):
            for policy, label in labels.items():
                with self.subTest(prefix=prefix, policy=policy):
                    scenario = f"{prefix}_{policy}"
                    self.assertEqual(climate_label(scenario), climate)
                    self.assertEqual(policy_label(scenario), label)

    def test_blank_regional_route_retains_global_missingness_or_explicit_zero_policy(self):
        source = self.source.copy()
        selected = ((source.Scenario == "baseline_noce") & (source.Region == "CHN")
                    & (source.Variable == STEEL + "|Primary") & (source.Unit == "Mt/yr"))
        source.loc[selected, "2050"] = np.nan
        for treatment, expected in (("preserve", np.nan), ("zero", 64)):
            data = self.data(source=source, config={**self.config, "missing_activity": treatment})
            with self.subTest(treatment=treatment):
                frame = data.production()
                total = self.values(frame, "baseline_noce", "Iron and steel", "Total")
                self.assertEqual(total[0], 51)
                if treatment == "preserve":
                    self.assertTrue(np.isnan(total[1]))
                else:
                    self.assertEqual(total[1], expected)
                self.assertEqual(self.values(frame, "baseline_noce", "Iron and steel", "Secondary"),
                                 [16, 24])
                self.assertEqual(data.results["baseline_noce"].coverage[0]["Treatment"], treatment)
                self.assertEqual(data.results["baseline_noce"].coverage[0]["Issue"], "blank cell")

    def test_absent_regional_rows_are_audited_and_entirely_absent_series_are_errors(self):
        selected = ((self.source.Scenario == "ndc_noce") & (self.source.Region == "CHN")
                    & (self.source.Variable == CLINKER) & (self.source.Unit == "Mt/yr"))
        source = self.source.loc[~selected]
        preserve = self.data(source=source)
        self.assertTrue(np.isnan(self.values(preserve.production(), "ndc_noce", "Cement", "Clinker")).all())
        zero = self.data(source=source, config={**self.config, "missing_activity": "zero"})
        self.assertEqual(self.values(zero.production(), "ndc_noce", "Cement", "Clinker"), [31, 45])
        self.assertEqual(zero.results["ndc_noce"].coverage[0]["Issue"], "absent row")
        self.assertEqual(len(zero.results["ndc_noce"].coverage), 2)
        selected = ((self.source.Scenario == "ndc_noce") & (self.source.Variable == CLINKER)
                    & (self.source.Unit == "Mt/yr"))
        for treatment in ("preserve", "zero"):
            with self.subTest(treatment=treatment), self.assertRaisesRegex(ValueError, "Required series is absent"):
                self.data(source=self.source.loc[~selected],
                          config={**self.config, "missing_activity": treatment}).production()

    def test_figure_uses_two_climate_rows_and_matching_sector_scales(self):
        spec = ConsolidationBuilder(self.data(), self.config).production()
        try:
            self.assertEqual(len(spec.figure.axes), 6)
            for column in range(3):
                top, bottom = spec.figure.axes[column], spec.figure.axes[column + 3]
                self.assertEqual(top.get_ylim(), bottom.get_ylim())
                self.assertTrue(top.get_title(loc="left").endswith("NDC"))
                self.assertTrue(bottom.get_title(loc="left").endswith("NDC+LTT"))
                for line in (*top.lines, *bottom.lines):
                    self.assertLessEqual(max(line.get_ydata()), top.get_ylim()[1])
            np.testing.assert_array_equal(spec.figure.axes[0].lines[0].get_ydata(), [115, 150])
            np.testing.assert_array_equal(spec.figure.axes[3].lines[0].get_ydata(), [30, 35])
        finally:
            plt.close(spec.figure)


if __name__ == "__main__":
    unittest.main()
