"""Independent raw-source checks for material output across CE scenarios."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np
import pandas as pd

from omnia_results.consolidation_data import ConsolidationData


POLICIES = ("noce", "medce_eu", "highce_eu", "medce_gbl", "highce_gbl")
CEMENT = "Production|Non-Metallic Minerals|Cement"
CLINKER = "Production|Non-Metallic Minerals|Cement Clinker"
STEEL = "Production|Iron and Steel"
ALUMINIUM = "Production|Non-Ferrous Metals|Aluminum"


class ConsolidationMaterialTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "input": "raw_materials.csv",
            "scenarios": [f"{climate}_{policy}" for climate in ("baseline", "ndc") for policy in POLICIES],
            "regions": ["EUE", "CHN"], "years": [2019, 2050], "missing_activity": "preserve",
            "historical_year": 2019, "comparison_year": 2050,
        }
        self.starts = {
            "baseline": {CEMENT: [20, 80], CLINKER: [15, 60],
                         STEEL + "|Primary": [20, 80], STEEL + "|Secondary": [5, 15],
                         ALUMINIUM + "|Primary": [2, 8], ALUMINIUM + "|Secondary": [.5, 1.5]},
            "ndc": {CEMENT: [10, 50], CLINKER: [8, 22],
                    STEEL + "|Primary": [10, 20], STEEL + "|Secondary": [1, 4],
                    ALUMINIUM + "|Primary": [1, 3], ALUMINIUM + "|Secondary": [.25, .75]},
        }
        rows = []
        for climate in ("baseline", "ndc"):
            for policy_index, policy in enumerate(POLICIES):
                p = policy_index
                endings = ({CEMENT: [40 - 4 * p, 160 - 16 * p],
                            CLINKER: [30 - 3 * p, 120 - 12 * p],
                            STEEL + "|Primary": [20 - 2 * p, 100 - 8 * p],
                            STEEL + "|Secondary": [10 + 3 * p, 20 + 7 * p],
                            ALUMINIUM + "|Primary": [4 - .25 * p, 12 - .75 * p],
                            ALUMINIUM + "|Secondary": [1 + .25 * p, 3 + .75 * p]}
                           if climate == "baseline" else
                           {CEMENT: [30 - p, 70 - 4 * p], CLINKER: [15 - p, 60 - 2 * p],
                            STEEL + "|Primary": [15 - p, 35 - 4 * p],
                            STEEL + "|Secondary": [3 + .5 * p, 7 + 1.5 * p],
                            ALUMINIUM + "|Primary": [2 - .1 * p, 6 - .4 * p],
                            ALUMINIUM + "|Secondary": [.5 + .05 * p, 1.5 + .15 * p]})
                for variable, regional_endings in endings.items():
                    for index, region in enumerate(self.config["regions"]):
                        # Non-noCE historical values deliberately disagree, so
                        # the historical bar must select its own noCE scenario.
                        start = self.starts[climate][variable][index] + p * 10000
                        rows.append([f"{climate}_{policy}", region, variable, "Mt/yr", start,
                                     regional_endings[index]])
                        rows.append([f"{climate}_{policy}", region, variable, "kt/yr", 99999, 99999])
                        rows.append([f"{climate}_{policy}", region, variable + "|Detail", "Mt/yr", 99999, 99999])
        self.source = pd.DataFrame(rows, columns=["Scenario", "Region", "Variable", "Unit", "2019", "2050"])

    def data(self, *, source=None, config=None):
        source = self.source if source is None else source
        config = self.config if config is None else config
        with patch("omnia_results.metrics.pd.read_csv", return_value=source.copy()) as read:
            data = ConsolidationData(config, Path("."))
        self.assertTrue(all(Path(call.args[0]).name == "raw_materials.csv" for call in read.call_args_list))
        return data

    @staticmethod
    def select(frame, scenario, sector, series, metric="Material production", year=2050):
        return frame[(frame.Scenario == scenario) & (frame.Sector == sector)
                     & (frame.Series == series) & (frame.Metric == metric) & (frame.Year == year)]

    def test_global_components_and_totals_are_extracted_independently_for_all_ten_scenarios(self):
        frame = self.data().material_production()
        self.assertEqual(len(frame), 150)
        for climate in ("ndc", "baseline"):
            for p, policy in enumerate(POLICIES):
                scenario = f"{climate}_{policy}"
                expected = ({("Cement", "Cement"): 200 - 20 * p,
                             ("Cement", "Clinker"): 150 - 15 * p,
                             ("Iron and steel", "Primary"): 120 - 10 * p,
                             ("Iron and steel", "Secondary"): 30 + 10 * p,
                             ("Iron and steel", "Total"): 150,
                             ("Aluminium", "Primary"): 16 - p,
                             ("Aluminium", "Secondary"): 4 + p,
                             ("Aluminium", "Total"): 20}
                            if climate == "baseline" else
                            {("Cement", "Cement"): 100 - 5 * p,
                             ("Cement", "Clinker"): 75 - 3 * p,
                             ("Iron and steel", "Primary"): 50 - 5 * p,
                             ("Iron and steel", "Secondary"): 10 + 2 * p,
                             ("Iron and steel", "Total"): 60 - 3 * p,
                             ("Aluminium", "Primary"): 8 - .5 * p,
                             ("Aluminium", "Secondary"): 2 + .2 * p,
                             ("Aluminium", "Total"): 10 - .3 * p})
                for (sector, series), value in expected.items():
                    with self.subTest(scenario=scenario, sector=sector, series=series):
                        selected = self.select(frame, scenario, sector, series)
                        self.assertEqual(len(selected), 1)
                        self.assertAlmostEqual(selected.Value.iloc[0], value)
                        self.assertEqual(selected.Unit.iloc[0], "Mt/yr")
                        self.assertEqual(selected.Role.iloc[0], "context" if series == "Total" else "plotted")
        cement = frame[(frame.Sector == "Cement") & (frame.Metric == "Material production")]
        self.assertEqual(set(cement.Series), {"Cement", "Clinker"})

    def test_historical_bars_use_each_climates_no_ce_source_without_scenario_averaging(self):
        frame = self.data().material_production()
        expected = {"ndc": {("Cement", "Cement"): 60, ("Cement", "Clinker"): 30,
                            ("Iron and steel", "Total"): 35, ("Aluminium", "Total"): 5},
                    "baseline": {("Cement", "Cement"): 100, ("Cement", "Clinker"): 75,
                                 ("Iron and steel", "Total"): 120, ("Aluminium", "Total"): 12}}
        history = frame[(frame.Metric == "Material production") & (frame.Year == 2019)]
        self.assertEqual(set(history.Scenario), {"ndc_noce", "baseline_noce"})
        self.assertEqual(history.BarOrder.unique().tolist(), [0])
        self.assertEqual(history.BarType.unique().tolist(), ["historical"])
        for climate, quantities in expected.items():
            for (sector, series), value in quantities.items():
                self.assertAlmostEqual(self.select(frame, f"{climate}_noce", sector, series,
                                                   year=2019).Value.iloc[0], value)

    def test_annotations_use_own_climate_2050_reference_and_secondary_share_of_total(self):
        frame = self.data().material_production()
        for climate in ("ndc", "baseline"):
            for p, policy in enumerate(POLICIES):
                scenario = f"{climate}_{policy}"
                reference = f"{climate}_noce"
                for sector, series in (("Cement", "Cement"), ("Iron and steel", "Total"), ("Aluminium", "Total")):
                    quantity = self.select(frame, scenario, sector, series).Value.iloc[0]
                    noce = self.select(frame, reference, sector, series).Value.iloc[0]
                    change = self.select(frame, scenario, sector, series, "Production change from no CE").iloc[0]
                    with self.subTest(scenario=scenario, sector=sector):
                        self.assertAlmostEqual(change.Value, (quantity / noce - 1) * 100)
                        self.assertAlmostEqual(change.Numerator, quantity)
                        self.assertAlmostEqual(change.Denominator, noce)
                        self.assertEqual(change.ConversionFactor, 100)
                        self.assertEqual(change.Offset, -100)
                        self.assertEqual(change.ReferenceScenario, reference)
                        self.assertEqual(change.ReferenceYear, 2050)
                # Denominators differ by climate; e.g. medium cement CE reduces
                # NDC output 5%, whereas NDC+LTT output falls 10%.
                if p == 1:
                    cement = self.select(frame, scenario, "Cement", "Cement", "Production change from no CE").iloc[0]
                    self.assertAlmostEqual(cement.Value, -5 if climate == "ndc" else -10)
            for sector in ("Iron and steel", "Aluminium"):
                shares = frame[(frame.Sector == sector) & (frame.Metric == "Secondary production share")
                               & frame.Scenario.str.startswith(climate + "_")]
                self.assertEqual(len(shares), 6)
                for observation in shares.itertuples():
                    secondary = self.select(frame, observation.Scenario, sector, "Secondary", year=observation.Year).Value.iloc[0]
                    total = self.select(frame, observation.Scenario, sector, "Total", year=observation.Year).Value.iloc[0]
                    self.assertAlmostEqual(observation.Value, secondary / total * 100)
                    self.assertAlmostEqual(observation.Numerator, secondary)
                    self.assertAlmostEqual(observation.Denominator, total)
                    self.assertEqual(observation.ReferenceScenario, observation.Scenario)
                    self.assertEqual(observation.ReferenceYear, observation.Year)

    def test_exact_units_and_parent_variables_and_region_provenance_exclude_distractors(self):
        frame = self.data().material_production()
        source = self.source[(self.source.Unit == "Mt/yr") & ~self.source.Variable.str.endswith("|Detail")]
        pd.testing.assert_frame_equal(frame, self.data(source=source).material_production())
        expected_variables = {("Cement", "Cement"): [CEMENT], ("Cement", "Clinker"): [CLINKER],
                              ("Iron and steel", "Primary"): [STEEL + "|Primary"],
                              ("Iron and steel", "Secondary"): [STEEL + "|Secondary"],
                              ("Iron and steel", "Total"): [STEEL + "|Primary", STEEL + "|Secondary"],
                              ("Aluminium", "Primary"): [ALUMINIUM + "|Primary"],
                              ("Aluminium", "Secondary"): [ALUMINIUM + "|Secondary"],
                              ("Aluminium", "Total"): [ALUMINIUM + "|Primary", ALUMINIUM + "|Secondary"]}
        for observation in frame[frame.Metric == "Material production"].itertuples():
            self.assertEqual(json.loads(observation.SourceVariables), expected_variables[observation.Sector, observation.Series])
            self.assertEqual(json.loads(observation.SourceRegions), ["EUE", "CHN"])
            self.assertAlmostEqual(observation.Numerator, observation.Value)
            self.assertEqual(observation.NumeratorUnit, "Mt/yr")
            self.assertEqual(observation.ConversionFactor, 1)

    def test_ndc_top_row_and_ndc_ltt_bottom_row_have_six_ordered_bar_categories(self):
        frame = self.data().material_production()
        for climate, panels, label in (("ndc", "abc", "NDC"), ("baseline", "def", "NDC+LTT")):
            selected = frame[frame.Panel.isin(list(panels))]
            self.assertTrue(selected.Scenario.str.startswith(climate + "_").all())
            self.assertEqual(selected.ClimatePathway.unique().tolist(), [label])
            for panel in panels:
                bars = selected[selected.Panel == panel][["BarOrder", "BarLabel", "Year"]].drop_duplicates().sort_values("BarOrder")
                self.assertEqual(bars.BarOrder.tolist(), list(range(6)))
                self.assertEqual(bars.BarLabel.tolist(), ["2019", "No CE", "M–EU", "H–EU", "M–EU+", "H–EU+"])
                self.assertEqual(bars.Year.tolist(), [2019, 2050, 2050, 2050, 2050, 2050])

    def test_missing_metal_route_preserves_unknown_totals_and_annotations_or_uses_explicit_zero(self):
        source = self.source.copy()
        selected = ((source.Scenario == "baseline_highce_gbl") & (source.Region == "EUE")
                    & (source.Variable == STEEL + "|Secondary") & (source.Unit == "Mt/yr"))
        source.loc[selected, "2050"] = np.nan
        for policy in ("preserve", "zero"):
            with self.subTest(policy=policy):
                data = self.data(source=source, config={**self.config, "missing_activity": policy})
                frame = data.material_production()
                total = self.select(frame, "baseline_highce_gbl", "Iron and steel", "Total").Value.iloc[0]
                share = self.select(frame, "baseline_highce_gbl", "Iron and steel", "Secondary", "Secondary production share").Value.iloc[0]
                change = self.select(frame, "baseline_highce_gbl", "Iron and steel", "Total", "Production change from no CE").Value.iloc[0]
                if policy == "preserve":
                    self.assertTrue(np.isnan(total))
                    self.assertTrue(np.isnan(share))
                    self.assertTrue(np.isnan(change))
                else:
                    self.assertAlmostEqual(total, 128)
                    self.assertAlmostEqual(share, 48 / 128 * 100)
                    self.assertAlmostEqual(change, (128 / 150 - 1) * 100)
                self.assertEqual(self.select(frame, "baseline_highce_gbl", "Iron and steel", "Primary").Value.iloc[0], 80)
                self.assertEqual(self.select(frame, "baseline_highce_gbl", "Cement", "Cement").Value.iloc[0], 120)
                self.assertEqual(data.results["baseline_highce_gbl"].coverage[0]["Treatment"], policy)

    def test_nonpositive_change_and_secondary_share_denominators_are_undefined(self):
        for value in (0, -1):
            source = self.source.copy()
            selected = ((source.Scenario == "baseline_noce") & (source.Unit == "Mt/yr")
                        & source.Variable.isin([CEMENT, STEEL + "|Primary", STEEL + "|Secondary"]))
            source.loc[selected, "2050"] = value
            with self.subTest(value=value):
                frame = self.data(source=source).material_production()
                changes = frame[(frame.Metric == "Production change from no CE")
                                & frame.Scenario.str.startswith("baseline_")
                                & frame.Sector.isin(["Cement", "Iron and steel"])]
                self.assertEqual(len(changes), 10)
                self.assertTrue(changes.Value.isna().all())
                share = self.select(frame, "baseline_noce", "Iron and steel", "Secondary", "Secondary production share")
                self.assertTrue(share.Value.isna().all())
                self.assertEqual(self.select(frame, "baseline_noce", "Cement", "Cement").Value.iloc[0], value * 2)
                ndc_change = self.select(frame, "ndc_medce_eu", "Cement", "Cement", "Production change from no CE").Value.iloc[0]
                self.assertAlmostEqual(ndc_change, -5)

    def test_all_ten_scenarios_are_required_for_the_complete_comparison(self):
        config = {**self.config, "scenarios": self.config["scenarios"][:-1]}
        with self.assertRaises(ValueError):
            self.data(config=config).material_production()

    def test_rendered_grouped_and_stacked_bars_match_raw_values_in_correct_climate_rows(self):
        from omnia_results.consolidation_materials import MaterialComparisonFigures

        class Builder(MaterialComparisonFigures):
            def __init__(self, data, config):
                self.data, self.config = data, config

        spec = Builder(self.data(), self.config).material_production()
        try:
            self.assertEqual(len(spec.figure.axes), 6)
            for index, (ax, panel) in enumerate(zip(spec.figure.axes, "abcdef")):
                self.assertFalse(any(isinstance(collection, PathCollection) for collection in ax.collections))
                self.assertFalse(any(line.get_marker() in {"D", "d"} for line in ax.lines))
                selected = spec.data[(spec.data.Panel == panel) & (spec.data.Metric == "Material production")
                                     & (spec.data.Role == "plotted")]
                actual = {container.get_label(): container for container in ax.containers}
                self.assertEqual(set(actual), set(selected.Series))
                for series, bars in actual.items():
                    expected = selected[selected.Series == series].sort_values("BarOrder").Value.to_numpy()
                    np.testing.assert_allclose([bar.get_height() for bar in bars], expected)
                    if series == "Secondary":
                        primary = selected[selected.Series == "Primary"].sort_values("BarOrder").Value.to_numpy()
                        np.testing.assert_allclose([bar.get_y() for bar in bars], primary)
                    else:
                        np.testing.assert_allclose([bar.get_y() for bar in bars], 0)
                if index < 3:
                    self.assertEqual(ax.get_ylim(), spec.figure.axes[index + 3].get_ylim())
        finally:
            plt.close(spec.figure)


if __name__ == "__main__":
    unittest.main()
