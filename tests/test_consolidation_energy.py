"""Raw-source examples for global energy and derived energy intensity."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from omnia_results.consolidation_data import ConsolidationData
from omnia_results.metrics import SECTORS


class ConsolidationEnergyTests(unittest.TestCase):
    def setUp(self):
        self.config = {"input": "raw_energy.csv", "scenarios": ["ndc_noce", "baseline_noce"],
                       "regions": ["EUE", "CHN"], "years": [2019, 2050],
                       "missing_activity": "preserve"}
        self.energy = {
            "baseline_noce": {"cement": [(3, 5), (7, 3)], "steel": [(2, 4), (6, 8)],
                              "aluminium": [(1, 3), (4, 6)]},
            "ndc_noce": {"cement": [(5, 6), (20, 10)], "steel": [(4, 9), (16, 21)],
                         "aluminium": [(4, 6), (6, 14)]},
        }
        self.production = {
            "baseline_noce": {
                "cement": {"": [(1, 4), (9, 16)]},
                "steel": {"|Primary": [(5, 7), (20, 30)], "|Secondary": [(1, 3), (4, 10)]},
                "aluminium": {"|Primary": [(1, 2), (5, 8)], "|Secondary": [(1, 1), (3, 4)]},
            },
            "ndc_noce": {
                "cement": {"": [(2, 3), (8, 5)]},
                "steel": {"|Primary": [(2, 3), (7, 9)], "|Secondary": [(1, 2), (2, 6)]},
                "aluminium": {"|Primary": [(2, 4), (3, 6)], "|Secondary": [(1, 2), (4, 8)]},
            },
        }
        rows = []
        for scenario in self.config["scenarios"]:
            for sector in SECTORS:
                weights = [.5, .1, .2, .2] if sector.key == "cement" else [.4, .1, .2, .2, .1]
                for index, region in enumerate(self.config["regions"]):
                    energies = self.energy[scenario][sector.key][index]
                    for fuel, fraction in zip(sector.fuels, weights):
                        variable = f"Final Energy|Industry|{sector.path}|{fuel}"
                        rows.append([scenario, region, variable, "EJ/yr", *[value * fraction for value in energies]])
                        rows.append([scenario, region, variable, "PJ/yr", 99999, 99999])
                        rows.append([scenario, region, variable + "|Detail", "EJ/yr", 99999, 99999])
                    rows.append([scenario, region, f"Final Energy|Industry|{sector.path}", "EJ/yr", 99999, 99999])
                    rows.append([scenario, region, f"Energy Intensity|Industry|{sector.path}", "EJ/Mt", 99999, 99999])
                    for route, regional_values in self.production[scenario][sector.key].items():
                        variable = f"Production|{sector.path}{route}"
                        rows.append([scenario, region, variable, "Mt/yr", *regional_values[index]])
                        rows.append([scenario, region, variable, "kt/yr", 99999, 99999])
                        rows.append([scenario, region, variable + "|Detail", "Mt/yr", 99999, 99999])
                    if sector.key == "cement":
                        rows.append([scenario, region, "Production|Non-Metallic Minerals|Cement Clinker",
                                     "Mt/yr", 99999, 99999])
        self.source = pd.DataFrame(rows, columns=["Scenario", "Region", "Variable", "Unit", "2019", "2050"])

    def data(self, *, source=None, config=None):
        source = self.source if source is None else source
        config = self.config if config is None else config
        with patch("omnia_results.metrics.pd.read_csv", return_value=source.copy()) as read:
            data = ConsolidationData(config, Path("."))
        self.assertTrue(all(Path(call.args[0]).name == "raw_energy.csv" for call in read.call_args_list))
        return data

    @staticmethod
    def metric(frame, scenario, sector, unit):
        return frame[(frame.Scenario == scenario) & (frame.Sector == sector) & (frame.Unit == unit)]

    def test_energy_and_intensity_are_independent_raw_source_global_values_for_each_pathway(self):
        frame = self.data().energy()
        self.assertEqual(len(frame), 24)
        expected = {
            ("baseline_noce", "Cement"): ([10, 8], [10, 20]),
            ("baseline_noce", "Iron and steel"): ([8, 12], [30, 50]),
            ("baseline_noce", "Aluminium"): ([5, 9], [10, 15]),
            ("ndc_noce", "Cement"): ([25, 16], [10, 8]),
            ("ndc_noce", "Iron and steel"): ([20, 30], [12, 20]),
            ("ndc_noce", "Aluminium"): ([10, 20], [10, 20]),
        }
        for (scenario, sector), (energy, production) in expected.items():
            with self.subTest(scenario=scenario, sector=sector):
                output = self.metric(frame, scenario, sector, "EJ/yr")
                intensity = self.metric(frame, scenario, sector, "GJ/t")
                np.testing.assert_allclose(output.Value, energy)
                np.testing.assert_allclose(intensity.Value, np.asarray(energy) / production * 1000)
                np.testing.assert_allclose(intensity.Numerator_EJ_yr, energy)
                np.testing.assert_allclose(intensity.Denominator_Mt_yr, production)
                self.assertEqual(intensity.ConversionFactor.tolist(), [1000, 1000])

    def test_global_intensity_is_ratio_of_sums_rather_than_mean_regional_or_reported_intensities(self):
        frame = self.data().energy()
        cement = self.metric(frame, "baseline_noce", "Cement", "GJ/t")
        self.assertAlmostEqual(cement.Value.iloc[0], 1000)
        mean_regional = (3 / 1 + 7 / 9) / 2 * 1000
        self.assertNotAlmostEqual(cement.Value.iloc[0], mean_regional)
        self.assertNotEqual(cement.Value.iloc[0], 99999 * 1000)
        self.assertEqual(cement.Denominator_Mt_yr.tolist(), [10, 20])
        steel = self.metric(frame, "baseline_noce", "Iron and steel", "GJ/t")
        self.assertEqual(steel.Denominator_Mt_yr.tolist(), [30, 50])

    def test_only_exact_fuel_parents_units_and_production_routes_enter_calculations(self):
        frame = self.data().energy()
        exact = self.source[~self.source.Variable.str.endswith("|Detail")
                            & ~self.source.Variable.str.startswith("Energy Intensity|")
                            & ~self.source.Variable.str.endswith("Cement Clinker")
                            & self.source.Unit.isin(["Mt/yr", "EJ/yr"])]
        parent_variables = [f"Final Energy|Industry|{sector.path}" for sector in SECTORS]
        exact = exact[~exact.Variable.isin(parent_variables)]
        pd.testing.assert_frame_equal(frame, self.data(source=exact).energy())
        for sector in SECTORS:
            energy_variables = [f"Final Energy|Industry|{sector.path}|{fuel}" for fuel in sector.fuels]
            production_variables = ([f"Production|{sector.path}"] if sector.key == "cement"
                                    else [f"Production|{sector.path}|Primary", f"Production|{sector.path}|Secondary"])
            energy = frame[(frame.Sector == sector.label) & (frame.Unit == "EJ/yr")]
            intensity = frame[(frame.Sector == sector.label) & (frame.Unit == "GJ/t")]
            self.assertTrue(all(json.loads(value) == energy_variables for value in energy.SourceVariables))
            self.assertTrue(all(set(json.loads(value)) == set(energy_variables + production_variables)
                                for value in intensity.SourceVariables))
            self.assertTrue(all(json.loads(value) == ["EUE", "CHN"] for value in energy.SourceRegions))

    def test_panel_assignment_and_source_labels_keep_both_no_ce_pathways(self):
        frame = self.data().energy()
        self.assertEqual(sorted(frame.Panel.unique()), list("abcdef"))
        for sector, top, bottom in zip(SECTORS, "abc", "def"):
            self.assertEqual(frame.loc[frame.Panel == top, "Unit"].unique().tolist(), ["EJ/yr"])
            self.assertEqual(frame.loc[frame.Panel == bottom, "Unit"].unique().tolist(), ["GJ/t"])
            self.assertEqual(frame.loc[frame.Panel == top, "Metric"].unique().tolist(), ["Final energy"])
            self.assertEqual(frame.loc[frame.Panel == bottom, "Metric"].unique().tolist(), ["Derived energy intensity"])
            self.assertEqual(frame.loc[frame.Panel == top, "Sector"].unique().tolist(), [sector.label])
            self.assertEqual(frame.loc[frame.Panel == top, "Scenario"].unique().tolist(), ["ndc_noce", "baseline_noce"])
        self.assertEqual(frame.ClimatePathway.unique().tolist(), ["NDC", "NDC+LTT"])
        self.assertEqual(frame.CE.unique().tolist(), ["No CE"])
        self.assertTrue((frame.Series == frame.Sector).all())

    def test_missing_energy_numerator_is_preserved_or_explicitly_zero_filled(self):
        source = self.source.copy()
        selected = ((source.Scenario == "baseline_noce") & (source.Region == "EUE")
                    & (source.Variable == "Final Energy|Industry|Non-Metallic Minerals|Cement|Gases")
                    & (source.Unit == "EJ/yr"))
        source.loc[selected, "2050"] = np.nan
        for policy in ("preserve", "zero"):
            with self.subTest(policy=policy):
                data = self.data(source=source, config={**self.config, "missing_activity": policy})
                frame = data.energy()
                energy = self.metric(frame, "baseline_noce", "Cement", "EJ/yr").iloc[1]
                intensity = self.metric(frame, "baseline_noce", "Cement", "GJ/t").iloc[1]
                self.assertEqual(intensity.Denominator_Mt_yr, 20)
                if policy == "preserve":
                    self.assertTrue(np.isnan(energy.Value))
                    self.assertTrue(np.isnan(intensity.Value))
                    self.assertTrue(np.isnan(intensity.Numerator_EJ_yr))
                else:
                    self.assertAlmostEqual(energy.Value, 7.5)
                    self.assertAlmostEqual(intensity.Value, 375)
                self.assertEqual(data.results["baseline_noce"].coverage[0]["Treatment"], policy)

    def test_missing_production_route_makes_intensity_undefined_without_erasing_energy(self):
        source = self.source.copy()
        selected = ((source.Scenario == "baseline_noce") & (source.Region == "EUE")
                    & (source.Variable == "Production|Iron and Steel|Secondary") & (source.Unit == "Mt/yr"))
        source.loc[selected, "2050"] = np.nan
        for policy in ("preserve", "zero"):
            with self.subTest(policy=policy):
                data = self.data(source=source, config={**self.config, "missing_activity": policy})
                frame = data.energy()
                energy = self.metric(frame, "baseline_noce", "Iron and steel", "EJ/yr").iloc[1]
                intensity = self.metric(frame, "baseline_noce", "Iron and steel", "GJ/t").iloc[1]
                self.assertAlmostEqual(energy.Value, 12)
                self.assertAlmostEqual(intensity.Numerator_EJ_yr, 12)
                if policy == "preserve":
                    self.assertTrue(np.isnan(intensity.Denominator_Mt_yr))
                    self.assertTrue(np.isnan(intensity.Value))
                else:
                    self.assertEqual(intensity.Denominator_Mt_yr, 47)
                    self.assertAlmostEqual(intensity.Value, 12 / 47 * 1000)
                self.assertEqual(data.results["baseline_noce"].coverage[0]["Treatment"], policy)

    def test_absent_regional_fuel_rows_are_audited_without_dropping_regions(self):
        selected = ((self.source.Scenario == "baseline_noce") & (self.source.Region == "CHN")
                    & (self.source.Variable == "Final Energy|Industry|Non-Ferrous Metals|Aluminum|Hydrogen")
                    & (self.source.Unit == "EJ/yr"))
        for policy in ("preserve", "zero"):
            with self.subTest(policy=policy):
                data = self.data(source=self.source.loc[~selected],
                                 config={**self.config, "missing_activity": policy})
                frame = data.energy()
                energy = self.metric(frame, "baseline_noce", "Aluminium", "EJ/yr")
                intensity = self.metric(frame, "baseline_noce", "Aluminium", "GJ/t")
                if policy == "preserve":
                    self.assertTrue(energy.Value.isna().all())
                    self.assertTrue(intensity.Value.isna().all())
                else:
                    np.testing.assert_allclose(energy.Value, [4.2, 7.8])
                    np.testing.assert_allclose(intensity.Value, [420, 520])
                audit = data.results["baseline_noce"].coverage
                self.assertEqual(len(audit), 2)
                self.assertTrue(all(record["Issue"] == "absent row" and record["Treatment"] == policy
                                    for record in audit))

    def test_nonpositive_global_production_is_undefined_and_zero_energy_is_a_valid_intensity(self):
        for production in (0, -2):
            source = self.source.copy()
            selected = ((source.Scenario == "baseline_noce")
                        & (source.Variable == "Production|Non-Metallic Minerals|Cement") & (source.Unit == "Mt/yr"))
            source.loc[selected, "2050"] = production
            with self.subTest(production=production):
                frame = self.data(source=source).energy()
                intensity = self.metric(frame, "baseline_noce", "Cement", "GJ/t").iloc[1]
                self.assertEqual(intensity.Denominator_Mt_yr, production * 2)
                self.assertTrue(np.isnan(intensity.Value))
        source = self.source.copy()
        selected = ((source.Scenario == "baseline_noce") & (source.Unit == "EJ/yr")
                    & source.Variable.str.startswith("Final Energy|Industry|Non-Metallic Minerals|Cement|"))
        source.loc[selected, "2050"] = 0
        frame = self.data(source=source).energy()
        intensity = self.metric(frame, "baseline_noce", "Cement", "GJ/t").iloc[1]
        self.assertEqual(intensity.Value, 0)
        self.assertEqual(intensity.Denominator_Mt_yr, 20)

    def test_rendered_panels_contain_two_curves_with_their_own_raw_values(self):
        from omnia_results.consolidation_energy import EnergyFigures

        class Builder(EnergyFigures):
            def __init__(self, data, config):
                self.data, self.config = data, config

        spec = Builder(self.data(), self.config).energy()
        try:
            self.assertEqual(len(spec.figure.axes), 6)
            for ax, panel in zip(spec.figure.axes, "abcdef"):
                observations = spec.data[spec.data.Panel == panel]
                lines = [line for line in ax.lines if len(line.get_xdata()) == len(self.config["years"])]
                self.assertEqual(len(lines), 2)
                self.assertEqual([line.get_linestyle() for line in lines], ["-", "--"])
                self.assertEqual([line.get_marker() for line in lines], ["o", "s"])
                for line, scenario in zip(lines, ("ndc_noce", "baseline_noce")):
                    expected = observations[observations.Scenario == scenario].Value
                    self.assertEqual(line.get_label(), "NDC+LTT" if scenario == "baseline_noce" else "NDC")
                    np.testing.assert_array_equal(line.get_xdata(), self.config["years"])
                    np.testing.assert_allclose(line.get_ydata(), expected)
            self.assertEqual([label.get_text() for label in spec.figure.legends[0].get_texts()],
                             ["NDC", "NDC+LTT"])
        finally:
            plt.close(spec.figure)


if __name__ == "__main__":
    unittest.main()
