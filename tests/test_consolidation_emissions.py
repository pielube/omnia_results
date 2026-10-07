"""Asymmetric source examples for emissions, industry shares and capture."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from omnia_results.consolidation_data import ConsolidationData
from omnia_results.metrics import SECTORS


EMISSIONS = "Sector GHG emissions"
SHARE = "Share of industrial GHG emissions"
INTENSITY = "Derived emissions intensity"
CAPTURE = "Carbon capture"


class ConsolidationEmissionsTests(unittest.TestCase):
    def setUp(self):
        self.config = {"input": "raw_emissions.csv", "scenarios": ["ndc_noce", "baseline_noce"],
                       "regions": ["EUE", "CHN"], "years": [2019, 2050],
                       "missing_activity": "preserve"}
        self.ghg = {
            "baseline_noce": {"cement": [(4, 6), (16, 14)], "steel": [(2, 4), (10, 11)],
                              "aluminium": [(3, 1), (2, -3)]},
            "ndc_noce": {"cement": [(10, 15), (20, 25)], "steel": [(8, 10), (4, 8)],
                         "aluminium": [(3, 6), (1, 2)]},
        }
        self.industry = {"baseline_noce": [(40, 60), (160, 140)],
                         "ndc_noce": [(10, 20), (50, 60)]}
        self.capture = {
            "baseline_noce": {"cement": [(1, 2), (3, 4)], "steel": [(2, 4), (4, 6)],
                              "aluminium": [(5, 6), (1, 2)]},
            "ndc_noce": {"cement": [(0, 1), (1, 4)], "steel": [(2, 3), (4, 6)],
                         "aluminium": [(1, 2), (0, 1)]},
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

        def add(scenario, region, variable, unit, values, wrong_unit):
            rows.append([scenario, region, variable, unit, *values])
            rows.append([scenario, region, variable, wrong_unit, 99999, 99999])
            rows.append([scenario, region, variable + "|Detail", unit, 99999, 99999])

        for scenario in self.config["scenarios"]:
            for index, region in enumerate(self.config["regions"]):
                add(scenario, region, "Emissions|GHG|Industry", "MtCO2e/yr",
                    self.industry[scenario][index], "MtCO2e/Mt")
                for sector in SECTORS:
                    add(scenario, region, f"Emissions|GHG|Industry|{sector.path}", "MtCO2e/yr",
                        self.ghg[scenario][sector.key][index], "MtCO2e/Mt")
                    add(scenario, region, f"Carbon Capture|Industry|{sector.path}", "MtCO2/yr",
                        self.capture[scenario][sector.key][index], "MtCO2e/yr")
                    rows.append([scenario, region, f"Emissions Intensity|Industry|{sector.path}",
                                 "MtCO2e/Mt", 99999, 99999])
                    for route, regional_values in self.production[scenario][sector.key].items():
                        add(scenario, region, f"Production|{sector.path}{route}", "Mt/yr",
                            regional_values[index], "kt/yr")
                    if sector.key == "cement":
                        rows.append([scenario, region, "Production|Non-Metallic Minerals|Cement Clinker",
                                     "Mt/yr", 99999, 99999])
        self.source = pd.DataFrame(rows, columns=["Scenario", "Region", "Variable", "Unit", "2019", "2050"])

    def data(self, *, source=None, config=None):
        source = self.source if source is None else source
        config = self.config if config is None else config
        with patch("omnia_results.metrics.pd.read_csv", return_value=source.copy()) as read:
            data = ConsolidationData(config, Path("."))
        self.assertTrue(all(Path(call.args[0]).name == "raw_emissions.csv" for call in read.call_args_list))
        return data

    @staticmethod
    def metric(frame, scenario, sector, metric):
        return frame[(frame.Scenario == scenario) & (frame.Sector == sector) & (frame.Metric == metric)]

    def test_global_emissions_share_intensity_and_capture_use_each_scenarios_raw_values(self):
        frame = self.data().emissions()
        self.assertEqual(len(frame), 48)
        expected = {
            ("baseline_noce", "Cement"): ([20, 20], [200, 200], [10, 20], [4, 6]),
            ("baseline_noce", "Iron and steel"): ([12, 15], [200, 200], [30, 50], [6, 10]),
            ("baseline_noce", "Aluminium"): ([5, -2], [200, 200], [10, 15], [6, 8]),
            ("ndc_noce", "Cement"): ([30, 40], [60, 80], [10, 8], [1, 5]),
            ("ndc_noce", "Iron and steel"): ([12, 18], [60, 80], [12, 20], [6, 9]),
            ("ndc_noce", "Aluminium"): ([4, 8], [60, 80], [10, 20], [1, 3]),
        }
        for (scenario, sector), (ghg, industry, production, capture) in expected.items():
            with self.subTest(scenario=scenario, sector=sector):
                np.testing.assert_allclose(self.metric(frame, scenario, sector, EMISSIONS).Value,
                                           np.asarray(ghg) / 1000)
                np.testing.assert_allclose(self.metric(frame, scenario, sector, SHARE).Value,
                                           np.asarray(ghg) / industry * 100)
                np.testing.assert_allclose(self.metric(frame, scenario, sector, INTENSITY).Value,
                                           np.asarray(ghg) / production)
                np.testing.assert_allclose(self.metric(frame, scenario, sector, CAPTURE).Value, capture)
                intensity = self.metric(frame, scenario, sector, INTENSITY)
                np.testing.assert_allclose(intensity.Numerator, ghg)
                np.testing.assert_allclose(intensity.Denominator, production)
                self.assertEqual(intensity.ConversionFactor.tolist(), [1, 1])
                self.assertEqual(intensity.NumeratorUnit.unique().tolist(), ["MtCO2e/yr"])
                self.assertEqual(intensity.DenominatorUnit.unique().tolist(), ["Mt/yr"])
                emissions = self.metric(frame, scenario, sector, EMISSIONS)
                shares = self.metric(frame, scenario, sector, SHARE)
                captured = self.metric(frame, scenario, sector, CAPTURE)
                np.testing.assert_allclose(emissions.Numerator, ghg)
                np.testing.assert_allclose(shares.Denominator, industry)
                np.testing.assert_allclose(captured.Numerator, capture)
                self.assertEqual(emissions.ConversionFactor.tolist(), [.001, .001])
                self.assertEqual(shares.ConversionFactor.tolist(), [100, 100])
                self.assertEqual(captured.ConversionFactor.tolist(), [1, 1])
                self.assertEqual(shares.DenominatorUnit.unique().tolist(), ["MtCO2e/yr"])
                self.assertEqual(captured.NumeratorUnit.unique().tolist(), ["MtCO2/yr"])

    def test_share_uses_total_industry_and_matching_climate_and_intensity_is_ratio_of_sums(self):
        frame = self.data().emissions()
        baseline_share = self.metric(frame, "baseline_noce", "Cement", SHARE).Value.iloc[0]
        ndc_share = self.metric(frame, "ndc_noce", "Cement", SHARE).Value.iloc[0]
        self.assertEqual(baseline_share, 10)
        self.assertEqual(ndc_share, 50)
        self.assertNotAlmostEqual(baseline_share, 20 / (20 + 12 + 5) * 100)
        self.assertNotAlmostEqual(ndc_share, 30 / 200 * 100)
        baseline_intensity = self.metric(frame, "baseline_noce", "Cement", INTENSITY).Value.iloc[0]
        self.assertEqual(baseline_intensity, 2)
        self.assertNotAlmostEqual(baseline_intensity, (4 / 1 + 16 / 9) / 2)
        self.assertNotEqual(baseline_intensity, 99999)

    def test_capture_is_not_subtracted_again_and_signed_emissions_are_retained(self):
        frame = self.data().emissions()
        self.assertEqual(self.metric(frame, "baseline_noce", "Cement", EMISSIONS).Value.tolist(), [.02, .02])
        self.assertEqual(self.metric(frame, "baseline_noce", "Aluminium", EMISSIONS).Value.iloc[1], -.002)
        self.assertEqual(self.metric(frame, "baseline_noce", "Aluminium", SHARE).Value.iloc[1], -1)
        self.assertAlmostEqual(self.metric(frame, "baseline_noce", "Aluminium", INTENSITY).Value.iloc[1], -2 / 15)
        source = self.source.copy()
        selected = (source.Variable.str.startswith("Carbon Capture|") & (source.Unit == "MtCO2/yr"))
        source.loc[selected, ["2019", "2050"]] = 10000
        changed = self.data(source=source).emissions()
        pd.testing.assert_frame_equal(frame[frame.Metric != CAPTURE], changed[changed.Metric != CAPTURE])

    def test_exact_source_units_and_variables_exclude_details_reported_intensity_and_clinker(self):
        frame = self.data().emissions()
        exact = self.source[(self.source.Unit.isin(["MtCO2e/yr", "MtCO2/yr", "Mt/yr"]))
                            & ~self.source.Variable.str.endswith("|Detail")
                            & ~self.source.Variable.str.startswith("Emissions Intensity|")
                            & ~self.source.Variable.str.endswith("Cement Clinker")]
        # Wrong capture units use the valid GHG unit, so exclude them separately.
        exact = exact[~(exact.Variable.str.startswith("Carbon Capture|") & (exact.Unit == "MtCO2e/yr"))]
        pd.testing.assert_frame_equal(frame, self.data(source=exact).emissions())
        for sector in SECTORS:
            ghg = f"Emissions|GHG|Industry|{sector.path}"
            production = ([f"Production|{sector.path}"] if sector.key == "cement" else
                          [f"Production|{sector.path}|Primary", f"Production|{sector.path}|Secondary"])
            capture = f"Carbon Capture|Industry|{sector.path}"
            expected = {EMISSIONS: [ghg], SHARE: [ghg, "Emissions|GHG|Industry"],
                        INTENSITY: [ghg, *production], CAPTURE: [capture]}
            for metric, variables in expected.items():
                selected = frame[(frame.Sector == sector.label) & (frame.Metric == metric)]
                self.assertTrue(all(set(json.loads(value)) == set(variables) for value in selected.SourceVariables))
                self.assertTrue(all(json.loads(value) == ["EUE", "CHN"] for value in selected.SourceRegions))

    def test_panels_units_and_labels_keep_both_climate_pathways_for_all_sectors(self):
        frame = self.data().emissions()
        settings = {"a": (EMISSIONS, "GtCO2e/yr"), "b": (SHARE, "%"),
                    "c": (INTENSITY, "tCO2e/t"), "d": (CAPTURE, "MtCO2/yr")}
        self.assertEqual(sorted(frame.Panel.unique()), list(settings))
        for panel, (metric, unit) in settings.items():
            selected = frame[frame.Panel == panel]
            self.assertEqual(selected.Metric.unique().tolist(), [metric])
            self.assertEqual(selected.Unit.unique().tolist(), [unit])
            self.assertEqual(set(selected.Sector), {sector.label for sector in SECTORS})
            self.assertEqual(set(selected.Scenario), {"baseline_noce", "ndc_noce"})
        self.assertEqual(set(frame.ClimatePathway), {"NDC+LTT", "NDC"})
        self.assertEqual(frame.CE.unique().tolist(), ["No CE"])

    def test_missing_ghg_numerators_preserve_unknown_ratios_or_use_explicit_zero_policy(self):
        source = self.source.copy()
        selected = ((source.Scenario == "baseline_noce") & (source.Region == "EUE")
                    & (source.Variable == "Emissions|GHG|Industry|Non-Metallic Minerals|Cement")
                    & (source.Unit == "MtCO2e/yr"))
        source.loc[selected, "2050"] = np.nan
        for policy in ("preserve", "zero"):
            with self.subTest(policy=policy):
                data = self.data(source=source, config={**self.config, "missing_activity": policy})
                frame = data.emissions()
                if policy == "preserve":
                    for metric in (EMISSIONS, SHARE, INTENSITY):
                        self.assertTrue(np.isnan(self.metric(frame, "baseline_noce", "Cement", metric).Value.iloc[1]))
                else:
                    self.assertEqual(self.metric(frame, "baseline_noce", "Cement", EMISSIONS).Value.iloc[1], .014)
                    self.assertAlmostEqual(self.metric(frame, "baseline_noce", "Cement", SHARE).Value.iloc[1], 7)
                    self.assertEqual(self.metric(frame, "baseline_noce", "Cement", INTENSITY).Value.iloc[1], .7)
                self.assertEqual(self.metric(frame, "baseline_noce", "Cement", CAPTURE).Value.iloc[1], 6)
                self.assertEqual(data.results["baseline_noce"].coverage[0]["Treatment"], policy)

    def test_missing_industry_production_and_capture_only_affect_dependent_metrics(self):
        cases = [
            ("ndc_noce", "CHN", "Emissions|GHG|Industry", "MtCO2e/yr", "Cement", SHARE, 200),
            ("baseline_noce", "EUE", "Production|Iron and Steel|Secondary", "Mt/yr", "Iron and steel", INTENSITY, 15 / 47),
            ("baseline_noce", "EUE", "Carbon Capture|Industry|Non-Ferrous Metals|Aluminum", "MtCO2/yr", "Aluminium", CAPTURE, 2),
        ]
        original = self.data().emissions()
        for scenario, region, variable, unit, sector, metric, zero_value in cases:
            source = self.source.copy()
            selected = ((source.Scenario == scenario) & (source.Region == region)
                        & (source.Variable == variable) & (source.Unit == unit))
            source.loc[selected, "2050"] = np.nan
            for policy in ("preserve", "zero"):
                with self.subTest(variable=variable, policy=policy):
                    data = self.data(source=source, config={**self.config, "missing_activity": policy})
                    frame = data.emissions()
                    value = self.metric(frame, scenario, sector, metric).Value.iloc[1]
                    if policy == "preserve":
                        self.assertTrue(np.isnan(value))
                    else:
                        self.assertAlmostEqual(value, zero_value)
                    pd.testing.assert_series_equal(self.metric(frame, scenario, sector, EMISSIONS).Value,
                                                   self.metric(original, scenario, sector, EMISSIONS).Value)
                    self.assertEqual(data.results[scenario].coverage[0]["Treatment"], policy)

    def test_nonpositive_industry_or_production_denominators_are_undefined(self):
        for variable, unit, metric in (
                ("Emissions|GHG|Industry", "MtCO2e/yr", SHARE),
                ("Production|Non-Metallic Minerals|Cement", "Mt/yr", INTENSITY)):
            for value in (0, -2):
                source = self.source.copy()
                selected = ((source.Scenario == "baseline_noce") & (source.Variable == variable)
                            & (source.Unit == unit))
                source.loc[selected, "2050"] = value
                with self.subTest(variable=variable, value=value):
                    frame = self.data(source=source).emissions()
                    row = self.metric(frame, "baseline_noce", "Cement", metric).iloc[1]
                    self.assertEqual(row.Denominator, value * 2)
                    self.assertTrue(np.isnan(row.Value))
                    self.assertEqual(self.metric(frame, "baseline_noce", "Cement", EMISSIONS).Value.iloc[1], .02)

    def test_rendered_panels_contain_six_curves_with_their_own_sector_and_climate_values(self):
        from omnia_results.consolidation_emissions import EmissionsFigures

        class Builder(EmissionsFigures):
            def __init__(self, data, config):
                self.data, self.config = data, config

        spec = Builder(self.data(), self.config).emissions()
        try:
            self.assertEqual(len(spec.figure.axes), 4)
            for ax, panel in zip(spec.figure.axes, "abcd"):
                curves = [line for line in ax.lines if line.get_label() != "_nolegend_"
                          and not line.get_label().startswith("_child")]
                self.assertEqual(len(curves), 6)
                self.assertEqual([line.get_label() for line in curves],
                                 [f"{sector.label} | {climate}" for sector in SECTORS
                                  for climate in ("NDC", "NDC+LTT")])
                self.assertEqual([line.get_linestyle() for line in curves], ["-", "--"] * 3)
                self.assertEqual([line.get_marker() for line in curves], ["o", "s"] * 3)
                expected = spec.data[spec.data.Panel == panel]
                actual = {line.get_label(): line for line in curves}
                expected_labels = {f"{sector.label} | {climate}" for sector in SECTORS
                                   for climate in ("NDC", "NDC+LTT")}
                self.assertEqual(set(actual), expected_labels)
                for (scenario, sector), rows in expected.groupby(["Scenario", "Sector"], sort=False):
                    climate = "NDC+LTT" if scenario == "baseline_noce" else "NDC"
                    np.testing.assert_allclose(actual[f"{sector} | {climate}"].get_ydata(), rows.Value)
                for line in curves:
                    np.testing.assert_array_equal(line.get_xdata(), self.config["years"])
            self.assertEqual([label.get_text() for label in spec.figure.legends[1].get_texts()],
                             ["NDC", "NDC+LTT"])
        finally:
            plt.close(spec.figure)


if __name__ == "__main__":
    unittest.main()
