"""Opt-in single-source production figures retain explicit raw scenario selection."""

from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np
import pandas as pd

from omnia_results.consolidation import ConsolidationBuilder
from omnia_results.consolidation_data import ConsolidationData
from omnia_results.consolidation_producers import ProducerFigures
from omnia_results.metrics import SECTORS


class ProducerBuilder(ProducerFigures):
    def __init__(self, data, config):
        self.data, self.config = data, config


class SimplifiedProductionTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "input": "raw_simplified.csv", "scenarios": ["baseline_noce", "ndc_noce"],
            "regions": ["EUE", "EUW", "CHN", "IND"], "years": [2019, 2030, 2050],
            "missing_activity": "preserve", "ranking_year": 2050, "price_weight_year": 2019,
            "top_n": 2, "producer_axis_scale": "log",
            "producer_groups": {"Europe": ["EUE", "EUW"], "China": ["CHN"], "India": ["IND"]},
        }
        self.simple = {**self.config, "simplified_production_scenario": "ndc_noce"}
        self.outputs = {
            "baseline_noce": {
                "cement": ([10, 10, 20, 5], [30, 30, 40, 20]),
                "steel": ([10, 10, 10, 10], [40, 40, 40, 60]),
                "aluminium": ([1, 2, 4, 8], [5, 10, 20, 40]),
            },
            "ndc_noce": {
                "cement": ([2, 3, 20, 10], [5, 10, 200, 100]),
                "steel": ([1, 2, 30, 20], [4, 6, 300, 200]),
                "aluminium": ([1, 1, 5, 3], [8, 12, 100, 50]),
            },
        }
        rows = []
        for scenario, sectors in self.outputs.items():
            for sector in SECTORS:
                start, end = sectors[sector.key]
                for index, region in enumerate(self.config["regions"]):
                    total = np.asarray([start[index], (start[index] + end[index]) / 2, end[index]])
                    parent = f"Production|{sector.path}"
                    routes = ([(parent, total), ("Production|Non-Metallic Minerals|Cement Clinker", total * .6)]
                              if sector.key == "cement" else
                              [(parent + "|Primary", total * .75), (parent + "|Secondary", total * .25)])
                    for variable, values in routes:
                        rows.append([scenario, region, variable, "Mt/yr", *values])
        self.source = pd.DataFrame(rows, columns=["Scenario", "Region", "Variable", "Unit", "2019", "2030", "2050"])

    def data(self, config=None):
        config = self.simple if config is None else config
        with patch("omnia_results.metrics.pd.read_csv", return_value=self.source.copy()):
            return ConsolidationData(config, Path("."))

    @staticmethod
    def visible_text(figure):
        texts = [text.get_text() for text in figure.texts]
        for ax in figure.axes:
            texts.extend([ax.get_title(loc="left"), ax.get_title(), ax.get_xlabel(), ax.get_ylabel()])
            texts.extend(text.get_text() for text in ax.texts)
            if ax.get_legend() is not None:
                texts.extend(text.get_text() for text in ax.get_legend().get_texts())
        for legend in figure.legends:
            texts.extend(text.get_text() for text in legend.get_texts())
        return " ".join(texts)

    def test_single_production_is_selected_scenario_output_without_cross_climate_averaging(self):
        frame = self.data().production()
        self.assertEqual(len(frame), 24)
        self.assertEqual(frame.Scenario.unique().tolist(), ["ndc_noce"])
        self.assertEqual(frame.Panel.unique().tolist(), list("abc"))
        expected = {("Cement", "Cement"): [35, 175, 315],
                    ("Cement", "Clinker"): [21, 105, 189],
                    ("Iron and steel", "Total"): [53, 281.5, 510],
                    ("Aluminium", "Total"): [10, 90, 170]}
        for (sector, series), values in expected.items():
            rows = frame[(frame.Sector == sector) & (frame.Series == series)]
            np.testing.assert_allclose(rows.Value, values)

    def test_single_producer_cohort_ranks_selected_scenario_and_retains_context_years(self):
        data = self.data()
        frame = data.leading_producers()
        self.assertEqual(len(frame), 27)
        self.assertEqual(frame.Scenario.unique().tolist(), ["ndc_noce"])
        self.assertEqual(frame.RankingScenario.unique().tolist(), ["ndc_noce"])
        ranks = data.producer_rankings()
        self.assertEqual(len(ranks), 6)
        self.assertEqual(ranks.Scenario.unique().tolist(), ["ndc_noce"])
        self.assertEqual(ranks.RankingScenario.unique().tolist(), ["ndc_noce"])
        for sector in SECTORS:
            plotted = frame[(frame.Sector == sector.label) & (frame.Role == "plotted")]
            self.assertEqual(plotted.Group.unique().tolist(), ["China", "India", "Other model regions"])
            self.assertEqual(plotted.Year.unique().tolist(), [2019, 2050])
            expected = np.asarray(self.outputs["ndc_noce"][sector.key]).sum(axis=1)
            np.testing.assert_allclose(plotted.groupby("Year").Value.sum(), expected)
            context = frame[(frame.Sector == sector.label) & (frame.Role == "context")]
            self.assertEqual(context.Year.tolist(), self.config["years"])
        cement = frame[(frame.Sector == "Cement") & (frame.Role == "context")]
        np.testing.assert_allclose(cement.Value, np.asarray([30 / 35, 165 / 175, 300 / 315]) * 100)

    def test_simplified_calculations_do_not_access_the_omitted_scenarios_values(self):
        data = self.data()
        omitted = data.results["baseline_noce"]
        with patch.object(omitted, "production", side_effect=AssertionError("Omitted production was read")), \
                patch.object(omitted, "values", side_effect=AssertionError("Omitted regional values were read")):
            self.assertEqual(data.production().Scenario.unique().tolist(), ["ndc_noce"])
            self.assertEqual(data.leading_producers().Scenario.unique().tolist(), ["ndc_noce"])
            self.assertEqual(data.producer_rankings().Scenario.unique().tolist(), ["ndc_noce"])
        self.assertEqual(omitted.coverage, [])

    def test_omitted_scenario_need_not_be_configured_and_selection_is_explicit(self):
        selected_only = {**self.simple, "scenarios": ["ndc_noce"]}
        data = self.data(selected_only)
        self.assertEqual(len(data.production()), 24)
        self.assertEqual(len(data.leading_producers()), 27)
        baseline = self.data({**self.config, "simplified_production_scenario": "baseline_noce"})
        self.assertEqual(baseline.production().Scenario.unique().tolist(), ["baseline_noce"])
        cement = baseline.producer_rankings().loc[lambda frame: frame.Sector == "Cement"]
        self.assertEqual(cement.Group.tolist(), ["Europe", "China"])
        self.assertEqual(cement.RankingScenario.unique().tolist(), ["baseline_noce"])
        with self.assertRaises(ValueError):
            self.data({**self.config, "simplified_production_scenario": "ndc_medce_eu"}).production()

    def test_default_mode_keeps_two_pathways_and_the_original_shared_baseline_ranking(self):
        data = self.data(self.config)
        production = data.production()
        self.assertEqual(len(production), 48)
        self.assertEqual(production.Scenario.unique().tolist(), ["ndc_noce", "baseline_noce"])
        self.assertEqual(production.Panel.unique().tolist(), list("abcdef"))
        producers, ranks = data.leading_producers(), data.producer_rankings()
        self.assertEqual(len(producers), 54)
        self.assertEqual(len(ranks), 12)
        self.assertEqual(producers.RankingScenario.unique().tolist(), ["baseline_noce"])
        for scenario in self.config["scenarios"]:
            cement = producers[(producers.Scenario == scenario) & (producers.Sector == "Cement")
                               & (producers.Role == "plotted")]
            self.assertEqual(cement.Group.unique().tolist(), ["Europe", "China", "Other model regions"])

    def test_single_production_artists_use_one_row_and_no_visible_climate_labels(self):
        spec = ConsolidationBuilder(self.data(), self.simple).production()
        try:
            self.assertEqual(len(spec.figure.axes), 3)
            for ax, sector, panel in zip(spec.figure.axes, SECTORS, "abc"):
                self.assertEqual(ax.get_title(loc="left"), sector.label)
                selected = spec.data[spec.data.Panel == panel]
                actual = {line.get_label(): line for line in ax.lines}
                self.assertEqual(set(actual), set(selected.Series))
                for series, rows in selected.groupby("Series", sort=False):
                    np.testing.assert_array_equal(actual[series].get_xdata(), self.config["years"])
                    np.testing.assert_allclose(actual[series].get_ydata(), rows.Value)
            text = self.visible_text(spec.figure)
            self.assertNotIn("NDC", text)
            self.assertNotIn("baseline", text.lower())
        finally:
            plt.close(spec.figure)

    def test_single_producer_artists_have_one_centred_endpoint_pair_per_group(self):
        spec = ProducerBuilder(self.data(), self.simple).leading_producers()
        try:
            self.assertEqual(len(spec.figure.axes), 3)
            for ax, sector in zip(spec.figure.axes, SECTORS):
                selected = spec.data[(spec.data.Sector == sector.label) & (spec.data.Role == "plotted")]
                labels = [label.get_text() for label in ax.get_yticklabels()]
                self.assertEqual(labels, ["China", "India", "Other model regions"])
                markers = np.vstack([collection.get_offsets() for collection in ax.collections
                                     if isinstance(collection, PathCollection)])
                self.assertEqual(len(markers), 6)
                for index, group in enumerate(labels):
                    points = markers[np.isclose(markers[:, 1], ax.get_yticks()[index])]
                    self.assertEqual(len(points), 2)
                    expected = selected[selected.Group == group].Value.to_numpy()
                    np.testing.assert_allclose(np.sort(points[:, 0]), np.sort(expected))
            text = self.visible_text(spec.figure)
            self.assertNotIn("NDC", text)
            self.assertNotIn("baseline", text.lower())
            self.assertIn("2019", text)
            self.assertIn("2050", text)
        finally:
            plt.close(spec.figure)


if __name__ == "__main__":
    unittest.main()
