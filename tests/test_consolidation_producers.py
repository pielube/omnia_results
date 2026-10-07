"""Independent examples for a shared producer cohort across climate pathways."""

import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np
import pandas as pd

from omnia_results.consolidation_data import ConsolidationData
from omnia_results.consolidation import _save_individual_figure
from omnia_results.consolidation_producers import ProducerFigures
from omnia_results.metrics import SECTORS


class ProducerBuilder(ProducerFigures):
    def __init__(self, data, config):
        self.data, self.config = data, config


class ConsolidationProducerTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "input": "raw_producers.csv", "scenarios": ["ndc_noce", "baseline_noce"],
            "regions": ["EUE", "EUW", "CHN", "IND", "AFE"],
            "years": [2019, 2030, 2050], "missing_activity": "preserve",
            "producer_groups": {"Europe": ["EUE", "EUW"], "China": ["CHN"],
                                "India": ["IND"], "Africa": ["AFE"]},
            "ranking_year": 2050, "price_weight_year": 2019, "top_n": 2,
            "producer_axis_scale": "log",
        }
        # Values are final output, in configured region order. For metals they
        # will be split into primary and secondary raw observations below.
        self.outputs = {
            "baseline_noce": {
                "cement": [(10, 14, 20), (5, 10, 30), (60, 50, 40), (100, 70, 30), (25, 15, 10)],
                "steel": [(11, 22, 50), (9, 18, 40), (25, 35, 60), (40, 60, 70), (100, 80, 20)],
                "aluminium": [(3, 4, 12), (7, 6, 8), (8, 20, 30), (6, 25, 40), (50, 40, 10)],
            },
            "ndc_noce": {
                "cement": [(8, 10, 9), (7, 8, 11), (55, 20, 10), (90, 110, 200), (20, 25, 15)],
                "steel": [(10, 10, 12), (8, 9, 8), (35, 100, 200), (35, 30, 10), (70, 80, 150)],
                "aluminium": [(6, 50, 200), (7, 40, 100), (9, 8, 7), (10, 9, 8), (40, 60, 200)],
            },
        }
        self.cohorts = {"Cement": ["Europe", "China"], "Iron and steel": ["Europe", "India"],
                        "Aluminium": ["India", "China"]}
        rows = []
        for scenario, sectors in self.outputs.items():
            for sector in SECTORS:
                for region, totals in zip(self.config["regions"], sectors[sector.key]):
                    parent = f"Production|{sector.path}"
                    routes = [(parent, totals)] if sector.key == "cement" else [
                        (parent + "|Primary", [value - 3 for value in totals]),
                        (parent + "|Secondary", [3, 3, 3]),
                    ]
                    for variable, values in routes:
                        rows.append([scenario, region, variable, "Mt/yr", *values])
                        rows.append([scenario, region, variable, "kt/yr", 99999, 99999, 99999])
                        rows.append([scenario, region, variable + "|Detail", "Mt/yr", 99999, 99999, 99999])
                    if sector.key == "cement":
                        rows.append([scenario, region, "Production|Non-Metallic Minerals|Cement Clinker",
                                     "Mt/yr", 99999, 99999, 99999])
        self.source = pd.DataFrame(rows, columns=["Scenario", "Region", "Variable", "Unit",
                                                 "2019", "2030", "2050"])

    def data(self, *, source=None, config=None):
        source = self.source if source is None else source
        config = self.config if config is None else config
        with patch("omnia_results.metrics.pd.read_csv", return_value=source.copy()) as read:
            data = ConsolidationData(config, Path("."))
        self.assertTrue(all(Path(call.args[0]).name == "raw_producers.csv"
                            for call in read.call_args_list))
        return data

    @staticmethod
    def plotted(data, scenario, sector):
        return data[(data.Scenario == scenario) & (data.Sector == sector) & (data.Role == "plotted")]

    def test_both_pathways_keep_the_ndc_ltt_cohort_and_order_when_ndc_leaders_differ(self):
        data = self.data()
        frame, rankings = data.leading_producers(), data.producer_rankings()
        for sector, cohort in self.cohorts.items():
            for scenario in self.config["scenarios"]:
                with self.subTest(sector=sector, scenario=scenario):
                    selected = self.plotted(frame, scenario, sector)
                    self.assertEqual(selected.Group.unique().tolist(), cohort + ["Other model regions"])
                    self.assertEqual(selected.DisplayOrder.unique().tolist(), [1, 2, 3])
                    ranked = rankings[(rankings.Scenario == scenario) & (rankings.Sector == sector)]
                    self.assertEqual(ranked.Group.tolist(), cohort)
                    self.assertEqual(ranked.Rank.tolist(), [1, 2])
                    self.assertEqual(ranked.RankingScenario.unique().tolist(), ["baseline_noce"])
        # NDC-only leaders must stay in the remainder, rather than replacing a
        # smaller producer in the fixed comparison cohort.
        ndc_cement = self.plotted(frame, "ndc_noce", "Cement")
        self.assertNotIn("India", ndc_cement.Group.tolist())
        self.assertEqual(ndc_cement.loc[ndc_cement.Group == "Other model regions", "Value"].tolist(), [110, 215])
        ndc_steel = self.plotted(frame, "ndc_noce", "Iron and steel")
        self.assertNotIn("China", ndc_steel.Group.tolist())
        self.assertEqual(ndc_steel.loc[ndc_steel.Group == "Other model regions", "Value"].tolist(), [105, 350])
        ndc_aluminium = self.plotted(frame, "ndc_noce", "Aluminium")
        self.assertNotIn("Europe", ndc_aluminium.Group.tolist())
        self.assertEqual(ndc_aluminium.loc[ndc_aluminium.Group == "Other model regions", "Value"].tolist(), [53, 500])

    def test_endpoints_use_the_first_configured_year_and_ranking_year(self):
        config = {**self.config, "years": [2030, 2050], "price_weight_year": 2030}
        frame = self.data(config=config).leading_producers()
        self.assertEqual(frame.loc[frame.Role == "plotted", "Year"].unique().tolist(), [2030, 2050])
        selected = self.plotted(frame, "baseline_noce", "Cement")
        self.assertEqual(selected.loc[selected.Group == "Europe", "Value"].tolist(), [24, 50])

    def test_merged_europe_and_metal_totals_use_exact_routes_and_source_units(self):
        data = self.data()
        frame = data.leading_producers()
        expected = {("baseline_noce", "Cement"): [15, 50], ("ndc_noce", "Cement"): [15, 20],
                    ("baseline_noce", "Iron and steel"): [20, 90],
                    ("ndc_noce", "Iron and steel"): [18, 20]}
        for (scenario, sector), values in expected.items():
            table = self.plotted(frame, scenario, sector)
            rows = table[table.Group == "Europe"]
            self.assertEqual(rows.Value.tolist(), values)
            self.assertTrue(all(json.loads(regions) == ["EUE", "EUW"] for regions in rows.SourceRegions))
        selected = frame[(frame.Role == "plotted") & (frame.Sector == "Iron and steel")]
        self.assertTrue(all(json.loads(variables) == ["Production|Iron and Steel|Primary",
                                                    "Production|Iron and Steel|Secondary"]
                            for variables in selected.SourceVariables))
        self.assertEqual(data.producer_membership().Region.tolist(), self.config["regions"])
        # Huge child-variable, wrong-unit and clinker distractors must have no effect.
        exact = self.source[(self.source.Unit == "Mt/yr")
                            & ~self.source.Variable.str.endswith("|Detail")
                            & ~self.source.Variable.str.endswith("Cement Clinker")]
        pd.testing.assert_frame_equal(frame, self.data(source=exact).leading_producers())

    def test_remainder_and_selected_groups_close_global_output_in_both_pathways(self):
        frame = self.data().leading_producers()
        for sector in SECTORS:
            for scenario in self.config["scenarios"]:
                table = self.plotted(frame, scenario, sector.label)
                actual = table.groupby("Year").Value.sum(min_count=3)
                expected = np.asarray(self.outputs[scenario][sector.key]).sum(axis=0)[[0, 2]]
                with self.subTest(sector=sector.key, scenario=scenario):
                    np.testing.assert_array_equal(actual.to_numpy(), expected)
                    remainder = table[table.Group == "Other model regions"]
                    self.assertTrue(remainder.Rank.isna().all())
                    regions = json.loads(remainder.SourceRegions.iloc[0])
                    selected_regions = set(region for group in self.cohorts[sector.label]
                                           for region in self.config["producer_groups"][group])
                    self.assertEqual(set(regions), set(self.config["regions"]) - selected_regions)

    def test_context_coverage_retains_all_years_and_the_same_fixed_cohort(self):
        frame = self.data().leading_producers()
        selected_totals = {
            ("baseline_noce", "cement"): [75, 74, 90], ("ndc_noce", "cement"): [70, 38, 30],
            ("baseline_noce", "steel"): [60, 100, 160], ("ndc_noce", "steel"): [53, 49, 30],
            ("baseline_noce", "aluminium"): [14, 45, 70], ("ndc_noce", "aluminium"): [19, 17, 15],
        }
        for sector in SECTORS:
            for scenario in self.config["scenarios"]:
                context = frame[(frame.Scenario == scenario) & (frame.Sector == sector.label)
                                & (frame.Role == "context")]
                denominator = np.asarray(self.outputs[scenario][sector.key]).sum(axis=0)
                self.assertEqual(context.Year.tolist(), self.config["years"])
                self.assertEqual(context.Unit.unique().tolist(), ["%"])
                np.testing.assert_allclose(context.Value, np.asarray(selected_totals[scenario, sector.key])
                                           / denominator * 100)
        # In 2019 the baseline leaders were India and China; fixed Europe/China
        # instead correctly yield 37.5% of cement, not 80%.
        context = frame[(frame.Scenario == "baseline_noce") & (frame.Sector == "Cement")
                        & (frame.Role == "context") & (frame.Year == 2019)]
        self.assertEqual(context.Value.iloc[0], 37.5)

    def test_partition_rejects_duplicate_missing_and_unexpected_regions(self):
        invalid = [
            {**self.config["producer_groups"], "China": ["CHN", "EUE"]},
            {**self.config["producer_groups"], "Europe": ["EUE", "EUE", "EUW"]},
            {label: members for label, members in self.config["producer_groups"].items() if label != "India"},
            {**self.config["producer_groups"], "Outside": ["WORLD"]},
        ]
        for groups in invalid:
            with self.subTest(groups=groups), self.assertRaisesRegex(ValueError, "partition.*exactly once"):
                self.data(config={**self.config, "producer_groups": groups}).leading_producers()

    def test_missing_selected_or_remainder_route_preserves_unknown_totals_and_audits_zero_policy(self):
        for region, group, baseline_total, zero_total, selected_sum in (
                ("EUW", "Europe", 20, 17, 27),
                ("CHN", "Other model regions", 350, 347, 30)):
            source = self.source.copy()
            selected = ((source.Scenario == "ndc_noce") & (source.Region == region)
                        & (source.Variable == "Production|Iron and Steel|Secondary") & (source.Unit == "Mt/yr"))
            source.loc[selected, "2050"] = np.nan
            for treatment in ("preserve", "zero"):
                with self.subTest(region=region, treatment=treatment):
                    data = self.data(source=source, config={**self.config, "missing_activity": treatment})
                    frame = data.leading_producers()
                    table = self.plotted(frame, "ndc_noce", "Iron and steel")
                    value = table.loc[(table.Group == group) & (table.Year == 2050), "Value"].iloc[0]
                    coverage = frame.loc[(frame.Scenario == "ndc_noce") & (frame.Sector == "Iron and steel")
                                         & (frame.Role == "context") & (frame.Year == 2050), "Value"].iloc[0]
                    if treatment == "preserve":
                        self.assertTrue(np.isnan(value))
                        self.assertTrue(np.isnan(coverage))
                    else:
                        self.assertEqual(value, zero_total)
                        self.assertAlmostEqual(coverage, selected_sum / 377 * 100)
                        self.assertNotEqual(value, baseline_total)
                    self.assertEqual(data.results["ndc_noce"].coverage[0]["Treatment"], treatment)
                    self.assertEqual(data.results["ndc_noce"].coverage[0]["Issue"], "blank cell")
                    self.assertEqual(self.plotted(frame, "baseline_noce", "Iron and steel")
                                     .loc[lambda rows: (rows.Group == "Europe") & (rows.Year == 2050), "Value"].iloc[0], 90)

    def test_rendered_markers_compare_two_offset_pathways_for_each_fixed_group(self):
        spec = ProducerBuilder(self.data(), self.config).leading_producers()
        try:
            for ax, sector in zip(spec.figure.axes, SECTORS):
                self.assertEqual(ax.get_xscale(), "log")
                labels = [label.get_text() for label in ax.get_yticklabels()]
                self.assertEqual(labels, self.cohorts[sector.label] + ["Other model regions"])
                markers = np.vstack([collection.get_offsets() for collection in ax.collections
                                     if isinstance(collection, PathCollection)])
                for order, group in enumerate(labels):
                    positions = markers[np.rint(markers[:, 1]).astype(int) == order]
                    levels = sorted(set(positions[:, 1]))
                    self.assertEqual(len(levels), 2)
                    for level, scenario in zip(levels, ("ndc_noce", "baseline_noce")):
                        expected = self.plotted(spec.data, scenario, sector.label)
                        expected = expected.loc[expected.Group == group, "Value"].to_numpy()
                        np.testing.assert_allclose(np.sort(positions[positions[:, 1] == level, 0]),
                                                   np.sort(expected))
        finally:
            plt.close(spec.figure)

    def test_log_axes_reject_nonpositive_endpoints_and_linear_axes_preserve_them(self):
        for value in (0, -2):
            source = self.source.copy()
            selected = ((source.Scenario == "ndc_noce") & (source.Region == "CHN")
                        & (source.Variable == "Production|Non-Metallic Minerals|Cement")
                        & (source.Unit == "Mt/yr"))
            source.loc[selected, "2019"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive endpoints"):
                ProducerBuilder(self.data(source=source), self.config).leading_producers()
            config = {**self.config, "producer_axis_scale": "linear"}
            spec = ProducerBuilder(self.data(source=source), config).leading_producers()
            try:
                ax = spec.figure.axes[0]
                self.assertEqual(ax.get_xscale(), "linear")
                markers = np.vstack([collection.get_offsets() for collection in ax.collections
                                     if isinstance(collection, PathCollection)])
                self.assertIn(value, markers[:, 0].tolist())
                self.assertLessEqual(ax.get_xlim()[0], value)
            finally:
                plt.close(spec.figure)

    def test_reviewed_exports_are_preserved_only_when_fresh_picture_and_data_both_match(self):
        fig, ax = plt.subplots(figsize=(2, 2))
        line, = ax.plot([2019, 2050], [10, 20])
        spec = SimpleNamespace(slug="reviewed_production", figure=fig,
                               data=pd.DataFrame({"Year": [2019, 2050], "Value": [10, 20]}))
        config = {"formats": ["png", "pdf"], "dpi": 72,
                  "preserve_individual_figures": [spec.slug]}
        try:
            with BytesIO() as image:
                fig.savefig(image, format="png", dpi=72)
                archived = {f"{spec.slug}.png": image.getvalue(),
                            f"{spec.slug}.csv": spec.data.to_csv(index=False).encode("utf-8"),
                            f"{spec.slug}.pdf": b"reviewed PDF must be retained"}
            # Simulate existing reviewed files in memory: the guard may read
            # them but every new render must stay in a buffer, never a file.
            with patch.object(Path, "is_file", return_value=True), \
                    patch.object(Path, "read_bytes", lambda path: archived[path.name]), \
                    patch.object(fig, "savefig", wraps=fig.savefig) as save:
                self.assertTrue(_save_individual_figure(spec, Path("reviewed"), config))
                spec.data.loc[1, "Value"] = 21
                with self.assertRaisesRegex(ValueError, "differs from the current build"):
                    _save_individual_figure(spec, Path("reviewed"), config)
                spec.data.loc[1, "Value"] = 20
                line.set_ydata([10, 21])
                with self.assertRaisesRegex(ValueError, "differs from the current build"):
                    _save_individual_figure(spec, Path("reviewed"), config)
                self.assertTrue(all(isinstance(call.args[0], BytesIO) for call in save.call_args_list))
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
