"""Route-response rendering retains observed zeroes and exposes missing references."""

import unittest

import matplotlib.pyplot as plt
from matplotlib.text import Annotation
import numpy as np

from omnia_results.europe_comparison_data import GCAM_PRODUCTION, OMNIA_REGIONS
from omnia_results.europe_relative import (COMPARISON_STYLE, EuropeRelativeBuilder,
                                           OFFSETS, POSITIONS)
from omnia_results.europe_relative_routes_data import PRODUCTION_GROUPS, ROUTE_MATERIAL_METRIC
from omnia_results.report import check_figure_bounds
import test_europe_relative_routes_data as fixtures


class EuropeRelativeRouteFigureTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.EuropeRelativeRouteDataTests()
        self.fixture.setUp()
        self.data = self.fixture.data()
        self.builder = EuropeRelativeBuilder(self.data, self.fixture.config)

    def tearDown(self):
        plt.close("all")

    def test_eight_panels_preserve_all_data_rows_and_share_one_percentage_scale(self):
        with plt.rc_context(COMPARISON_STYLE):
            spec = self.builder.material_response_by_steel_route()
            spec.figure.canvas.draw()
            check_figure_bounds(spec.figure, spec.slug)
        self.assertEqual(spec.slug, "fig04_material_production_by_steel_route")
        self.assertEqual(spec.panel_count, 8)
        self.assertEqual(len(spec.figure.axes), 8)
        np.testing.assert_allclose(spec.figure.get_size_inches() * 25.4, [240, 175])
        self.assertEqual(len(spec.data), 32)
        self.assertTrue(spec.data.Plotted.all())
        self.assertEqual(set(spec.data.Metric), {ROUTE_MATERIAL_METRIC})
        self.assertEqual(set(spec.data.Role), {"net"})
        self.assertEqual(set(spec.data.Panel), set("abcdefgh"))
        self.assertEqual(set(spec.data.ProductionGroup), set(PRODUCTION_GROUPS))
        self.assertEqual(len({ax.get_ylim() for ax in spec.figure.axes}), 1)
        for group, ax in zip(PRODUCTION_GROUPS * 2, spec.figure.axes):
            self.assertIn(group, ax.get_title(loc="left"))
            self.assertEqual(len(ax.patches), 4)
            self.assertEqual([len(collection.get_offsets()) for collection in ax.collections], [2, 2])

    def test_route_bars_use_own_route_percentage_changes_and_correct_provenance(self):
        with plt.rc_context(COMPARISON_STYLE):
            spec = self.builder.material_response_by_steel_route()
        for climate_row in (0, 1):
            primary_ax = spec.figure.axes[climate_row * 4 + 1]
            secondary_ax = spec.figure.axes[climate_row * 4 + 2]
            np.testing.assert_allclose([patch.get_height() for patch in primary_ax.patches],
                                       [-25, -50, -25, -50])
            np.testing.assert_allclose([patch.get_height() for patch in secondary_ax.patches],
                                       [75, 150, 75, 150])
        primary = fixtures.EuropeRelativeRouteDataTests.select(spec.data, "Primary steel")
        secondary = fixtures.EuropeRelativeRouteDataTests.select(spec.data, "Secondary steel")
        self.assertEqual((primary.Denominator, secondary.Denominator), (400, 100))
        self.assertEqual((primary.Value, secondary.Value), (-25, 75))
        self.assertIn("route-specific no-CE", spec.caption)
        self.assertIn("not additive", spec.caption)
        notes = " ".join(text.get_text() for text in spec.figure.texts)
        self.assertIn("each use their own route reference", notes)

    def test_cement_and_aluminium_draw_the_original_total_response(self):
        original = self.data.material_response()
        with plt.rc_context(COMPARISON_STYLE):
            spec = self.builder.material_response_by_steel_route()
        for row_index, climate in enumerate(("ndc", "baseline")):
            for column, sector in ((0, "Cement"), (3, "Aluminium")):
                axis = spec.figure.axes[row_index * 4 + column]
                expected = []
                for model in ("OMNIA-ENGAGE", "GCAM"):
                    for policy in ("medce_eu", "highce_eu"):
                        source = original.loc[original.Model.eq(model) & original.Climate.eq(climate)
                                              & original.Policy.eq(policy) & original.Sector.eq(sector)
                                              & original.Role.eq("net")].iloc[0]
                        expected.append(source.Value)
                np.testing.assert_allclose([patch.get_height() for patch in axis.patches], expected)

    def test_missing_ltt_secondary_reference_has_no_omnia_endpoints_and_explicit_annotation(self):
        source = self.fixture.omnia.copy()
        source.loc[source.Scenario.eq("baseline_noce") & source.Region.isin(OMNIA_REGIONS)
                   & source.Variable.eq("Production|Iron and Steel|Secondary"), "2050"] = np.nan
        for treatment in ("zero", "preserve"):
            data = self.fixture.data(omnia=source, missing_activity=treatment)
            builder = EuropeRelativeBuilder(data, {**self.fixture.config, "missing_activity": treatment})
            with plt.rc_context(COMPARISON_STYLE):
                spec = builder.material_response_by_steel_route()
                spec.figure.canvas.draw()
                check_figure_bounds(spec.figure, spec.slug)
            axis = spec.figure.axes[6]
            self.assertEqual(len(axis.collections[0].get_offsets()), 0)
            self.assertEqual(len(axis.collections[1].get_offsets()), 2)
            self.assertTrue(np.isnan([patch.get_height() for patch in axis.patches[:2]]).all())
            annotation = " ".join(text.get_text() for text in axis.texts)
            self.assertIn("OMNIA-ENGAGE:\nno-CE reference unavailable", annotation)
            notes = " ".join(text.get_text() for text in spec.figure.texts)
            self.assertIn("OMNIA no-CE inputs are missing", notes)
            selected = spec.data.loc[spec.data.Model.eq("OMNIA-ENGAGE")
                                     & spec.data.ProductionGroup.eq("Secondary steel")
                                     & spec.data.Climate.eq("baseline")]
            self.assertTrue(selected.Value.isna().all())
            self.assertTrue(selected.ReferenceMissingInputCount.eq(3).all())
            self.assertFalse(selected.ResponseDefined.any())

    def test_observed_zero_route_change_retains_its_endpoint_marker(self):
        source = self.fixture.gcam.copy()
        policy = source.Scenario.eq("NDC_MedCE_EU")
        # Increase the medium-CE primary route from 60*5 to its no-CE value of
        # 80*5, and update its reported parent consistently in each geography.
        weights = source.Region.map({"EU-27": 1, "Austria": .4, "Belgium": .6})
        changed = policy & source.Variable.isin([GCAM_PRODUCTION["steel"],
                                               GCAM_PRODUCTION["steel"] + "|Primary"])
        source.loc[changed, "2050"] += 100 * weights[changed]
        data = self.fixture.data(gcam=source)
        with plt.rc_context(COMPARISON_STYLE):
            spec = EuropeRelativeBuilder(data, self.fixture.config).material_response_by_steel_route()
        endpoints = spec.figure.axes[1].collections[1].get_offsets()
        self.assertEqual(len(endpoints), 2)
        np.testing.assert_allclose(endpoints[0], [POSITIONS[0] + OFFSETS[1], 0])
        row = fixtures.EuropeRelativeRouteDataTests.select(spec.data, "Primary steel")
        self.assertTrue(row.ResponseDefined)
        self.assertEqual(row.ReferenceMissingInputCount, 0)
        self.assertEqual(row.Value, 0)

    def test_separated_primary_endpoints_have_nonoverlapping_high_ce_labels(self):
        frame = self.data.material_response_by_steel_route()
        primary = frame.ProductionGroup.eq("Primary steel")
        frame.loc[primary & frame.Model.eq("OMNIA-ENGAGE"), "Value"] = -82
        frame.loc[primary & frame.Model.eq("GCAM"), "Value"] = frame.loc[
            primary & frame.Model.eq("GCAM"), "Policy"].map({"medce_eu": -34, "highce_eu": -55})
        secondary = frame.ProductionGroup.eq("Secondary steel")
        frame.loc[secondary, "Value"] = frame.loc[secondary, "Value"].clip(upper=91)
        self.data.material_response_by_steel_route = lambda: frame.copy()
        with plt.rc_context(COMPARISON_STYLE):
            spec = self.builder.material_response_by_steel_route()
            spec.figure.canvas.draw()
        renderer = spec.figure.canvas.get_renderer()
        for axis_index in (1, 5):
            axis = spec.figure.axes[axis_index]
            self.assertEqual(axis.get_ylim(), (-120, 120))
            labels = []
            for offset in OFFSETS:
                matches = [text for text in axis.texts if isinstance(text, Annotation)
                           and np.isclose(text.xy[0], POSITIONS[1] + offset)]
                self.assertEqual(len(matches), 1)
                labels.append(matches[0].get_bbox_patch().get_window_extent(renderer))
            self.assertFalse(labels[0].overlaps(labels[1]),
                             "High-CE primary-route labels overlap despite separated model responses")


if __name__ == "__main__":
    unittest.main()
