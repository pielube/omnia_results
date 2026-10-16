"""Verify percentage marks, shared scales and signed fuel accounting."""

import unittest

import matplotlib.pyplot as plt
import numpy as np

from omnia_results.europe_relative import (COMPARISON_STYLE, EuropeRelativeBuilder,
                                           POSITIONS, OFFSETS)
from omnia_results.report import check_figure_bounds
import test_europe_relative_data as fixtures


class EuropeRelativeFigureTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.EuropeRelativeDataTests()
        fixture.setUp()
        self.data = fixture.data()
        self.builder = EuropeRelativeBuilder(self.data, fixture.config)

    def tearDown(self):
        plt.close("all")

    def test_production_marks_plot_net_responses_with_one_scale_and_context_routes(self):
        with plt.rc_context(COMPARISON_STYLE):
            spec = self.builder.material_response()
            check_figure_bounds(spec.figure, spec.slug)
        ax = spec.figure.axes[1]  # NDC steel.
        heights = [patch.get_height() for patch in ax.patches]
        np.testing.assert_allclose(heights, [100 / 12, 100 / 12, -10, -25])
        self.assertEqual(len({ax.get_ylim() for ax in spec.figure.axes}), 1)
        self.assertEqual(spec.data.loc[spec.data.Plotted, "Role"].unique().tolist(), ["net"])
        self.assertTrue((~spec.data.loc[spec.data.Role == "component", "Plotted"]).all())
        self.assertEqual(set(spec.data.Policy), {"medce_eu", "highce_eu"})
        self.assertEqual(set(spec.data.Year), {2050})

    def test_energy_component_geometry_reconstructs_net_markers_and_retains_increases(self):
        with plt.rc_context(COMPARISON_STYLE):
            spec = self.builder.energy_response()
            check_figure_bounds(spec.figure, spec.slug)
        self.assertEqual(len({ax.get_ylim() for ax in spec.figure.axes}), 1)
        for ax in spec.figure.axes:
            for model_index in range(2):
                marks = ax.collections[model_index].get_offsets()
                for policy_index, position in enumerate(POSITIONS + OFFSETS[model_index]):
                    components = [p.get_height() for p in ax.patches
                                  if np.isclose(p.get_x() + p.get_width() / 2, position)]
                    self.assertAlmostEqual(sum(components), marks[policy_index, 1])
        self.assertTrue(any(p.get_height() > 0 for ax in spec.figure.axes for p in ax.patches))
        self.assertTrue(any(p.get_height() < 0 for ax in spec.figure.axes for p in ax.patches))

    def test_missing_response_has_no_endpoint_but_observed_zero_retains_a_marker(self):
        original = self.data.material_response()
        frame = original.copy()
        zero = (frame.Model == "GCAM") & (frame.Climate == "ndc") & (frame.Sector == "Cement") & (frame.Role == "net")
        frame.loc[zero & (frame.Policy == "medce_eu"), "Value"] = 0.
        frame.loc[zero & (frame.Policy == "highce_eu"), "Value"] = np.nan
        self.data.material_response = lambda: frame.copy()
        with plt.rc_context(COMPARISON_STYLE):
            spec = self.builder.material_response()
        observed = spec.figure.axes[0].collections[1].get_offsets()
        self.assertEqual(len(observed), 1)
        np.testing.assert_allclose(observed[0], [POSITIONS[0] + OFFSETS[1], 0.])


if __name__ == "__main__":
    unittest.main()
