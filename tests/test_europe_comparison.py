"""Check that exported chart geometry preserves the source definitions."""

import unittest

import matplotlib.pyplot as plt
import numpy as np

from omnia_results.europe_comparison import EuropeComparisonBuilder, COMPARISON_STYLE
from omnia_results.report import check_figure_bounds
import test_europe_comparison_data as fixtures


class EuropeComparisonFigureTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.EuropeComparisonDataTests()
        fixture.setUp()
        # Include a genuine earlier GCAM observation to verify it cannot enter
        # the plotted historical reference period accidentally.
        fixture.gcam['2015'] = fixture.gcam['2021'] * 2
        self.data = fixture.data()
        self.builder = EuropeComparisonBuilder(self.data, fixture.config)

    def tearDown(self):
        plt.close('all')

    def test_trajectories_retain_each_models_years_and_exclude_earlier_context(self):
        with plt.rc_context(COMPARISON_STYLE):
            spec = self.builder.emissions_response()
            check_figure_bounds(spec.figure, spec.slug)
        for ax in spec.figure.axes[:2]:
            self.assertEqual(len(ax.lines), 6)
            self.assertEqual(list(ax.get_xticks()), [2019, 2050])
            for index, line in enumerate(ax.lines):
                expected = [2019, 2050] if index < 3 else [2021, 2050]
                self.assertEqual(list(line.get_xdata()), expected)
        self.assertIn('source emissions scopes differ', spec.figure.texts[1].get_text())

    def test_carrier_stack_heights_reproduce_reported_energy_including_residual(self):
        with plt.rc_context(COMPARISON_STYLE):
            spec = self.builder.energy_mix()
            check_figure_bounds(spec.figure, spec.slug)
        ax = spec.figure.axes[2]  # Aluminium, NDC.
        for center, expected in ((1.5 - .19, 90), (1.5 + .19, 32)):
            heights = [p.get_height() for p in ax.patches
                       if np.isclose(p.get_x() + p.get_width() / 2, center)]
            self.assertAlmostEqual(sum(heights), expected)
        self.assertIn('Unallocated', [t.get_text() for legend in spec.figure.legends for t in legend.get_texts()])

    def test_signed_component_bars_reconstruct_net_instead_of_absolute_savings(self):
        with plt.rc_context(COMPARISON_STYLE):
            spec = self.builder.emissions_response()
        ax = spec.figure.axes[2]  # NDC model-specific components.
        expected = [-36, 36, -85, 55]  # OMNIA medium/high, GCAM medium/high.
        centers = [-.23, 1.5 - .23, .23, 1.5 + .23]
        for center, net in zip(centers, expected):
            components = [p.get_height() for p in ax.patches
                          if np.isclose(p.get_x() + p.get_width() / 2, center)]
            self.assertAlmostEqual(sum(components), net)
        actual_nets = np.concatenate([c.get_offsets()[:, 1] for c in ax.collections])
        np.testing.assert_allclose(actual_nets, expected)
        self.assertTrue(any(p.get_height() > 0 for p in ax.patches))
        self.assertTrue(any(p.get_height() < 0 for p in ax.patches))


if __name__ == '__main__':
    unittest.main()
