"""Emissions chart shows all-industry responses and keeps sector detail as context."""

import unittest

import matplotlib.pyplot as plt
import numpy as np

from omnia_results.europe_relative import COMPARISON_STYLE, EuropeRelativeBuilder
from omnia_results.report import check_figure_bounds
import test_europe_relative_emissions_data as fixtures


class EuropeRelativeEmissionsFigureTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.EuropeRelativeEmissionsDataTests()
        fixture.setUp()
        self.data = fixture.data()
        self.builder = EuropeRelativeBuilder(self.data, fixture.config)

    def tearDown(self):
        plt.close("all")

    def test_industry_bars_match_own_model_responses_and_use_common_pathway_scale(self):
        with plt.rc_context(COMPARISON_STYLE):
            spec = self.builder.emissions_response()
            check_figure_bounds(spec.figure, spec.slug)
        np.testing.assert_allclose([p.get_height() for p in spec.figure.axes[0].patches], [-10, -20, -13, -23])
        self.assertEqual(len(spec.figure.axes), 2)
        self.assertEqual(spec.panel_count, 2)
        self.assertEqual(spec.figure.axes[0].get_ylim(), spec.figure.axes[1].get_ylim())
        context = spec.data.loc[(spec.data.PanelKind == "industry") & (spec.data.Role == "component")]
        self.assertFalse(context.Plotted.any())

    def test_all_plotted_rows_use_industry_parent_scope_and_sector_detail_is_context_only(self):
        with plt.rc_context(COMPARISON_STYLE):
            spec = self.builder.emissions_response()
        plotted = spec.data.loc[spec.data.Plotted]
        self.assertEqual(len(plotted), 8)
        self.assertEqual(set(plotted.Model), {"OMNIA-ENGAGE", "GCAM"})
        self.assertEqual(set(plotted.PanelKind), {"industry"})
        self.assertEqual(set(plotted.Role), {"net"})
        focus = spec.data.loc[spec.data.PanelKind == "three_sector"]
        self.assertEqual(set(focus.Model), {"OMNIA-ENGAGE"})
        self.assertFalse(focus.Plotted.any())
        self.assertTrue(focus.ContextOnly.all())
        self.assertEqual(set(focus.DisplayRole), {"context"})
        self.assertTrue(focus.Panel.eq("").all())
        self.assertEqual(set(focus.loc[focus.Climate == "ndc", "Denominator"]), {240})
        omnia = plotted.loc[plotted.Model.eq("OMNIA-ENGAGE")]
        self.assertEqual(set(omnia.loc[omnia.Climate == "ndc", "Denominator"]), {600})
        self.assertIn("including sectors beyond cement, steel and aluminium",
                      " ".join(t.get_text() for t in spec.figure.texts))
        self.assertIn("reported all-industry parent Emissions|GHG|Industry", spec.caption)

    def test_missing_sector_inputs_do_not_replace_independently_reported_industry_bars(self):
        fixture = fixtures.EuropeRelativeEmissionsDataTests()
        fixture.setUp()
        source = fixture.omnia.copy()
        mask = source.Scenario.str.startswith("baseline_") & source.Variable.eq(fixtures.ALUMINIUM_GHG) & source.Unit.eq("MtCO2e/yr")
        source.loc[mask, "2050"] = np.nan
        data = fixture.data(omnia=source, missing_activity="zero")
        builder = EuropeRelativeBuilder(data, {**fixture.config, "missing_activity": "zero"})
        with plt.rc_context(COMPARISON_STYLE):
            spec = builder.emissions_response()
        net = spec.data.loc[(spec.data.PanelKind == "three_sector") & (spec.data.Climate == "baseline") & (spec.data.Role == "net")]
        self.assertTrue(net.AllEUAluminiumMissingAsZero.all())
        self.assertFalse(net.Plotted.any())
        self.assertFalse(spec.data.loc[spec.data.Plotted, "UsesProvisionalZeros"].any())
        np.testing.assert_allclose([p.get_height() for p in spec.figure.axes[0].patches], [-10, -20, -13, -23])
        self.assertIn("do not replace the independently reported industry parent", spec.caption)


if __name__ == "__main__":
    unittest.main()
