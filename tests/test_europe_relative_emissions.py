"""Emissions chart shows combined three-sector responses with explicit gas scopes."""

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

    def test_three_sector_bars_match_own_model_responses_and_use_common_pathway_scale(self):
        with plt.rc_context(COMPARISON_STYLE):
            spec = self.builder.emissions_response()
            check_figure_bounds(spec.figure, spec.slug)
        np.testing.assert_allclose([p.get_height() for p in spec.figure.axes[0].patches], [-5, -65, -16, -38])
        self.assertEqual(len(spec.figure.axes), 2)
        self.assertEqual(spec.panel_count, 2)
        self.assertEqual(spec.figure.axes[0].get_ylim(), spec.figure.axes[1].get_ylim())
        context = spec.data.loc[(spec.data.PanelKind == "industry") & (spec.data.Role == "component")]
        self.assertFalse(context.Plotted.any())

    def test_only_combined_three_sector_nets_are_plotted_and_gas_scopes_are_explicit(self):
        with plt.rc_context(COMPARISON_STYLE):
            spec = self.builder.emissions_response()
        plotted = spec.data.loc[spec.data.Plotted]
        self.assertEqual(len(plotted), 8)
        self.assertEqual(set(plotted.Model), {"OMNIA-ENGAGE", "GCAM"})
        self.assertEqual(set(plotted.PanelKind), {"three_sector"})
        self.assertEqual(set(plotted.Role), {"net"})
        self.assertEqual(set(plotted.Panel), {"a", "b"})
        industry = spec.data.loc[spec.data.PanelKind == "industry"]
        self.assertFalse(industry.Plotted.any())
        self.assertTrue(industry.ContextOnly.all())
        self.assertEqual(set(industry.DisplayRole), {"context"})
        self.assertTrue(industry.Panel.eq("").all())
        components = spec.data.loc[spec.data.Role.eq("component")]
        self.assertFalse(components.Plotted.any())
        self.assertTrue(components.ContextOnly.all())
        omnia = plotted.loc[plotted.Model.eq("OMNIA-ENGAGE")]
        gcam = plotted.loc[plotted.Model.eq("GCAM")]
        self.assertEqual(set(omnia.loc[omnia.Climate == "ndc", "Denominator"]), {240})
        self.assertEqual(set(gcam.loc[gcam.Climate == "ndc", "Denominator"]), {250})
        notes = " ".join(t.get_text() for t in spec.figure.texts)
        self.assertIn("cement, steel and aluminium", notes)
        self.assertIn("OMNIA GHG", notes)
        self.assertIn("GCAM CO2", notes.replace("\u2082", "2"))
        self.assertIn("Other industry is excluded from numerator and denominator", spec.caption)
        self.assertIn("Sector percentages are not averaged", spec.caption)

    def test_plotted_omnia_totals_explicitly_flag_provisional_missing_aluminium_zeros(self):
        fixture = fixtures.EuropeRelativeEmissionsDataTests()
        fixture.setUp()
        source = fixture.omnia.copy()
        mask = source.Scenario.str.startswith("baseline_") & source.Variable.eq(fixtures.ALUMINIUM_GHG) & source.Unit.eq("MtCO2e/yr")
        source.loc[mask, "2050"] = np.nan
        data = fixture.data(omnia=source, missing_activity="zero")
        builder = EuropeRelativeBuilder(data, {**fixture.config, "missing_activity": "zero"})
        with plt.rc_context(COMPARISON_STYLE):
            spec = builder.emissions_response()
        net = spec.data.loc[(spec.data.PanelKind == "three_sector") & (spec.data.Model == "OMNIA-ENGAGE")
                            & (spec.data.Climate == "baseline") & (spec.data.Role == "net")]
        self.assertTrue(net.AllEUAluminiumMissingAsZero.all())
        self.assertTrue(net.Plotted.all())
        self.assertTrue(net.UsesProvisionalZeros.all())
        self.assertFalse(spec.data.loc[spec.data.Plotted & spec.data.Model.eq("GCAM"), "UsesProvisionalZeros"].any())
        np.testing.assert_allclose([p.get_height() for p in spec.figure.axes[0].patches], [-5, -65, -16, -38])
        np.testing.assert_allclose([p.get_height() for p in spec.figure.axes[1].patches], [10, -80, -16, -38])
        notes = " ".join(t.get_text() for t in spec.figure.texts)
        self.assertIn("aluminium blanks", notes)
        self.assertIn("zero (provisional)", notes)
        self.assertIn("without establishing physical zero emissions", spec.caption)


if __name__ == "__main__":
    unittest.main()
