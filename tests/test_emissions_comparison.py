"""Independent response examples for the two scopes in the combined figure."""

import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from omnia_results.emissions_comparison import build_combined_emissions
from omnia_results.openprom_comparison import COMPARISON_STYLE
from omnia_results.report import check_figure_bounds


def _europe_frame():
    rows = []
    values = {("ndc", "OMNIA-ENGAGE"): (-12., -27.),
              ("ndc", "GCAM"): (-3., -8.),
              ("baseline", "OMNIA-ENGAGE"): (5., -13.),
              ("baseline", "GCAM"): (-7., -18.)}
    for (climate, model), responses in values.items():
        reference = (600. if model == "OMNIA-ENGAGE" else 2300.) * (2 if climate == "baseline" else 1)
        for policy, response in zip(("medce_eu", "highce_eu"), responses):
            common = dict(Model=model, Climate=climate, Policy=policy, Year=2050,
                          Panel="a" if climate == "ndc" else "b", PanelKind="industry",
                          Role="net", Value=response, Unit="%", SourceFile=f"{model}.raw",
                          SourceVariables=f"{model} industry GHG", Denominator=reference,
                          ReferenceValue=reference, ScenarioValue=reference * (1 + response / 100),
                          DenominatorUnit="MtCO2e/yr", Coverage="EU", ReferenceYear=2050)
            rows.append(common)
            # These context-only rows must not change either bar heights or limits.
            rows.append({**common, "Role": "component", "Value": 90000., "Unit": "percentage points"})
            rows.append({**common, "PanelKind": "three_sector", "Value": -90000.})
    return pd.DataFrame(rows).sample(frac=1., random_state=17).reset_index(drop=True)


def _openprom_frame():
    rows = []
    values = {("ndc", "OMNIA-ENGAGE"): (-40., -80.),
              ("ndc", "OPEN-PROM"): (-20., -50.),
              ("baseline", "OMNIA-ENGAGE"): (-24., -56.),
              ("baseline", "OPEN-PROM"): (7., -10.)}
    for (climate, model), responses in values.items():
        reference = (4000. if model == "OMNIA-ENGAGE" else 1000.) * (3 if climate == "baseline" else 1)
        common = dict(Model=model, Climate=climate, Year=2050, Metric="Emissions",
                      Unit="Mt CO2e/yr" if model == "OMNIA-ENGAGE" else "Mt CO2/yr",
                      SourceFile=f"{model}.raw", SourceVariables=f"{model} steel emissions",
                      ReferenceValue=reference, ReferenceYear=2050)
        rows.append({**common, "Ambition": "Reference", "Coverage": "Reference",
                     "Value": reference, "PercentChange": 0.})
        for ambition, response in zip(("Medium", "High"), responses):
            rows.append({**common, "Ambition": ambition,
                         "Coverage": "EU+" if model == "OMNIA-ENGAGE" else "Unspecified",
                         "Value": reference * (1 + response / 100), "PercentChange": response})
            if model == "OMNIA-ENGAGE":
                rows.append({**common, "Ambition": ambition, "Coverage": "EU",
                             "Value": 1., "PercentChange": -99.})
    frame = pd.DataFrame(rows)
    return pd.concat([frame, frame.assign(Metric="Energy", PercentChange=-999.),
                      frame.assign(Year=2030, PercentChange=999.)], ignore_index=True).sample(
                          frac=1., random_state=24).reset_index(drop=True)


class EmissionsComparisonFigureTests(unittest.TestCase):
    def setUp(self):
        self.europe, self.openprom = _europe_frame(), _openprom_frame()

    def tearDown(self):
        plt.close("all")

    def _build(self):
        with plt.rc_context(COMPARISON_STYLE):
            return build_combined_emissions(self.europe, self.openprom)

    def test_asymmetric_responses_use_four_bars_per_panel_with_independent_row_scales(self):
        spec = self._build()
        self.assertEqual(spec.panel_count, 4)
        self.assertEqual([ax.get_title(loc="left") for ax in spec.figure.axes],
                         ["NDC", "NDC+LTT", "NDC", "NDC+LTT"])
        expected = [(-12, -27, -3, -8), (5, -13, -7, -18),
                    (-40, -80, -20, -50), (-24, -56, 7, -10)]
        for ax, heights in zip(spec.figure.axes, expected):
            np.testing.assert_allclose([patch.get_height() for patch in ax.patches], heights)
            self.assertTrue(all(not patch.get_hatch() for patch in ax.patches[:2]))
            self.assertTrue(all(patch.get_hatch() for patch in ax.patches[2:]))
            self.assertEqual(sum(len(marks.get_offsets()) for marks in ax.collections), 4)
        axes = spec.figure.axes
        self.assertEqual(axes[0].get_ylim(), axes[1].get_ylim())
        self.assertEqual(axes[2].get_ylim(), axes[3].get_ylim())
        self.assertNotEqual(axes[0].get_ylim(), axes[2].get_ylim())
        self.assertLess(axes[0].get_ylim()[0], -27)
        self.assertGreater(axes[0].get_ylim()[1], 5)
        self.assertLess(axes[2].get_ylim()[0], -80)
        self.assertGreater(axes[2].get_ylim()[1], 7)
        labels = " ".join(text.get_text() for ax in axes for text in ax.texts)
        self.assertIn("+5.0%", labels)
        self.assertIn("+7.0%", labels)

    def test_exports_preserve_model_specific_units_references_and_context(self):
        spec = self._build()
        plotted = spec.data.loc[spec.data.PlotRole.eq("plotted")]
        self.assertEqual(len(plotted), 16)
        self.assertEqual(set(plotted.Panel), {"a", "b", "c", "d"})
        self.assertEqual(set(spec.data.Year), {2050})
        self.assertTrue(plotted.PlotUnit.eq("%").all())
        europe = plotted.loc[plotted.ComparisonRow.eq("European all industry")]
        np.testing.assert_allclose(europe.PlotValue, europe.Value)
        self.assertTrue(europe.PanelKind.eq("industry").all())
        self.assertTrue(europe.Role.eq("net").all())
        self.assertEqual(set(europe.loc[europe.Model.eq("GCAM"), "ReferenceValue"]), {2300, 4600})
        steel = plotted.loc[plotted.ComparisonRow.eq("Global steel")]
        np.testing.assert_allclose(steel.PlotValue, steel.PercentChange)
        self.assertEqual(set(steel.Unit), {"Mt CO2e/yr", "Mt CO2/yr"})
        self.assertEqual(set(steel.loc[steel.Model.eq("OMNIA-ENGAGE"), "Coverage"]), {"EU+"})
        self.assertEqual(set(steel.loc[steel.Model.eq("OPEN-PROM"), "Coverage"]), {"Unspecified"})
        self.assertEqual(set(steel.loc[steel.Model.eq("OMNIA-ENGAGE"), "ReferenceValue"]), {4000, 12000})
        self.assertEqual(set(steel.loc[steel.Model.eq("OPEN-PROM"), "ReferenceValue"]), {1000, 3000})
        self.assertTrue(spec.data.SourceFile.notna().all())
        refs = spec.data.loc[spec.data.ComparisonRow.eq("Global steel") & spec.data.Ambition.eq("Reference")]
        self.assertEqual(len(refs), 4)
        self.assertTrue(refs.ContextOnly.all())
        self.assertTrue(refs.Panel.eq("").all())
        self.assertTrue(spec.data.loc[spec.data.PanelKind.eq("three_sector"), "ContextOnly"].all())

    def test_visible_scope_labels_and_bounds_are_explicit(self):
        spec = self._build()
        with plt.rc_context(COMPARISON_STYLE):
            check_figure_bounds(spec.figure, spec.slug)
        visible = " ".join(text.get_text() for text in spec.figure.texts)
        for phrase in ("Europe", "all-industry GHG", "EU-only CE", "OPEN-PROM global", "CE in EU+",
                       "intervention area unspecified", "energy-related CO", "boundaries differ",
                       "scales match within rows"):
            self.assertIn(phrase, visible)
        self.assertIn("carbon capture is not subtracted again", spec.caption)
        self.assertIn("do not establish policy equivalence", spec.caption)
        self.assertIn("independent between rows", spec.caption)
        self.assertIn("no steel-specific process-emissions or GHG total", spec.caption)


if __name__ == "__main__":
    unittest.main()
