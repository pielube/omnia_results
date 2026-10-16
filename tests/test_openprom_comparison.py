"""Check the comparison artwork against independent, signed source examples."""

from types import SimpleNamespace
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from omnia_results.openprom_comparison import (COMPARISON_STYLE,
                                               OpenPromComparisonBuilder)
from omnia_results.report import check_figure_bounds


class OpenPromComparisonFigureTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "years": [2019, 2024, 2030, 2035, 2040, 2050],
            "historical_year": 2019, "comparison_year": 2050,
            "missing_activity": "zero",
        }
        self.frame = self._source_frame()
        self.builder = self._builder(self.frame)

    def tearDown(self):
        plt.close("all")

    def _builder(self, frame):
        return OpenPromComparisonBuilder(SimpleNamespace(frame=lambda: frame.copy()),
                                         self.config)

    def _source_frame(self):
        rows = []
        for metric, unit, metric_scale in (("Energy", "EJ/yr", 1.),
                                            ("Emissions", "Mt CO2e/yr", 30.)):
            for climate, climate_scale in (("ndc", 1.), ("baseline", 2.)):
                for model, coverages, model_scale in (
                        ("OPEN-PROM", ("Unspecified",), 1.),
                        ("OMNIA-ENGAGE", ("EU", "EU+"), 3.)):
                    years = (range(2019, 2051) if model == "OPEN-PROM"
                             else self.config["years"])
                    for year in years:
                        reference = (20. + .25 * (year - 2019)) * metric_scale * climate_scale * model_scale
                        source_unit = "Mt CO2/yr" if metric == "Emissions" and model == "OPEN-PROM" else unit
                        common = {"Metric": metric, "Climate": climate, "Model": model,
                                  "Year": year, "Unit": source_unit}
                        rows.append({**common, "Ambition": "Reference", "Coverage": "Reference",
                                     "Value": reference, "ReferenceValue": reference,
                                     "PercentChange": 0.})
                        for ambition, ambition_scale in (("Medium", 1.), ("High", 2.)):
                            for coverage in coverages:
                                # Distinct model/area responses include both increases and reductions.
                                rate = {"Unspecified": -.35, "EU": .12, "EU+": -.2}[coverage]
                                rate *= ambition_scale * (year - 2019) / 31.
                                value = reference * (1. + rate)
                                rows.append({**common, "Ambition": ambition, "Coverage": coverage,
                                             "Value": value, "ReferenceValue": reference,
                                             "PercentChange": 100. * rate})
        # The renderer must order actual observations, not depend on source row order.
        return pd.DataFrame(rows).sample(frac=1., random_state=18).reset_index(drop=True)

    def _build(self, number, metric, relative, builder=None):
        with plt.rc_context(COMPARISON_STYLE):
            return (builder or self.builder).build(number, metric, relative)

    def _selected(self, metric, climate, ambition, model, coverage):
        frame = self.frame
        return frame.loc[frame.Metric.eq(metric) & frame.Climate.eq(climate) &
                         frame.Ambition.eq(ambition) & frame.Model.eq(model) &
                         frame.Coverage.eq(coverage)].sort_values("Year")

    def test_four_figures_keep_panel_order_legible_bounds_and_shared_scales(self):
        with plt.rc_context(COMPARISON_STYLE):
            specs = list(self.builder.figures())
            for spec in specs:
                check_figure_bounds(spec.figure, spec.slug)
        self.assertEqual([spec.slug for spec in specs], [
            "fig01_energy_levels", "fig02_energy_response",
            "fig03_emissions_levels", "fig04_emissions_response",
        ])
        for spec in specs:
            self.assertEqual(spec.panel_count, 4)
            self.assertEqual([ax.get_title(loc="left") for ax in spec.figure.axes], [
                "Medium | NDC", "Medium | NDC+LTT", "High | NDC", "High | NDC+LTT",
            ])
            self.assertEqual(len({ax.get_ylim() for ax in spec.figure.axes}), 1)
            for panel, climate, ambition in (("a", "ndc", "Medium"),
                                              ("b", "baseline", "Medium"),
                                              ("c", "ndc", "High"),
                                              ("d", "baseline", "High")):
                observations = spec.data.loc[spec.data.Panel.eq(panel)]
                self.assertEqual(set(observations.Climate), {climate})
                self.assertEqual(set(observations.Ambition), {"Reference", ambition})

    def test_absolute_curves_use_source_values_and_years_with_each_reference_once(self):
        for number, metric in ((1, "Energy"), (3, "Emissions")):
            spec = self._build(number, metric, False)
            for ax, climate, ambition in zip(spec.figure.axes,
                                             ("ndc", "baseline", "ndc", "baseline"),
                                             ("Medium", "Medium", "High", "High")):
                self.assertEqual(len(ax.lines), 5)
                settings = (("OPEN-PROM", "Reference", "Reference"),
                            ("OMNIA-ENGAGE", "Reference", "Reference"),
                            ("OPEN-PROM", ambition, "Unspecified"),
                            ("OMNIA-ENGAGE", ambition, "EU"),
                            ("OMNIA-ENGAGE", ambition, "EU+"))
                for line, (model, setting, coverage) in zip(ax.lines, settings):
                    expected = self._selected(metric, climate, setting, model, coverage)
                    np.testing.assert_array_equal(line.get_xdata(), expected.Year)
                    np.testing.assert_allclose(line.get_ydata(), expected.Value)
                    expected_years = range(2019, 2051) if model == "OPEN-PROM" else self.config["years"]
                    np.testing.assert_array_equal(line.get_xdata(), list(expected_years))
                self.assertEqual([line.get_linestyle() for line in ax.lines], ["--", "--", "-", "-", "-"])
            reference = spec.data.loc[spec.data.Ambition.eq("Reference")]
            self.assertFalse(reference.duplicated(["Panel", "Model", "Year"]).any())
            self.assertEqual(set(spec.data.Role), {"plotted"})
            np.testing.assert_allclose(spec.data.PlotValue, spec.data.Value)
            self.assertTrue(spec.data.PlotUnit.eq(spec.data.Unit).all())

    def test_response_curves_keep_signed_percentages_and_context_references(self):
        for number, metric in ((2, "Energy"), (4, "Emissions")):
            spec = self._build(number, metric, True)
            for ax, climate, ambition in zip(spec.figure.axes,
                                             ("ndc", "baseline", "ndc", "baseline"),
                                             ("Medium", "Medium", "High", "High")):
                self.assertEqual(len(ax.lines), 4)
                for line, (model, coverage) in zip(ax.lines[:3], (("OPEN-PROM", "Unspecified"),
                                                                 ("OMNIA-ENGAGE", "EU"),
                                                                 ("OMNIA-ENGAGE", "EU+"))):
                    expected = self._selected(metric, climate, ambition, model, coverage)
                    np.testing.assert_array_equal(line.get_xdata(), expected.Year)
                    np.testing.assert_allclose(line.get_ydata(), expected.PercentChange)
                np.testing.assert_array_equal(ax.lines[3].get_ydata(), [0., 0.])
                self.assertLess(ax.get_ylim()[0], -70.)
                self.assertGreater(ax.get_ylim()[1], 24.)
            context = spec.data.Ambition.eq("Reference")
            self.assertTrue(spec.data.loc[context, "Role"].eq("context").all())
            self.assertTrue(spec.data.loc[~context, "Role"].eq("plotted").all())
            np.testing.assert_allclose(spec.data.PlotValue, spec.data.PercentChange)
            self.assertEqual(set(spec.data.PlotUnit), {"%"})
            # Exported absolute observations remain available alongside plotted percentages.
            self.assertTrue(spec.data.loc[~context, "Value"].gt(0.).all())
            self.assertTrue(spec.data.loc[~context, "PlotValue"].lt(0.).any())

    def test_missing_observation_remains_gap_and_observed_zero_remains_zero(self):
        frame = self.frame.copy()
        selected = (frame.Metric.eq("Energy") & frame.Climate.eq("ndc") &
                    frame.Model.eq("OPEN-PROM") & frame.Ambition.eq("Medium"))
        frame.loc[selected & frame.Year.eq(2030), "PercentChange"] = np.nan
        frame.loc[selected & frame.Year.eq(2040), "PercentChange"] = 0.
        frame.loc[selected & frame.Year.eq(2050), "PercentChange"] = np.nan
        spec = self._build(2, "Energy", True, self._builder(frame))
        line = spec.figure.axes[0].lines[0]
        self.assertEqual(len(line.get_xdata()), 32)
        self.assertTrue(np.isnan(line.get_ydata()[11]))
        self.assertEqual(line.get_ydata()[21], 0.)
        self.assertTrue(np.isnan(line.get_ydata()[-1]))
        endpoint_notes = " ".join(text.get_text() for text in spec.figure.axes[0].texts)
        self.assertIn("OPEN unavailable", endpoint_notes)

    def test_emissions_figures_state_different_scopes_in_visible_text_and_caption(self):
        for number, relative in ((3, False), (4, True)):
            spec = self._build(number, "Emissions", relative)
            visible = " ".join(text.get_text() for text in spec.figure.texts)
            self.assertIn("reported steel GHG", visible)
            self.assertIn("energy-related CO", visible)
            self.assertIn("boundaries differ", visible)
            self.assertIn("no steel-specific process-emissions or GHG total", spec.caption)
            self.assertIn("different gas and accounting scopes", spec.caption)
            units = set(spec.data.Unit)
            self.assertEqual(units, {"Mt CO2/yr", "Mt CO2e/yr"})

    def test_bottom_response_annotations_leave_space_above_figure_notes(self):
        for number, metric in ((2, "Energy"), (4, "Emissions")):
            spec = self._build(number, metric, True)
            spec.figure.canvas.draw()
            renderer = spec.figure.canvas.get_renderer()
            first_note = spec.figure.texts[3].get_window_extent(renderer)
            for ax in spec.figure.axes[2:]:
                annotations = [text for text in ax.texts
                               if text.get_text().startswith("2050:")]
                self.assertEqual(len(annotations), 1)
                endpoint_note = annotations[0].get_window_extent(renderer)
                self.assertFalse(endpoint_note.overlaps(first_note))
                self.assertGreaterEqual(endpoint_note.y0 - first_note.y1, 3.)


if __name__ == "__main__":
    unittest.main()
