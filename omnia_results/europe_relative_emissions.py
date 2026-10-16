"""Reported all-industry GHG responses in the European model comparison."""

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

from .europe_comparison import CLIMATES, CLIMATE_LABELS, HATCH, MODELS, _clean_axes
from .report import ReportFigure


class RelativeEmissionsFigures:
    def emissions_response(self):
        from .europe_relative import (MODEL_COLORS, OFFSETS, POLICIES, POSITIONS, WIDTH,
                                      _model_legend, _net_labels)

        data = self.data.emissions_response().copy()
        data["Plotted"] = data.PanelKind.eq("industry") & data.Role.eq("net")
        data["ContextOnly"] = ~data.Plotted
        data["DisplayRole"] = np.where(data.Plotted, "plotted", "context")
        data.loc[data.PanelKind.ne("industry"), "Panel"] = ""
        title = "Circular economy effects on European industrial GHG emissions"
        year = self.config["comparison_year"]
        fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 105 / 25.4), squeeze=False)
        for column, ax in enumerate(axes[0]):
            ax.set_position([.105 if column == 0 else .610, .290, .365, .380])
        fig.text(.035, .970, title, fontsize=8, weight="bold", va="top")
        fig.text(.035, .900, f"EU-only CE in {year} | reported all industry | change from each model's own no-CE case",
                 fontsize=6.7, color="#52616A", va="top")
        _model_legend(fig, production=True)
        fig.legends[0].set_bbox_to_anchor((.035, .835))
        plotted_values = []
        for column, climate in enumerate(CLIMATES):
            ax = axes[0, column]
            _clean_axes(ax, chr(97 + column), CLIMATE_LABELS[climate],
                        "Industry GHG change\nfrom no CE (%)")
            ax.set_xlim(-.65, 2.0)
            ax.set_xticks(POSITIONS, ["Medium CE\nEU", "High CE\nEU"], fontsize=6.3)
            ax.tick_params(axis="x", length=0, pad=5)
            ax.axhline(0, color="#53626B", linewidth=.7, zorder=2)
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
            for index, model in enumerate(MODELS):
                selected = data.loc[(data.Model == model) & (data.Climate == climate) & data.Plotted]
                values = selected.set_index("Policy").Value.reindex(POLICIES).to_numpy(dtype=float)
                plotted_values.extend(values[np.isfinite(values)])
                x = POSITIONS + OFFSETS[index]
                ax.bar(x, values, WIDTH, color=MODEL_COLORS[model],
                       edgecolor="#53626B" if index else "white", linewidth=.45,
                       hatch=HATCH if index else None, zorder=3)
                present = np.isfinite(values)
                ax.scatter(x[present], values[present], marker="D" if index else "o", s=13,
                           color=MODEL_COLORS[model], edgecolors="white", linewidths=.4, zorder=5)
                _net_labels(ax, x, values, index, stagger=False)
        plotted_values = np.asarray([0., *plotted_values], dtype=float)
        lo, hi = float(plotted_values.min()), float(plotted_values.max())
        span = max(hi - lo, 5.)
        limits = (5 * np.floor((lo - .20 * span) / 5), 5 * np.ceil((hi + .15 * span) / 5))
        for ax in axes[0]:
            ax.set_ylim(*limits)
        notes = [
            "OMNIA: reported all-industry GHG, including sectors beyond cement, steel and aluminium.",
            "GCAM: industry energy-demand + industrial-process Kyoto gases; gas and process coverage are not harmonised.",
            f"Each model uses its own {year} industry no-CE reference. EU-region boundaries differ slightly.",
        ]
        for y, note in zip((.130, .085, .040), notes):
            fig.text(.035, y, note, fontsize=5.4, color="#52616A", va="bottom")
        caption = (
            f"EU-only CE effects on reported European industrial GHG emissions in {year}, with NDC on the left "
            "and NDC+LTT on the right. Panels a,b compare source-defined all-industry percentage changes for "
            "OMNIA-ENGAGE (left solid bar) and GCAM (right hatched bar), calculated as "
            "100 × (CE industry GHG − no-CE industry GHG) / no-CE industry GHG in the same model, pathway "
            "and year. OMNIA selects the reported all-industry parent Emissions|GHG|Industry, including sectors "
            "beyond cement, iron and steel, and aluminium; it is not the sum of those three sectors. GCAM sums "
            "Emissions|Kyoto Gases|Energy|Demand|Industry and Emissions|Kyoto Gases|Industrial Processes. "
            "The inventories' gas coverage, GWP conventions and process boundaries are not documented as "
            "harmonised. Scales match between climate columns. Reported negative GHG is retained, and carbon "
            "capture is not subtracted again. GCAM has no matching sector GHG/Kyoto aggregates, so no sector "
            "breakdown is plotted. Its partial sector CO2 variables cannot reconstruct complete cement, steel "
            "and aluminium GHG inventories and are not substituted. Industry decompositions and the separate "
            "OMNIA three-sector calculations remain unplotted CSV context, with Plotted=False. Missing sector "
            "inputs in that context do not replace the independently reported industry parent. Original source "
            "quantities, reference totals, units, variables, regions, missing treatments and coefficients accompany "
            "the CSVs. Only medium/high EU-only CE is selected. Geographic boundaries retain the documented "
            "OMNIA EU-region versus GCAM EU-27 approximation. No CE is zero by construction, which does not "
            "establish agreement in absolute emissions or harmonise source inventories.")
        return ReportFigure("fig03_emissions_response", "3", title,
                            "How do reported all-industry GHG emissions respond to circular economy?",
                            caption, fig, data, 2)
