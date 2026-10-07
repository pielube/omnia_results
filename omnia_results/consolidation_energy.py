"""Global energy trajectories for the two consolidated no-CE pathways."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np

from .consolidation_data import PRODUCTION_SCENARIOS, climate_label
from .metrics import SECTORS
from .report import ReportFigure


class EnergyFigures:
    """Compare final energy and derived intensity in aligned sector panels."""

    def energy(self) -> ReportFigure:
        data = self.data.energy()
        title = "Industrial energy requirements"
        fig, axes = plt.subplots(2, 3, figsize=(180 / 25.4, 150 / 25.4), squeeze=False)
        fig.subplots_adjust(left=.09, right=.985, bottom=.14, top=.78,
                            wspace=.51, hspace=.66)
        years = self.config["years"]
        first, last = min(years), max(years)
        ticks = sorted({first, last, *[year for year in years if year % 10 == 0]})
        fig.text(.035, .977, title, fontsize=8, weight="bold", va="top")
        fig.text(.035, .936, f"No circular economy | {first}–{last}",
                 fontsize=6.7, color="#52616A", va="top")
        for column, sector in enumerate(SECTORS):
            for row in range(2):
                ax, panel = axes[row, column], chr(97 + row * 3 + column)
                selected = data.loc[(data.Sector == sector.label) & (data.Panel == panel) &
                                    (data.Role == "plotted")]
                ax.spines[["top", "right"]].set_visible(False)
                ax.set_axisbelow(True)
                ax.grid(axis="y", color="#DDE3E6", linewidth=.35)
                ax.tick_params(labelsize=6.3, pad=3)
                ax.set_xlim(first - .6, last + .6)
                ax.set_xticks(ticks)
                ax.set_title(sector.label, loc="left", fontsize=7, pad=9)
                ax.text(-.14, 1.04, panel, transform=ax.transAxes, fontsize=8,
                        weight="bold", va="bottom", ha="left")
                label = ("Final energy (EJ yr$^{-1}$)" if row == 0 else
                         "Energy intensity (GJ t$^{-1}$)")
                ax.set_ylabel(label, fontsize=6.8, labelpad=3)
                if row == 1:
                    ax.set_xlabel("Year", fontsize=6.8, labelpad=4)
                finite = selected.Value.to_numpy(dtype=float)
                finite = finite[np.isfinite(finite)]
                low = min(0., float(finite.min())) if finite.size else 0.
                high = max(0., float(finite.max())) if finite.size else 1.
                if high <= low:
                    high = low + 1.
                span = high - low
                ax.set_ylim(low - .03 * span if low < 0 else 0., high + .08 * span)
                ax.yaxis.set_major_locator(ticker.MaxNLocator(4, min_n_ticks=3))
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.4g}"))
                if low < 0:
                    ax.axhline(0, color="#63717A", linewidth=.6)
                for index, scenario in enumerate(PRODUCTION_SCENARIOS):
                    values = (selected.loc[selected.Scenario == scenario]
                              .set_index("Year")["Value"].reindex(years))
                    ax.plot(values.index, values.to_numpy(dtype=float), color=sector.color,
                            linestyle="-" if index == 0 else "--",
                            marker="o" if index == 0 else "s",
                            markersize=3.8 if index == 0 else 2.0,
                            markerfacecolor="white" if index == 0 else sector.color,
                            markeredgewidth=.7 if index == 0 else .55,
                            linewidth=1.15 if index == 0 else 1.05,
                            label=climate_label(scenario))

        handles = [
            Line2D([], [], color="#53626A", linestyle="-", marker="o", markersize=3.8,
                   markerfacecolor="white", markeredgewidth=.7, linewidth=1.15,
                   label="NDC"),
            Line2D([], [], color="#53626A", linestyle="--", marker="s", markersize=2,
                   markerfacecolor="#53626A", markeredgewidth=.55, linewidth=1.05,
                   label="NDC+LTT"),
        ]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.035, .879), ncol=2,
                   fontsize=6, columnspacing=2.0, handlelength=2.5, handletextpad=.5,
                   frameon=False)
        fig.text(.035, .060, "Energy intensity is the ratio of global final energy to global material production.",
                 fontsize=5.8, color="#52616A")
        assumption = ("Draft assumption: missing additive activity = zero."
                      if self.config["missing_activity"] == "zero" else
                      "Missing activity retained; incomplete global totals appear as gaps.")
        fig.text(.035, .025, assumption, fontsize=5.5, color="#63717A", va="bottom")
        caption = (
            "Global final energy (a–c) and derived energy intensity (d–f) without circular-economy "
            "measures under NDC (ndc_noce; solid lines with open circles) and NDC+LTT "
            "(baseline_noce; dashed lines with filled squares). Columns show cement, iron and steel, "
            "and aluminium; colours identify sectors consistently across the two climate pathways. "
            f"Values are extracted directly from {self.config['input']}, summing each of the "
            f"{len(self.config['regions'])} configured model regions once. Final energy sums the "
            "exact Electricity, Gases, Liquids and Solids parent variables in EJ/yr, with Hydrogen "
            "also included for steel and aluminium. Nested carrier subcategories are excluded "
            "to avoid double counting. Derived intensity is global final energy divided by global "
            "production, multiplied by 1000 to convert EJ/Mt to GJ/t; it is a ratio of global "
            "totals rather than an average of regional intensities or the separately reported "
            "source intensity series. Cement uses cement output excluding clinker; steel and "
            "aluminium use primary plus secondary production. The source data records the "
            "energy numerator, production denominator, conversion factor, original variables "
            "and constituent model regions for audit. Each panel has its own vertical scale "
            "spanning both pathways. Markers show the supplied model years and straight lines "
            "join those values without smoothing, extrapolation or displacement. Curves may "
            "overlap where source values coincide or are very close; a filled square within "
            "an open circle denotes coincident pathway observations. A nonpositive "
            "production denominator yields an undefined intensity. " + assumption)
        return ReportFigure("fig03_energy", "3", title,
                            "How do global final energy and energy per tonne evolve under NDC and NDC+LTT?",
                            caption, fig, data, 6)
