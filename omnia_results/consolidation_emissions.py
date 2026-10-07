"""Emissions and carbon-capture comparisons for the consolidated report."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np

from .consolidation_data import PRODUCTION_SCENARIOS, climate_label
from .metrics import SECTORS
from .report import ReportFigure


class EmissionsFigures:
    """Compare both climate pathways within each emissions metric panel."""

    def emissions(self) -> ReportFigure:
        data = self.data.emissions()
        title = "Industrial emissions and carbon capture"
        fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 150 / 25.4), squeeze=False)
        fig.subplots_adjust(left=.095, right=.985, bottom=.14, top=.78,
                            wspace=.36, hspace=.66)
        years = self.config["years"]
        first, last = min(years), max(years)
        ticks = sorted({first, last, *[year for year in years if year % 10 == 0]})
        fig.text(.035, .977, title, fontsize=8, weight="bold", va="top")
        fig.text(.035, .936, f"No circular economy | {first}–{last}",
                 fontsize=6.7, color="#52616A", va="top")
        specifications = (
            ("Sector greenhouse-gas emissions", "GHG emissions (Gt CO$_2$e yr$^{-1}$)"),
            ("Share of total industry emissions", "Share of industry emissions (%)"),
            ("Emissions per tonne of product", "GHG intensity (t CO$_2$e t$^{-1}$)"),
            ("Carbon capture", "CO$_2$ captured (Mt CO$_2$ yr$^{-1}$)"),
        )
        for index, (ax, (panel_title, ylabel)) in enumerate(zip(axes.flat, specifications)):
            panel = chr(97 + index)
            selected = data.loc[(data.Panel == panel) & (data.Role == "plotted")]
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#DDE3E6", linewidth=.35)
            ax.tick_params(labelsize=6.3, pad=3)
            ax.set_xlim(first - .6, last + .6)
            ax.set_xticks(ticks)
            ax.set_title(panel_title, loc="left", fontsize=7, pad=9)
            ax.text(-.14, 1.04, panel, transform=ax.transAxes, fontsize=8,
                    weight="bold", va="bottom", ha="left")
            ax.set_ylabel(ylabel, fontsize=6.8, labelpad=3)
            if index >= 2:
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
            for sector in SECTORS:
                for pathway, scenario in enumerate(PRODUCTION_SCENARIOS):
                    values = (selected.loc[(selected.Sector == sector.label) &
                                           (selected.Scenario == scenario)]
                              .set_index("Year")["Value"].reindex(years))
                    ax.plot(values.index, values.to_numpy(dtype=float), color=sector.color,
                            linestyle="-" if pathway == 0 else "--",
                            marker="o" if pathway == 0 else "s",
                            markersize=3.8 if pathway == 0 else 2.,
                            markerfacecolor="white" if pathway == 0 else sector.color,
                            markeredgewidth=.7 if pathway == 0 else .55,
                            linewidth=1.15 if pathway == 0 else 1.05,
                            clip_on=False, label=f"{sector.label} | {climate_label(scenario)}")

        sector_handles = [Line2D([], [], color=sector.color, linewidth=1.6,
                                 label=sector.label) for sector in SECTORS]
        fig.legend(handles=sector_handles, loc="upper left", bbox_to_anchor=(.035, .879),
                   ncol=3, fontsize=6, columnspacing=1.2, handlelength=1.5,
                   handletextpad=.45, frameon=False)
        pathway_handles = [
            Line2D([], [], color="#53626A", linestyle="-", marker="o", markersize=3.8,
                   markerfacecolor="white", markeredgewidth=.7, linewidth=1.15,
                   label="NDC"),
            Line2D([], [], color="#53626A", linestyle="--", marker="s", markersize=2,
                   markerfacecolor="#53626A", markeredgewidth=.55, linewidth=1.05,
                   label="NDC+LTT"),
        ]
        fig.legend(handles=pathway_handles, loc="upper left", bbox_to_anchor=(.59, .879),
                   ncol=2, fontsize=6, columnspacing=1.5, handlelength=2.5,
                   handletextpad=.5, frameon=False)
        fig.text(.035, .060,
                 "GHG emissions are the reported model values. CO2 capture is shown separately.",
                 fontsize=5.8, color="#52616A")
        assumption = ("Draft assumption: missing additive activity = zero."
                      if self.config["missing_activity"] == "zero" else
                      "Missing activity retained; incomplete global totals appear as gaps.")
        fig.text(.035, .025, assumption, fontsize=5.5, color="#63717A", va="bottom")
        caption = (
            "Global sector greenhouse-gas emissions (a), shares of all-industry greenhouse-gas "
            "emissions (b), derived emissions per tonne of material (c), and carbon capture (d) "
            "without circular-economy measures. Each panel contains cement, iron and steel, and "
            "aluminium under NDC (ndc_noce; solid lines with open circles) and NDC+LTT "
            "(baseline_noce; dashed lines with filled squares). Colours identify sectors; line and "
            "marker styles identify climate pathways consistently with the energy figure. "
            f"Values are extracted directly from {self.config['input']} using exact variable "
            f"and unit selections, summing the {len(self.config['regions'])} model regions once. "
            "Panel a converts reported sector GHG from MtCO2e/yr to GtCO2e/yr. These reported "
            "emissions are not reduced again by the separately reported capture values. "
            "Panel b divides each sector's global GHG emissions by Emissions|GHG|Industry and "
            "multiplies by 100; the three plotted sectors do not exhaust total industry. "
            "Panel c divides global sector GHG in MtCO2e/yr by global production in Mt/yr, "
            "giving tCO2e/t. It is a ratio of global totals rather than an average of regional "
            "intensities. Cement uses cement output excluding clinker; steel and aluminium use "
            "primary plus secondary output. A nonpositive denominator yields an undefined "
            "ratio. Panel d uses Carbon Capture|Industry|sector in MtCO2/yr, a distinct CO2 "
            "quantity from the CO2-equivalent GHG measure. Source data includes original "
            "variables, constituent regions, numerators, denominators and conversion factors "
            "for audit. Each panel's vertical scale spans both pathways and all three sectors. "
            "Markers show supplied years joined by straight lines, without smoothing or "
            "displacement. Curves and markers can overlap where source values coincide or "
            "are very close; a filled square within an open circle represents overlapping "
            "pathway observations. Very small aluminium capture values lie close to zero on "
            "the common capture scale. " + assumption)
        return ReportFigure("fig04_emissions", "4", title,
                            "How do industrial emissions, emissions intensity and carbon capture differ under NDC and NDC+LTT?",
                            caption, fig, data, 4)
