"""Climate-matched annualised sector-cost changes across CE settings."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np

from .comparison_common import POLICY_COLORS, POLICY_MARKERS, POLICY_SHORT, POLICY_STYLES
from .consolidation_data import CE_POLICIES, MATERIAL_CLIMATES, climate_label
from .metrics import SECTORS
from .report import ReportFigure


class SectorCostFigures:
    """Compare CE sector costs against the matching no-CE cost in each year."""

    def sector_costs(self) -> ReportFigure:
        data = self.data.sector_costs()
        start = self.config.get("main_cost_start_year", 2024)
        years = [year for year in self.config["years"] if year >= start]
        if len(years) < 2:
            raise ValueError("Sector-cost trajectories require at least two supplied years in the main cost period")
        first, last = min(years), max(years)
        plotted = data.loc[(data.Role == "plotted") & (data.Year.isin(years))]
        title = "Circular economy changes annualised industrial sector costs"
        fig, axes = plt.subplots(2, 3, figsize=(180 / 25.4, 150 / 25.4), squeeze=False)
        fig.subplots_adjust(left=.09, right=.985, bottom=.14, top=.78,
                            wspace=.51, hspace=.66)
        fig.text(.035, .977, title, fontsize=8, weight="bold", va="top")
        fig.text(.035, .936, f"Change from the same year's no-CE cost | {first}–{last}",
                 fontsize=6.7, color="#52616A", va="top")
        ticks = sorted({first, last, *[year for year in years if year % 10 == 0]})

        for column, sector in enumerate(SECTORS):
            sector_data = plotted.loc[plotted.Sector == sector.label]
            finite = sector_data.Value.to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            low = min(0., float(finite.min())) if finite.size else 0.
            high = max(0., float(finite.max())) if finite.size else 0.
            span = max(high - low, 1.)
            limits = (low - .10 * span, high + .12 * span)
            for row, climate in enumerate(MATERIAL_CLIMATES):
                ax, panel = axes[row, column], chr(97 + row * 3 + column)
                selected = sector_data.loc[sector_data.Panel == panel]
                ax.spines[["top", "right"]].set_visible(False)
                ax.set_axisbelow(True)
                ax.grid(axis="y", color="#DDE3E6", linewidth=.35)
                ax.tick_params(labelsize=6.3, pad=3)
                ax.set_xlim(first - .6, last + .6)
                ax.set_xticks(ticks)
                ax.set_ylim(*limits)
                ax.yaxis.set_major_locator(ticker.MaxNLocator(4, min_n_ticks=3))
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.4g}"))
                ax.set_ylabel("Cost change vs no CE (%)", fontsize=6.8, labelpad=3)
                if row == 1:
                    ax.set_xlabel("Year", fontsize=6.8, labelpad=4)
                ax.set_title(f"{sector.label} | {climate_label(climate + '_noce')}",
                             loc="left", fontsize=7, pad=9)
                ax.text(-.14, 1.04, panel, transform=ax.transAxes, fontsize=8,
                        weight="bold", va="bottom", ha="left")
                ax.axhline(0, color="#B7C1C6", linewidth=.55, zorder=2, label="_zero_axis")
                for policy in CE_POLICIES:
                    scenario = f"{climate}_{policy}"
                    values = (selected.loc[selected.Scenario == scenario]
                              .set_index("Year")["Value"].reindex(years))
                    ax.plot(values.index, values.to_numpy(dtype=float), color=POLICY_COLORS[policy],
                            linestyle=POLICY_STYLES[policy], marker=POLICY_MARKERS[policy],
                            markerfacecolor="white" if policy.endswith("_eu") else POLICY_COLORS[policy],
                            markeredgewidth=.65, markersize=2.6, linewidth=1.0,
                            label=scenario, zorder=4 if policy.endswith("gbl") else 3)

        handles = [Line2D([], [], color=POLICY_COLORS[policy], marker=POLICY_MARKERS[policy],
                          linestyle=POLICY_STYLES[policy], linewidth=1, markersize=3,
                          markerfacecolor="white" if policy.endswith("_eu") else POLICY_COLORS[policy],
                          label=POLICY_SHORT[policy]) for policy in CE_POLICIES]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.035, .879), ncol=5,
                   fontsize=6, columnspacing=1.7, handlelength=2.2,
                   handletextpad=.5, frameon=False)
        fig.text(.035, .071,
                 "M/H = medium/high CE; EU+ = EU + international partners (17 model regions).",
                 fontsize=5.5, color="#52616A")
        fig.text(.035, .046,
                 "Reference: same sector, climate pathway and year. Source 2019 costs are retained in the data.",
                 fontsize=5.5, color="#52616A")
        fig.text(.035, .019,
                 "Costs retain missingness; unavailable or nonpositive no-CE costs make percentage changes undefined.",
                 fontsize=5.5, color="#63717A", va="bottom")
        caption = (
            f"Annualised sector-cost changes across five CE settings in {first}–{last}, with "
            "NDC in the upper row and NDC+LTT in the lower row. Columns show cement, iron and "
            "steel, and aluminium. Each curve gives 100 × (scenario cost − no-CE cost) / "
            "no-CE cost, using the same sector, climate pathway and model year for the "
            "reference. No CE is therefore zero wherever its reference cost is finite and "
            "positive. The same policy colours, markers and line styles are used as in the "
            "emissions-response figure. EU+ denotes CE adoption in the EU plus international "
            "partners, covering 17 model regions rather than global adoption. "
            f"Values are extracted directly from {self.config['input']} through exact "
            "Total Annualised Cost|Industry|sector parent variables in Millions USD_2010/yr, "
            f"summing the {len(self.config['regions'])} configured model regions once. "
            "These are annualised sector costs in constant 2010 USD. Cost inputs always "
            "retain missingness, independently of the additive-activity policy. Incomplete "
            "global costs or nonpositive no-CE reference costs give undefined percentages "
            "and gaps; finite negative scenario costs and percentage changes are retained. "
            "Source data contains all supplied years, with observations before the main "
            "plot period retained as context. Original cost variables, "
            "constituent model regions, scenario and reference costs, reference scenario "
            "and year, numerators, denominators, conversion factors and offsets are recorded "
            "for audit. Vertical limits are identical within each sector across the two "
            "climate pathways, include zero and all finite changes, and differ across "
            "sectors. Markers show supplied model years joined by straight segments without "
            "smoothing, extrapolation or displacement."
        )
        return ReportFigure("fig08_sector_costs", "8", title,
                            "How do circular-economy settings change annualised costs within each industrial sector?",
                            caption, fig, data, 6)
