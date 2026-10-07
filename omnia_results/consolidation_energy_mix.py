"""Final-energy carrier bars across CE settings and a historical reference."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Patch
import numpy as np

from .consolidation_data import ENERGY_CARRIERS, MATERIAL_CLIMATES, climate_label
from .metrics import SECTORS
from .report import ReportFigure


FUEL_COLORS = {"Electricity": "#0072B2", "Gases": "#E69F00",
               "Hydrogen": "#009E73", "Liquids": "#CC79A7",
               "Solids": "#66717A"}


def _percentage_label(value: float) -> str:
    if abs(value) < .05:
        return "0%"
    return f"{value:+.1f}%".replace("-", "\N{MINUS SIGN}")


class EnergyMixFigures:
    """Compare global final-energy levels and carrier composition."""

    def energy_mix(self) -> ReportFigure:
        data = self.data.energy_mix()
        energy = data.loc[data.Metric == "Final energy"]
        title = "Circular economy reduces final energy demand and changes the fuel mix"
        historical = self.config.get("historical_year", 2019)
        future = self.config.get("comparison_year", 2050)
        fig, axes = plt.subplots(2, 3, figsize=(180 / 25.4, 175 / 25.4), squeeze=False)
        fig.subplots_adjust(left=.085, right=.985, bottom=.20, top=.76,
                            wspace=.48, hspace=.60)
        fig.text(.035, .977, title, fontsize=8, weight="bold", va="top")
        fig.text(.035, .941, f"Global final energy | {historical} reference and {future} CE scenarios",
                 fontsize=6.7, color="#52616A", va="top")
        positions = np.array([0., 1.45, 2.45, 3.45, 4.45, 5.45])
        for column, sector in enumerate(SECTORS):
            sector_data = energy.loc[energy.Sector == sector.label]
            finite = sector_data.Value.to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            low = min(0., float(finite.min())) if finite.size else 0.
            high = max(0., float(finite.max())) if finite.size else 1.
            if high <= low:
                high = low + 1.
            span = high - low
            limits = (low - .04 * span if low < 0 else 0., high + .18 * span)
            for row, climate in enumerate(MATERIAL_CLIMATES):
                ax, panel = axes[row, column], chr(97 + row * 3 + column)
                selected = sector_data.loc[sector_data.Panel == panel]
                bars = (selected[["BarOrder", "BarLabel"]].drop_duplicates()
                        .set_index("BarOrder").reindex(range(6)))
                labels = [label.replace("–", "\n").replace("No CE", "No\nCE")
                          for label in bars.BarLabel.fillna("").tolist()]
                ax.spines[["top", "right"]].set_visible(False)
                ax.set_axisbelow(True)
                ax.grid(axis="y", color="#DDE3E6", linewidth=.35)
                ax.tick_params(labelsize=6.1, pad=3)
                ax.set_xlim(-.52, 5.99)
                ax.set_ylim(*limits)
                ax.yaxis.set_major_locator(ticker.MaxNLocator(4, min_n_ticks=3))
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.4g}"))
                ax.set_ylabel("Final energy (EJ yr$^{-1}$)", fontsize=6.8, labelpad=3)
                ax.set_xticks(positions, labels=labels, fontsize=6)
                ax.tick_params(axis="x", length=0, pad=4)
                ax.get_xticklabels()[0].set_weight("bold")
                ax.set_title(f"{sector.label} | {climate_label(climate + '_noce')}",
                             loc="left", fontsize=7, pad=9)
                ax.text(-.14, 1.04, panel, transform=ax.transAxes, fontsize=8,
                        weight="bold", va="bottom", ha="left")
                ax.axvline(.725, color="#B7C1C6", linewidth=.55, linestyle=(0, (2, 2)))
                ax.text(3.45, -.19, str(future), transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=5.8, color="#52616A")
                bottom = np.zeros(6, dtype=float)
                for carrier in ENERGY_CARRIERS:
                    if carrier not in sector.fuels:
                        continue
                    values = (selected.loc[selected.Series == carrier]
                              .set_index("BarOrder")["Value"].reindex(range(6))
                              .to_numpy(dtype=float))
                    ax.bar(positions, values, .64, bottom=bottom,
                           color=FUEL_COLORS[carrier], edgecolor="white", linewidth=.35,
                           label=carrier)
                    # Missing components propagate to the cumulative stack:
                    # unknown quantities are never silently replaced with zero.
                    bottom = bottom + values
                totals = (selected.loc[selected.Series == "Total"]
                          .set_index("BarOrder")["Value"].reindex(range(6)))
                changes = (data.loc[(data.Panel == panel) &
                                    (data.Metric == "Final energy change from no CE")]
                           .set_index("BarOrder")["Value"].reindex(range(1, 6)))
                for order in range(1, 6):
                    value, change = float(totals.loc[order]), float(changes.loc[order])
                    if np.isfinite(value) and np.isfinite(change):
                        ax.text(positions[order], value + span * .036,
                                "ref." if order == 1 else _percentage_label(change),
                                ha="center", va="bottom", fontsize=5.5, color="#202D35")

        handles = [Patch(facecolor=FUEL_COLORS[carrier], label=carrier)
                   for carrier in ENERGY_CARRIERS]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.035, .882), ncol=5,
                   fontsize=6, columnspacing=1.7, handlelength=1.4, handletextpad=.5,
                   frameon=False)
        fig.text(.035, .112,
                 f"{historical}: no CE. All other bars: {future}. Labels above: change from the same pathway's {future} no CE.",
                 fontsize=5.5, color="#52616A")
        fig.text(.035, .087, "Each sector uses the same scale across both climate pathways; sectors have different scales.",
                 fontsize=5.5, color="#52616A")
        fig.text(.035, .057,
                 "M/H = medium/high CE; EU+ = EU + international partners (17 model regions).",
                 fontsize=5.5, color="#52616A")
        assumption = ("Draft assumption: missing additive activity = zero."
                      if self.config["missing_activity"] == "zero" else
                      "Missing activity retained; incomplete stacks and annotations appear as gaps.")
        fig.text(.035, .023, assumption, fontsize=5.5, color="#63717A", va="bottom")
        caption = (
            f"Global final energy in {historical} without CE, followed by five {future} CE "
            "settings. The upper row shows NDC (ndc_* scenarios) and the lower row NDC+LTT "
            f"(baseline_* scenarios). Each pathway's {historical} no-CE observation is separated "
            f"visually from its {future} policy bars. Future bars are ordered no CE, medium CE "
            "in the EU, high CE in the EU, medium CE in the EU plus international partners, "
            "and high CE in the EU plus partners. EU+ indicates these 17 participating model "
            "regions rather than global CE adoption; all plotted quantities are global. "
            "Stacks show Electricity, Gases, Hydrogen, Liquids and Solids in a consistent "
            "order and palette. Cement includes its four reported parent carriers; steel "
            "and aluminium also include Hydrogen. Final energy is calculated from the exact "
            "Final Energy|Industry|sector|carrier parent variables in EJ/yr; nested carrier "
            f"subcategories are excluded to avoid double counting. Values are extracted "
            f"directly from {self.config['input']}, summing each of the "
            f"{len(self.config['regions'])} model regions once. Each sector has the same "
            "vertical limits across pathways, including its historical reference, while "
            "sectors have different scales. Future labels give the change in total sector "
            f"energy relative to no CE under the same climate pathway in {future}: "
            "100 × (scenario final energy / reference final energy − 1). Ref. marks the "
            "future no-CE reference; no change label is assigned to the historical bar. "
            "Source data retains exact carrier variables, constituent regions, scenario "
            "and year references, and calculation numerators, denominators, conversion "
            "factors and offsets. Missing components propagate through the stack and its "
            "total; an incomplete or nonpositive reference yields an undefined percentage. " +
            assumption)
        return ReportFigure("fig06_energy_mix", "6", title,
                            "How do circular-economy settings change global industrial energy demand and its carrier composition?",
                            caption, fig, data, 6)
