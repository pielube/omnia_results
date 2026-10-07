"""Material-production bars across CE settings with a historical reference."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Patch
import numpy as np

from .consolidation_data import MATERIAL_CLIMATES, climate_label
from .metrics import SECTORS
from .report import ReportFigure


PRODUCT_COLORS = {"Cement": "#0072B2", "Clinker": "#A7B3BA",
                  "Primary": "#0072B2", "Secondary": "#72C4AF"}


def _percentage_label(value: float) -> str:
    if abs(value) < .05:
        return "0%"
    return f"{value:+.1f}%".replace("-", "\N{MINUS SIGN}")


class MaterialComparisonFigures:
    """Show six production bars per panel, ordered consistently across pathways."""

    def material_production(self) -> ReportFigure:
        data = self.data.material_production()
        production = data.loc[data.Metric == "Material production"]
        title = "Circular economy changes material demand and primary supply"
        historical = self.config.get("historical_year", 2019)
        future = self.config.get("comparison_year", 2050)
        fig, axes = plt.subplots(2, 3, figsize=(180 / 25.4, 175 / 25.4), squeeze=False)
        fig.subplots_adjust(left=.085, right=.985, bottom=.20, top=.76,
                            wspace=.48, hspace=.60)
        fig.text(.035, .977, title, fontsize=8, weight="bold", va="top")
        fig.text(.035, .941, f"Global production | {historical} reference and {future} CE scenarios",
                 fontsize=6.7, color="#52616A", va="top")
        # The extra space after the historical observation separates dates
        # without displacing any values or implying a continuous time axis.
        positions = np.array([0., 1.45, 2.45, 3.45, 4.45, 5.45])
        for column, sector in enumerate(SECTORS):
            sector_data = production.loc[production.Sector == sector.label]
            finite = sector_data.Value.to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            low = min(0., float(finite.min())) if finite.size else 0.
            high = max(0., float(finite.max())) if finite.size else 1.
            if high <= low:
                high = low + 1.
            span = high - low
            # Reserve consistent annotation space within the same sector scale.
            limits = (low - .04 * span if low < 0 else 0., high + .18 * span)
            for row, climate in enumerate(MATERIAL_CLIMATES):
                ax, panel = axes[row, column], chr(97 + row * 3 + column)
                selected = sector_data.loc[sector_data.Panel == panel]
                bars = (selected[["BarOrder", "BarLabel"]].drop_duplicates()
                        .set_index("BarOrder").reindex(range(6)))
                ax.spines[["top", "right"]].set_visible(False)
                ax.set_axisbelow(True)
                ax.grid(axis="y", color="#DDE3E6", linewidth=.35)
                ax.tick_params(labelsize=6.1, pad=3)
                ax.set_xlim(-.52, 5.99)
                ax.set_ylim(*limits)
                ax.yaxis.set_major_locator(ticker.MaxNLocator(4, min_n_ticks=3))
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.4g}"))
                ax.set_ylabel("Production (Mt yr$^{-1}$)", fontsize=6.8, labelpad=3)
                labels = [label.replace("–", "\n").replace("No CE", "No\nCE")
                          for label in bars.BarLabel.fillna("").tolist()]
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

                def component(name):
                    return (selected.loc[selected.Series == name]
                            .set_index("BarOrder")["Value"].reindex(range(6))
                            .to_numpy(dtype=float))

                if sector.key == "cement":
                    cement, clinker = component("Cement"), component("Clinker")
                    ax.bar(positions - .155, cement, .275, color=PRODUCT_COLORS["Cement"],
                           edgecolor="white", linewidth=.4, label="Cement")
                    ax.bar(positions + .155, clinker, .275, color=PRODUCT_COLORS["Clinker"],
                           edgecolor="white", linewidth=.4, label="Clinker")
                    totals = cement
                    annotation_x = positions - .155
                else:
                    primary, secondary = component("Primary"), component("Secondary")
                    ax.bar(positions, primary, .64, color=PRODUCT_COLORS["Primary"],
                           edgecolor="white", linewidth=.4, label="Primary")
                    ax.bar(positions, secondary, .64, bottom=primary,
                           color=PRODUCT_COLORS["Secondary"], edgecolor="white", linewidth=.4,
                           label="Secondary")
                    totals = component("Total")
                    annotation_x = positions
                    shares = (data.loc[(data.Panel == panel) &
                                       (data.Metric == "Secondary production share")]
                              .set_index("BarOrder")["Value"].reindex(range(6)))
                    for order, x, p, s in zip(range(6), positions, primary, secondary):
                        share = float(shares.loc[order])
                        if np.isfinite(share) and np.isfinite(p) and np.isfinite(s) and s > .12 * span:
                            ax.text(x, p + s / 2, f"{share:.0f}%", fontsize=5.6,
                                    ha="center", va="center", color="#173F39")

                changes = (data.loc[(data.Panel == panel) &
                                    (data.Metric == "Production change from no CE")]
                           .set_index("BarOrder")["Value"].reindex(range(1, 6)))
                for order in range(1, 6):
                    value, change = float(totals[order]), float(changes.loc[order])
                    if np.isfinite(value) and np.isfinite(change):
                        ax.text(annotation_x[order], value + span * .036,
                                "ref." if order == 1 else _percentage_label(change),
                                ha="center", va="bottom", fontsize=5.5, color="#202D35")

        handles = [Patch(facecolor=PRODUCT_COLORS["Cement"], label="Cement / primary metal"),
                   Patch(facecolor=PRODUCT_COLORS["Clinker"], label="Clinker"),
                   Patch(facecolor=PRODUCT_COLORS["Secondary"], label="Secondary metal")]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.035, .882), ncol=3,
                   fontsize=6, columnspacing=1.6, handlelength=1.4, handletextpad=.5,
                   frameon=False)
        fig.text(.035, .112,
                 f"{historical}: no CE. All other bars: {future}. Labels above: change from the same pathway's {future} no CE.",
                 fontsize=5.5, color="#52616A")
        fig.text(.035, .087,
                 "Changes refer to cement or total metal output. Within metal stacks: secondary-production shares.",
                 fontsize=5.5, color="#52616A")
        fig.text(.035, .057,
                 "M/H = medium/high CE; EU+ = EU + international partners (17 model regions).",
                 fontsize=5.5, color="#52616A")
        assumption = ("Draft assumption: missing additive activity = zero."
                      if self.config["missing_activity"] == "zero" else
                      "Missing activity retained; unavailable bars and annotations appear as gaps.")
        fig.text(.035, .023, assumption, fontsize=5.5, color="#63717A", va="bottom")
        caption = (
            f"Global production in {historical} without CE, followed by five {future} CE settings. "
            "The upper row shows NDC (ndc_* scenarios) and the lower row NDC+LTT (baseline_* "
            f"scenarios). The historical {historical} observation comes from each pathway's "
            f"no-CE scenario; it is separated visually from the {future} policy bars. The "
            "future bars are ordered no CE, medium CE in the EU, high CE in the EU, medium "
            "CE in the EU plus international partners, and high CE in the EU plus partners. "
            "EU+ indicates these 17 participating model regions rather than global CE adoption. "
            "Cement and clinker are distinct products shown side by side and never added. "
            "Steel and aluminium bars stack primary and secondary output; their totals are "
            "the sum of the two routes. Labels within sufficiently large metal segments "
            "give the secondary-production share, including the historical reference. "
            f"Future labels above bars give the percentage change in cement or total metal "
            f"production from no CE under the same climate pathway in {future}, using "
            "100 × (scenario production / reference production − 1); ref. identifies the "
            "no-CE reference. No percentage change is assigned to the historical bar. "
            "Each sector uses the same vertical limits across rows, including all historical "
            f"and future values. Production is extracted directly from {self.config['input']} "
            f"using exact variable and Mt/yr selections and summing each of the "
            f"{len(self.config['regions'])} model regions once. Source variables, source "
            "regions, reference scenario and year, and the numerator, denominator, conversion "
            "factor and offset for each annotation are retained in the source data. An "
            "unavailable value is not plotted, and a nonpositive reference denominator yields "
            "an undefined percentage. " + assumption)
        return ReportFigure("fig05_material_production", "5", title,
                            "How do circular-economy settings change material production relative to no CE and the historical reference?",
                            caption, fig, data, 6)
