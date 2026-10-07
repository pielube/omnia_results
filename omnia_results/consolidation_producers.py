"""Paired producer endpoints for the two consolidated no-CE pathways."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np

from .consolidation_data import PRODUCTION_SCENARIOS
from .metrics import SECTORS
from .report import ReportFigure


class ProducerFigures:
    """Plot directly calculated producer groups with a fixed ranking cohort."""

    def leading_producers(self) -> ReportFigure:
        data = self.data.leading_producers()
        plotted = data.loc[data.Role == "plotted"]
        first, last = min(self.config["years"]), self.config["ranking_year"]
        top_n = self.config["top_n"]
        scale = self.config.get("producer_axis_scale", "log")
        if scale not in {"log", "linear"}:
            raise ValueError("producer_axis_scale must be 'log' or 'linear'")

        title = "The geography of industrial production"
        fig, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 165 / 25.4), squeeze=False)
        fig.subplots_adjust(left=.178, right=.985, bottom=.185, top=.73, wspace=1.09)
        fig.text(.035, .977, title, fontsize=8, weight="bold", va="top")
        fig.text(.035, .936,
                 f"No circular economy | {first}–{last} | fixed {last} ranking: NDC+LTT",
                 fontsize=6.7, color="#52616A", va="top")

        for column, sector in enumerate(SECTORS):
            ax, panel = axes[0, column], chr(97 + column)
            selected = plotted.loc[plotted.Sector == sector.label]
            ordered = (selected.loc[selected.Scenario == "baseline_noce", ["Group", "DisplayOrder"]]
                       .drop_duplicates().sort_values("DisplayOrder"))
            labels = ordered.Group.tolist()
            if not labels:
                raise ValueError(f"No producer groups available for {sector.label}")
            positions = np.arange(len(labels), dtype=float)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#DDE3E6", linewidth=.35)
            ax.tick_params(labelsize=6, pad=3)
            ax.set_yticks(positions, labels=labels, fontsize=6)
            ax.tick_params(axis="y", length=0, pad=4)
            ax.set_ylim(len(labels) - .35, -.65)
            ax.set_title(sector.label, loc="left", fontsize=7, pad=25)
            ax.text(-.14, 1.085, panel, transform=ax.transAxes, fontsize=8,
                    weight="bold", va="bottom", ha="left")

            finite = selected.Value.to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            if scale == "log":
                if (finite <= 0).any():
                    plt.close(fig)
                    raise ValueError(
                        f"Log producer axes require positive endpoints ({sector.label}); "
                        "set producer_axis_scale to 'linear' for zero or negative values")
                ax.set_xscale("log")
                if finite.size:
                    ax.set_xlim(float(finite.min()) / 1.7, float(finite.max()) * 1.45)
                else:
                    ax.set_xlim(.1, 1.)
                low, high = ax.get_xlim()
                decades = np.arange(np.ceil(np.log10(low)), np.floor(np.log10(high)) + 1)
                if len(decades):
                    stride = max(1, int(np.ceil(len(decades) / 3)))
                    ax.xaxis.set_major_locator(ticker.FixedLocator(10. ** decades[::stride]))
                else:
                    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=3))
                ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(2, 5), numticks=12))
                ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            else:
                low = min(float(finite.min()), 0.) if finite.size else 0.
                high = max(float(finite.max()), 0.) if finite.size else 1.
                span = max(high - low, 1.)
                ax.set_xlim(low - .05 * span if low < 0 else 0., high + .12 * span)
                ax.xaxis.set_major_locator(ticker.MaxNLocator(3, min_n_ticks=3))
                if low < 0:
                    ax.axvline(0, color="#63717A", linewidth=.6)
            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.4g}"))
            ax.set_xlabel("Production (Mt yr$^{-1}$)" +
                          ("\nLogarithmic scale" if scale == "log" else ""),
                          fontsize=6.5, labelpad=4)

            for pathway, scenario in enumerate(PRODUCTION_SCENARIOS):
                values = (selected.loc[selected.Scenario == scenario]
                          .pivot(index="Group", columns="Year", values="Value")
                          .reindex(index=labels, columns=[first, last]))
                y = positions + (-.145 if pathway == 0 else .145)
                earlier = values[first].to_numpy(dtype=float)
                later = values[last].to_numpy(dtype=float)
                for position, a, b in zip(y, earlier, later):
                    if np.isfinite(a) and np.isfinite(b):
                        ax.plot([a, b], [position, position], color="#A7B3BA",
                                linestyle="-" if pathway == 0 else (0, (3, 2)),
                                linewidth=.8, zorder=2)
                early_present = np.isfinite(earlier)
                late_present = np.isfinite(later)
                ax.scatter(earlier[early_present], y[early_present], facecolors="white",
                           edgecolors="#53626A", marker="o", s=11.5, linewidths=.65,
                           zorder=3)
                ax.scatter(later[late_present], y[late_present], facecolors=sector.color,
                           edgecolors="white", marker="D", s=13, linewidths=.35,
                           zorder=4)

            # The unranked remainder follows the fixed producer cohort.
            remainder = np.flatnonzero(ordered.Group.to_numpy() == "Other model regions")
            if len(remainder):
                ax.axhline(float(remainder[0]) - .5, color="#B7C1C6",
                           linewidth=.55, linestyle=(0, (2, 2)))
            coverage = data.loc[(data.Sector == sector.label) & (data.Role == "context") &
                                (data.Series == "Selected producer coverage") & (data.Year == last)]
            shares = []
            for scenario in PRODUCTION_SCENARIOS:
                observation = coverage.loc[coverage.Scenario == scenario, "Value"]
                value = float(observation.iloc[0]) if len(observation) else float("nan")
                shares.append(f"{value:.1f}%" if np.isfinite(value) else "n/a")
            ax.text(0, 1.022, f"Top {top_n} share: {' / '.join(shares)}",
                    transform=ax.transAxes, fontsize=5.5, color="#52616A", va="bottom")

        handles = [
            Line2D([], [], color="#53626A", marker="o", markerfacecolor="white",
                   markeredgewidth=.65, markersize=3.4, linestyle="none", label=str(first)),
            Line2D([], [], color="#53626A", marker="D", markeredgecolor="white",
                   markeredgewidth=.35, markersize=3.6, linestyle="none",
                   label=f"{last} (sector colour)"),
            Line2D([], [], color="#A7B3BA", linewidth=.9, linestyle="-", label="Upper: NDC"),
            Line2D([], [], color="#A7B3BA", linewidth=.9, linestyle=(0, (3, 2)), label="Lower: NDC+LTT"),
        ]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.035, .879), ncol=4,
                   fontsize=6, columnspacing=1.5, handlelength=2.1, handletextpad=.5,
                   frameon=False)
        fig.text(.035, .088, f"Selected producer shares in {last}: NDC / NDC+LTT. Fixed regions in both pathways.",
                 fontsize=5.7, color="#52616A")
        fig.text(.035, .054,
                 "Europe combines five model regions. Indonesia group = Indonesia, Philippines and Viet Nam.",
                 fontsize=5.5, color="#52616A")
        assumption = ("Draft assumption: missing additive activity = zero."
                      if self.config["missing_activity"] == "zero" else
                      "Missing activity retained; missing endpoints appear as gaps.")
        fig.text(.035, .021, assumption, fontsize=5.5, color="#63717A", va="bottom")
        caption = (
            f"Production in {first} and {last} in the {top_n} largest producer groups, selected "
            f"and ordered once by {last} output under NDC+LTT (baseline_noce). The same fixed "
            "groups and ordering are used under NDC (ndc_noce), so differences between pathways "
            "are compared for the same regions. Each group has two vertically offset endpoint "
            "pairs: NDC above with a solid connector, and NDC+LTT below with a dashed connector. "
            f"Open circles denote {first}; filled sector-coloured diamonds denote {last}. "
            "Panels show cement output (a), total primary-plus-secondary steel output (b), "
            "and total primary-plus-secondary aluminium output (c). Other model regions sums "
            "all producer groups outside the selected cohort and is not included in the ranking. "
            f"Panel annotations give the fixed cohort's shares of global {last} production "
            "in NDC / NDC+LTT order. Europe combines five model regions (ENE, ENW, EUE, EUM "
            "and EUW). Indonesia group combines Indonesia, Philippines and Viet Nam. "
            f"Values are extracted directly from {self.config['input']} using exact production "
            "variable and Mt/yr selections; source variables and constituent model regions are "
            "recorded with every observation. Horizontal ranges include both pathways within "
            "each sector, and sectors have different ranges. " +
            ("Logarithmic axes compare production magnitudes. " if scale == "log" else
             "Linear axes retain zero and negative source values. ") + assumption)
        return ReportFigure("fig02_leading_producers", "2", title,
                            "How does production change in the same leading producer regions under NDC and NDC+LTT?",
                            caption, fig, data, 3)
