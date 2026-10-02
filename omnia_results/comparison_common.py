"""Shared layout and source-data recording for the CE comparison figures."""

import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd

from .report import ReportFigure
from .plots import STYLE


CLIMATE_LABELS = {"baseline": "Baseline", "ndc": "NDC"}
POLICY_SHORT = {"noce": "No CE", "medce_eu": "M–EU", "highce_eu": "H–EU",
                "medce_gbl": "M–EU+", "highce_gbl": "H–EU+"}
POLICY_COLORS = {"noce": "#66717A", "medce_eu": "#0072B2", "medce_gbl": "#0072B2",
                 "highce_eu": "#D55E00", "highce_gbl": "#D55E00"}
POLICY_MARKERS = {"noce": "s", "medce_eu": "o", "highce_eu": "o",
                  "medce_gbl": "D", "highce_gbl": "D"}
POLICY_STYLES = {"noce": "-", "medce_eu": "--", "highce_eu": "--",
                 "medce_gbl": "-", "highce_gbl": "-"}
COMPARISON_STYLE = {**STYLE, "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"]}


class ComparisonBuilderBase:
    def __init__(self, data, config):
        self.data, self.config = data, config
        self.year = data.comparison_year
        self.rows = []

    def canvas(self, title, height_mm, rows=1, cols=1, *, left=.09, right=.98,
               bottom=.18, top=.78, wspace=.4, hspace=.55, subtitle=None):
        self.rows = []
        fig, axes = plt.subplots(rows, cols, figsize=(180 / 25.4, height_mm / 25.4), squeeze=False)
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                            wspace=wspace, hspace=hspace)
        fig.text(.025, .975, title, fontsize=7.5, weight="bold", va="top")
        fig.text(.025, .924, subtitle or f"Global outcomes · {self.year} · Baseline and NDC",
                 fontsize=6.3, va="top", color="#52616A")
        fig.text(.025, .044, "M/H = medium/high CE; EU+ = EU + international partners (17 model regions).",
                 fontsize=5.5, color="#52616A", va="bottom")
        missing = ("Draft assumption: missing additive activity = zero." if self.config["missing_activity"] == "zero"
                   else "Missing activity preserved; incomplete aggregates remain gaps.")
        fig.text(.025, .014, missing + " Baseline and NDC are model climate pathways.",
                 fontsize=5.3, color="#52616A", va="bottom")
        return fig, axes

    def record(self, panel, scenario, metric, component, value, unit, *, year=None,
               scope="Global", role="plotted", reference=None):
        climate, policy = scenario.split("_", 1)
        self.rows.append({"Scenario": scenario, "Climate": self.data.climate_labels[climate],
                          "Policy": self.data.policy_labels[policy], "Panel": panel, "Scope": scope,
                          "Metric": metric, "Component": component, "Year": int(year or self.year),
                          "Value": float(value) if pd.notna(value) else float("nan"), "Unit": unit,
                          "ReferenceScenario": reference or "", "Role": role})

    def record_series(self, panel, scenario, metric, component, series, unit, **kwargs):
        for year, value in series.items():
            self.record(panel, scenario, metric, component, value, unit, year=int(year), **kwargs)

    def finish(self, fig, number, slug, title, question, caption, panels):
        return ReportFigure(slug, str(number), title, question, caption, fig,
                            pd.DataFrame(self.rows), panels)

    @staticmethod
    def clean_axes(ax, *, time=False, years=None):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#DFE4E6", linewidth=.35)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=6, pad=3)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4, min_n_ticks=3))
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.4g}"))
        if time:
            lo, hi = min(years), max(years)
            ax.set_xticks(sorted({lo, hi, *[y for y in years if y % 10 == 0]}))
            ax.set_xlim(lo - .6, hi + .6)
            ax.set_xlabel("Year", fontsize=6.5)

    @staticmethod
    def panel_title(ax, letter, title):
        ax.set_title(title, loc="left", fontsize=6.7, pad=9)
        ax.text(-.135, 1.035, letter, transform=ax.transAxes, weight="bold", fontsize=7.5,
                va="bottom", ha="left")

    def policy_ticks(self, ax):
        ax.set_xticks(range(len(self.data.policies)), [POLICY_SHORT[p] for p in self.data.policies],
                      fontsize=5.5)
        ax.tick_params(axis="x", length=0, pad=4)

    def scenario_legend(self, fig, **kwargs):
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=POLICY_COLORS[p], marker=POLICY_MARKERS[p],
                          linestyle=POLICY_STYLES[p], linewidth=1, markersize=3,
                          markerfacecolor="white" if p.endswith("_eu") else POLICY_COLORS[p],
                          label=POLICY_SHORT[p]) for p in self.data.policies]
        defaults = dict(loc="upper left", bbox_to_anchor=(.07, .865), ncol=5,
                        fontsize=6, handlelength=2.4, columnspacing=1.4)
        defaults.update(kwargs)
        return fig.legend(handles=handles, **defaults)
