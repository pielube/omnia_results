"""A curated report: seven main figures and two supporting figures."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib import ticker
import numpy as np
import pandas as pd

from .metrics import Results, SECTORS
from .plots import STYLE
from .report_data import ReportData
from .report_gallery import write_report_gallery


@dataclass
class ReportFigure:
    slug: str
    number: str
    title: str
    question: str
    caption: str
    figure: object
    data: pd.DataFrame
    panel_count: int
    section: str = "Main report"


class ReportBuilder:
    def __init__(self, results: Results, config: dict):
        self.results, self.config = results, config
        self.data = ReportData(results, config)
        self.label = config.get("scenario_labels", {}).get(results.scenario, results.scenario)
        self.rows: list[dict] = []

    def record(self, panel, series, values, unit, scope="Global", role="plotted"):
        self.rows.extend({"Scenario": self.results.scenario, "Panel": panel,
                          "Scope": scope, "Series": series, "Year": int(year),
                          "Value": value, "Unit": unit, "Role": role}
                         for year, value in values.items())

    def canvas(self, title, height_mm, rows=1, cols=1, *, left=.08, right=.97,
               bottom=.17, top=.78, wspace=.44, hspace=.60):
        self.rows = []
        fig, axes = plt.subplots(rows, cols, figsize=(183 / 25.4, height_mm / 25.4), squeeze=False)
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                            wspace=wspace, hspace=hspace)
        fig.text(.035, .975, title, va="top", weight="bold", fontsize=8)
        fig.text(.035, .925, f"{self.label}  |  {min(self.results.years)}–{max(self.results.years)}",
                 va="top", fontsize=6.5, color="#52616A")
        if self.results.missing_activity == "zero":
            note = "Draft assumption: blank activity values = zero; reported prices and intensities retain gaps."
        else:
            note = "Missing activity values retained; incomplete aggregates appear as gaps."
        fig.text(.035, .022, note, fontsize=5.5, color="#63717A", va="bottom")
        return fig, axes

    def finish(self, fig, number, slug, title, question, caption, panels, support=False):
        return ReportFigure(slug, number, title, question, caption, fig, pd.DataFrame(self.rows),
                            panels, "Supporting figures" if support else "Main report")

    @staticmethod
    def clean_axes(ax, *, time=True, years=None):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y" if time else "x", color="#DDE3E6", linewidth=.35)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=6, pad=3)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4, min_n_ticks=3))
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.5g}"))
        if time:
            years = list(years)
            lo, hi = min(years), max(years)
            ticks = [lo, *[y for y in years if y % 10 == 0 and y not in (lo, hi)], hi]
            ax.set_xticks(sorted(set(ticks)))
            ax.set_xlim(lo - .7, hi + .7)
            ax.set_xlabel("Year", fontsize=6.5, labelpad=3)

    @staticmethod
    def panel_title(ax, letter, title, *, pad=9, letter_y=1.055):
        ax.set_title(title, loc="left", fontsize=7, pad=pad)
        ax.text(-.14, letter_y, letter, transform=ax.transAxes, fontweight="bold", fontsize=8,
                va="bottom", ha="left")

    def line(self, ax, panel, label, values, unit, color, marker="o", style="-", scope="Global"):
        ax.plot(values.index, values.to_numpy(dtype=float), color=color, linestyle=style,
                marker=marker, markersize=2.6, markerfacecolor="white", markeredgewidth=.65,
                linewidth=1.0, label=label, clip_on=False)
        self.record(panel, label, values, unit, scope)

    @staticmethod
    def zero_axis(ax):
        low, high = ax.get_ylim()
        ax.set_ylim(min(0, low), max(high, .01))

    def production(self):
        title = "Material production and production routes"
        fig, axes = self.canvas(title, 83, cols=3, top=.71, bottom=.19, wspace=.45)
        for i, sector in enumerate(SECTORS):
            ax, panel = axes[0, i], chr(97 + i)
            self.clean_axes(ax, years=self.results.years)
            self.panel_title(ax, panel, sector.label)
            for j, (label, values) in enumerate(self.results.production(sector).items()):
                self.line(ax, panel, label, values, "Mt/yr", sector.color if j == 0 else "#53626A",
                          ["o", "s", "^"][j], ["-", "--", ":"][j])
            ax.set_ylabel("Production (Mt yr⁻¹)", fontsize=6.5)
            self.zero_axis(ax)
            ax.legend(loc="lower left", bbox_to_anchor=(0, 1.15), borderaxespad=0,
                      ncol=len(ax.lines), fontsize=5.5, columnspacing=.7, handlelength=1.6,
                      handletextpad=.3)
        return self.finish(fig, "1", "fig01_production", title,
                           "How do global material output and production routes evolve?",
                           "Global production of (a) cement and clinker, (b) iron and steel, and (c) aluminium. "
                           "Steel and aluminium totals are primary plus secondary production. Cement and clinker are distinct products and are not added. "
                           "Each sector has its own vertical scale; markers show the supplied model years.", 3)

    def endpoint_plot(self, ax, sector, panel, metric="production"):
        if metric not in {"production", "reported_intensity", "emissions"}:
            raise ValueError(f"Unknown endpoint metric: {metric}")
        intensity, emissions = metric == "reported_intensity", metric == "emissions"
        groups = self.data.emissions_groups(sector) if emissions else self.data.production_groups(sector)
        cohort = self.data.ranked_emitters(sector) if emissions else self.data.ranked_producers(sector)
        first, last = min(self.results.years), self.config["ranking_year"]
        if intensity:
            values = self.data.reported_intensity_groups(sector).loc[cohort, [first, last]]
            labels = list(cohort)
            unit = "GJ/t"
        else:
            values = groups.loc[cohort, [first, last]].copy()
            remaining = groups.drop(index=cohort)
            values.loc["Other model regions"] = remaining.sum(axis=0, min_count=len(remaining)).loc[[first, last]] if len(remaining) else 0
            labels = list(values.index)
            unit = "MtCO2e/yr" if emissions else "Mt/yr"
        self.clean_axes(ax, time=False)
        positions = np.arange(len(labels))
        x0, x1 = values[first].to_numpy(dtype=float), values[last].to_numpy(dtype=float)
        for j, (a, b) in enumerate(zip(x0, x1)):
            if np.isfinite(a) and np.isfinite(b):
                ax.plot([a, b], [j, j], color="#ABB5BB", linewidth=.8, zorder=2)
        ax.scatter(x0, positions, facecolors="white", edgecolors="#56636B", marker="o", s=13, linewidths=.7, zorder=3)
        ax.scatter(x1, positions, facecolors=sector.color, edgecolors="white", marker="D", s=15, linewidths=.35, zorder=4)
        ax.set_yticks(positions, labels=labels, fontsize=5.8)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.set_ylim(len(labels) - .3, -.75)
        scale_key = "emitter_axis_scale" if emissions else "producer_axis_scale"
        scale = "linear" if intensity else self.config.get(scale_key, "log")
        if scale not in {"linear", "log"}:
            raise ValueError(f"{scale_key} must be linear or log")
        if scale == "log":
            finite = values.to_numpy()[np.isfinite(values.to_numpy())]
            if (finite <= 0).any():
                raise ValueError(f"Logarithmic axes require positive endpoints; set {scale_key} to linear")
            ax.set_xscale("log")
            if finite.size:
                ax.set_xlim(finite.min() / 1.6, finite.max() * 1.5)
            ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=6))
            if emissions and finite.size:
                lo, hi = ax.get_xlim()
                decades = np.arange(np.ceil(np.log10(lo)), np.floor(np.log10(hi)) + 1)
                stride = max(1, int(np.ceil(len(decades) / 3)))
                ax.xaxis.set_major_locator(ticker.FixedLocator(10.0 ** decades[::stride]))
            ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(2, 5), numticks=12))
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        else:
            finite = values.to_numpy()[np.isfinite(values.to_numpy())]
            if finite.size and finite.min() < 0:
                ax.axvline(0, color="#63717A", linewidth=.6)
            else:
                ax.set_xlim(left=0)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(3, min_n_ticks=3))
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.4g}"))
        xlabel = ("Reported intensity (GJ t⁻¹)" if intensity else
                  "GHG emissions (Mt CO₂e yr⁻¹)" if emissions else "Production (Mt yr⁻¹)")
        ax.set_xlabel(xlabel + ("\nLogarithmic scale" if scale == "log" else ""), fontsize=6.5)
        self.panel_title(ax, panel, sector.label, pad=22 if not intensity else 9,
                         letter_y=1.105 if not intensity else 1.055)
        if not intensity:
            ax.axhline(len(labels) - 1.5, color="#B7C1C6", linewidth=.55, linestyle=(0, (2, 2)))
            coverage = self.data.emitter_coverage(sector) if emissions else self.data.producer_coverage(sector)
            quantity = "emissions" if emissions else "output"
            share_label = f"{coverage.loc[last]:.2f}" if emissions else f"{coverage.loc[last]:.1f}"
            coverage_text = (f"Top {len(cohort)}: {share_label}% " +
                             (f"in {last}" if emissions else f"of {last} {quantity}")
                             if pd.notna(coverage.loc[last]) else f"{len(cohort)} groups; coverage unavailable")
            ax.text(0, 1.025, coverage_text,
                    transform=ax.transAxes, fontsize=5.5, color="#52616A", va="bottom")
            self.record(panel, "Selected emitter coverage" if emissions else "Selected producer coverage", coverage,
                        "%", "Selected emitters" if emissions else "Selected producers", role="context")
        for label in labels:
            self.record(panel, label, values.loc[label], unit, "Emitter regions" if emissions else "Producer regions")
        return first, last

    def ranked_figure(self, metric="production"):
        intensity, emissions = metric == "reported_intensity", metric == "emissions"
        title = ("Energy intensity in the leading producer regions" if intensity else
                 "The geography of industrial emissions" if emissions else "The geography of industrial production")
        # Each column reserves its own left label margin; wide gaps are intentional.
        fig, axes = self.canvas(title, 149, cols=3, left=.175, right=.985,
                                top=.755, bottom=.16, wspace=1.04)
        for i, sector in enumerate(SECTORS):
            first, last = self.endpoint_plot(axes[0, i], sector, chr(97 + i), metric)
        handles = [Line2D([], [], color="#56636B", marker="o", markerfacecolor="white", linestyle="none", markersize=4, label=str(first)),
                   Line2D([], [], color="#53626A", marker="D", linestyle="none", markersize=4, label=f"{last} (sector colour)")]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.035, .872), ncol=2,
                   fontsize=6, handletextpad=.5, columnspacing=2)
        fig.text(.035, .066, "Europe combines five model regions. Indonesia group = Indonesia, Philippines and Viet Nam.",
                 fontsize=5.5, color="#52616A")
        group_type, quantity = ("emitter", "emissions") if emissions else ("producer", "production")
        cohort_description = (f"The {self.config['top_n']} largest {group_type} groups in {last}" if self.results.missing_activity == "zero" else
                              f"Up to {self.config['top_n']} largest groups with complete {last} {quantity} data")
        caption = (f"{cohort_description}, ranked separately for each sector. "
                   f"The same {last} cohort and ordering are used for both endpoint years ({first} and {last}). "
                   "Europe aggregates ENE, ENW, EUE, EUM and EUW; remaining groups preserve individual model regions. "
                   "China denotes the CHN model region (including Hong Kong, Macao and Taiwan). "
                   "Indonesia group includes Indonesia, Philippines and Viet Nam. Exact memberships accompany the figures. ")
        if intensity:
            caption += ("Dots show supplied regional energy intensity, converted to GJ/t. Europe's source intensities are weighted by contemporaneous sector production. "
                        "These reported intensities use the source definition and differ from the derived global final-energy/production intensities in Figure 4. Missing intensity values remain gaps.")
            return self.finish(fig, "S1", "figS1_producer_intensity", title,
                               "How do energy intensities compare within the same leading-producer cohort?", caption, 3, True)
        caption += ("The final row is the unranked sum of all remaining groups, retaining complete global accounting. "
                    "Open circles show the first year; filled diamonds show the ranking year. Thin connectors indicate endpoint changes, not intervening trajectories.")
        scale_key = "emitter_axis_scale" if emissions else "producer_axis_scale"
        if self.config.get(scale_key, "log") == "log":
            caption += f" {quantity.capitalize()} axes are logarithmic to make changes visible across groups spanning several orders of magnitude."
        if emissions:
            caption += (" Rankings use absolute sector GHG emissions in MtCO2e/yr, independently of the production ranking in Figure 2. "
                        "Coverage is the selected groups' share of global emissions for that sector, not all-industry emissions. "
                        "The source emissions totals are used as reported; CO2 capture is not subtracted again.")
            return self.finish(fig, "7", "fig07_leading_emitters", title,
                               "Where are sector emissions concentrated, and how do they change between 2019 and 2050?", caption, 3)
        return self.finish(fig, "2", "fig02_leading_producers", title,
                           "Where is production concentrated, and how does it move between 2019 and 2050?", caption, 3)

    def carbon_prices(self):
        title = "Regional carbon-price indices"
        fig, axes = self.canvas(title, 108, left=.105, right=.69, top=.81, bottom=.18)
        ax = axes[0, 0]
        prices = self.data.carbon_prices()
        colors = ["#0072B2", "#D55E00", "#222222", "#AA4499", "#009E73", "#6A8B45", "#8C564B", "#5E69AE", "#9B7000"]
        markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
        self.clean_axes(ax, years=self.results.years)
        for i, (label, values) in enumerate(prices.iterrows()):
            self.line(ax, "a", label, values, "USD_2010/tCO2e", colors[i % len(colors)],
                      markers[i % len(markers)], ["-", "--", "-."][i % 3], scope="Macroregions")
        ax.set_ylabel("Carbon price (USD₂₀₁₀ per t CO₂e)", fontsize=7)
        self.zero_axis(ax)
        # Place labels in the right margin, spaced independently of data values.
        last = max(self.results.years)
        endpoints = prices[last].dropna().sort_values()
        low, high = ax.get_ylim()
        positions = (endpoints.to_numpy() - low) / (high - low)
        gap = .071
        for i in range(1, len(positions)):
            positions[i] = max(positions[i], positions[i - 1] + gap)
        if len(positions) and positions[-1] > .96:
            positions[-1] = .96
            for i in range(len(positions) - 2, -1, -1):
                positions[i] = min(positions[i], positions[i + 1] - gap)
        for (name, endpoint), position in zip(endpoints.items(), positions):
            i = prices.index.get_loc(name)
            ax.annotate(name, xy=(last, endpoint), xycoords="data", xytext=(1.06, position),
                        textcoords="axes fraction", va="center", fontsize=6.3, color="#233139",
                        arrowprops={"arrowstyle": "-", "color": colors[i % len(colors)], "lw": .7},
                        annotation_clip=False)
        fig.text(.105, .845, f"Fixed {self.config['price_weight_year']} industrial-emissions weights", fontsize=6, color="#52616A")
        caption = (f"{len(prices)} disjoint groups cover all {len(self.results.regions)} model regions. Each line is an index calculated as the weighted mean of model-region carbon prices, "
                   f"using fixed {self.config['price_weight_year']} industrial greenhouse-gas emissions as weights. "
                   "The weights do not change with decarbonisation or relocation of industry, so changes reflect the price paths. "
                   "These indices are not uniform regional policies, population-weighted prices, or averages across countries. "
                   "North America includes the United States, Canada and Mexico; Europe contains the five European model regions; China is the CHN model region. "
                   "Exact membership and numerical weights are supplied in CSV files.")
        return self.finish(fig, "3", "fig03_carbon_prices", title,
                           "How do carbon-price paths differ across economically meaningful regional groups?", caption, 1)

    def energy(self):
        title = "Industrial energy requirements"
        fig, axes = self.canvas(title, 141, rows=2, cols=3, top=.815, bottom=.115, hspace=.69)
        for col, sector in enumerate(SECTORS):
            for row, (metric, label, unit) in enumerate([
                    ("energy", "Final energy (EJ yr⁻¹)", "EJ/yr"),
                    ("energy_intensity", "Energy intensity (GJ t⁻¹)", "GJ/t")]):
                ax, panel = axes[row, col], chr(97 + row * 3 + col)
                self.clean_axes(ax, years=self.results.years)
                self.panel_title(ax, panel, sector.label)
                self.line(ax, panel, sector.label, self.results.global_metric(metric, sector), unit,
                          sector.color, sector.marker, sector.linestyle)
                self.zero_axis(ax)
                ax.set_ylabel(label, fontsize=6.5)
        return self.finish(fig, "4", "fig04_energy", title,
                           "How do total energy requirements and energy per tonne evolve together?",
                           "Global sectoral final energy (a–c) and derived energy intensity (d–f), with sector columns aligned. "
                           "Energy sums exactly the carrier-level variables listed in ResultsMetrics.docx, excluding nested fuel subcategories. "
                           "Global intensity is global final energy divided by global sector production, converted from EJ/Mt to GJ/t. "
                           "Cement uses cement production; steel and aluminium use primary plus secondary production. "
                           "Each panel has its own vertical scale. Reported regional intensities, a different source definition, are in Figure S1.", 6)

    def emissions(self):
        title = "Industrial emissions and carbon capture"
        fig, axes = self.canvas(title, 141, rows=2, cols=2, top=.785, bottom=.12, wspace=.30, hspace=.66)
        specs = [("emissions", "Sector greenhouse-gas emissions", "GHG emissions (Gt CO₂e yr⁻¹)", "GtCO2e/yr"),
                 ("emissions_share", "Share of total industry emissions", "Share of industry emissions (%)", "%"),
                 ("emissions_intensity", "Emissions per tonne of product", "GHG intensity (t CO₂e t⁻¹)", "tCO2e/t"),
                 ("capture", "Carbon capture", "CO₂ captured (Mt CO₂ yr⁻¹)", "MtCO2/yr")]
        for i, (key, title_i, ylabel, unit) in enumerate(specs):
            ax, panel = axes.flat[i], chr(97 + i)
            self.clean_axes(ax, years=self.results.years)
            self.panel_title(ax, panel, title_i)
            for sector in SECTORS:
                self.line(ax, panel, sector.label, self.results.global_metric(key, sector), unit,
                          sector.color, sector.marker, sector.linestyle)
            ax.set_ylabel(ylabel, fontsize=6.5)
            self.zero_axis(ax)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(.075, .889),
                   ncol=3, fontsize=6.5, columnspacing=2)
        return self.finish(fig, "5", "fig05_emissions", title,
                           "How do sector emissions, emissions intensity and capture change?",
                           "Global sectoral (a) greenhouse-gas emissions, (b) share of total industry greenhouse-gas emissions, "
                           "(c) emissions per tonne of production, and (d) CO₂ capture. Industry shares use Emissions|GHG|Industry as denominator; "
                           "the three sectors are not assumed to exhaust industry. Emissions intensity is a ratio of global sums. "
                           "Capture is shown separately and is not subtracted again from reported emissions. GHG emissions and CO₂ capture use their distinct source units.", 4)

    def costs(self, full_period=False):
        first = min(self.results.years) if full_period else self.config["main_cost_start_year"]
        years = [year for year in self.results.years if year >= first]
        if len(years) < 2:
            raise ValueError("The main cost period needs at least two supplied years")
        title = "Annualised costs: full source period" if full_period else f"Annualised costs, {years[0]}–{years[-1]}"
        fig, axes = self.canvas(title, 90, cols=2, top=.72, bottom=.205, wspace=.35)
        fig.texts[1].set_text(f"{self.label}  |  {years[0]}–{years[-1]}")
        for i, ax in enumerate(axes.flat):
            self.clean_axes(ax, years=years)
            self.panel_title(ax, chr(97 + i), ["Whole system", "Industrial sectors"][i])
        system = self.results.series("Total Annualised Cost", "Millions USD_2010/yr", activity=False) / 1e6
        self.line(axes[0, 0], "a", "System", system.loc[years], "trillion USD_2010/yr", "#273E4B")
        axes[0, 0].set_ylabel("Cost (trillion USD₂₀₁₀ yr⁻¹)", fontsize=6.5)
        for sector in SECTORS:
            self.line(axes[0, 1], "b", sector.label, self.results.global_metric("sector_costs", sector).loc[years],
                      "billion USD_2010/yr", sector.color, sector.marker, sector.linestyle)
        axes[0, 1].set_ylabel("Cost (billion USD₂₀₁₀ yr⁻¹)", fontsize=6.5)
        for ax in axes.flat:
            self.zero_axis(ax)
        axes[0, 1].legend(loc="lower left", bbox_to_anchor=(0, 1.15), ncol=3, borderaxespad=0,
                          fontsize=5.5, columnspacing=.8, handlelength=1.5, handletextpad=.3)
        note = ("Full source period shown; 2019 costs require checking before interpretation." if full_period else
                f"Main view starts in {years[0]} because of anomalous initial-year costs; the full period is retained in Figure S2.")
        fig.text(.035, .073, note, fontsize=5.5, color="#52616A")
        caption = ("Global annualised (a) whole-system and (b) sector costs, in constant 2010 USD. "
                   "The system axis uses trillions and the sector axis billions; costs are not stacked or interpreted as an exhaustive system breakdown. ")
        if full_period:
            caption += ("All years and signed source values are retained. In the supplied baseline export, 2019 steel costs exceed total-system costs, "
                        "and several regional system costs are negative. This figure makes the anomalous values visible without compressing the main report's later-period trends.")
        else:
            caption += (f"The main report shows {years[0]}–{years[-1]} to keep later trends legible. Initial-year cost anomalies are not corrected or discarded: "
                        "the full-period series appear in Figure S2 and the audit identifies negative regional costs. Source cost accounting needs checking before substantive interpretation.")
        return self.finish(fig, "S2" if full_period else "6", "figS2_costs_full_period" if full_period else "fig06_costs",
                           title, "How do system and industrial costs evolve?", caption, 2, full_period)

    def figures(self):
        yield self.production()
        yield self.ranked_figure()
        yield self.carbon_prices()
        yield self.energy()
        yield self.emissions()
        yield self.costs()
        yield self.ranked_figure("emissions")
        yield self.ranked_figure("reported_intensity")
        yield self.costs(full_period=True)


def check_figure_bounds(fig, slug):
    """Fail on cropped labels while preserving exact physical figure dimensions."""
    from matplotlib.text import Text
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    visible_text = list(fig.texts)
    for ax in fig.axes:
        visible_text.extend(child for child in ax.get_children() if isinstance(child, Text))
        visible_text.extend([ax.xaxis.label, ax.yaxis.label])
        for locations, labels, limits in [(ax.get_xticks(), ax.get_xticklabels(), ax.get_xlim()),
                                           (ax.get_yticks(), ax.get_yticklabels(), ax.get_ylim())]:
            lo, hi = sorted(limits)
            visible_text.extend(label for loc, label in zip(locations, labels) if lo <= loc <= hi)
        if ax.get_legend():
            visible_text.extend(ax.get_legend().get_texts())
    for legend in fig.legends:
        visible_text.extend(legend.get_texts())
    for text in visible_text:
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.width and (box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1):
            raise ValueError(f"Label outside canvas in {slug}: {text.get_text()!r}")


def generate_report(config: dict, base: Path):
    source = (base / config["input"]).resolve()
    mapping_path = (base / config["region_mapping"]).resolve()
    mapping = pd.read_csv(mapping_path, encoding="utf-8-sig")
    if not {"region", "country_OMNIA", "ISO3"} <= set(mapping.columns):
        raise ValueError("Region mapping needs region, country_OMNIA and ISO3 columns")
    if not set(config["regions"]) <= set(mapping.region):
        raise ValueError("The supplied country mapping does not cover all model regions")
    for scenario in config["scenarios"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", scenario):
            raise ValueError("Scenario names must be simple identifiers")
        results = Results(source, scenario, config["regions"], config["years"], config["missing_activity"])
        builder = ReportBuilder(results, config)
        directory = (base / config["output"] / scenario).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        records, all_data, captions = [], [], []
        with plt.rc_context(STYLE), PdfPages(directory / "report_figures.pdf") as book:
            for spec in builder.figures():
                check_figure_bounds(spec.figure, spec.slug)
                for fmt in config["formats"]:
                    metadata = {"Title": spec.title, "Description": spec.caption} if fmt == "svg" else None
                    spec.figure.savefig(directory / f"{spec.slug}.{fmt}", dpi=config["dpi"], metadata=metadata)
                book.savefig(spec.figure)
                width, height = spec.figure.get_size_inches() * 25.4
                plt.close(spec.figure)
                spec.data.to_csv(directory / f"{spec.slug}.csv", index=False)
                all_data.append(spec.data.assign(Figure=spec.slug))
                assumption = ("Blank cells and absent regional rows for additive activity variables are provisionally treated as zero; "
                              "missing prices, costs and reported intensities remain missing." if results.missing_activity == "zero" else
                              "Missing activity inputs remain missing; additive sums require complete regional coverage.")
                caption = f"Figure {spec.number}. {spec.title}. {builder.label}. {spec.caption} {assumption} No uncertainty estimates were supplied."
                records.append({"slug": spec.slug, "number": spec.number, "title": spec.title,
                                "question": spec.question, "section": spec.section, "caption": caption,
                                "width_mm": float(width), "height_mm": float(height), "panel_count": spec.panel_count,
                                "formats": config["formats"]})
                captions.append(f"## Figure {spec.number} · {spec.title}\n\n{caption}\n")
                print(f"Rendered {scenario}/{spec.slug}", flush=True)
        builder.data.carbon_price_weights().to_csv(directory / "carbon_price_weights.csv", index=False)
        membership = builder.data.membership()
        membership.to_csv(directory / "region_membership.csv", index=False)
        membership.merge(mapping.drop_duplicates(), left_on="Region", right_on="region", how="left").to_csv(directory / "country_membership.csv", index=False)
        builder.data.producer_ranking_table().to_csv(directory / "producer_rankings.csv", index=False)
        builder.data.emitter_ranking_table().to_csv(directory / "emitter_rankings.csv", index=False)
        comparison = results.intensity_comparison()
        comparison.to_csv(directory / "intensity_comparison.csv", index=False)
        costs = results.raw[results.raw.Variable.str.startswith("Total Annualised Cost")]
        negative = costs.melt(id_vars=["Scenario", "Region", "Variable", "Unit"], var_name="Year", value_name="Value")
        negative = negative[negative.Value < 0]
        negative.to_csv(directory / "negative_costs.csv", index=False)
        pd.DataFrame(results.coverage, columns=["Scenario", "Region", "Variable", "Unit", "Year", "Issue", "Treatment"]).drop_duplicates().to_csv(directory / "missing_inputs.csv", index=False)
        pd.concat(all_data, ignore_index=True).to_csv(directory / "source_data.csv", index=False)
        main_count = sum(record["section"] == "Main report" for record in records)
        support_count = len(records) - main_count
        notes = [
            f"{main_count} main figures tell the report story; Figures S1 and S2 hold regional-intensity detail and full-period costs.",
            "Blank activity values are provisionally treated as zero. This is a draft assumption, not a confirmed export convention. Prices, costs and reported intensities are never zero-filled." if results.missing_activity == "zero" else "Missing values are preserved and aggregates require complete coverage.",
            f"Carbon-price indices use fixed {config['price_weight_year']} industrial-GHG-emissions weights. The nine geographic groups partition the 28 model regions without overlap.",
            f"Leading producers are selected independently by sector using {config['ranking_year']} total production. The same top-{config['top_n']} cohort is shown at both endpoints. Other model regions preserve the global remainder.",
            f"Figure 7 selects the top {config['top_n']} emitters independently by sector using {config['ranking_year']} absolute GHG emissions. It uses Figure 2's geographic groups, with a separate emissions ranking and an unranked remainder. Coverage uses each sector's global emissions as denominator.",
            "Europe combines ENE, ENW, EUE, EUM and EUW. North America contains USA, CAN and MEX. Regional definitions follow whole model regions, including the territories assigned to them in the mapping.",
            "China is the CHN model region, including Hong Kong, Macao and Taiwan. Indonesia group includes Indonesia, Philippines and Viet Nam. The mapping's ZijieRegion column crosses model boundaries and is not used for aggregation.",
            "Global intensities are ratios of global sums. Reported producer-region energy intensities retain the source definition; Europe's regional intensities are production-weighted. These definitions do not agree in this export.",
            f"There are {len(negative)} negative regional cost entries. Main costs start in {config['main_cost_start_year']}; all source years, including anomalous initial costs, remain in Figure S2 and its data.",
        ]
        audit = {"scenario": scenario, "scenario_label": builder.label, "missing_activity": results.missing_activity,
                 "generated_utc": datetime.now(timezone.utc).isoformat(), "source": source.name,
                 "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                 "region_mapping": config["region_mapping"], "region_mapping_sha256": hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
                 "figure_count": len(records), "main_figure_count": main_count, "supporting_figure_count": support_count,
                 "selected_source_rows": len(results.raw), "notes": notes, "config": config,
                 "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__}}
        (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (directory / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (directory / "captions.md").write_text("# Report figure captions\n\n" + "\n".join(captions), encoding="utf-8")
        write_methods(directory, config, notes)
        write_report_gallery(directory, records, audit)
        print(f"Saved {main_count} main and {support_count} supporting figures to {directory}", flush=True)


def write_methods(directory, config, notes):
    text = "# Report design and aggregation methods\n\n" + "\n\n".join(notes)
    text += "\n\n## Geographic definitions\n\n| Carbon-price group | Model regions |\n| --- | --- |\n"
    text += "\n".join(f"| {name} | {', '.join(codes)} |" for name, codes in config["macroregions"].items())
    text += ("\n\nThe country mapping is included as a local snapshot under data/. Country rows are used to describe model-region membership, "
             "not as weights or replicated result rows. Duplicate country rows do not change any aggregate. "
             "country_membership.csv lists the exact countries and territories for both grouping systems. Labels describe model regions rather than strict political borders. "
             "For example, Central Asia group is ASC (including the Caucasus, Mongolia and Afghanistan); "
             "Other Asia–Pacific in producer comparisons is ASE; Gulf & Iran is MEA, which also includes Iraq and Yemen.\n\n"
             "## Formulas\n\n"
             "- Global additive quantities: sum once across the 28 source model regions.\n"
             "- Producer output: cement production; primary + secondary for steel/aluminium. Europe is summed before ranking.\n"
             "- Emitter rankings: absolute Emissions|GHG|Industry|sector rows in MtCO2e/yr, summed into the same groups as production. Select leading groups by emissions, not production. Coverage = selected emissions / global sector emissions × 100; the remainder includes every unselected group.\n"
             "- Carbon-price index: sum(price in year t × industry GHG in weight year) / sum(industry GHG in weight year), within each group. Weights are fixed for all plotted years.\n"
             "- Global energy intensity: sum(final energy, EJ/yr) / sum(production, Mt/yr) × 1,000 = GJ/t.\n"
             "- Reported producer-group energy intensity: sum(reported regional GJ/t × current production) / sum(current production). A missing intensity with positive production invalidates the weighted mean.\n"
             "- Emissions intensity: sum(MtCO2e/yr) / sum(Mt/yr) = tCO2e/t.\n"
             "- Emissions share: sector GHG / all-industry GHG × 100.\n"
             "- Costs: millions USD_2010 divided by 1,000 for billions or 1,000,000 for trillions. Negative values are retained.\n\n"
             "Each figure has an accompanying CSV. Role=plotted identifies points actually drawn; Role=context records producer/emitter coverage, annotated in Figures 2 and 7. "
             "Caption text and axis labels specify when quantities or scales differ. All lines connect supplied years directly; no annual interpolation, smoothing, extrapolation or uncertainty is invented.\n\n"
             "## Figure choices\n\n"
             "Figures 1–2 establish global output and geography; Figure 3 presents the regional price context; Figure 4 connects energy demand to intensity; "
             "Figure 5 shows emissions and capture; Figure 6 shows costs; Figure 7 compares the leading emitter regions using the endpoint layout of Figure 2. Figure S1 uses the same leading-producer cohort for reported energy intensities, "
             "and Figure S2 preserves the entire cost period. The main cost view deliberately excludes the anomalous first year, with that choice stated on the figure.\n\n"
             "All figures are 183 mm wide; the tallest is 149 mm. PDF/SVG preserve editable text and vector lines; PNG uses the configured resolution. "
             "Titles, panel letters, sector colours, marker styles and typography are consistent across the report.\n")
    (directory / "methods.md").write_text(text, encoding="utf-8")
