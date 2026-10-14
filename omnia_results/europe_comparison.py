"""Publication figures comparing EU-only OMNIA-ENGAGE and GCAM scenarios."""

from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import platform

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import ticker
import numpy as np
import pandas as pd

from .comparison_common import POLICY_COLORS
from .consolidation_energy_mix import FUEL_COLORS
from .consolidation_materials import PRODUCT_COLORS
from .consolidation_impacts import INDUSTRY_COLORS, _row_limits
from .metrics import SECTORS
from .plots import STYLE
from .report import ReportFigure, check_figure_bounds
from .report_gallery import _figure_card


MODELS = ("OMNIA-ENGAGE", "GCAM")
CLIMATES = ("ndc", "baseline")
CLIMATE_LABELS = {"ndc": "NDC", "baseline": "NDC+LTT"}
POLICIES = ("noce", "medce_eu", "highce_eu")
POLICY_LABELS = {"noce": "No CE", "medce_eu": "M–EU", "highce_eu": "H–EU"}
COMPARISON_STYLE = {**STYLE, "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"]}
HATCH = "////"
POSITIONS = np.array([0., 1.5, 2.8, 4.1])
OFFSETS = (-.19, .19)
BAR_WIDTH = .34
ENERGY_COLORS = {**FUEL_COLORS, "Unallocated": "#C9B79C"}
EMISSIONS_COLORS = {**INDUSTRY_COLORS, "Industrial energy demand": "#6A3D9A",
                    "Industrial processes": "#C7A600"}


def _percentage(value):
    return "0%" if abs(value) < .05 else f"{value:+.1f}%".replace("-", "\N{MINUS SIGN}")


def _clean_axes(ax, letter, title, ylabel):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DDE3E6", linewidth=.35)
    ax.tick_params(labelsize=6.1, pad=3)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4, min_n_ticks=3))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.4g}"))
    ax.set_ylabel(ylabel, fontsize=6.6, labelpad=3)
    ax.set_title(title, loc="left", fontsize=7, pad=9)
    ax.text(-.14, 1.04, letter, transform=ax.transAxes,
            fontsize=8, weight="bold", va="bottom", ha="left")


def _model_handles():
    return [Patch(facecolor="#FFFFFF", edgecolor="#53626B", label="OMNIA-ENGAGE · left"),
            Patch(facecolor="#FFFFFF", edgecolor="#53626B", hatch=HATCH, label="GCAM · right")]


def _bar_values(frame, series):
    return (frame.loc[frame.Series == series].set_index("BarOrder").Value
            .reindex(range(4)).to_numpy(dtype=float))


class EuropeComparisonBuilder:
    def __init__(self, data, config):
        self.data, self.config = data, config

    def figures(self):
        yield self.material_production()
        yield self.energy_mix()
        yield self.emissions_response()

    def _bar_canvas(self, title, subtitle):
        fig, axes = plt.subplots(2, 3, figsize=(180 / 25.4, 185 / 25.4), squeeze=False)
        fig.subplots_adjust(left=.085, right=.985, bottom=.205, top=.765,
                            wspace=.48, hspace=.64)
        fig.text(.035, .977, title, fontsize=8, weight="bold", va="top")
        fig.text(.035, .941, subtitle, fontsize=6.7, color="#52616A", va="top")
        fig.legend(handles=_model_handles(), loc="upper left", bbox_to_anchor=(.035, .903),
                   ncol=2, fontsize=6, columnspacing=2, handlelength=1.8, frameon=False)
        return fig, axes

    def _bar_axes(self, ax, letter, sector, climate, ylabel, limits):
        _clean_axes(ax, letter, f"{sector.label} | {CLIMATE_LABELS[climate]}", ylabel)
        ax.set_xlim(-.65, 4.7)
        ax.set_ylim(*limits)
        history = f"{self.config['historical_year']} /\n{self.config['gcam_historical_year']}"
        ax.set_xticks(POSITIONS, [history, "No\nCE", "M\nEU", "H\nEU"], fontsize=6)
        ax.tick_params(axis="x", length=0, pad=4)
        ax.get_xticklabels()[0].set_weight("bold")
        ax.axvline(.72, color="#B7C1C6", linewidth=.55, linestyle=(0, (2, 2)))
        ax.text(2.8, -.20, str(self.config["comparison_year"]),
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=5.8, color="#52616A")

    @staticmethod
    def _limits(frame):
        values = frame.Value.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        low = min(0., float(values.min())) if values.size else 0.
        high = max(0., float(values.max())) if values.size else 1.
        span = max(high - low, .001)
        return (low - .04 * span if low < 0 else 0., high + .22 * span), span

    @staticmethod
    def _annotations(ax, data, model, climate, sector, x, totals, span, metric):
        frame = data.loc[(data.Model == model) & (data.Climate == climate) &
                         (data.Sector == sector.label) & (data.Metric == metric)]
        changes = frame.set_index("BarOrder").Value.reindex(range(1, 4))
        for order in range(1, 4):
            if np.isfinite(totals[order]) and np.isfinite(changes.loc[order]):
                # A vertical label fits over each narrow bar and preserves the
                # two model-specific references without adjacent text collisions.
                ax.text(x[order], totals[order] + .025 * span,
                        "ref." if order == 1 else _percentage(changes.loc[order]),
                        ha="center", va="bottom", rotation=90, fontsize=5.5,
                        color="#202D35")

    def material_production(self):
        data = self.data.material_production()
        production = data.loc[data.Metric == "Material production"]
        title = "European material production: OMNIA-ENGAGE and GCAM"
        fig, axes = self._bar_canvas(title, f"EU-only CE settings | historical references and {self.config['comparison_year']} production")
        handles = [Patch(facecolor=PRODUCT_COLORS["Primary"], label="Cement / primary metal"),
                   Patch(facecolor=PRODUCT_COLORS["Secondary"], label="Secondary metal (OMNIA)"),
                   Patch(facecolor="#ADB8BE", edgecolor="#53626B", hatch=HATCH, label="Total metal (GCAM)")]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.035, .862), ncol=3,
                   fontsize=5.8, columnspacing=1.3, handlelength=1.5, frameon=False)
        for column, sector in enumerate(SECTORS):
            sector_data = production.loc[production.Sector == sector.label]
            totals_data = sector_data.loc[sector_data.Series == ("Cement" if sector.key == "cement" else "Total")]
            limits, span = self._limits(totals_data)
            for row, climate in enumerate(CLIMATES):
                ax = axes[row, column]
                self._bar_axes(ax, chr(97 + row * 3 + column), sector, climate,
                               "Production (Mt yr$^{-1}$)", limits)
                for model_index, model in enumerate(MODELS):
                    selected = sector_data.loc[(sector_data.Model == model) & (sector_data.Climate == climate)]
                    x = POSITIONS + OFFSETS[model_index]
                    hatch = HATCH if model == "GCAM" else None
                    total = _bar_values(selected, "Cement" if sector.key == "cement" else "Total")
                    if sector.key == "cement" or model == "GCAM":
                        color = PRODUCT_COLORS["Cement"] if sector.key == "cement" else "#ADB8BE"
                        ax.bar(x, total, BAR_WIDTH, color=color, edgecolor="#53626B" if hatch else "white",
                               linewidth=.35, hatch=hatch, zorder=3)
                    else:
                        primary, secondary = _bar_values(selected, "Primary"), _bar_values(selected, "Secondary")
                        ax.bar(x, primary, BAR_WIDTH, color=PRODUCT_COLORS["Primary"], edgecolor="white", linewidth=.35)
                        ax.bar(x, secondary, BAR_WIDTH, bottom=primary, color=PRODUCT_COLORS["Secondary"],
                               edgecolor="white", linewidth=.35)
                        shares = data.loc[(data.Model == model) & (data.Climate == climate) &
                                          (data.Sector == sector.label) & (data.Metric == "Secondary production share")]
                        shares = shares.set_index("BarOrder").Value.reindex(range(4))
                        for order in range(4):
                            if np.isfinite(shares.loc[order]) and secondary[order] > .15 * span:
                                ax.text(x[order], primary[order] + secondary[order] / 2,
                                        f"{shares.loc[order]:.0f}%", ha="center", va="center",
                                        fontsize=5.0, color="#173F39", rotation=90)
                    self._annotations(ax, data, model, climate, sector, x, total, span,
                                      "Production change from no CE")
        self._bar_notes(fig, production=True)
        caption = (
            "European production in a 2 × 3 layout: NDC above NDC+LTT, with cement, steel and aluminium columns. "
            "Each category pairs OMNIA-ENGAGE on the left with hatched GCAM on the right. "
            "Historical bars use the model's own no-CE scenario at 2019 (OMNIA) or 2021 (GCAM); "
            "future categories show no, medium and high CE in the EU in 2050. EU+ scenarios are excluded. "
            "OMNIA metal output stacks primary and secondary routes; GCAM reports total metal production only. "
            "GCAM's grey hatched total is not assigned to a production route. Cement excludes clinker; "
            "clinker is omitted because GCAM provides no corresponding series. OMNIA secondary shares are "
            "shown within sufficiently large segments. Future annotations use each model's own no-CE total "
            "under the same climate pathway in 2050: 100 × (scenario/reference − 1). "
            "Sector scales match across pathways and models. " + self._common_caption())
        return ReportFigure("fig01_material_production", "1", title,
                            "How do EU-only CE settings change material output in the two models?",
                            caption, fig, data, 6)

    def energy_mix(self):
        data = self.data.energy_mix()
        energy = data.loc[data.Metric == "Final energy"]
        title = "European final-energy mix: OMNIA-ENGAGE and GCAM"
        fig, axes = self._bar_canvas(title, f"EU-only CE settings | historical references and {self.config['comparison_year']} final energy")
        fig.legend(handles=[Patch(facecolor=color, label=carrier) for carrier, color in ENERGY_COLORS.items()],
                   loc="upper left", bbox_to_anchor=(.035, .862), ncol=6, fontsize=5.8,
                   columnspacing=1.2, handlelength=1.4, frameon=False)
        for column, sector in enumerate(SECTORS):
            sector_data = energy.loc[energy.Sector == sector.label]
            limits, span = self._limits(sector_data.loc[sector_data.Series == "Total"])
            for row, climate in enumerate(CLIMATES):
                ax = axes[row, column]
                self._bar_axes(ax, chr(97 + row * 3 + column), sector, climate,
                               "Final energy (EJ yr$^{-1}$)", limits)
                for model_index, model in enumerate(MODELS):
                    selected = sector_data.loc[(sector_data.Model == model) & (sector_data.Climate == climate)]
                    x, bottom = POSITIONS + OFFSETS[model_index], np.zeros(4)
                    for carrier, color in ENERGY_COLORS.items():
                        if carrier not in set(selected.Series):
                            continue
                        values = _bar_values(selected, carrier)
                        ax.bar(x, values, BAR_WIDTH, bottom=bottom, color=color,
                               edgecolor="#53626B" if model_index else "white", linewidth=.3,
                               hatch=HATCH if model_index else None)
                        bottom += values
                    self._annotations(ax, data, model, climate, sector, x,
                                      _bar_values(selected, "Total"), span,
                                      "Final energy change from no CE")
        self._bar_notes(fig, production=False)
        caption = (
            "European final energy in a 2 × 3 layout with NDC above NDC+LTT and cement, steel and aluminium columns. "
            "Solid OMNIA-ENGAGE bars are paired with hatched GCAM bars in each category. Colours identify "
            "electricity, gases, hydrogen, liquids, solids and unallocated energy. Exact carrier parents enter "
            "each stack once. GCAM's reported sector total is retained; Unallocated is this total minus the "
            "sum of its supplied carrier parents (nonzero for aluminium). Missing fuel variables are not "
            "assigned to a carrier or silently imputed as zero. OMNIA totals sum the documented carrier parents. Historical "
            "references use 2019 (OMNIA) and 2021 (GCAM), with no interpolation. Future bars compare no, "
            "medium and high EU-only CE in 2050. Each future percentage uses the same model, pathway and "
            "year's no-CE final energy. Sector scales match across pathways and models. " + self._common_caption())
        return ReportFigure("fig02_energy_mix", "2", title,
                            "How do EU-only CE settings change final-energy demand and carrier mixes?",
                            caption, fig, data, 6)

    def emissions_response(self):
        data = self.data.emissions_response()
        future = self.config["comparison_year"]
        title = "European industrial GHG response: OMNIA-ENGAGE and GCAM"
        fig, axes = plt.subplots(3, 2, figsize=(180 / 25.4, 235 / 25.4), squeeze=False)
        for row, bottom in enumerate((.650, .375, .100)):
            for column in range(2):
                axes[row, column].set_position([.095 if column == 0 else .605, bottom, .370, .175])
        fig.text(.035, .978, title, fontsize=8, weight="bold", va="top")
        fig.text(.035, .943, "Reported industry totals | EU-only CE; source emissions scopes differ",
                 fontsize=6.7, color="#52616A", va="top")
        policy_handles = [Line2D([], [], color=POLICY_COLORS[p], linewidth=1.2,
                                 label=POLICY_LABELS[p]) for p in POLICIES]
        model_handles = [Line2D([], [], color="#53626B", linestyle="-", marker="o", markerfacecolor="white",
                               markersize=3, label="OMNIA-ENGAGE"),
                         Line2D([], [], color="#53626B", linestyle="--", marker="D", markerfacecolor="#53626B",
                                markersize=3, label="GCAM")]
        fig.legend(handles=policy_handles + model_handles, loc="upper center",
                   bbox_to_anchor=(.535, .902), ncol=5, fontsize=5.8,
                   handlelength=2, columnspacing=1.1, title="CE settings and models (a, b)",
                   title_fontsize=6, frameon=False)
        trajectory_values, bounds, nets, percentages = [], [], [], []
        positions = np.array([0., 1.5])
        changes_metric = "Industrial GHG emissions change"
        for column, climate in enumerate(CLIMATES):
            climate_name = CLIMATE_LABELS[climate]
            for row in range(3):
                ax, panel = axes[row, column], chr(97 + row * 2 + column)
                _clean_axes(ax, panel,
                            f"{climate_name} | " + ("reported industry GHG" if row == 0 else
                                                   f"change by reported component, {future}" if row == 1 else
                                                   f"relative industrial response, {future}"),
                            "Industry GHG\n(Mt CO$_2$e yr$^{-1}$)" if row == 0 else
                            "Industry GHG change\n(Mt CO$_2$e yr$^{-1}$)" if row == 1 else
                            "Change from no CE (%)")
                # Keep labels inside the canvas even when large source values
                # require wider tick labels; both columns share the same offset.
                ax.yaxis.set_label_coords(-.14, .5)
                if row == 0:
                    first = min(self.config['historical_year'], self.config['gcam_historical_year'])
                    ax.set_xlim(first - .6, future + .6)
                    ticks = sorted({first, future, *[y for years in self.data.years.values() for y in years
                                                   if first <= y <= future and y % 10 == 0]})
                    ax.set_xticks(ticks)
                    ax.set_xlabel("Year", fontsize=6.5, labelpad=3)
                    for model_index, model in enumerate(MODELS):
                        for policy in POLICIES:
                            selected = data.loc[(data.Model == model) & (data.Climate == climate) &
                                                (data.Policy == policy) &
                                                (data.Metric == "Industrial GHG emissions") &
                                                (data.Series == "Total industry") & (data.Role == "plotted") &
                                                (data.Year <= future)].sort_values("Year")
                            trajectory_values.extend(selected.Value)
                            ax.plot(selected.Year, selected.Value, color=POLICY_COLORS[policy],
                                    linestyle="--" if model_index else "-", marker="D" if model_index else "o",
                                    markerfacecolor=POLICY_COLORS[policy] if model_index else "white",
                                    markeredgewidth=.65, markersize=2.4, linewidth=1.0,
                                    zorder=3 if model_index else 4)
                else:
                    ax.set_xlim(-.65, 2.15)
                    ax.set_xticks(positions, ["M–EU", "H–EU"], fontsize=6)
                    ax.tick_params(axis="x", length=0, pad=4)
                    ax.axhline(0, color="#56636B", linewidth=.65, zorder=2)
                    for model_index, model in enumerate(MODELS):
                        x = positions + (-.23 if model_index == 0 else .23)
                        selected = data.loc[(data.Model == model) & (data.Climate == climate)]
                        if row == 1:
                            components = selected.loc[(selected.Metric == changes_metric) &
                                                       (selected.Role == "plotted")]
                            positive, negative = np.zeros(2), np.zeros(2)
                            for series in components.Series.drop_duplicates():
                                values = (components.loc[components.Series == series].set_index("Policy").Value
                                          .reindex(POLICIES[1:]).to_numpy(dtype=float))
                                bottom = np.where(values >= 0, positive, negative)
                                ax.bar(x, values, .39, bottom=bottom, color=EMISSIONS_COLORS[series],
                                       edgecolor="#53626B" if model_index else "white", linewidth=.3,
                                       hatch=HATCH if model_index else None, zorder=3)
                                positive += np.where(values >= 0, values, 0.)
                                negative += np.where(values < 0, values, 0.)
                                positive[~np.isfinite(values)] = np.nan
                                negative[~np.isfinite(values)] = np.nan
                            net = (selected.loc[(selected.Metric == changes_metric) & (selected.Role == "net")]
                                   .set_index("Policy").Value.reindex(POLICIES[1:]).to_numpy(dtype=float))
                            bounds.append((negative.copy(), positive.copy()))
                            nets.append(net)
                            ax.scatter(x, net, marker="D", s=14, color="#1D262C", edgecolors="white",
                                       linewidths=.45, zorder=5)
                            for position, value in zip(x, net):
                                if np.isfinite(value):
                                    ax.annotate(f"{value:+.1f}".replace("-", "\N{MINUS SIGN}"),
                                                (position, value), xytext=(0, -8 if value <= 0 else 7),
                                                textcoords="offset points", ha="center",
                                                va="top" if value <= 0 else "bottom", fontsize=5.3,
                                                bbox={"facecolor": "white", "edgecolor": "none", "pad": .25},
                                                zorder=6)
                        else:
                            values = (selected.loc[selected.Metric == "Industrial GHG emissions change from no CE"]
                                      .set_index("Policy").Value.reindex(POLICIES[1:]).to_numpy(dtype=float))
                            percentages.extend(values)
                            ax.bar(x, values, .39, color=[POLICY_COLORS[p] for p in POLICIES[1:]],
                                   edgecolor="#53626B" if model_index else "white", linewidth=.35,
                                   hatch=HATCH if model_index else None, zorder=3)
                            for position, value in zip(x, values):
                                if np.isfinite(value):
                                    ax.annotate(_percentage(value), (position, value),
                                                xytext=(0, -8 if value <= 0 else 7), textcoords="offset points",
                                                ha="center", va="top" if value <= 0 else "bottom", fontsize=5.6)
        finite = np.asarray(trajectory_values, dtype=float)
        finite = finite[np.isfinite(finite)]
        low = min(0., float(finite.min())) if finite.size else 0.
        high = max(1., float(finite.max())) if finite.size else 1.
        span = high - low
        for ax in axes[0]:
            ax.set_ylim(low - .03 * span if low < 0 else 0., high + .10 * span)
        for ax in axes[1]:
            ax.set_ylim(*_row_limits(bounds, nets))
        finite = np.asarray(percentages, dtype=float)
        finite = finite[np.isfinite(finite)]
        low = min(0., float(finite.min())) if finite.size else 0.
        high = max(0., float(finite.max())) if finite.size else 0.
        span = max(high - low, 1.)
        for ax in axes[2]:
            ax.set_ylim(low - .27 * span, high + .27 * span)
        component_handles = [Patch(facecolor=color, label=series) for series, color in EMISSIONS_COLORS.items()]
        component_handles.append(Line2D([], [], marker="D", color="#1D262C", linestyle="none",
                                        markersize=3.5, label="Industry net"))
        # Two rows retain readable labels and keep model-specific components explicit.
        fig.legend(handles=component_handles, loc="upper center", bbox_to_anchor=(.535, .612),
                   ncol=4, fontsize=5.5, columnspacing=1.15, handlelength=1.4,
                   title="Reported components (c, d): OMNIA sectors / GCAM energy and processes",
                   title_fontsize=5.6, frameon=False)
        fig.legend(handles=_model_handles(), loc="upper center", bbox_to_anchor=(.535, .332),
                   ncol=2, fontsize=6, columnspacing=2.5,
                   title="Paired bars (c–f); percentages use each model's own no-CE industry total",
                   title_fontsize=5.6, frameon=False)
        notes = [
            "OMNIA: reported industry GHG. GCAM: industry energy-demand + industrial-process Kyoto gases.",
            "Emissions accounting and gas coverage are not harmonised; sector GHG is unavailable in GCAM.",
            "EU-region boundaries differ slightly. Model years are retained; carbon capture is not subtracted again.",
            self._missing_note(),
        ]
        for y, note in zip((.065, .048, .031, .014), notes):
            fig.text(.035, y, note, fontsize=5.3, color="#52616A", va="bottom")
        caption = (
            "Reported European industrial GHG under NDC (left) and NDC+LTT (right). "
            "Panels a,b compare no, medium and high EU-only CE trajectories, with solid/open-circle OMNIA "
            "and dashed/filled-diamond GCAM lines at each model's supplied years from its historical "
            "reference onward. No interpolation, extrapolation or year displacement is used. "
            "OMNIA selects Emissions|GHG|Industry; GCAM sums Emissions|Kyoto Gases|Energy|Demand|Industry "
            "and Emissions|Kyoto Gases|Industrial Processes, both in Mt CO2-equiv/yr. These are source-defined "
            "industrial totals with different accounting scopes and potentially different gas coverage/GWP "
            "conventions, rather than harmonised sector totals. Panels c,d show signed 2050 CE-minus-no-CE "
            "changes, separately decomposed into OMNIA's cement, steel, aluminium and Other industry "
            "accounting residual, and GCAM's industrial energy demand and industrial processes. "
            "The component legends are model-specific; they do not imply a mapping between those categories. "
            "Positive and negative components stack independently from zero; black diamonds and signed "
            "labels give each model's net industrial change. Panels e,f show 100 × (CE − no CE) / no CE "
            "using each model's own source-defined industrial total in the same climate pathway and year. "
            "Both decomposition and percentage bars place OMNIA on the left and hatched GCAM on the right. "
            "Scales match across pathways within each row. Regional/partner decomposition from the reference "
            "Figure 7 is replaced because only EU-only data and unmatched sector emissions are available. "
            "Carbon capture is not subtracted again. " + self._common_caption())
        return ReportFigure("fig03_emissions_response", "3", title,
                            "How does EU-only CE change each model's reported industrial GHG?",
                            caption, fig, data, 6)

    def _bar_notes(self, fig, *, production):
        historical, gcam_historical, future = (self.config[k] for k in
                                               ('historical_year', 'gcam_historical_year', 'comparison_year'))
        notes = [
            f"Historical bars: OMNIA {historical} / GCAM {gcam_historical}. All other bars: {future}; no interpolation.",
            f"Labels above: change from each model's {future} no CE in the same pathway. M/H = medium/high CE.",
            ("OMNIA metal stacks show routes; GCAM totals have no route breakdown. Cement excludes clinker."
             if production else "GCAM unallocated energy = reported sector total minus supplied carrier parents (aluminium)."),
            "OMNIA: EUE + EUM + EUW; GCAM: EU-27 aggregate. Model geographic boundaries differ slightly.",
            self._missing_note(),
        ]
        for y, note in zip((.124, .100, .076, .052, .028), notes):
            fig.text(.035, y, note, fontsize=5.5, color="#52616A", va="bottom")

    def _missing_note(self):
        return ("OMNIA missing activity = zero (provisional); GCAM missing values remain gaps."
                if self.config['missing_activity'] == 'zero' else
                "Both models retain missing activity; incomplete aggregates and bars remain gaps.")

    def _common_caption(self):
        return (
            "OMNIA totals sum EUE, EUM and EUW; GCAM selects its supplied EU-27 aggregate once, "
            "without adding country rows. These aggregates are checked against available-country sums. "
            "OMNIA regional boundaries include some small non-EU "
            "territories; membership is exported explicitly, so the two boundaries are not assumed identical. "
            "Source scenario names are retained: OMNIA baseline_* and GCAM LTT_* are displayed as NDC+LTT; "
            "ndc_* and NDC_* are displayed as NDC. Matching labels do not establish identical policy assumptions. "
            + self._missing_note() + " Exact source variables, units, region memberships, "
            "conversion factors and calculation references accompany the source CSVs.")


def generate_europe_comparison(config, base):
    """Build all three figures directly from the configured raw sources."""
    from .europe_comparison_data import EuropeComparisonData
    import openpyxl

    base = Path(base)
    data = EuropeComparisonData(config, base)
    builder = EuropeComparisonBuilder(data, config)
    directory = (base / config["output"]).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    records, frames, captions = [], [], []
    with plt.rc_context(COMPARISON_STYLE), PdfPages(directory / config.get("combined_pdf", "report_figures.pdf"),
            metadata={"Title": "European industrial results: OMNIA-ENGAGE and GCAM"}) as book:
        for spec in builder.figures():
            check_figure_bounds(spec.figure, spec.slug)
            for fmt in config["formats"]:
                metadata = {"Title": spec.title, "Description": spec.caption, "Date": None} if fmt == "svg" else None
                spec.figure.savefig(directory / f"{spec.slug}.{fmt}", dpi=config["dpi"], metadata=metadata)
            book.savefig(spec.figure)
            spec.data.to_csv(directory / f"{spec.slug}.csv", index=False)
            width, height = spec.figure.get_size_inches() * 25.4
            plt.close(spec.figure)
            records.append({"number": spec.number, "slug": spec.slug, "title": spec.title,
                            "question": spec.question, "caption": spec.caption, "section": spec.section,
                            "panel_count": spec.panel_count, "width_mm": float(width), "height_mm": float(height),
                            "formats": config["formats"]})
            frames.append(spec.data.assign(Figure=spec.slug))
            captions.append(f"## Figure {spec.number}: {spec.title}\n\n{spec.caption}\n")
            print(f"Rendered Europe comparison/{spec.slug}", flush=True)
    all_data = pd.concat(frames, ignore_index=True)
    all_data.to_csv(directory / "source_data.csv", index=False)
    data.scenario_mapping().to_csv(directory / "scenario_mapping.csv", index=False)
    membership = data.geography_membership()
    membership.to_csv(directory / "geography_membership.csv", index=False)
    data.variable_mapping().to_csv(directory / "variable_mapping.csv", index=False)
    data.aggregation_checks.to_csv(directory / "gcam_aggregation_checks.csv", index=False)
    data.missing_inputs.to_csv(directory / "missing_inputs.csv", index=False)
    sources = {model: {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
               for model, path in data.sources.items()}
    mapping_path = (base / config["region_mapping"]).resolve()
    audit = {"generated_utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
             "region_mapping_sha256": hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
             "config": config, "figure_count": len(records),
             "source_row_counts": {model: len(raw) for model, raw in data.raw.items()},
             "historical_years": data.historical_years, "source_years": data.years,
             "missing_input_count": len(data.missing_inputs),
             "gcam_aggregate_check_count": len(data.aggregation_checks),
             "max_gcam_aggregate_country_difference": float(data.aggregation_checks.Difference.abs().max()),
             "versions": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                          "numpy": np.__version__, "pandas": pd.__version__, "openpyxl": openpyxl.__version__},
             "notes": [*data.notes,
                       "Emissions gas coverage and GWP/accounting conventions are not documented as harmonised between the models.",
                       "All source data are calculated from the raw CSV and workbook; existing figures are visual references only.",
                       "The GCAM aggregate is used directly; available-country sums verify its accounting without imputing absent country rows."]}
    (directory / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (directory / "captions.md").write_text("# European model comparison captions\n\n" + "\n".join(captions), encoding="utf-8")
    _write_methods(directory, config, membership)
    _write_gallery(directory, config, records)
    print(f"Saved three European comparison figures to {directory}", flush=True)


def _write_methods(directory, config, membership):
    extras = membership.loc[(membership.Model == "OMNIA-ENGAGE") & ~membership.IncludedInGCAM, "Country"].tolist()
    text = f"""# European OMNIA-ENGAGE / GCAM comparison methods

Regenerate all three figures with Python 3.11+ and the project dependencies:

```powershell
python -m omnia_results --config {config['config_file']}
```

## Sources and selection

Calculations read `{config['input']}` and `{config['gcam_input']}` (`{config.get('gcam_sheet', 'Sheet1')}`) directly. Previously exported figures and CSVs are not calculation inputs. `audit.json` records source and regional-mapping hashes, software versions and the complete configuration.

Six scenarios are selected from each model: NDC and NDC+LTT, each with no, medium and high CE applied in the EU only. No EU+ scenarios enter the collection. OMNIA `baseline_*` and GCAM `LTT_*` are paired under the display label NDC+LTT; OMNIA `ndc_*` and GCAM `NDC_*` are paired as NDC. These source labels do not establish identical policy assumptions. `scenario_mapping.csv` retains every original identifier.

## Geographic scope

OMNIA sums EUE, EUM and EUW once. This is the repository's EU policy region, rather than the original report's five-region Europe producer group. The supplied country mapping assigns all EU-27 countries and these additional countries/territories to the three inseparable OMNIA regions: {', '.join(extras)}. They cannot be subtracted using the available regional results, so OMNIA is a geographic approximation to GCAM's EU-27.

GCAM selects the supplied **EU-27 aggregate once**, without adding overlapping country rows. Its aggregate is checked against the sum of available country observations for every selected source variable and year. Some production and process-emissions variables have fewer than 27 country rows; an absent country observation is not silently assigned zero. The comparison uses the authoritative reported aggregate, while `gcam_aggregation_checks.csv` documents available-country coverage and reconciliation. `geography_membership.csv` records both model boundaries.

## Historical observations and references

Historical bars use each model's own no-CE scenario: {config['historical_year']} for OMNIA and {config['gcam_historical_year']} for GCAM. Future bars use {config['comparison_year']}. No historical-year interpolation or calibration is introduced. Emissions trajectories retain all supplied observations from each model's historical reference to the comparison year; markers use their actual numerical years.

Every future CE percentage uses **the same model, climate pathway, sector and future year** as its no-CE denominator: `100 × (scenario / no_CE − 1)`. It is not a change from the historical bar. Positive, finite denominators are required; unavailable references produce gaps. All numerator/denominator values and source scenarios remain in the CSVs.

## Figure 1: material production

The 2 × 3 layout has NDC above NDC+LTT and cement, iron and steel, and aluminium columns. Within each historical/future category, OMNIA is on the left and hatched GCAM on the right. Scales match across models and climate rows within each sector.

OMNIA cement selects `Production|Non-Metallic Minerals|Cement`; metal totals sum exact primary and secondary production parents. GCAM selects the exact reported cement parent, `Production|Iron and Steel|Steel`, and `Production|Non-Ferrous Metals|Aluminum`, in `Mt/yr`. GCAM has no production-route split, so metal totals are shown as grey hatched bars, not primary or secondary production. OMNIA secondary shares are annotated where legible. Clinker has no matching GCAM series and is omitted from the figure; any exported OMNIA clinker values are explicitly context.

## Figure 2: final-energy mix

The layout and bar categories match Figure 1. Exact carrier parents select electricity, gases, hydrogen, liquids and solids without adding overlapping detailed fuel children. OMNIA totals sum its documented carrier parents; its cement export has no hydrogen parent. GCAM totals select the reported sector final-energy parent in `EJ/yr`. GCAM cement and steel report five carrier parents; aluminium reports electricity, liquids and solids, without gas or hydrogen parents.

**Unallocated energy** is GCAM's reported sector total minus the sum of its supplied carrier parents. The residual is positive for aluminium in the supplied export and is shown as a separate stack component. No unreported carrier is invented and no residual is assigned to gases or hydrogen. Near-zero arithmetic residue below `1e-12 EJ/yr` is recorded as zero. The reported totals and exact residual coefficients remain in the source CSV. All stack components reproduce the totals used for percentage annotations.

## Figure 3: reported industrial GHG response

This adaptation retains Figure 7's trajectory and signed-response approach. Columns show NDC and NDC+LTT. The top row compares industrial GHG trajectories; the middle row decomposes {config['comparison_year']} changes into each model's own reported components; the bottom row compares percentage changes against each model's own no-CE industrial total. Regional/partner contributions are omitted because the requested scope is EU-only and GCAM has no matched sector GHG decomposition.

OMNIA selects `Emissions|GHG|Industry` in `MtCO2e/yr`. GCAM sums `Emissions|Kyoto Gases|Energy|Demand|Industry` and `Emissions|Kyoto Gases|Industrial Processes`, both reported in `Mt CO2-equiv/yr`. The unit scale is the same; **the accounting scope, gas coverage and GWP conventions are not documented as harmonised**. GCAM's industrial-process aggregate is source-defined and is not presented as an all-sector GHG decomposition. GCAM sector CO2 variables cannot supply the missing sector GHG comparison, and no CO2/GHG substitution is made.

OMNIA's components are cement, iron and steel, aluminium and Other industry (reported industry minus the three sectors). GCAM's components are industrial energy demand and industrial processes. The component colours and legend distinguish these model-specific accounts. Signed components stack separately above and below zero; their sum equals the net change shown by the black diamond. Carbon capture is never subtracted again. The relative-response row compares `100 × (CE − no_CE) / no_CE` within each model, rather than treating the two source totals as identical inventories.

## Missing values and export quality

OMNIA additive activity follows the configured `{config['missing_activity']}` convention, preserving the existing report's provisional zero assumption by default. GCAM blank aggregate values remain missing and missing required aggregate rows/variables fail validation. Prices and costs are not used. `missing_inputs.csv` lists affected selected observations and their treatment.

Figures retain 180 mm width, embedded PDF fonts, editable SVG text, and 600 dpi PNG export. The first two figures are 185 mm high; Figure 3 is 235 mm high. Automated canvas bounds checks prevent cropped labels. Common sector/row limits preserve model comparisons; lines connect actual observations without smoothing, uncertainty estimates or extrapolation. The collection includes a three-page PDF, individual artwork, full-precision source CSVs, captions, this methods document, scenario/geography/variable mappings and an audit.
"""
    (directory / "methods.md").write_text(text, encoding="utf-8")


def _write_gallery(directory, config, records):
    cards = "\n".join(_figure_card(record) for record in records)
    combined = escape(config.get("combined_pdf", "report_figures.pdf"), quote=True)
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>European industrial model comparison</title>
<style>
*{{box-sizing:border-box}}body{{font-family:Arial,Helvetica,sans-serif;background:#f4f5f3;color:#233139;margin:0;line-height:1.6}}
main{{max-width:1080px;margin:auto;padding:42px 26px}}h1{{font-size:32px;line-height:1.2}}a{{color:#006887;text-underline-offset:3px}}
.intro{{max-width:880px;color:#52616a}}.download{{display:inline-block;background:#173f4b;color:white;padding:10px 16px;border-radius:4px}}
.scope{{padding:12px 16px;background:#edf1f0;border-left:3px solid #667d83}}.figure-card{{margin:32px 0;background:white;border:1px solid #d7e0df;border-radius:5px}}
.figure-heading{{padding:20px 26px 0}}.figure-number{{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:#607780}}h3{{margin:5px 0;font-size:22px}}
.purpose{{color:#607780}}figure{{margin:0;padding:16px 26px}}figure img{{display:block;width:100%;height:auto}}figcaption{{font-size:13px;line-height:1.65;margin-top:20px}}
.figure-links,.downloads{{padding:0 26px 20px;font-size:13px}}.downloads{{display:flex;flex-wrap:wrap;gap:14px}}footer{{color:#607780;font-size:12px;margin-top:30px}}
@media(max-width:600px){{main{{padding:24px 12px}}.figure-heading,figure{{padding-left:12px;padding-right:12px}}h1{{font-size:25px}}}}
</style></head><body><main>
<p>OMNIA-ENGAGE / GCAM · European industry</p><h1>Comparing EU-only circular-economy scenarios</h1>
<p class="intro">Three figures compare material production, final-energy mixes and reported industrial GHG under NDC and NDC+LTT, with no, medium and high CE in the EU.</p>
<p><a class="download" href="{combined}">Download all three figures · PDF</a></p>
<p class="scope">OMNIA sums its EU model regions (EUE, EUM, EUW); GCAM uses its EU-27 aggregate. The boundaries differ slightly. Historical references use 2019 / 2021. GCAM reports no production-route split or matching sector GHG; Figure 3 compares source-defined industrial totals with explicit scope notes.</p>
<p><a href="source_data.csv">All source data</a> · <a href="captions.md">Captions</a> · <a href="methods.md">Methods</a> · <a href="audit.json">Audit</a><br>
<a href="scenario_mapping.csv">Scenario mapping</a> · <a href="geography_membership.csv">Geography</a> · <a href="variable_mapping.csv">Variables</a> · <a href="gcam_aggregation_checks.csv">GCAM aggregate checks</a> · <a href="missing_inputs.csv">Missing inputs</a></p>
{cards}
<footer>Calculated directly from {escape(config['input'])} and {escape(config['gcam_input'])}. Vector PDF, editable SVG, 600 dpi PNG and full-precision CSV accompany each figure.</footer>
</main></body></html>'''
    (directory / "index.html").write_text(page, encoding="utf-8")

