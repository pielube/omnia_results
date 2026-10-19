"""Global steel comparisons that retain each model's reported boundaries."""

from datetime import datetime, timezone
import hashlib
from html import escape
import json
import os
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

from .plots import STYLE
from .report import ReportFigure, check_figure_bounds
from .report_gallery import _figure_card


CLIMATES = ("ndc", "baseline")
CLIMATE_LABELS = {"ndc": "NDC", "baseline": "NDC+LTT"}
AMBITIONS = ("Medium", "High")
COLORS = {"OPEN-PROM": "#0072B2", "EU": "#D55E00", "EU+": "#009E73"}
COMPARISON_STYLE = {**STYLE, "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"]}


def _signed(value, precision=1):
    if not np.isfinite(value):
        return "unavailable"
    if abs(value) < .5 * 10 ** -precision:
        return f"{0:.{precision}f}%"
    return f"{value:+.{precision}f}%".replace("-", "\N{MINUS SIGN}")


def response_summary(frame, year):
    """Keep all absolute/reference quantities alongside the plotted responses."""
    return (frame.loc[frame.Year.eq(year)]
            .sort_values(["Metric", "Climate", "Model", "Ambition", "Coverage"])
            .reset_index(drop=True))


def draw_openprom_emissions_panels(axes, frame, year=2050, letters=("a", "b")):
    """Draw paired 2050 bars with the Europe comparison's model encoding.

    EU+ describes OMNIA's adoption area; both models' quantities remain global
    steel. Each bar retains its own model/climate reference denominator.
    """
    from .europe_comparison import HATCH, _clean_axes
    from .europe_relative import OFFSETS, POSITIONS, WIDTH, _net_labels

    selected = frame.loc[frame.Metric.eq("Emissions") & frame.Year.eq(year) &
                         frame.Ambition.isin(AMBITIONS) &
                         ((frame.Model.eq("OMNIA-ENGAGE") & frame.Coverage.eq("EU+")) |
                          (frame.Model.eq("OPEN-PROM") & frame.Coverage.eq("Unspecified")))].copy()
    if selected.empty:
        raise ValueError(f"No global steel emissions comparisons for {year}")
    if selected.duplicated(["Model", "Climate", "Ambition"]).any():
        raise ValueError("Duplicate global steel emissions comparison bars")
    values = selected.PercentChange.to_numpy(dtype=float)
    finite = np.r_[0., values[np.isfinite(values)]]
    low, high = float(finite.min()), float(finite.max())
    span = max(high - low, 5.)
    limits = (5 * np.floor((low - .20 * span) / 5),
              5 * np.ceil((high + .15 * span) / 5))
    selected["Panel"] = selected.Climate.map(dict(zip(CLIMATES, letters)))
    selected["PlotValue"] = selected.PercentChange
    selected["PlotUnit"] = "%"
    selected["Role"] = "plotted"
    selected["Plotted"] = True
    for column, climate in enumerate(CLIMATES):
        ax = axes[column]
        _clean_axes(ax, letters[column], CLIMATE_LABELS[climate],
                    "Steel emissions change\nfrom own reference (%)")
        ax.set_xlim(-.65, 2.)
        ax.set_xticks(POSITIONS, ["Medium\nsetting", "High\nsetting"], fontsize=6.3)
        ax.tick_params(axis="x", length=0, pad=5)
        ax.axhline(0, color="#53626B", linewidth=.7, zorder=2)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(20 if span > 40 else 10 if span > 20 else 5))
        for index, model in enumerate(("OMNIA-ENGAGE", "OPEN-PROM")):
            bars = selected.loc[selected.Model.eq(model) & selected.Climate.eq(climate)]
            heights = bars.set_index("Ambition").PercentChange.reindex(AMBITIONS).to_numpy(dtype=float)
            x = POSITIONS + OFFSETS[index]
            color = "#0072B2" if index == 0 else "#89969E"
            ax.bar(x, heights, WIDTH, color=color, edgecolor="#53626B" if index else "white",
                   linewidth=.45, hatch=HATCH if index else None, zorder=3)
            present = np.isfinite(heights)
            ax.scatter(x[present], heights[present], marker="D" if index else "o", s=13,
                       color=color, edgecolors="white", linewidths=.4, zorder=5)
            _net_labels(ax, x, heights, index, stagger=False)
        ax.set_ylim(*limits)
    return selected


class OpenPromComparisonBuilder:
    """Rows separate settings; columns separate climate pathways."""

    def __init__(self, data, config):
        self.data, self.config = data, config
        self.frame = data.frame()

    def figures(self):
        for number, metric, relative in ((1, "Energy", False), (2, "Energy", True),
                                          (3, "Emissions", False), (4, "Emissions", True)):
            yield self.build(number, metric, relative)

    @staticmethod
    def _handles(relative):
        handles = [
            Line2D([], [], color=COLORS["OPEN-PROM"], marker="D", markersize=3,
                   linewidth=1.2, label="OPEN-PROM"),
            Line2D([], [], color=COLORS["EU"], marker="o", markerfacecolor="white",
                   markersize=3, linewidth=1.2, label="OMNIA: CE in EU"),
            Line2D([], [], color=COLORS["EU+"], marker="s", markerfacecolor="white",
                   markersize=3, linewidth=1.2, label="OMNIA: CE in EU+"),
        ]
        if relative:
            handles = [handles[0], handles[2]]
        if not relative:
            handles += [Line2D([], [], color=COLORS["OPEN-PROM"], linestyle="--",
                               linewidth=.9, label="OPEN-PROM reference"),
                        Line2D([], [], color="#555C62", linestyle="--",
                               linewidth=.9, label="OMNIA no CE (both areas)")]
        return handles

    def _limits(self, frame, relative):
        values = frame["PercentChange" if relative else "Value"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        low, high = (min(0., float(finite.min())), max(0., float(finite.max()))) if finite.size else (0., 1.)
        span = max(high - low, 1.)
        return (low - .09 * span if low < 0 else 0., high + .12 * span)

    def _plot(self, ax, selected, *, model, coverage, reference, relative):
        line = selected.loc[selected.Model.eq(model)]
        line = line.loc[line.Ambition.eq("Reference") if reference else line.Coverage.eq(coverage)]
        if line.empty:
            raise ValueError(f"Missing plotted series: {model}, {coverage}, reference={reference}")
        line = line.sort_values("Year")
        if line.Year.duplicated().any():
            raise ValueError("A plotted series contains duplicate years")
        color = (COLORS["OPEN-PROM"] if model == "OPEN-PROM" else
                 "#555C62" if reference else COLORS[coverage])
        marker = None if reference else ("D" if model == "OPEN-PROM" else "o" if coverage == "EU" else "s")
        common_years = set(self.config["years"])
        markevery = [i for i, year in enumerate(line.Year) if year in common_years]
        ax.plot(line.Year.to_numpy(), line["PercentChange" if relative else "Value"].to_numpy(dtype=float),
                color=color, linewidth=.85 if reference else 1.2,
                linestyle="--" if reference else "-", marker=marker,
                markersize=3, markeredgewidth=.65,
                markerfacecolor=color if model == "OPEN-PROM" else "white",
                markevery=markevery, zorder=2 if reference else 3)

    def build(self, number, metric, relative):
        if metric == "Emissions" and relative:
            return self.emissions_response()
        frame = self.frame.loc[self.frame.Metric.eq(metric)].copy()
        if frame.empty:
            raise ValueError(f"No comparison observations for {metric}")
        energy = metric == "Energy"
        slug = f"fig{number:02d}_{'energy' if energy else 'emissions'}_{'response' if relative else 'levels'}"
        title = ("Global steel final energy" if energy else "Global steel: source-defined emissions")
        title += " | response to medium/high settings" if relative else " | absolute levels"
        fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 175 / 25.4), squeeze=False)
        fig.subplots_adjust(left=.11, right=.975, bottom=.205, top=.765,
                            wspace=.32, hspace=.61)
        fig.text(.035, .978, title, fontsize=8.2, weight="bold", va="top")
        subtitle = ("Change from each model's own reference in the same pathway and year"
                    if relative else "Source observations, 2019–2050 | dashed lines show the matching references")
        fig.text(.035, .945, subtitle, fontsize=6.6, color="#52616A", va="top")
        scope = ("Final energy: OPEN-PROM Mtoe converted to EJ; OMNIA sums five exact carrier parents."
                 if energy else "OMNIA = reported steel GHG (Mt CO₂e/yr); OPEN-PROM = energy-related CO₂ (Mt CO₂/yr).")
        fig.text(.035, .915, scope, fontsize=6, color="#374B55", va="top")
        fig.legend(handles=self._handles(relative), loc="upper left", bbox_to_anchor=(.035, .884),
                   ncol=3, fontsize=6, handlelength=2.4, columnspacing=1.7,
                   labelspacing=.9, frameon=False)
        limit_frame = frame.loc[~frame.Ambition.eq("Reference") & ~frame.Coverage.eq("EU")] if relative else frame
        limits = self._limits(limit_frame, relative)
        plot_frames = []
        for row, ambition in enumerate(AMBITIONS):
            for col, climate in enumerate(CLIMATES):
                ax = axes[row, col]
                panel = chr(97 + row * 2 + col)
                selected = frame.loc[frame.Climate.eq(climate) &
                                     frame.Ambition.isin(["Reference", ambition])].copy()
                if not relative:
                    self._plot(ax, selected, model="OPEN-PROM", coverage="Unspecified", reference=True, relative=False)
                    self._plot(ax, selected, model="OMNIA-ENGAGE", coverage="Reference", reference=True, relative=False)
                policies = selected.loc[selected.Ambition.eq(ambition)]
                self._plot(ax, policies, model="OPEN-PROM", coverage="Unspecified", reference=False, relative=relative)
                for coverage in (("EU+",) if relative else ("EU", "EU+")):
                    self._plot(ax, policies, model="OMNIA-ENGAGE", coverage=coverage, reference=False, relative=relative)
                if relative:
                    ax.axhline(0, color="#7C898E", linewidth=.55, zorder=1)
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="y", color="#DDE3E6", linewidth=.35, zorder=0)
                ax.set_axisbelow(True)
                ax.set_title(f"{ambition} | {CLIMATE_LABELS[climate]}", loc="left", fontsize=7, pad=9)
                ax.text(-.14, 1.045, panel, transform=ax.transAxes, fontsize=8, weight="bold")
                ax.set_xlim(self.config["historical_year"] - .7, self.config["comparison_year"] + .7)
                ax.set_ylim(*limits)
                ax.set_xticks([self.config["historical_year"], 2030, 2040, self.config["comparison_year"]])
                ax.tick_params(labelsize=6.2, pad=3)
                ax.yaxis.set_major_locator(ticker.MaxNLocator(5, min_n_ticks=3))
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}" if relative or not energy else "{x:,.1f}"))
                ax.set_xlabel("Year", fontsize=6.5, labelpad=3)
                ylabel = ("Change from own reference (%)" if relative else
                          "Final energy (EJ yr$^{-1}$)" if energy else "Source-defined emissions (Mt yr$^{-1}$)")
                ax.set_ylabel(ylabel, fontsize=6.4, labelpad=4)
                if relative:
                    final = policies.loc[policies.Year.eq(self.config["comparison_year"])]
                    labels = []
                    for model, coverage, label in (("OPEN-PROM", "Unspecified", "OPEN"),
                                                   ("OMNIA-ENGAGE", "EU+", "EU+")):
                        values = final.loc[final.Model.eq(model) & final.Coverage.eq(coverage), "PercentChange"]
                        labels.append(f"{label} {_signed(values.iloc[0])}" if len(values) else f"{label} unavailable")
                    ax.text(0, -.325, f"{self.config['comparison_year']}: " + "  ·  ".join(labels),
                            transform=ax.transAxes, fontsize=5.5, color="#374B55", va="top")
                selected["Panel"] = panel
                selected["PlotValue"] = selected["PercentChange" if relative else "Value"]
                selected["PlotUnit"] = "%" if relative else selected.Unit
                selected["Role"] = np.where(relative & (selected.Ambition.eq("Reference") | selected.Coverage.eq("EU")),
                                            "context", "plotted")
                plot_frames.append(selected)
        notes = [
            ("All values are global. OMNIA EU+ = CE in EU plus partners (17 regions)."
             if relative else
             "All values are global. EU/EU+ identify OMNIA's CE adoption area; EU+ = EU plus partners (17 regions)."),
            "OPEN-PROM medium/high policy content and adoption geography are undocumented; labels are paired descriptively.",
            ("Different historical levels are retained; no calibration or production-scope adjustment is imposed."
             if energy else "CO₂/GHG and process-accounting boundaries differ. Percentages do not harmonise the emissions scopes."),
            ("OMNIA missing activity = zero (provisional); OPEN-PROM uses complete reported World totals."
             if self.config["missing_activity"] == "zero" else
             "OMNIA retains missing activity; OPEN-PROM requires complete reported World totals."),
            "OPEN-PROM retains annual observations; OMNIA markers show supplied years. Lines connect observations without smoothing.",
        ]
        for y, note in zip((.108, .086, .064, .042, .020), notes):
            fig.text(.035, y, note, fontsize=5.2, color="#52616A", va="bottom")
        caption = self._caption(metric, relative)
        return ReportFigure(slug, str(number), title,
                            "How do the global steel responses compare across model, climate and OMNIA adoption area?",
                            caption, fig, pd.concat(plot_frames, ignore_index=True), 4)

    def emissions_response(self):
        """Match the reference emissions figure's two-panel paired-bar style."""
        from .europe_comparison import HATCH

        year = self.config["comparison_year"]
        title = "Global steel emissions: response to medium/high settings"
        fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 105 / 25.4), squeeze=False)
        for column, ax in enumerate(axes[0]):
            ax.set_position([.105 if column == 0 else .610, .300, .365, .370])
        fig.text(.035, .970, title, fontsize=8, weight="bold", va="top")
        fig.text(.035, .900, f"{year} | OMNIA CE in EU+ | change from each model's own pathway reference",
                 fontsize=6.7, color="#52616A", va="top")
        fig.legend(handles=[Patch(facecolor="#0072B2", edgecolor="#53626B", linewidth=.45,
                                  label="OMNIA-ENGAGE · EU+ · left"),
                            Patch(facecolor="#89969E", edgecolor="#53626B", linewidth=.45,
                                  hatch=HATCH, label="OPEN-PROM · right")],
                   loc="upper left", bbox_to_anchor=(.035, .835), ncol=2,
                   fontsize=6.2, handlelength=1.8, handletextpad=.6, columnspacing=2, frameon=False)
        plotted = draw_openprom_emissions_panels(axes[0], self.frame, year)
        data = self.frame.loc[self.frame.Metric.eq("Emissions")].copy()
        mask = data.Year.eq(year) & data.Ambition.isin(AMBITIONS) & (
            (data.Model.eq("OMNIA-ENGAGE") & data.Coverage.eq("EU+")) |
            (data.Model.eq("OPEN-PROM") & data.Coverage.eq("Unspecified")))
        data["Plotted"] = mask
        data["ContextOnly"] = ~mask
        data["Role"] = np.where(mask, "plotted", "context")
        data["Panel"] = np.where(mask, data.Climate.map({"ndc": "a", "baseline": "b"}), "")
        data["PlotValue"] = data.PercentChange.where(mask)
        data["PlotUnit"] = "%"
        if len(plotted) != int(mask.sum()):
            raise ValueError("Global steel emissions plot/export selections disagree")
        notes = [
            "OMNIA: reported steel GHG; OPEN-PROM: energy-related steel CO₂. Gas/process boundaries differ.",
            f"Each model uses its own {year} pathway reference; capture is not subtracted again.",
            "OMNIA CE in EU+ (17 regions); OPEN-PROM intervention and adoption geography are undocumented.",
            ("OMNIA missing activity = zero (provisional). Medium/high labels do not establish equivalent policies."
             if self.config["missing_activity"] == "zero" else
             "OMNIA missing activity is retained. Medium/high labels do not establish equivalent policies."),
        ]
        for y, note in zip((.160, .120, .080, .040), notes):
            fig.text(.035, y, note, fontsize=5.4, color="#52616A", va="bottom")
        caption = (
            f"Global steel emissions responses in {year}, with NDC on the left and NDC+LTT on the right. "
            "Each medium/high category pairs OMNIA-ENGAGE CE in EU plus partners (left solid blue bar) "
            "with OPEN-PROM's corresponding ambition setting (right grey hatched bar). EU-only OMNIA cases "
            "are not plotted. Each value is 100 × (setting − reference) / reference in the same model, climate "
            "and year. OMNIA uses the pathway's no-CE scenario; OPEN-PROM uses its supplied matching CSV reference. "
            "All totals are global; EU+ denotes the 17-region CE adoption area, not the area aggregated. "
            "OMNIA selects Emissions|GHG|Industry|Iron and Steel in MtCO2e/yr; OPEN-PROM selects "
            "Emissions|CO2|Energy|Demand|Industry|Iron and Steel in Mt CO2/yr. OPEN-PROM has no steel-specific "
            "process-emissions or GHG total. The different gas and accounting scopes are not harmonised by "
            "percentages. The medium/high labels do not establish equivalent interventions or adoption geography. "
            "Capture is not subtracted again. Markers identify observed endpoints, including zero; missing values "
            "remain gaps. The two panels share a percentage scale. Absolute quantities and references remain in "
            "the source CSV, and unplotted EU-only cases and other years are explicitly marked as context. "
            + notes[3])
        return ReportFigure("fig04_emissions_response", "4", title,
                            "How do global steel emissions respond under OMNIA EU+ and OPEN-PROM settings?",
                            caption, fig, data, 2)

    def _caption(self, metric, relative):
        text = ("Global steel final-energy demand" if metric == "Energy" else
                "Global steel source-defined emissions: OMNIA reported GHG versus OPEN-PROM energy-related CO2")
        text += (", shown as 100 × (setting − reference) / reference in the same model, climate pathway and year. "
                 if relative else ", shown in absolute source quantities with dashed matching reference trajectories. ")
        text += (
            "Rows show medium then high settings; columns show NDC then NDC+LTT. "
            "OPEN-PROM's corresponding NDC/NDC+LTT CSV supplies its reference; OMNIA uses the same climate's no-CE scenario. "
            + ("Only OMNIA EU+ and OPEN-PROM are plotted; EU-only OMNIA remains unplotted CSV context. Every plotted quantity is global. "
               if relative else "The two OMNIA curves change CE adoption area (EU or EU plus partners); every plotted quantity is global. ") +
            "OPEN-PROM medium/high scenario labels do not establish equivalent interventions or geographic coverage. "
            "OPEN-PROM uses the supplied World aggregate once; OMNIA sums its 28 model regions once. "
            "OPEN-PROM annual observations and OMNIA source years are retained from 2019 through 2050, "
            "without interpolation onto a common grid, smoothing, extrapolation or historical recalibration. "
        )
        text += (
            "OPEN-PROM selects Final Energy|Industry|Iron and Steel in Mtoe and multiplies by 0.041868 for EJ/yr. "
            "OMNIA sums exactly Electricity, Gases, Hydrogen, Liquids and Solids under the steel final-energy parent, in EJ/yr. "
            "Fuel children and energy-transformation inputs are not added. Final-energy units are aligned; full process boundaries remain source-defined. "
            if metric == "Energy" else
            "OMNIA selects Emissions|GHG|Industry|Iron and Steel in MtCO2e/yr. OPEN-PROM selects "
            "Emissions|CO2|Energy|Demand|Industry|Iron and Steel in Mt CO2/yr. The latter equals its gross series minus "
            "reported capture within rounding; capture is not subtracted again. OPEN-PROM supplies no steel-specific "
            "process-emissions or GHG total. The two source quantities have different gas and accounting scopes; neither "
            "absolute levels nor percentage responses establish harmonised steel-emissions comparisons. "
        )
        return text + (
            "OMNIA missing additive activity follows the configured provisional zero treatment; OPEN-PROM requires complete reported World totals. "
            if self.config["missing_activity"] == "zero" else
            "Missing OMNIA observations are preserved; incomplete aggregates appear as gaps. OPEN-PROM requires complete reported World totals. "
        ) + "Raw quantities, units, references, source scenarios and conversion factors accompany the figure data."


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_openprom_comparison(config, base):
    """Recalculate and render into a dedicated collection directory."""
    from .openprom_comparison_data import OpenPromComparisonData
    from .consolidation import _save_individual_figure
    import openpyxl

    base = Path(base)
    data = OpenPromComparisonData(config, base)
    builder = OpenPromComparisonBuilder(data, config)
    directory = (base / config["output"]).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    records, plotted, captions = [], [], []
    with plt.rc_context(COMPARISON_STYLE), PdfPages(directory / config.get("combined_pdf", "report_figures.pdf"),
            metadata={"Title": "Global steel: OMNIA-ENGAGE and OPEN-PROM"}) as book:
        for spec in builder.figures():
            check_figure_bounds(spec.figure, spec.slug)
            preserved = _save_individual_figure(spec, directory, config)
            book.savefig(spec.figure)
            width, height = spec.figure.get_size_inches() * 25.4
            plt.close(spec.figure)
            records.append({"number": spec.number, "slug": spec.slug, "title": spec.title,
                            "question": spec.question, "caption": spec.caption, "section": spec.section,
                            "panel_count": spec.panel_count, "width_mm": float(width), "height_mm": float(height),
                            "formats": config["formats"], "individual_artifacts_preserved": preserved})
            plotted.append(spec.data.assign(Figure=spec.slug))
            captions.append(f"## Figure {spec.number}: {spec.title}\n\n{spec.caption}\n")
            print(f"{'Preserved' if preserved else 'Rendered'} global steel comparison/{spec.slug}", flush=True)
    frame = data.frame()
    frame.to_csv(directory / "global_steel_metrics.csv", index=False)
    pd.concat(plotted, ignore_index=True).to_csv(directory / "source_data.csv", index=False)
    summary = response_summary(frame, config["comparison_year"])
    summary.to_csv(directory / "comparison_2050.csv", index=False)
    data.diagnostics().to_csv(directory / "capture_diagnostics.csv", index=False)
    data.scenario_mapping().to_csv(directory / "scenario_mapping.csv", index=False)
    data.missing_inputs.to_csv(directory / "missing_inputs.csv", index=False)
    checks = data.aggregation_checks
    checks.to_csv(directory / "aggregation_checks.csv", index=False)
    audit = {"generated_utc": datetime.now(timezone.utc).isoformat(), "config": config,
             "sources": {str(key): {"file": str(path.relative_to(base.resolve())) if path.is_relative_to(base.resolve()) else str(path),
                                    "sha256": _sha256(path)} for key, path in data.sources.items()},
             "figure_count": len(records), "metric_row_count": len(frame),
             "missing_input_count": len(data.missing_inputs), "aggregation_check_count": len(checks),
             "versions": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                          "numpy": np.__version__, "pandas": pd.__version__, "openpyxl": openpyxl.__version__},
             "notes": [*data.notes,
                       "The source-defined emissions have different gas/process boundaries; percentages do not harmonise them.",
                       "OPEN-PROM medium/high is paired descriptively with OMNIA medium/high CE; policy and adoption geography equivalence is unconfirmed.",
                       "All calculations read the raw sources directly; previous exploration caches and generated figures are not inputs."]}
    (directory / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (directory / "captions.md").write_text("# Global steel comparison captions\n\n" + "\n".join(captions), encoding="utf-8")
    _write_methods(directory, config)
    _write_gallery(directory, config, records, summary)
    if config.get("combined_comparison_config"):
        _export_combined_emissions(data, config, base)
    print(f"Saved four global steel comparison figures to {directory}", flush=True)


def _export_combined_emissions(openprom_data, config, base):
    """Recalculate the European row without modifying its original collection."""
    from .emissions_comparison import build_combined_emissions
    from .europe_relative_data import EuropeRelativeData

    europe_config_path = (base / config["combined_comparison_config"]).resolve()
    europe_config = json.loads(europe_config_path.read_text(encoding="utf-8"))
    if europe_config["comparison_year"] != config["comparison_year"]:
        raise ValueError("The two emissions comparisons must use the same comparison year")
    europe_data = EuropeRelativeData(europe_config, europe_config_path.parent)
    directory = (base / config["combined_comparison_output"]).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(COMPARISON_STYLE):
        spec = build_combined_emissions(europe_data.emissions_response(), openprom_data.frame(),
                                        year=config["comparison_year"])
        check_figure_bounds(spec.figure, spec.slug)
        for fmt in config["formats"]:
            metadata = {"Title": spec.title, "Description": spec.caption, "Date": None} if fmt == "svg" else None
            spec.figure.savefig(directory / f"{spec.slug}.{fmt}", dpi=config["dpi"], metadata=metadata)
        width, height = spec.figure.get_size_inches() * 25.4
        plt.close(spec.figure)
    spec.data.to_csv(directory / f"{spec.slug}.csv", index=False)
    spec.data.to_csv(directory / "source_data.csv", index=False)
    plotted = spec.data.loc[spec.data.Plotted]
    plotted.to_csv(directory / "comparison_2050.csv", index=False)
    record = {"number": spec.number, "slug": spec.slug, "title": spec.title,
              "question": spec.question, "caption": spec.caption, "section": spec.section,
              "panel_count": spec.panel_count, "width_mm": float(width), "height_mm": float(height),
              "formats": config["formats"]}
    source_paths = {**openprom_data.sources,
                    "GCAM": europe_data.source_data.gcam_source,
                    "European region mapping": (europe_config_path.parent / europe_config["region_mapping"]).resolve()}
    audit = {"generated_utc": datetime.now(timezone.utc).isoformat(),
             "sources": {name: {"file": str(path), "sha256": _sha256(path)} for name, path in source_paths.items()},
             "openprom_config": config, "europe_config": europe_config,
             "figure_count": 1, "panel_count": 4, "plotted_row_count": len(plotted),
             "notes": ["Panels a,b retain the European all-industry EU-only CE comparison with GCAM.",
                       "Panels c,d retain global steel, OMNIA EU+ versus OPEN-PROM; EU-only OMNIA is not plotted.",
                       "Each row has its own shared percentage scale and source-defined emissions inventories.",
                       "Existing European figures and other collections are not modified by this export.",
                       "Both rows calculate from raw sources through their existing adapters; figure CSVs and artwork are not calculation inputs."]}
    (directory / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps([record], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (directory / "captions.md").write_text(f"# Combined emissions comparison\n\n{spec.caption}\n", encoding="utf-8")
    methods = (
        "# Combined emissions comparison methods\n\n"
        "Regenerate with `python -m omnia_results --config figures.openprom_comparison.261016.json`. "
        "The OPEN-PROM config names the European comparison config and the separate combined-output directory.\n\n"
        "The top row reproduces the European all-industry comparison at 2050 for EU-only medium/high CE: "
        "OMNIA reported Emissions|GHG|Industry versus GCAM industry energy-demand plus industrial-process Kyoto gases. "
        "The bottom row compares global steel under OMNIA EU+ and OPEN-PROM medium/high settings: "
        "OMNIA sector GHG versus OPEN-PROM steel energy-related CO2. Both rows use their original source adapters, "
        "scenarios, exact units and own-model same-year references. Gas/process boundaries are not harmonised. "
        "EU/EU+ denotes policy adoption; output geography differs between the rows. The percentage scales match "
        "within each row and are independent between rows. Only the OMNIA EU+ cases enter the lower row. "
        "The OPEN-PROM intervention and adoption geography remain unspecified.\n\n"
        "Original absolute quantities and denominator provenance accompany every bar. PlotValue/PlotUnit "
        "identify percentages; Value/Unit retain each input adapter's original row definition. Plotted and "
        "PlotRole distinguish the sixteen plotted bars from reference and decomposition context. "
        "ComparisonRow and Panel identify the figure's two distinct scopes. Carbon capture is not subtracted again. "
        "Missing activity retains the configurations' provisional zero assumption for OMNIA; unavailable "
        "responses remain gaps. Source hashes and both configurations are recorded in audit.json. "
        "The existing European figure is preserved.\n")
    (directory / "methods.md").write_text(methods, encoding="utf-8")
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Emissions comparison · GCAM and OPEN-PROM</title>
<style>body{{font:15px/1.6 Arial,sans-serif;color:#233139;background:#f4f5f3;margin:0}}main{{max-width:1080px;margin:auto;padding:36px 24px}}a{{color:#006887}}img{{width:100%;height:auto}}.figure-card{{background:white;padding:22px;margin:24px 0}}figure{{margin:0}}.downloads{{display:flex;gap:16px}}.scope{{background:#e7eeed;padding:16px}}figcaption{{font-size:13px}}h3{{font-size:23px}}</style>
</head><body><main><h1>Emissions responses: GCAM and OPEN-PROM comparisons</h1>
<p class="scope">Top: European all-industry emissions, EU-only CE. Bottom: global steel, OMNIA CE in EU+ versus OPEN-PROM. Gas/process inventories and policy coverage remain different. Each model uses its own 2050 reference; scales match within each row.</p>
<p><a href="{spec.slug}.pdf">Download four-panel figure · PDF</a> · <a href="comparison_2050.csv">2050 plotted values</a> · <a href="methods.md">Methods</a> · <a href="audit.json">Audit</a></p>
{_figure_card(record)}</main></body></html>'''
    (directory / "index.html").write_text(page, encoding="utf-8")
    print(f"Rendered four-panel emissions comparison to {directory}", flush=True)


def _write_methods(directory, config):
    text = f"""# Global steel: OMNIA-ENGAGE and OPEN-PROM

Regenerate from the raw files with Python 3.11+ and the project dependencies:

```powershell
python -m omnia_results --config {config.get('config_file', 'figures.openprom_comparison.261016.json')}
```

## Purpose and source selection

The collection compares global final energy and source-defined steel emissions under NDC and NDC+LTT. Absolute-level Figures 1 and 3 and the summary tables retain both EU and EU-plus-partner OMNIA CE adoption areas. Response Figures 2 and 4 show only OMNIA EU+ alongside OPEN-PROM; EU-only cases remain unplotted context. Figure 2 retains four time-series panels. Figure 4 uses two paired-bar panels at 2050, styled like the European all-industry comparison. The four sheets of `{config['openprom_workbook']}` supply OPEN-PROM medium/high cases. Its two configured CSVs supply the matching climate-pathway references. OMNIA reads `{config['input']}` directly; `ndc_*` is NDC and `baseline_*` is NDC+LTT. Exact identifiers, sheets and reference scenarios are retained in `scenario_mapping.csv`. No previous figure data or temporary exploration cache is a calculation input.

The OPEN-PROM workbook does not document the intervention behind Medium/High or its adoption geography. Matching these labels with OMNIA medium/high CE is descriptive and does not establish equivalent policies, scenario assumptions or causal effects. The two OPEN-PROM CSVs are treated as the supplied pathway references, without assuming an independently confirmed no-CE definition.

## Global accounting and years

OPEN-PROM selects `World` once. Country rows, the EU aggregate and World are never added together. Aggregation checks compare World with the disjoint 13-group partition that includes EU once. OMNIA sums its 28 source regions once. Both OMNIA EU and EU+ curves are **global totals**: EU/EU+ specifies the CE adoption area, not the aggregation boundary. EU is EUE/EUM/EUW (three regions); EU+ also includes 14 partner regions (17 in total). The source suffix `_gbl` does not mean worldwide CE adoption.

The trajectory figures retain OPEN-PROM's annual 2019–2050 observations and OMNIA's 2019, 2024, 2030, 2035, 2040 and 2050 observations. Markers indicate common supplied comparison years; all other OPEN-PROM annual observations remain on the line. Lines connect observed model outputs; no interpolation onto a common grid, smoothing, extrapolation or calibration is introduced. Figure 4 instead selects only 2050 emissions responses, with NDC left and NDC+LTT right; each medium/high category pairs solid blue OMNIA EU+ on the left with grey hatched OPEN-PROM on the right, plus endpoint markers and signed percentage labels. All 2050 comparisons use actual 2050 observations. Historical diagnostics in the metric table use the model/climate reference's 2019 observation.

## Final energy

OPEN-PROM uses the exact parent `Final Energy|Industry|Iron and Steel` in Mtoe. Multiplication by **0.041868** converts Mtoe to EJ. Its 26 supplied carrier components provide an accounting check; they are not added to the total. OMNIA sums exactly `Final Energy|Industry|Iron and Steel|Electricity`, `|Gases`, `|Hydrogen`, `|Liquids` and `|Solids` in EJ/yr. Detailed children overlap those parents and are excluded. This is final energy, not primary energy, electricity-generation input or lifecycle energy. Unit conversion does not establish identical process or feedstock coverage.

## Emissions boundaries and carbon capture

OMNIA uses `Emissions|GHG|Industry|Iron and Steel`, matched exactly to **MtCO2e/yr**. OPEN-PROM uses `Emissions|CO2|Energy|Demand|Industry|Iron and Steel` in **Mt CO2/yr**. OPEN-PROM's steel variable is energy-related CO2, while OMNIA reports sector GHG. OPEN-PROM has aggregate industrial-process CO2 and economy-wide Kyoto-gas totals, but no steel-specific process CO2 or steel GHG total. Those aggregates cannot be assigned to steel from this export. Electricity-related upstream emissions and full sector process boundaries are not documented as harmonised.

The absolute-emissions figure retains both source units explicitly. Its shared numerical mass axis is a descriptive display of source quantities, not a harmonised emissions inventory or an estimate of model bias. The percentage figure compares each source quantity with its own reference; it also retains the differing scopes. Differences between models must not be attributed solely to CE assumptions or technical performance.

OPEN-PROM's reported net series equals its gross steel energy-CO2 series minus reported steel capture within rounding. Net emissions are used directly; capture is not subtracted again. `capture_diagnostics.csv` retains net, gross and capture quantities, reference values and responses where available. OMNIA capture is separate; no gross GHG inventory is reconstructed by adding captured CO2 to GHG.

## Responses and reference denominators

Each response is `100 × (setting − reference) / reference`, using the **same model, climate pathway, steel variable and model year**. OMNIA uses its climate's no-CE scenario for both adoption areas. OPEN-PROM uses the corresponding supplied CSV reference. The percentage is not a change from 2019. Missing, nonfinite or nonpositive references produce gaps. Positive values indicate increases, negative values decreases; signed outcomes are preserved.

Absolute figures retain the references as dashed curves. Energy response Figure 2 shows only OPEN-PROM and OMNIA EU+, with 2050 values printed below each panel; the orange EU-only curve, its legend entry and endpoint text are omitted. Emissions response Figure 4 also excludes EU-only OMNIA and shows eight 2050 bars across two climate panels, with medium/high categories. Common axis limits apply across the four panels in Figures 1–3 and across the two panels in Figure 4. Unplotted observations, including EU-only cases and other years, are flagged as context in the response CSVs.

## Missing inputs and validation

OMNIA follows the configured `{config['missing_activity']}` missing-activity treatment. The default provisionally treats missing additive activity as zero; blank emissions and absent regional carrier rows are documented in `missing_inputs.csv`. This remains an assumption about the OMNIA export convention. Entirely absent required variable/unit combinations fail. OPEN-PROM required World totals must be complete; blanks are never filled with zero. Infinite inputs, duplicate required source keys and mismatched scenarios/units are rejected. Missing or nonfinite reference quantities never produce a defined percentage.

The XLSX reader decodes shared/inline strings and converts numeric strings before calculation. It streams actual data rows rather than trusting trailing formatted worksheet ranges. `aggregation_checks.csv` records source accounting checks, including OPEN-PROM fuel totals, gross/net/capture and regional reconciliation. Checks use tolerances for the supplied four-decimal source rounding; no source values are corrected to force closure.

## Artifacts

Figures 1/2 show final-energy levels/responses; Figures 3/4 show source-defined emissions levels/responses. Each has PDF, editable SVG, 600 dpi PNG and figure CSV. The four-page PDF collects them in that order. `global_steel_metrics.csv` contains one row per metric/model/scenario/year; `source_data.csv` also records figure/panel, plotted value/unit and context roles. `comparison_2050.csv` retains levels, absolute changes, references and percentages at full precision. The gallery presents 2050 totals and responses together. Captions, scenario mapping, diagnostics, missing inputs, checks and source SHA-256 hashes accompany the exports. The configured preserved Figures 1 and 3 retain their reviewed individual files only after fresh source CSV and PNG verification.

The same command also exports a separate four-panel emissions figure in `{config.get('combined_comparison_output', 'report_figures_emissions_comparison_261019')}`. It places the original European all-industry/EU-only comparison with GCAM above the global-steel/OMNIA-EU+ comparison with OPEN-PROM. Scales match within rows; geography, sectors and emissions inventories remain different between rows. The original GCAM artwork is not modified. Both rows are calculated from raw inputs, and the combined figure includes source data, methods and an audit.
"""
    (directory / "methods.md").write_text(text, encoding="utf-8")


def _summary_table(summary, metric):
    rows = []
    for climate in CLIMATES:
        for ambition in AMBITIONS:
            cells = []
            for model, coverage in (("OPEN-PROM", "Unspecified"), ("OMNIA-ENGAGE", "EU"), ("OMNIA-ENGAGE", "EU+")):
                selected = summary.loc[summary.Metric.eq(metric) & summary.Climate.eq(climate) &
                                       summary.Ambition.eq(ambition) & summary.Model.eq(model) &
                                       summary.Coverage.eq(coverage)]
                if len(selected) != 1:
                    raise ValueError("2050 gallery summary requires exactly one observation per comparison")
                item = selected.iloc[0]
                value = f"{item.Value:,.2f}" if np.isfinite(item.Value) else "unavailable"
                cells.append(f"<td>{value}<br><strong>{escape(_signed(item.PercentChange))}</strong></td>")
            rows.append(f"<tr><th scope='row'>{CLIMATE_LABELS[climate]}</th><td>{ambition}</td>{''.join(cells)}</tr>")
    units = "Final energy: all levels in EJ/yr." if metric == "Energy" else "Emissions: OPEN-PROM levels in Mt CO₂/yr; OMNIA levels in Mt CO₂e/yr. Scopes differ."
    return (f"<div class='table-wrap'><table><caption>{escape(units)}<br>2050 level, with change from own 2050 reference below.</caption>"
            "<thead><tr><th>Climate</th><th>Setting</th><th>OPEN-PROM</th><th>OMNIA · EU</th><th>OMNIA · EU+</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def _write_gallery(directory, config, records, summary):
    cards = "\n".join(_figure_card(record) for record in records)
    combined = escape(config.get("combined_pdf", "report_figures.pdf"), quote=True)
    missing_note = ("OMNIA missing activity follows the provisional zero assumption."
                    if config["missing_activity"] == "zero" else
                    "Missing activity is retained; incomplete aggregates appear as gaps.")
    combined_link = ""
    if config.get("combined_comparison_output"):
        relative = os.path.relpath(config["combined_comparison_output"], config["output"]).replace("\\", "/")
        combined_link = (f'<p><a href="{escape(relative, quote=True)}/index.html">'
                         'Combined four-panel GCAM and OPEN-PROM emissions comparison</a></p>')
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Global steel · OMNIA-ENGAGE / OPEN-PROM</title>
<style>
*{{box-sizing:border-box}}body{{font-family:Arial,Helvetica,sans-serif;background:#f4f5f3;color:#233139;margin:0;line-height:1.65}}
main{{max-width:1080px;margin:auto;padding:42px 26px}}h1{{font-size:32px;line-height:1.2}}a{{color:#006887;text-underline-offset:3px}}
.intro{{max-width:900px;color:#52616a}}.download{{display:inline-block;background:#173f4b;color:white;padding:10px 16px;border-radius:4px}}
.scope{{padding:14px 18px;background:#e7eeed;border-left:3px solid #557b82}}.figure-card{{margin:32px 0;background:white;border:1px solid #d7e0df;border-radius:5px}}
.figure-heading{{padding:20px 26px 0}}.figure-number{{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:#607780}}h3{{margin:5px 0;font-size:22px}}
.purpose{{color:#607780}}figure{{margin:0;padding:16px 26px}}figure img{{display:block;width:100%;height:auto}}figcaption{{font-size:13px;line-height:1.65;margin-top:20px}}
.downloads{{padding:0 26px 20px;display:flex;flex-wrap:wrap;gap:14px;font-size:13px}}footer{{color:#607780;font-size:12px;margin-top:30px}}
.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;background:white;width:100%;margin:18px 0 28px;font-size:14px}}caption{{text-align:left;font-size:13px;color:#52616a;margin-bottom:8px}}
th,td{{text-align:left;padding:12px;border-bottom:1px solid #dce3e2}}td strong{{font-size:12px;color:#52616a}}nav a{{margin-right:18px}}
@media(max-width:600px){{main{{padding:24px 12px}}.figure-heading,figure{{padding-left:12px;padding-right:12px}}h1{{font-size:25px}}th,td{{padding:8px;font-size:12px}}}}
</style></head><body><main>
<p>OMNIA-ENGAGE / OPEN-PROM · Global steel · 2019–2050</p>
<h1>Steel energy and emissions across scenarios</h1>
<p class="intro">Absolute levels and changes from each model's own reference. Absolute figures and tables retain both OMNIA EU and EU+ cases. Response figures show only OMNIA EU+ alongside OPEN-PROM: energy trajectories in Figure 2 and paired 2050 emissions bars in Figure 4, under NDC and NDC+LTT.</p>
<p><a class="download" href="{combined}">Download all four figures · PDF</a></p>
{combined_link}
<p class="scope"><strong>Every quantity is global.</strong> EU/EU+ identifies OMNIA's CE adoption area, not the area being summed. OPEN-PROM's medium/high intervention and adoption geography are undocumented; the labels do not establish equivalent policies.<br><strong>Emissions scopes differ:</strong> OMNIA reports steel GHG; OPEN-PROM reports steel energy-related CO₂. Its export has no matching steel process-emissions or GHG total. Percentage responses retain this difference. {missing_note}</p>
<nav aria-label="Figures"><a href="#figure1">Energy levels</a><a href="#figure2">Energy response</a><a href="#figure3">Emissions levels</a><a href="#figure4">Emissions response</a></nav>
<h2>2050 final energy</h2>{_summary_table(summary, 'Energy')}
<h2>2050 source-defined emissions</h2>{_summary_table(summary, 'Emissions')}
<p><a href="comparison_2050.csv">2050 comparison CSV</a> · <a href="global_steel_metrics.csv">All metric values</a> · <a href="source_data.csv">Figure source data</a> · <a href="methods.md">Methods</a> · <a href="captions.md">Captions</a><br>
<a href="scenario_mapping.csv">Scenario mapping</a> · <a href="capture_diagnostics.csv">Capture diagnostics</a> · <a href="aggregation_checks.csv">Accounting checks</a> · <a href="missing_inputs.csv">Missing inputs</a> · <a href="audit.json">Audit</a></p>
{cards}
<footer>Calculated directly from the raw OMNIA CSV and OPEN-PROM workbook/reference CSVs. Each figure includes vector PDF, editable SVG, {int(config['dpi'])} dpi PNG and full-precision source data.</footer>
</main></body></html>'''
    (directory / "index.html").write_text(page, encoding="utf-8")
