"""Global steel comparisons that retain each model's reported boundaries."""

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
        limits = self._limits(frame.loc[~frame.Ambition.eq("Reference")] if relative else frame, relative)
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
                for coverage in ("EU", "EU+"):
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
                                                   ("OMNIA-ENGAGE", "EU", "EU"),
                                                   ("OMNIA-ENGAGE", "EU+", "EU+")):
                        values = final.loc[final.Model.eq(model) & final.Coverage.eq(coverage), "PercentChange"]
                        labels.append(f"{label} {_signed(values.iloc[0])}" if len(values) else f"{label} unavailable")
                    ax.text(0, -.325, f"{self.config['comparison_year']}: " + "  ·  ".join(labels),
                            transform=ax.transAxes, fontsize=5.5, color="#374B55", va="top")
                selected["Panel"] = panel
                selected["PlotValue"] = selected["PercentChange" if relative else "Value"]
                selected["PlotUnit"] = "%" if relative else selected.Unit
                selected["Role"] = np.where(relative & selected.Ambition.eq("Reference"), "context", "plotted")
                plot_frames.append(selected)
        notes = [
            "All values are global. EU/EU+ identify OMNIA's CE adoption area; EU+ = EU plus partners (17 regions).",
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

    def _caption(self, metric, relative):
        text = ("Global steel final-energy demand" if metric == "Energy" else
                "Global steel source-defined emissions: OMNIA reported GHG versus OPEN-PROM energy-related CO2")
        text += (", shown as 100 × (setting − reference) / reference in the same model, climate pathway and year. "
                 if relative else ", shown in absolute source quantities with dashed matching reference trajectories. ")
        text += (
            "Rows show medium then high settings; columns show NDC then NDC+LTT. "
            "OPEN-PROM's corresponding NDC/NDC+LTT CSV supplies its reference; OMNIA uses the same climate's no-CE scenario. "
            "The two OMNIA curves change CE adoption area (EU or EU plus partners); every plotted quantity is global. "
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
            plotted.append(spec.data.assign(Figure=spec.slug))
            captions.append(f"## Figure {spec.number}: {spec.title}\n\n{spec.caption}\n")
            print(f"Rendered global steel comparison/{spec.slug}", flush=True)
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
    print(f"Saved four global steel comparison figures to {directory}", flush=True)


def _write_methods(directory, config):
    text = f"""# Global steel: OMNIA-ENGAGE and OPEN-PROM

Regenerate from the raw files with Python 3.11+ and the project dependencies:

```powershell
python -m omnia_results --config {config.get('config_file', 'figures.openprom_comparison.261016.json')}
```

## Purpose and source selection

The collection compares global final energy and source-defined steel emissions under NDC and NDC+LTT, with medium/high OPEN-PROM settings and both EU and EU-plus-partner OMNIA CE adoption areas. The four sheets of `{config['openprom_workbook']}` supply OPEN-PROM medium/high cases. Its two configured CSVs supply the matching climate-pathway references. OMNIA reads `{config['input']}` directly; `ndc_*` is NDC and `baseline_*` is NDC+LTT. Exact identifiers, sheets and reference scenarios are retained in `scenario_mapping.csv`. No previous figure data or temporary exploration cache is a calculation input.

The OPEN-PROM workbook does not document the intervention behind Medium/High or its adoption geography. Matching these labels with OMNIA medium/high CE is descriptive and does not establish equivalent policies, scenario assumptions or causal effects. The two OPEN-PROM CSVs are treated as the supplied pathway references, without assuming an independently confirmed no-CE definition.

## Global accounting and years

OPEN-PROM selects `World` once. Country rows, the EU aggregate and World are never added together. Aggregation checks compare World with the disjoint 13-group partition that includes EU once. OMNIA sums its 28 source regions once. Both OMNIA EU and EU+ curves are **global totals**: EU/EU+ specifies the CE adoption area, not the aggregation boundary. EU is EUE/EUM/EUW (three regions); EU+ also includes 14 partner regions (17 in total). The source suffix `_gbl` does not mean worldwide CE adoption.

The main figures retain OPEN-PROM's annual 2019–2050 observations and OMNIA's 2019, 2024, 2030, 2035, 2040 and 2050 observations. Markers indicate common supplied comparison years; all other OPEN-PROM annual observations remain on the line. Lines connect observed model outputs; no interpolation onto a common grid, smoothing, extrapolation or calibration is introduced. All 2050 comparisons use actual 2050 observations. Historical diagnostics in the metric table use the model/climate reference's 2019 observation.

## Final energy

OPEN-PROM uses the exact parent `Final Energy|Industry|Iron and Steel` in Mtoe. Multiplication by **0.041868** converts Mtoe to EJ. Its 26 supplied carrier components provide an accounting check; they are not added to the total. OMNIA sums exactly `Final Energy|Industry|Iron and Steel|Electricity`, `|Gases`, `|Hydrogen`, `|Liquids` and `|Solids` in EJ/yr. Detailed children overlap those parents and are excluded. This is final energy, not primary energy, electricity-generation input or lifecycle energy. Unit conversion does not establish identical process or feedstock coverage.

## Emissions boundaries and carbon capture

OMNIA uses `Emissions|GHG|Industry|Iron and Steel`, matched exactly to **MtCO2e/yr**. OPEN-PROM uses `Emissions|CO2|Energy|Demand|Industry|Iron and Steel` in **Mt CO2/yr**. OPEN-PROM's steel variable is energy-related CO2, while OMNIA reports sector GHG. OPEN-PROM has aggregate industrial-process CO2 and economy-wide Kyoto-gas totals, but no steel-specific process CO2 or steel GHG total. Those aggregates cannot be assigned to steel from this export. Electricity-related upstream emissions and full sector process boundaries are not documented as harmonised.

The absolute-emissions figure retains both source units explicitly. Its shared numerical mass axis is a descriptive display of source quantities, not a harmonised emissions inventory or an estimate of model bias. The percentage figure compares each source quantity with its own reference; it also retains the differing scopes. Differences between models must not be attributed solely to CE assumptions or technical performance.

OPEN-PROM's reported net series equals its gross steel energy-CO2 series minus reported steel capture within rounding. Net emissions are used directly; capture is not subtracted again. `capture_diagnostics.csv` retains net, gross and capture quantities, reference values and responses where available. OMNIA capture is separate; no gross GHG inventory is reconstructed by adding captured CO2 to GHG.

## Responses and reference denominators

Each response is `100 × (setting − reference) / reference`, using the **same model, climate pathway, steel variable and model year**. OMNIA uses its climate's no-CE scenario for both adoption areas. OPEN-PROM uses the corresponding supplied CSV reference. The percentage is not a change from 2019. Missing, nonfinite or nonpositive references produce gaps. Positive values indicate increases, negative values decreases; signed outcomes are preserved.

Absolute figures retain the references as dashed curves. Response figures show only the three medium/high comparisons (OPEN-PROM, OMNIA EU, OMNIA EU+), with 2050 values printed below each panel. Common axis limits across the four panels within each figure allow comparisons between settings and climate pathways.

## Missing inputs and validation

OMNIA follows the configured `{config['missing_activity']}` missing-activity treatment. The default provisionally treats missing additive activity as zero; blank emissions and absent regional carrier rows are documented in `missing_inputs.csv`. This remains an assumption about the OMNIA export convention. Entirely absent required variable/unit combinations fail. OPEN-PROM required World totals must be complete; blanks are never filled with zero. Infinite inputs, duplicate required source keys and mismatched scenarios/units are rejected. Missing or nonfinite reference quantities never produce a defined percentage.

The XLSX reader decodes shared/inline strings and converts numeric strings before calculation. It streams actual data rows rather than trusting trailing formatted worksheet ranges. `aggregation_checks.csv` records source accounting checks, including OPEN-PROM fuel totals, gross/net/capture and regional reconciliation. Checks use tolerances for the supplied four-decimal source rounding; no source values are corrected to force closure.

## Artifacts

Figures 1/2 show final-energy levels/responses; Figures 3/4 show source-defined emissions levels/responses. Each has PDF, editable SVG, 600 dpi PNG and figure CSV. The four-page PDF collects them in that order. `global_steel_metrics.csv` contains one row per metric/model/scenario/year; `source_data.csv` also records figure/panel, plotted value/unit and context roles. `comparison_2050.csv` retains levels, absolute changes, references and percentages at full precision. The gallery presents 2050 totals and responses together. Captions, scenario mapping, diagnostics, missing inputs, checks and source SHA-256 hashes accompany the exports.
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
<p class="intro">Absolute levels and changes from each model's own reference. Four OPEN-PROM medium/high scenarios are compared with OMNIA's medium/high CE in the EU and EU plus partners, under both NDC and NDC+LTT.</p>
<p><a class="download" href="{combined}">Download all four figures · PDF</a></p>
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
