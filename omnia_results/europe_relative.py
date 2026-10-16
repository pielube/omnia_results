"""Within-model CE response figures for the European model comparison."""

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

from .europe_comparison import (CLIMATES, CLIMATE_LABELS, COMPARISON_STYLE,
                                ENERGY_COLORS, HATCH, MODELS, _clean_axes)
from .metrics import SECTORS
from .report import ReportFigure, check_figure_bounds
from .report_gallery import _figure_card
from .europe_relative_emissions import RelativeEmissionsFigures


POLICIES = ("medce_eu", "highce_eu")
POSITIONS = np.array([0., 1.35])
OFFSETS = (-.21, .21)
WIDTH = .35
MODEL_COLORS = {"OMNIA-ENGAGE": "#0072B2", "GCAM": "#89969E"}


def _label(value):
    if abs(value) < .05:
        return "0.0%"
    return f"{value:+.1f}%".replace("-", "\N{MINUS SIGN}")


def _model_legend(fig, *, production):
    handles = [Patch(facecolor=MODEL_COLORS[model] if production else "white",
                     edgecolor="#53626B", linewidth=.45, hatch=HATCH if index else None,
                     label=f"{model} · {'right' if index else 'left'}")
               for index, model in enumerate(MODELS)]
    if not production:
        handles.append(Line2D([], [], color="#202A30", marker="D", linestyle="none",
                              markeredgecolor="white", markeredgewidth=.4,
                              markersize=4, label="Total energy change"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.035, .891), ncol=len(handles),
               fontsize=6.2, handlelength=1.8, handletextpad=.6, columnspacing=2, frameon=False)


def _net_values(selected):
    return (selected.loc[selected.Role == "net"].set_index("Policy").Value
            .reindex(POLICIES).to_numpy(dtype=float))


def _net_labels(ax, x, values, model_index, *, stagger=True):
    for position, value in zip(x, values):
        if not np.isfinite(value):
            continue
        # Stagger negative labels to keep adjacent near-equal responses legible.
        shift = (-15 if stagger and model_index == 1 else -5) if value < -.05 else 5
        ax.annotate(_label(value), (position, value), xytext=(0, shift),
                    textcoords="offset points", ha="center", va="top" if shift < 0 else "bottom",
                    fontsize=5.9, color="#202A30", zorder=7,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": .35})


class EuropeRelativeBuilder(RelativeEmissionsFigures):
    def __init__(self, data, config):
        self.data, self.config = data, config

    def figures(self):
        yield self.material_response()
        yield self.energy_response()
        yield self.emissions_response()

    def _canvas(self, title, *, energy=False):
        fig, axes = plt.subplots(2, 3, figsize=(180 / 25.4, 175 / 25.4), squeeze=False)
        fig.subplots_adjust(left=.100, right=.985, bottom=.215, top=.745,
                            wspace=.47, hspace=.63)
        year = self.config["comparison_year"]
        fig.text(.035, .977, title, fontsize=8, weight="bold", va="top")
        fig.text(.035, .941, f"EU-only CE in {year} | change from each model's own {year} no-CE case",
                 fontsize=6.7, color="#52616A", va="top")
        _model_legend(fig, production=not energy)
        if energy:
            fig.legend(handles=[Patch(facecolor=color, label=series) for series, color in ENERGY_COLORS.items()],
                       loc="upper left", bbox_to_anchor=(.035, .843), ncol=6, fontsize=5.8,
                       columnspacing=1.1, handlelength=1.4, handletextpad=.5, frameon=False)
        for row, climate in enumerate(CLIMATES):
            for column, sector in enumerate(SECTORS):
                ax = axes[row, column]
                _clean_axes(ax, chr(97 + 3 * row + column), f"{sector.label} | {CLIMATE_LABELS[climate]}",
                            "Change from no CE (%)")
                ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
                ax.tick_params(labelsize=6.1)
                ax.set_xlim(-.65, 2.0)
                ax.set_xticks(POSITIONS, ["Medium CE\nEU", "High CE\nEU"], fontsize=6.3)
                ax.tick_params(axis="x", length=0, pad=5)
                ax.axhline(0, color="#53626B", linewidth=.7, zorder=2)
        return fig, axes

    def material_response(self):
        data = self.data.material_response().copy()
        data["Plotted"] = data.Role.eq("net")
        title = "Circular-economy effects on European material production"
        fig, axes = self._canvas(title)
        net = data.loc[data.Role == "net", "Value"].to_numpy(dtype=float)
        finite = net[np.isfinite(net)]
        low = min(0., float(finite.min())) if finite.size else 0.
        high = max(0., float(finite.max())) if finite.size else 0.
        span = max(high - low, 5.)
        limits = (5 * np.floor((low - .22 * span) / 5),
                  5 * np.ceil((high + .18 * span) / 5))
        for row, climate in enumerate(CLIMATES):
            for column, sector in enumerate(SECTORS):
                ax = axes[row, column]
                ax.set_ylim(*limits)
                ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
                for index, model in enumerate(MODELS):
                    selected = data.loc[(data.Model == model) & (data.Climate == climate) &
                                        (data.Sector == sector.label)]
                    values, x = _net_values(selected), POSITIONS + OFFSETS[index]
                    ax.bar(x, values, WIDTH, color=MODEL_COLORS[model],
                           edgecolor="#53626B" if index else "white", linewidth=.45,
                           hatch=HATCH if index else None, zorder=3)
                    # A zero-height bar is otherwise invisible. This marker also
                    # distinguishes an observed near-zero response from missing data.
                    present = np.isfinite(values)
                    ax.scatter(x[present], values[present], marker="o" if index == 0 else "D",
                               s=12, color=MODEL_COLORS[model], edgecolors="white", linewidths=.4, zorder=5)
                    _net_labels(ax, x, values, index)
        self._notes(fig, energy=False)
        caption = (
            f"Percentage changes in European material production in {self.config['comparison_year']}, "
            "with NDC above NDC+LTT and cement, iron and steel, and aluminium columns. Each policy category "
            "pairs OMNIA-ENGAGE on the left with hatched GCAM on the right. The medium and high EU-only CE "
            "scenarios are compared with each model's own no-CE scenario in the same pathway, sector and year: "
            "100 × (CE output − no-CE output) / no-CE output. No CE is the zero reference by construction. "
            "The same percentage scale is used across all six panels. Markers identify observed values, including "
            "near-zero responses; labels are rounded to 0.1% and are not the plotting coordinates. Missing values "
            "remain gaps. Cement excludes clinker; OMNIA metal output sums primary and secondary routes, while "
            "GCAM uses its source-reported metal series. GCAM aluminium's route coverage is unconfirmed. "
            "Route contributions are exported as context but are not stacked in this figure, because GCAM has "
            "no matching route decomposition. " + self._common_caption())
        return ReportFigure("fig01_material_production_response", "1", title,
                            "How much does EU-only CE change material output relative to each model's no-CE case?",
                            caption, fig, data, 6)

    def energy_response(self):
        data = self.data.energy_response().copy()
        data["Plotted"] = True
        title = "Circular-economy effects on European final-energy demand"
        fig, axes = self._canvas(title, energy=True)
        bounds, nets = [], []
        for row, climate in enumerate(CLIMATES):
            for column, sector in enumerate(SECTORS):
                ax = axes[row, column]
                for index, model in enumerate(MODELS):
                    selected = data.loc[(data.Model == model) & (data.Climate == climate) &
                                        (data.Sector == sector.label)]
                    positive, negative = np.zeros(2), np.zeros(2)
                    x = POSITIONS + OFFSETS[index]
                    components = selected.loc[selected.Role == "component"]
                    for series, color in ENERGY_COLORS.items():
                        if series not in set(components.Series):
                            continue
                        values = (components.loc[components.Series == series].set_index("Policy").Value
                                  .reindex(POLICIES).to_numpy(dtype=float))
                        bottom = np.where(values >= 0, positive, negative)
                        ax.bar(x, values, WIDTH, bottom=bottom, color=color,
                               edgecolor="#53626B" if index else "white", linewidth=.35,
                               hatch=HATCH if index else None, zorder=3)
                        positive += np.where(values >= 0, values, 0.)
                        negative += np.where(values < 0, values, 0.)
                        bounds.extend(positive[np.isfinite(positive)])
                        bounds.extend(negative[np.isfinite(negative)])
                        positive[~np.isfinite(values)], negative[~np.isfinite(values)] = np.nan, np.nan
                    values = _net_values(selected)
                    nets.extend(values[np.isfinite(values)])
                    ax.scatter(x, values, marker="D", s=17, color="#202A30", edgecolors="white",
                               linewidths=.5, zorder=6)
                    _net_labels(ax, x, values, index, stagger=False)
        all_values = np.asarray([0., *bounds, *nets], dtype=float)
        low, high = float(all_values.min()), float(all_values.max())
        span = max(high - low, 10.)
        limits = (10 * np.floor((low - .12 * span) / 10),
                  10 * np.ceil((high + .12 * span) / 10))
        for ax in axes.flat:
            ax.set_ylim(*limits)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(30))
        self._notes(fig, energy=True)
        caption = (
            f"Changes in European final-energy demand in {self.config['comparison_year']}, with NDC above "
            "NDC+LTT and cement, iron and steel, and aluminium columns. Paired bars place OMNIA-ENGAGE on the "
            "left and hatched GCAM on the right. Every fuel segment is 100 × (CE carrier energy − no-CE carrier "
            "energy) / no-CE total sector energy, using the same model, pathway, sector and year. These segments "
            "are percentage-point contributions to the total-energy percentage change; they are not "
            "carrier-specific percentage changes. Positive and negative contributions stack independently "
            "from zero. Black diamonds and signed labels show the net total-energy percentage change. "
            "All six panels share one percentage scale, retaining increases as well as decreases. "
            "Exact carrier parents are selected once. GCAM's total-energy reference retains its reported "
            "sector parent; Unallocated is the reported parent minus supplied carrier parents in each scenario. "
            "Its change remains an unidentified contribution, with no fuel or process attribution. GCAM "
            "aluminium's alumina-refining coverage is unconfirmed. " + self._common_caption())
        return ReportFigure("fig02_energy_response", "2", title,
                            "How do fuels contribute to the total-energy response to EU-only CE?",
                            caption, fig, data, 6)

    def _notes(self, fig, *, energy):
        year = self.config["comparison_year"]
        notes = [f"Reference: each model's own {year} no-CE case in the same pathway and sector (0%).",
                 ("Fuel segments: contributions in percentage points. Diamonds and labels: total energy change (%)."
                  if energy else "Bars and labels: total output change. Cement excludes clinker; OMNIA metals include both production routes."),
                 ("Aluminium: GCAM alumina-refining coverage is unconfirmed; Unallocated retains unidentified energy."
                  if energy else "Aluminium: GCAM production-route coverage is unconfirmed. Labels rounded to 0.1%."),
                 "Within-model normalisation; no absolute-level alignment assumed. EU-region boundaries differ slightly.",
                 ("OMNIA missing activity = zero (provisional); GCAM missing values remain gaps."
                  if self.config["missing_activity"] == "zero" else
                  "Missing activity retained in both models; incomplete references and responses remain gaps.")]
        for y, note in zip((.130, .105, .080, .055, .030), notes):
            fig.text(.035, y, note, fontsize=5.4, color="#52616A", va="bottom")

    @staticmethod
    def _common_caption():
        return (
            "Normalisation focuses on within-model CE responses and does not establish agreement in absolute "
            "production or energy levels, or harmonise the models' process boundaries. Historical bars are "
            "omitted because these are matched-year CE contrasts. OMNIA sums EUE, EUM and EUW; GCAM selects "
            "its EU-27 aggregate once. The OMNIA regions include some small non-EU countries and territories. "
            "Only medium/high CE applied in the EU enters; EU+ cases are excluded. Raw source quantities, "
            "reference totals, exact variables, regions and calculation coefficients accompany every result. "
            "A missing, nonfinite or nonpositive reference makes the response undefined.")


def generate_europe_relative(config, base):
    """Export the percentage-response collection from the raw model sources."""
    import openpyxl
    from .europe_relative_data import EuropeRelativeData
    from .consolidation import _save_individual_figure

    base = Path(base)
    data = EuropeRelativeData(config, base)
    source = data.source_data
    directory = (base / config["output"]).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    records, frames, captions = [], [], []
    with plt.rc_context(COMPARISON_STYLE), PdfPages(directory / config["combined_pdf"],
            metadata={"Title": "EU-only circular economy: within-model percentage responses"}) as book:
        for spec in EuropeRelativeBuilder(data, config).figures():
            check_figure_bounds(spec.figure, spec.slug)
            preserved = _save_individual_figure(spec, directory, config)
            if not preserved and "svg" in config["formats"]:
                path = directory / f"{spec.slug}.svg"
                text = path.read_text(encoding="utf-8")
                path.write_text("\n".join(line.rstrip(" \t") for line in text.splitlines()) + "\n", encoding="utf-8")
            book.savefig(spec.figure)
            width, height = spec.figure.get_size_inches() * 25.4
            plt.close(spec.figure)
            records.append({"number": spec.number, "slug": spec.slug, "title": spec.title,
                            "question": spec.question, "caption": spec.caption, "section": spec.section,
                            "panel_count": spec.panel_count, "width_mm": float(width), "height_mm": float(height),
                            "formats": config["formats"], "individual_artifacts_preserved": preserved})
            frames.append(spec.data.assign(Figure=spec.slug))
            captions.append(f"## Figure {spec.number}: {spec.title}\n\n{spec.caption}\n")
            print(f"{'Preserved' if preserved else 'Rendered'} Europe relative/{spec.slug}", flush=True)
    all_data = pd.concat(frames, ignore_index=True)
    all_data.to_csv(directory / "source_data.csv", index=False)
    all_data.loc[all_data.Role == "net"].to_csv(directory / "comparison_summary.csv", index=False)
    data.closure_checks.to_csv(directory / "contribution_checks.csv", index=False)
    data.emissions_availability().to_csv(directory / "gcam_emissions_availability.csv", index=False)
    all_data.loc[(all_data.Figure == "fig03_emissions_response") & (all_data.Model == "GCAM") &
                 (all_data.Role == "component")].to_csv(directory / "gcam_emissions_context.csv", index=False)
    source.scenario_mapping().to_csv(directory / "scenario_mapping.csv", index=False)
    source.geography_membership().to_csv(directory / "geography_membership.csv", index=False)
    variables = source.variable_mapping()
    variables.to_csv(directory / "variable_mapping.csv", index=False)
    source.aggregation_checks.to_csv(directory / "gcam_aggregation_checks.csv", index=False)
    source.missing_inputs.to_csv(directory / "missing_inputs.csv", index=False)
    closure_error = data.closure_checks.Difference.abs().max()
    audit = {"generated_utc": datetime.now(timezone.utc).isoformat(), "figure_count": len(records),
             "sources": {model: {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for model, path in source.sources.items()},
             "region_mapping_sha256": hashlib.sha256((base / config["region_mapping"]).read_bytes()).hexdigest(),
             "config": config,
             "normalisation": {"formula": "100 * (CE - no_CE) / no_CE_total",
                               "reference_matches": ["Model", "Climate", "Sector", "Year"],
                               "reference_year": config["comparison_year"],
                               "absolute_alignment_assumed": False,
                               "emissions_reference": "Each model's source-defined all-industry no-CE GHG",
                               "emissions_context_reference": "Unplotted OMNIA three-sector context uses combined cement, steel and aluminium no-CE GHG",
                               "component_unit": "percentage points", "net_unit": "%"},
             "source_row_counts": {model: len(raw) for model, raw in source.raw.items()},
             "missing_input_count": len(source.missing_inputs),
             "complete_contribution_checks": int(data.closure_checks.Complete.sum()),
             "max_contribution_closure_error_pp": float(closure_error) if pd.notna(closure_error) else None,
             "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                          "matplotlib": matplotlib.__version__, "openpyxl": openpyxl.__version__},
             "notes": [*source.notes, *data.notes,
                       "No historical bars or absolute quantities are plotted. Their raw comparison-year values remain in the source CSVs.",
                       "All six production panels share one percentage scale; all six energy panels share another common percentage scale.",
                       "Production plots only the net response; route contributions are exported as context to avoid implying matching GCAM routes.",
                       "Emissions panels a,b compare reported all-industry totals only; OMNIA selects Emissions|GHG|Industry directly, including sectors beyond cement, steel and aluminium.",
                       "OMNIA-only three-sector responses and industry decompositions remain unplotted CSV context with ContextOnly=True and Plotted=False.",
                       "GCAM provides partial sector CO2 variables but no matching cement/steel/aluminium GHG totals; these are not substituted for GHG.",
                       "Under the configured zero policy, missing NDC+LTT aluminium GHG in all three OMNIA EU regions is provisional zero in unplotted sector context; it does not replace the reported industry-parent inputs.",
                       "The companion absolute-value figures remain in report_figures_europe_comparison_261014."]}
    (directory / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (directory / "captions.md").write_text("# European CE percentage-response captions\n\n" + "\n".join(captions), encoding="utf-8")
    _write_methods(directory, config)
    _write_gallery(directory, config, records)
    print(f"Saved {len(records)} percentage-response figures to {directory}", flush=True)


def _write_methods(directory, config):
    text = f"""# European CE percentage-response methods

Regenerate the collection:

```powershell
python -m omnia_results --config {config['config_file']}
```

## What the normalisation compares

All three figures compare EU-only medium/high CE with **the same model's own {config['comparison_year']} no-CE result**, matching climate pathway and emissions/sector scope. The formula is **100 × (CE − no CE) / no CE**. A model's reference is never taken from the other model or the other climate pathway.

No CE is 0% by construction. Equal zero references do not establish agreement in absolute production or energy. These figures focus on CE responses; the [companion absolute collection](../report_figures_europe_comparison_261014/index.html) documents the underlying level differences. Normalisation also does not establish identical process boundaries, CE assumptions or calibration. Historical observations are omitted because these are CE-versus-no-CE comparisons at one future year.

The sources remain `{config['input']}` and `{config['gcam_input']}`, sheet `{config.get('gcam_sheet', 'Sheet1')}`. Both are read directly; earlier figure CSVs and artwork are not calculation inputs. Input and regional-mapping SHA-256 hashes, software versions and the complete configuration are in `audit.json`.

## Scenario and geographic selection

NDC appears above NDC+LTT in Figures 1–2 and to its left in Figure 3. `ndc_*` and `NDC_*` identify the NDC cases; OMNIA `baseline_*` and GCAM `LTT_*` are paired under NDC+LTT. Matching display labels do not establish identical climate-policy assumptions. Only medium/high CE applied in the EU is compared, each against its own no-CE reference; EU+ cases are excluded.

OMNIA sums EUE, EUM and EUW. GCAM selects its reported EU-27 aggregate once, without adding overlapping country rows. The aggregate is checked against available-country sums. OMNIA's model regions also contain Greenland, Faroe Islands, Andorra, Monaco and San Marino, so its boundary is an approximation to GCAM's EU-27. `scenario_mapping.csv`, `geography_membership.csv` and `gcam_aggregation_checks.csv` retain the definitions and checks.

## Figure 1: total material-production response

The 2 × 3 layout retains cement, iron and steel, and aluminium columns, with paired OMNIA/GCAM bars for medium and high CE. Every panel uses the same percentage axis. Endpoint markers identify observed changes, including near-zero values; missing responses have neither bar nor marker. Labels round to 0.1%, while coordinates and source CSVs retain the full values.

Cement excludes clinker. OMNIA steel and aluminium total output sums primary and secondary routes. GCAM uses its source-reported metal production. **GCAM aluminium's primary/secondary route coverage is unconfirmed.** The chart shows total reported-output responses without claiming equivalent route coverage.

OMNIA route contributions can be large and offsetting even when net production changes little. Only net percentages are plotted, so these internal movements do not dominate the total-output comparison. Component contributions remain in the figure CSV with `Role=component` and `Plotted=False`. Net rows have `Role=net` and `Plotted=True`.

## Figure 2: carrier contributions to total-energy response

Each coloured segment is **100 × (carrier CE energy − carrier no-CE energy) / no-CE total sector energy**. The denominator is shared by every carrier in the same model, pathway and sector. Segments are percentage-point contributions to the total percentage change. They are **not percentages relative to the carrier's own no-CE energy**, which would not add to the total-energy response.

Positive and negative contributions stack separately around zero, retaining increases as well as decreases. Their algebraic sum equals the net total-energy percentage change shown by a black diamond and label. All six panels share one percentage scale. `contribution_checks.csv` records component closure, including whether each comparison is complete.

OMNIA final energy sums exact parent carriers, excluding nested children. GCAM retains its reported sector-energy parent for its total and reference. Its Unallocated component is reported total minus its supplied carrier parents, calculated separately for CE and no CE; the plotted contribution is the difference between those residuals divided by the no-CE total. The residual's fuel/process attribution remains unidentified. **GCAM aluminium's alumina-refining coverage is unconfirmed.** The figure therefore compares source-defined responses rather than harmonised energy intensities.

## Figure 3: reported all-industry responses

Figure 3 uses a **1 × 2** layout with NDC on the left and NDC+LTT on the right. **Panels a,b** reproduce the earlier emissions figure's industry-level percentage comparison: paired OMNIA and GCAM bars, each divided by its own {config['comparison_year']} no-CE industry total. **OMNIA selects the reported all-industry parent `Emissions|GHG|Industry` directly**, in `MtCO2e/yr`. It includes sectors beyond cement, iron and steel, and aluminium; it is not formed by adding those three sectors. GCAM sums `Emissions|Kyoto Gases|Energy|Demand|Industry` and `Emissions|Kyoto Gases|Industrial Processes`, both in `Mt CO2-equiv/yr`. The two panels share a common percentage scale. This compares source-defined industrial responses. Their gas coverage, GWP conventions and process boundaries are not documented as harmonised.

No sector breakdown is plotted. The earlier OMNIA-only three-sector calculations remain **unplotted CSV context**, with `PanelKind=three_sector`, `ContextOnly=True`, `DisplayRole=context` and `Plotted=False`. Their contributions use `100 × (sector CE GHG − sector no-CE GHG) / combined three-sector no-CE GHG`; their net rows use the same combined three-sector reference. This contextual denominator differs from the industry denominator used by every displayed bar. These context rows have no displayed panel assignment.

In that unplotted context, individual-sector percentage changes are not used: the NDC+LTT no-CE steel GHG value is negative, and aluminium GHG inputs are blank. Contributions use the positive combined three-sector reference instead, while preserving negative reported emissions. Carbon capture is not subtracted again. Under the configured zero policy, **NDC+LTT aluminium GHG blanks in all three OMNIA EU regions, in no CE and both CE cases, are provisionally treated as zero in sector context**. This does not establish zero physical emissions, and these sector imputations do not replace the independently reported industry-parent inputs used in the chart. `UsesProvisionalZeros`, `AllEUAluminiumMissingAsZero`, raw missing-input lists and scenario flags retain the context assumption in the CSV. Under `preserve`, incomplete three-sector references and contributions remain gaps; the independently reported industry comparison can remain available.

### Available GCAM detail and the limit to sector matching

The supplied workbook contains no matched cement/steel/aluminium GHG or Kyoto-gas aggregates. Its available sector CO2 snippets have different boundaries:

| GCAM variable | Available coverage | Why it cannot substitute for OMNIA sector GHG |
| --- | --- | --- |
| `Emissions|CO2|Industrial Processes|Cement` | Cement process CO2 | Missing cement energy emissions and non-CO2 GHG. |
| `Emissions|CO2|Energy|Demand|Industry|Iron and Steel` | Steel energy-demand CO2 | Not an all-GHG sector total. |
| `Emissions|CO2|Energy|Demand|Industry|Non-Ferrous Metals` | Broader non-ferrous energy-demand CO2 | Neither aluminium-specific nor all-GHG. |

These quantities are not substituted for GHG or assigned undocumented warming potentials. `gcam_emissions_availability.csv` inventories every source emissions variable/unit and records its coverage limitations. GCAM's available energy-demand versus industrial-process GHG contributions are retained in `gcam_emissions_context.csv`, using its own no-CE industry total. OMNIA's industry-sector and Other industry residual contributions also remain in the figure CSV as context. `Plotted=False` identifies all emissions context; the chart plots industry nets only.

## Missing inputs and provenance

OMNIA follows the configured `{config['missing_activity']}` activity policy; the default provisional zero treatment is retained from the existing report and identified where plotted inputs use it. In Figure 3, missing sector inputs affect unplotted decomposition context, while the bars use independently reported industry-parent inputs. GCAM missing aggregate values remain gaps. Required variables/units and GCAM aggregate rows cannot be fabricated. Nonfinite or nonpositive total references produce undefined contributions and net responses. `missing_inputs.csv` records affected selected inputs and their treatment.

Each source-data row retains its CE and no-CE absolute quantities, signed numerator, total reference denominator, conversion factor, original scenarios, year, variables, units, coefficients and regions. `comparison_summary.csv` collects the net responses. Components use `Unit=percentage points`; net rows use `Unit=%`. The `Plotted` flag distinguishes chart marks from context. Emissions records additionally identify `PanelKind`, `EmissionsScope` and `EmissionsCoverage`.

Figures 1–2 are 180 × 175 mm; the single-row Figure 3 is 180 × 105 mm. All include vector PDF, editable SVG text and 600 dpi PNG exports, plus a combined three-page PDF (`{config['combined_pdf']}`). The reviewed individual Figures 1–2 are retained byte-for-byte after their freshly calculated CSV and PNG match the existing exports. Automated bounds checks and visual review check label placement. No values are rescaled between models, displaced, clipped, smoothed or extrapolated to improve apparent agreement.
"""
    (directory / "methods.md").write_text(text, encoding="utf-8")


def _write_gallery(directory, config, records):
    cards = "\n".join(_figure_card(record) for record in records)
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>European circular-economy percentage responses</title>
<style>
*{{box-sizing:border-box}}body{{font-family:Arial,Helvetica,sans-serif;background:#f4f5f3;color:#233139;margin:0;line-height:1.6}}
main{{max-width:1080px;margin:auto;padding:42px 26px}}h1{{font-size:32px;line-height:1.2}}a{{color:#006887;text-underline-offset:3px}}
.intro{{max-width:880px;color:#52616a}}.download{{display:inline-block;background:#173f4b;color:white;padding:10px 16px;border-radius:4px}}
.scope{{padding:12px 16px;background:#edf1f0;border-left:3px solid #667d83}}.figure-card{{margin:32px 0;background:white;border:1px solid #d7e0df;border-radius:5px}}
.figure-heading{{padding:20px 26px 0}}.figure-number{{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:#607780}}h3{{margin:5px 0;font-size:22px}}
.purpose{{color:#607780}}figure{{margin:0;padding:16px 26px}}figure img{{display:block;width:100%;height:auto}}figcaption{{font-size:13px;line-height:1.65;margin-top:20px}}
.downloads{{display:flex;flex-wrap:wrap;gap:14px;padding:0 26px 20px;font-size:13px}}footer{{color:#607780;font-size:12px}}
@media(max-width:600px){{main{{padding:24px 12px}}.figure-heading,figure{{padding-left:12px;padding-right:12px}}h1{{font-size:25px}}}}
</style></head><body><main><p>OMNIA-ENGAGE / GCAM · European industry</p>
<h1>Comparing the response to circular economy</h1>
<p class="intro">Three figures show {config['comparison_year']} percentage changes from each model's own no-CE case: production, final energy, and emissions. The emissions figure compares reported all-industry responses in two panels, with NDC on the left and NDC+LTT on the right. Both pathways retain medium/high EU-only CE.</p>
<p><a class="download" href="{escape(config['combined_pdf'], quote=True)}">Download all three figures · PDF</a></p>
<p class="scope">No CE is 0% by construction within each model, pathway and reference scope. This normalisation does not establish agreement in absolute levels or harmonise process boundaries. OMNIA's emissions bars use its reported all-industry parent, including sectors beyond cement, steel and aluminium. GCAM uses industry energy-demand plus industrial-process Kyoto gases and has no matching three-sector GHG totals. Aluminium production and energy process coverage remain unconfirmed. The <a href="../report_figures_europe_comparison_261014/index.html">absolute-value comparison</a> remains available.</p>
<p><a href="comparison_summary.csv">Net percentage responses</a> · <a href="source_data.csv">All source data</a> · <a href="contribution_checks.csv">Contribution checks</a> · <a href="methods.md">Methods</a> · <a href="captions.md">Captions</a> · <a href="audit.json">Audit</a><br><a href="gcam_emissions_availability.csv">GCAM emissions coverage</a> · <a href="gcam_emissions_context.csv">GCAM energy/process contributions</a></p>
{cards}<footer>Calculated directly from {escape(config['input'])} and {escape(config['gcam_input'])}. PDF, editable SVG, 600 dpi PNG and full-precision CSV accompany each figure.</footer>
</main></body></html>'''
    (directory / "index.html").write_text(page, encoding="utf-8")

