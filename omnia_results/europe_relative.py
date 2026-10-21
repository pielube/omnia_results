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
from .europe_relative_routes import RelativeRouteProductionFigures


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


class EuropeRelativeBuilder(RelativeEmissionsFigures, RelativeRouteProductionFigures):
    def __init__(self, data, config):
        self.data, self.config = data, config

    def figures(self):
        yield self.material_response()
        yield self.energy_response()
        yield self.emissions_response()
        if self.config.get("include_steel_route_figure", False):
            yield self.material_response_by_steel_route()

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
            "Route contributions are exported as context but are not stacked in this total-output figure. "
            "The updated GCAM workbook also reports primary and secondary steel; the optional Figure 4 "
            "compares those routes using their own no-CE production references. " + self._common_caption())
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
    data.emissions_scope_checks.to_csv(directory / "gcam_sector_scope_checks.csv", index=False)
    all_data.loc[(all_data.Figure == "fig03_emissions_response") & (all_data.Model == "GCAM") &
                 (all_data.Role == "component")].to_csv(directory / "gcam_emissions_context.csv", index=False)
    source.scenario_mapping().to_csv(directory / "scenario_mapping.csv", index=False)
    source.geography_membership().to_csv(directory / "geography_membership.csv", index=False)
    variables = pd.concat([source.variable_mapping(), data.sector_emissions_mapping()], ignore_index=True)
    if config.get("include_steel_route_figure", False):
        routes = data.material_response_by_steel_route()
        route_mapping = routes.loc[routes.Route.isin(["Primary", "Secondary"]),
                                   ["Model", "Metric", "Sector", "Route", "SourceVariables",
                                    "SourceCoefficients"]].drop_duplicates().copy()
        route_mapping["SourceUnit"], route_mapping["FigureUnit"] = "Mt/yr", "%"
        route_mapping["Definition"] = "Percentage change relative to the same model/pathway steel route's own no-CE output"
        variables = pd.concat([variables, route_mapping], ignore_index=True)
        data.steel_route_checks.to_csv(directory / "steel_route_checks.csv", index=False)
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
                               "emissions_reference": "Each model's combined cement, iron-and-steel and aluminium same-year no-CE emissions",
                               "emissions_inventories": {"OMNIA-ENGAGE": "Reported sector GHG in MtCO2e/yr", "GCAM": "Reported sector CO2 in Mt CO2/yr"},
                               "emissions_context_reference": "Unplotted all-industry context retains each model's own all-industry GHG denominator",
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
                       "Figure 1 plots total production responses, with unplotted route contributions retained as context.",
                       "Emissions panels a,b compare the combined cement, iron-and-steel and aluminium response for both models; other industry is excluded from numerator and denominator.",
                       "Sector contributions and all-industry responses remain unplotted CSV context with ContextOnly=True and Plotted=False.",
                       "OMNIA sector GHG is compared with GCAM reported sector CO2; no GHG equivalence or undocumented GWP conversion is assumed.",
                       "GCAM cement sums non-metallic-minerals energy CO2 and cement-process CO2; aluminium selects the specific child, without adding its overlapping non-ferrous parent.",
                       "gcam_sector_scope_checks.csv verifies the non-metallic-minerals final-energy parent equals its cement child for all selected source years and scenarios.",
                       "Under the configured zero policy, missing NDC+LTT aluminium GHG in all three OMNIA EU regions remains provisional zero and affects the plotted three-sector sum; flags and source rows retain that assumption.",
                       "The companion absolute-value figures remain in report_figures_europe_comparison_261014."]}
    if config.get("include_steel_route_figure", False):
        checks = data.steel_route_checks
        error = checks.Difference.abs().max()
        audit["normalisation"]["steel_route_reference"] = "Figure 4 uses each model/pathway/year steel route's own no-CE output, not total steel"
        audit["normalisation"]["steel_route_percentages_additive"] = False
        audit["complete_steel_route_checks"] = int(checks.Complete.sum())
        audit["max_steel_route_closure_error_mt"] = float(error) if pd.notna(error) else None
        audit["undefined_route_response_count"] = int((~routes.ResponseDefined).sum())
        audit["route_responses_with_missing_reference"] = int(routes.ReferenceMissingInputCount.gt(0).sum())
        audit["notes"].extend([
            "Figure 4 adds separate primary and secondary steel columns alongside unchanged cement/aluminium responses; all eight panels share one percentage scale.",
            "Figure 4 steel percentages use route-specific denominators and are not additive; they are excluded from percentage-point contribution checks.",
            "steel_route_checks.csv verifies GCAM reported primary plus secondary steel equals its total in every complete selected scenario/year.",
            "OMNIA NDC+LTT no-CE secondary steel is missing in all three EU regions at the comparison year; the percentage is undefined under both zero and preserve policies. Panel g shows a gap and explicit annotation.",
            "Existing Figures 1-3 individual PDF/SVG/PNG/CSV artifacts are preserved after source-CSV and rendered-PNG equality checks when listed in preserve_individual_figures.",
        ])
    (directory / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (directory / "captions.md").write_text("# European CE percentage-response captions\n\n" + "\n".join(captions), encoding="utf-8")
    _write_methods(directory, config)
    _write_gallery(directory, config, records)
    print(f"Saved {len(records)} percentage-response figures to {directory}", flush=True)


def _write_methods(directory, config):
    include_routes = config.get("include_steel_route_figure", False)
    route_methods = ""
    if include_routes:
        route_methods = """## Figure 4: material production with separate steel routes

The new **2 × 4** figure retains cement and aluminium from Figure 1, with separate primary and secondary iron-and-steel columns between them. NDC appears above NDC+LTT. Paired medium/high EU-only CE bars retain the model colours, hatch and endpoint markers. All eight panels share one percentage scale.

For steel, the formula is **100 × (CE route output − no-CE route output) / no-CE route output**. Each route uses its own same-model, same-pathway, same-year no-CE production. These are production changes, not route shares or changes in shares. The two percentages are not additive and do not use the total-steel denominator from Figure 1's unplotted contribution rows. Cement and aluminium responses equal Figure 1 exactly.

OMNIA selects `Production|Iron and Steel|Primary` and `Production|Iron and Steel|Secondary`; GCAM selects `Production|Iron and Steel|Steel|Primary` and `Production|Iron and Steel|Steel|Secondary`, all in Mt/yr. GCAM's reported primary-plus-secondary sum is checked against its reported steel total for each selected source scenario/year in `steel_route_checks.csv`. Incomplete checks remain flagged; no missing route output is fabricated. These checks are distinct from percentage-point contribution closure.

**OMNIA NDC+LTT no-CE secondary steel is blank in all three EU regions in 2050.** Under the configured zero policy its route reference becomes provisional zero, so the percentage remains undefined. Preserving the blanks also leaves it undefined. Panel g therefore contains only GCAM bars and explicitly identifies the unavailable OMNIA reference; it does not substitute a total-steel denominator or infer a production value. Unavailable, nonfinite and nonpositive aggregate route references produce gaps, while observed zero changes retain endpoint markers. Partly missing OMNIA inputs follow the configured activity policy and retain explicit completeness and provisional-zero flags. `ResponseDefined`, `ReferenceComplete`, `ReferenceIssue`, `ReferenceMissingInputCount` and the full source-input lists distinguish missing references from reported physical zero production.

"""
    artifact_description = (
        "Figure 4 is 240 × 175 mm. The four-page combined PDF is "
        f"`{config['combined_pdf']}`; the earlier three-page collection remains available. "
        "Existing Figures 1–3 individual artwork and CSVs are preserved after equality checks; "
        "the new figure and collection metadata are generated from the raw sources."
        if include_routes else
        f"The combined three-page PDF is `{config['combined_pdf']}`. Figures are generated directly from the raw sources.")
    text = f"""# European CE percentage-response methods

Regenerate the collection:

```powershell
python -m omnia_results --config {config['config_file']}
```

## What the normalisation compares

All figures compare EU-only medium/high CE with **the same model's own {config['comparison_year']} no-CE result**, matching climate pathway and production/energy/emissions scope. The formula is **100 × (CE − no CE) / no CE**. A model's reference is never taken from the other model or the other climate pathway. Separate steel routes, when shown, each use their own route's reference.

No CE is 0% by construction. Equal zero references do not establish agreement in absolute production or energy. These figures focus on CE responses; the [companion absolute collection](../report_figures_europe_comparison_261014/index.html) documents the underlying level differences. Normalisation also does not establish identical process boundaries, CE assumptions or calibration. Historical observations are omitted because these are CE-versus-no-CE comparisons at one future year.

The sources remain `{config['input']}` and `{config['gcam_input']}`, sheet `{config.get('gcam_sheet', 'Sheet1')}`. Both are read directly; earlier figure CSVs and artwork are not calculation inputs. Input and regional-mapping SHA-256 hashes, software versions and the complete configuration are in `audit.json`.

## Scenario and geographic selection

NDC appears above NDC+LTT in the production and energy figures and to its left in Figure 3. `ndc_*` and `NDC_*` identify the NDC cases; OMNIA `baseline_*` and GCAM `LTT_*` are paired under NDC+LTT. Matching display labels do not establish identical climate-policy assumptions. Only medium/high CE applied in the EU is compared, each against its own no-CE reference; EU+ cases are excluded.

OMNIA sums EUE, EUM and EUW. GCAM selects its reported EU-27 aggregate once, without adding overlapping country rows. The aggregate is checked against available-country sums. OMNIA's model regions also contain Greenland, Faroe Islands, Andorra, Monaco and San Marino, so its boundary is an approximation to GCAM's EU-27. `scenario_mapping.csv`, `geography_membership.csv` and `gcam_aggregation_checks.csv` retain the definitions and checks.

## Figure 1: total material-production response

The 2 × 3 layout retains cement, iron and steel, and aluminium columns, with paired OMNIA/GCAM bars for medium and high CE. Every panel uses the same percentage axis. Endpoint markers identify observed changes, including near-zero values; missing responses have neither bar nor marker. Labels round to 0.1%, while coordinates and source CSVs retain the full values.

Cement excludes clinker. OMNIA steel and aluminium total output sums primary and secondary routes. GCAM uses its source-reported metal production. **GCAM aluminium's primary/secondary route coverage is unconfirmed.** The chart shows total reported-output responses without claiming equivalent route coverage.

OMNIA route contributions can be large and offsetting even when net production changes little. Only net percentages are plotted, so these internal movements do not dominate the total-output comparison. Component contributions remain in the figure CSV with `Role=component` and `Plotted=False`. Net rows have `Role=net` and `Plotted=True`.

## Figure 2: carrier contributions to total-energy response

Each coloured segment is **100 × (carrier CE energy − carrier no-CE energy) / no-CE total sector energy**. The denominator is shared by every carrier in the same model, pathway and sector. Segments are percentage-point contributions to the total percentage change. They are **not percentages relative to the carrier's own no-CE energy**, which would not add to the total-energy response.

Positive and negative contributions stack separately around zero, retaining increases as well as decreases. Their algebraic sum equals the net total-energy percentage change shown by a black diamond and label. All six panels share one percentage scale. `contribution_checks.csv` records component closure, including whether each comparison is complete.

OMNIA final energy sums exact parent carriers, excluding nested children. GCAM retains its reported sector-energy parent for its total and reference. Its Unallocated component is reported total minus its supplied carrier parents, calculated separately for CE and no CE; the plotted contribution is the difference between those residuals divided by the no-CE total. The residual's fuel/process attribution remains unidentified. **GCAM aluminium's alumina-refining coverage is unconfirmed.** The figure therefore compares source-defined responses rather than harmonised energy intensities.

## Figure 3: combined cement, steel and aluminium responses

Figure 3 retains its **1 × 2** layout, paired model bars, colours and endpoint markers, with NDC on the left and NDC+LTT on the right. Each bar now shows **100 × (sum of three-sector CE emissions − sum of three-sector no-CE emissions) / sum of three-sector no-CE emissions** within the same model, pathway, geography and {config['comparison_year']} year. The sum covers cement, iron and steel, and aluminium only. Other industry is excluded from both numerator and denominator. Individual-sector percentages are not averaged. The two panels share one percentage scale.

**OMNIA reports sector GHG in MtCO2e/yr; GCAM reports sector CO2 in Mt CO2/yr.** The updated workbook adds useful CO2 disaggregation, but still supplies no complete three-sector Kyoto/GHG inventory. This is a comparison of responses within the reported quantities, with gas coverage, GWP conventions and process boundaries explicitly source-defined.

| Model/sector | Exact source selection |
| --- | --- |
| OMNIA cement | Emissions\\|GHG\\|Industry\\|Non-Metallic Minerals\\|Cement |
| OMNIA steel | Emissions\\|GHG\\|Industry\\|Iron and Steel |
| OMNIA aluminium | Emissions\\|GHG\\|Industry\\|Non-Ferrous Metals\\|Aluminum |
| GCAM cement energy | Emissions\\|CO2\\|Energy\\|Demand\\|Industry\\|Non-Metallic Minerals |
| GCAM cement processes | Emissions\\|CO2\\|Industrial Processes\\|Non-Metallic Minerals\\|Cement |
| GCAM steel | Emissions\\|CO2\\|Energy\\|Demand\\|Industry\\|Iron and Steel |
| GCAM aluminium | Emissions\\|CO2\\|Energy\\|Demand\\|Industry\\|Non-Ferrous Metals\\|Aluminum |

GCAM's non-metallic-minerals energy category is assigned to cement: the reported final-energy parent equals its cement child in this workbook. A runtime check verifies this identity for each selected scenario/year and exports the values in `gcam_sector_scope_checks.csv`. The cement-process parent overlaps the selected cement child; the non-ferrous-metals energy-CO2 parent overlaps the selected aluminium child. These parents are not added again. No additional steel/aluminium process-CO2 child is exported, so none is fabricated. Capture is not subtracted again.

Only combined three-sector net responses are displayed: `PanelKind=three_sector`, `Role=net`, `Plotted=True`. Sector contributions remain context, each divided by its own model's combined three-sector no-CE total; they reconstruct the net response. All-industry GHG responses are also retained as unplotted context, with their separate industry denominators. `ContextOnly=True`, `DisplayRole=context` and `Plotted=False` identify unplotted rows, which have no displayed panel assignment. GCAM's unplotted contributions are also exported in `gcam_emissions_context.csv`, and `gcam_emissions_availability.csv` inventories the available source quantities.

Negative reported emissions are retained. The NDC+LTT no-CE steel GHG value in OMNIA is negative, so individual-sector percentages are not used. Under the configured zero policy, **NDC+LTT aluminium GHG blanks in all three OMNIA EU regions, in no CE and both CE cases, are provisionally treated as zero**. This affects the displayed combined sum and does not establish physical zero emissions. `UsesProvisionalZeros`, `AllEUAluminiumMissingAsZero`, raw missing-input lists and scenario flags retain this assumption in the CSV. Under `preserve`, incomplete sector sums remain gaps.

{route_methods}## Missing inputs and provenance

OMNIA follows the configured `{config['missing_activity']}` activity policy; the default provisional zero treatment is retained from the existing report and identified where plotted inputs use it. In Figure 3, missing sector inputs affect the displayed combined three-sector response and are flagged explicitly. GCAM missing aggregate values remain gaps. Required variables/units and GCAM aggregate rows cannot be fabricated. Nonfinite or nonpositive total references produce undefined contributions and net responses. `missing_inputs.csv` records affected selected inputs and their treatment.

Each source-data row retains its CE and no-CE absolute quantities, signed numerator, reference denominator, conversion factor, original scenarios, year, variables, units, coefficients and regions. `comparison_summary.csv` collects the net responses. Components use `Unit=percentage points`; net rows use `Unit=%`. The `Plotted` flag identifies series selected for display; an undefined response still has no mark. Emissions records additionally identify `PanelKind`, `EmissionsScope` and `EmissionsCoverage`. Separate steel-route records identify `ProductionGroup`, `Route` and their own-route `DenominatorScope`.

Figures 1–2 are 180 × 175 mm; the single-row Figure 3 is 180 × 105 mm. All include vector PDF, editable SVG text and 600 dpi PNG exports. {artifact_description} The updated GCAM production values retain their supplied precision. Automated bounds checks and visual review check label placement. No values are rescaled between models, displaced, clipped, smoothed or extrapolated to improve apparent agreement.
"""
    (directory / "methods.md").write_text(text, encoding="utf-8")


def _write_gallery(directory, config, records):
    cards = "\n".join(_figure_card(record) for record in records)
    count = len(records)
    route_intro = (" An additional production figure separates primary and secondary steel, each relative to its own route's no-CE output."
                   if config.get("include_steel_route_figure", False) else "")
    route_link = (' · <a href="steel_route_checks.csv">Steel route checks</a>'
                  if config.get("include_steel_route_figure", False) else "")
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
<p class="intro">{count} figures show {config['comparison_year']} percentage changes from each model's own no-CE case: production, final energy, and emissions. The emissions figure compares combined cement, steel and aluminium responses in two panels, with NDC on the left and NDC+LTT on the right. Both pathways retain medium/high EU-only CE.{route_intro}</p>
<p><a class="download" href="{escape(config['combined_pdf'], quote=True)}">Download all {count} figures · PDF</a></p>
<p class="scope">No CE is 0% by construction within each model, pathway and reference scope. This normalisation does not establish agreement in absolute levels or harmonise process boundaries. Emissions bars sum cement, iron and steel, and aluminium only in both models. OMNIA reports sector GHG; GCAM reports sector CO2, with cement energy and processes plus reported steel/aluminium energy CO2. These inventories remain source-defined. Aluminium production and energy process coverage remain unconfirmed. The <a href="../report_figures_europe_comparison_261014/index.html">absolute-value comparison</a> remains available.</p>
<p><a href="comparison_summary.csv">Net percentage responses</a> · <a href="source_data.csv">All source data</a> · <a href="contribution_checks.csv">Contribution checks</a> · <a href="methods.md">Methods</a> · <a href="captions.md">Captions</a> · <a href="audit.json">Audit</a><br><a href="gcam_emissions_availability.csv">GCAM emissions coverage</a> · <a href="gcam_emissions_context.csv">GCAM context contributions</a> · <a href="gcam_sector_scope_checks.csv">Cement scope checks</a>{route_link}</p>
{cards}<footer>Calculated directly from {escape(config['input'])} and {escape(config['gcam_input'])}. PDF, editable SVG, 600 dpi PNG and full-precision CSV accompany each figure.</footer>
</main></body></html>'''
    (directory / "index.html").write_text(page, encoding="utf-8")

