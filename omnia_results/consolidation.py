"""Refined report figures, calculated directly from the all-CE results source."""

from datetime import datetime, timezone
import hashlib
from html import escape
from io import BytesIO
import json
from pathlib import Path
import platform

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from .consolidation_data import CLIMATE_LABELS, ConsolidationData, climate_label
from .consolidation_producers import ProducerFigures
from .consolidation_energy import EnergyFigures
from .consolidation_emissions import EmissionsFigures
from .consolidation_materials import MaterialComparisonFigures
from .consolidation_energy_mix import EnergyMixFigures
from .consolidation_impacts import EmissionsResponseFigures
from .consolidation_costs import SectorCostFigures
from .consolidation_policy_coverage import PolicyCoverageFigures
from .metrics import SECTORS
from .plots import STYLE
from .report import ReportFigure, check_figure_bounds
from .report_gallery import _figure_card


CONSOLIDATION_STYLE = {**STYLE, "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"]}


class ConsolidationBuilder(ProducerFigures, EnergyFigures, EmissionsFigures,
                           MaterialComparisonFigures, EnergyMixFigures, EmissionsResponseFigures,
                           SectorCostFigures, PolicyCoverageFigures):
    def __init__(self, data: ConsolidationData, config: dict):
        self.data, self.config = data, config

    def figures(self):
        builders = {"production": self.production, "leading_producers": self.leading_producers,
                    "energy": self.energy, "emissions": self.emissions,
                    "material_production": self.material_production, "energy_mix": self.energy_mix,
                    "emissions_response": self.emissions_response, "sector_costs": self.sector_costs,
                    "material_policy_coverage": self.material_policy_coverage,
                    "energy_policy_coverage": self.energy_policy_coverage}
        requested = self.config.get("figures", ["production"])
        if not requested or len(set(requested)) != len(requested) or set(requested) - builders.keys():
            raise ValueError("Select unique supported consolidation figures: "
                             "production, leading_producers, energy, emissions, material_production, "
                             "energy_mix, emissions_response, sector_costs, "
                             "material_policy_coverage, energy_policy_coverage")
        for name in requested:
            yield builders[name]()

    def production(self) -> ReportFigure:
        data = self.data.production()
        scenarios = self.data.production_scenarios()
        simplified = len(scenarios) == 1
        title = "Global material production and production routes"
        fig, axes = plt.subplots(len(scenarios), 3,
                                 figsize=(180 / 25.4, (100 if simplified else 150) / 25.4),
                                 squeeze=False)
        fig.subplots_adjust(left=.09, right=.985, bottom=.20 if simplified else .14,
                            top=.68 if simplified else .78, wspace=.51, hspace=.63)
        fig.text(.035, .977, title, fontsize=8, weight="bold", va="top")
        fig.text(.035, .936,
                 f"No circular economy | {min(self.config['years'])}–{max(self.config['years'])}",
                 fontsize=6.7, color="#52616A", va="top")
        years = self.config["years"]
        ticks = sorted({min(years), max(years), *[year for year in years if year % 10 == 0]})
        for column, sector in enumerate(SECTORS):
            sector_data = data[data.Sector == sector.label]
            finite = sector_data.Value.to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            low = min(0., float(finite.min())) if finite.size else 0.
            high = max(float(finite.max()), 1.) if finite.size else 1.
            locator = ticker.MaxNLocator(4, min_n_ticks=3)
            limits = locator.tick_values(low, high * 1.05)
            for row, scenario in enumerate(scenarios):
                ax, panel = axes[row, column], chr(97 + row * 3 + column)
                ax.spines[["top", "right"]].set_visible(False)
                ax.set_axisbelow(True)
                ax.grid(axis="y", color="#DDE3E6", linewidth=.35)
                ax.tick_params(labelsize=6.3, pad=3)
                ax.set_xlim(min(years) - .6, max(years) + .6)
                ax.set_xticks(ticks)
                ax.set_ylim(float(limits[0]), float(limits[-1]))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(4, min_n_ticks=3))
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
                ax.set_ylabel("Production (Mt yr$^{-1}$)", fontsize=6.8, labelpad=3)
                if row == len(scenarios) - 1:
                    ax.set_xlabel("Year", fontsize=6.8, labelpad=4)
                ax.set_title(sector.label if simplified else f"{sector.label} | {climate_label(scenario)}", loc="left",
                             fontsize=7, pad=9)
                ax.text(-.14, 1.04, panel, transform=ax.transAxes, va="bottom",
                        ha="left", fontsize=8, weight="bold")
                selected = sector_data[sector_data.Scenario == scenario]
                for index, (series, values) in enumerate(selected.groupby("Series", sort=False)):
                    ax.plot(values.Year, values.Value,
                            color=sector.color if index == 0 else ("#53626A" if index == 1 else "#7B8E96"),
                            linestyle=("-", "--", ":")[index], marker=("o", "s", "^")[index],
                            linewidth=1.15 if index == 0 else 1.0, markersize=2.8,
                            markerfacecolor="white", markeredgewidth=.7, label=series)
                if row == 0:
                    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.16), borderaxespad=0,
                              ncol=selected.Series.nunique(), fontsize=5.8,
                              columnspacing=.8, handlelength=1.7, handletextpad=.35)
        figure_note = ("Metals: total = primary + secondary. Cement and clinker are distinct products."
                       if simplified else
                       "Top row: NDC. Bottom row: NDC+LTT. Matching sector scales across rows.")
        fig.text(.035, .060, figure_note,
                 fontsize=5.8, color="#52616A")
        assumption = ("Draft assumption: missing additive activity = zero."
                      if self.config["missing_activity"] == "zero"
                      else "Missing activity retained; incomplete global totals appear as gaps.")
        fig.text(.035, .025, assumption, fontsize=5.5, color="#63717A", va="bottom")
        caption = (
            "Global production without circular-economy measures under NDC (a–c, "
            "ndc_noce in the source) and NDC+LTT (d–f, baseline_noce). Columns show cement and "
            "clinker (a,d), iron and steel (b,e), and aluminium (c,f). Cement and clinker are "
            "distinct products and are not added. Steel and aluminium totals equal primary "
            "plus secondary production. Each global value sums the 28 model regions once, "
            f"using exact variable and Mt/yr unit selections from {self.config['input']}. "
            "Vertical scales match within each sector across the two pathways; sectors have "
            "different scales. Markers denote supplied model years, joined by straight line "
            "segments, with no smoothing or extrapolation. The global production trajectories "
            "are extracted independently for the two pathways. " + assumption +
            " No uncertainty estimates were supplied."
        )
        if simplified:
            caption = (
                "Global material production without circular-economy measures, shown in one "
                "row with cement and clinker (a), iron and steel (b), and aluminium (c). "
                "Cement and clinker are distinct products and are not added. Steel and "
                "aluminium totals equal primary plus secondary production. "
                f"The figure selects {scenarios[0]} once from {self.config['input']}, with "
                "no averaging or duplication of climate pathways. The selected scenario "
                "remains explicit in the source CSV. Each global value sums the "
                f"{len(self.config['regions'])} configured model regions once, using exact "
                "production parents and Mt/yr units. Each sector has its own vertical "
                "scale. Markers show supplied model years joined by straight segments "
                "without smoothing or extrapolation. " + assumption +
                " No uncertainty estimates were supplied."
            )
        return ReportFigure("fig01_production", "1", title,
                            ("How do global material output and production routes evolve?" if simplified else
                             "How do material output and production routes evolve under NDC and NDC+LTT?"),
                            caption, fig, data, 3 if simplified else 6)


def _save_individual_figure(spec, directory, config):
    """Retain a reviewed figure only when its fresh raw-data build still matches."""
    filenames = [directory / f"{spec.slug}.{fmt}" for fmt in [*config["formats"], "csv"]]
    preserve = spec.slug in config.get("preserve_individual_figures", [])
    if preserve and all(path.is_file() for path in filenames):
        if "png" not in config["formats"]:
            raise ValueError("Preserving individual figures requires a PNG for visual verification")
        with BytesIO() as buffer:
            spec.figure.savefig(buffer, format="png", dpi=config["dpi"])
            matches_picture = buffer.getvalue() == (directory / f"{spec.slug}.png").read_bytes()
        matches_data = spec.data.to_csv(index=False).encode("utf-8") == (directory / f"{spec.slug}.csv").read_bytes()
        if not (matches_picture and matches_data):
            raise ValueError(f"Preserved {spec.slug} differs from the current build; remove it from "
                             "preserve_individual_figures to export an updated version")
        return True
    for fmt in config["formats"]:
        metadata = {"Title": spec.title, "Description": spec.caption, "Date": None} if fmt == "svg" else None
        spec.figure.savefig(directory / f"{spec.slug}.{fmt}", dpi=config["dpi"], metadata=metadata)
    spec.data.to_csv(directory / f"{spec.slug}.csv", index=False)
    return False


def generate_consolidation(config: dict, base: Path):
    data = ConsolidationData(config, base)
    builder = ConsolidationBuilder(data, config)
    directory = (base / config["output"]).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    records, all_data, captions = [], [], []
    # PDF font embedding occurs on close, inside the same style context as saving.
    with plt.rc_context(CONSOLIDATION_STYLE), PdfPages(directory / config.get("combined_pdf", "report_figures.pdf"),
            metadata={"Title": "Consolidated OMNIA report figures: NDC and NDC+LTT"}) as book:
        for spec in builder.figures():
            check_figure_bounds(spec.figure, spec.slug)
            preserved = _save_individual_figure(spec, directory, config)
            book.savefig(spec.figure)
            width, height = spec.figure.get_size_inches() * 25.4
            plt.close(spec.figure)
            all_data.append(spec.data.assign(Figure=spec.slug))
            captions.append(f"## Figure {spec.number}: {spec.title}\n\n{spec.caption}\n")
            records.append({"number": spec.number, "slug": spec.slug, "title": spec.title,
                            "question": spec.question, "caption": spec.caption, "section": spec.section,
                            "width_mm": float(width), "height_mm": float(height),
                            "panel_count": spec.panel_count, "formats": config["formats"],
                            "individual_artifacts_preserved": preserved})
            print(f"{'Preserved' if preserved else 'Rendered'} consolidation/{spec.slug}", flush=True)
    pd.concat(all_data, ignore_index=True).to_csv(directory / "source_data.csv", index=False)
    if "leading_producers" in config.get("figures", []):
        data.producer_membership().to_csv(directory / "producer_membership.csv", index=False)
        data.producer_rankings().to_csv(directory / "producer_rankings.csv", index=False)
    if "emissions_response" in config.get("figures", []):
        data.geographic_membership().to_csv(directory / "geographic_membership.csv", index=False)
        data.partner_rankings().to_csv(directory / "partner_rankings.csv", index=False)
    coverage_figures = {"material_policy_coverage", "energy_policy_coverage"} & set(config.get("figures", []))
    if coverage_figures:
        data.policy_coverage_membership().to_csv(directory / "policy_coverage_membership.csv", index=False)
    missing = [record for results in data.results.values() for record in results.coverage]
    pd.DataFrame(missing, columns=["Scenario", "Region", "Variable", "Unit", "Year", "Issue", "Treatment"]).drop_duplicates().to_csv(directory / "missing_inputs.csv", index=False)
    production_note = (
        f"Figures 1–2 select {config['simplified_production_scenario']} once without averaging climate pathways. "
        "Figure 1 has one row; Figure 2 has one endpoint pair per producer group, ranked by that same source scenario. "
        "Figures 3–4 retain both pathways with NDC first."
        if config.get("simplified_production_scenario") is not None else
        "Figures 1–4 present NDC before NDC+LTT: top/bottom rows in Figure 1, upper/lower producer pairs in Figure 2, and first/second legend entries and curve styles in Figures 3–4. Figure 2 keeps one cohort ranked by baseline_noce production in the configured ranking year, with the same producer groups and order for NDC.")
    audit = {"generated_utc": datetime.now(timezone.utc).isoformat(), "source": config["input"],
             "source_sha256": hashlib.sha256(data.source.read_bytes()).hexdigest(),
             "scenario_interpretation": CLIMATE_LABELS,
             "scenario_labels": {scenario: climate_label(scenario) for scenario in data.results},
             "source_rows_selected": {scenario: len(results.raw) for scenario, results in data.results.items()},
             "region_count": len(config["regions"]), "figure_count": len(records),
             "missing_input_count": len(missing), "config": config,
             "versions": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                          "numpy": np.__version__, "pandas": pd.__version__},
             "notes": ["All figures calculate directly from the configured raw results CSV. Earlier exported data and artwork are used only to verify preservation of reviewed individual figures.",
                       "Every baseline_* scenario in this source is interpreted as NDC+LTT. Source scenario identifiers remain unchanged.",
                       "Previously generated report and comparison collections are preserved.",
                       production_note,
                       "Figure 3 sums exact parent energy carriers; global energy intensity is global final energy divided by global production, multiplied by 1000 to convert EJ/Mt to GJ/t.",
                       "Figure 4 uses reported sector GHG emissions without subtracting capture again. Shares use each pathway's total industrial GHG, intensities use global production, and capture retains its separate CO2 unit.",
                       "Figure 5 puts NDC above NDC+LTT and compares each pathway's no-CE historical production with all five future CE settings. Cement and clinker remain separate; metals show primary and secondary stacks. Future percentage changes use the same pathway's future no-CE total, not the historical bar.",
                       "Figure 6 keeps the same historical/future bars and climate row order as Figure 5. It stacks exact final-energy carrier parents, excludes nested subcategories, and compares future totals with each pathway's future no-CE energy. Cement has no hydrogen parent in this source.",
                       "Figure 7 has NDC on the left and NDC+LTT on the right. Top trajectories and bottom geographic changes cover three sectors; the middle net change covers all industry, with Other industry calculated as a residual. Partner regions are ranked once by mean absolute comparison-year three-sector change across all eight CE cases; the same three regions and order apply throughout.",
                       "Figure 8 puts NDC above NDC+LTT and compares all five CE settings using 100 * (sector cost - no-CE sector cost) / no-CE sector cost. Every reference matches the sector, climate and year. Exact annualised-cost parents use Millions USD_2010/yr and preserve missingness regardless of the additive-activity policy. Main plots start in the configured main cost year; earlier costs are retained as context."]}
    if coverage_figures:
        audit["notes"].extend([
            "Figures 9–10 put NDC above NDC+LTT and compare four CE cases inside and outside each scenario's policy area. Each percentage is 100 * (group scenario total - group no-CE total) / group no-CE total, matching climate, sector, group and comparison year. EU-only adoption uses EU and its complement; EU+ adoption uses EU plus partners and its complement.",
            "Figure 9 measures material production, not consumption or trade-adjusted demand: the raw export has no material consumption or trade variables. Cement excludes clinker; metals sum exact primary and secondary parents. Figure 10 measures final-energy demand by summing exact carrier parents, excluding nested children.",
            "Inside/outside percentages have different denominators and are not additive. Context absolute changes sum to the global sector change. Signed effects outside the adoption area do not establish a specific leakage or rebound mechanism. Each group's completeness is evaluated independently under preserved missingness.",
        ])
        membership = data.policy_coverage_membership()
        audit["policy_coverage"] = {
            "year": config.get("comparison_year", 2050),
            "group_region_counts": {
                area: group.groupby("Scope").size().to_dict()
                for area, group in membership.groupby("PolicyArea", sort=False)
            },
            "membership_file": "policy_coverage_membership.csv",
            "material_measure": "Production, not consumption",
        }
    if "emissions_response" in config.get("figures", []) or coverage_figures:
        policy_source = (base / config["policy_regions_csv"]).resolve()
        audit["policy_regions_source"] = config["policy_regions_csv"]
        audit["policy_regions_sha256"] = hashlib.sha256(policy_source.read_bytes()).hexdigest()
    if "emissions_response" in config.get("figures", []):
        audit["partner_ranking"] = {
            "metric": "Mean absolute three-sector GHG change across eight climate-matched CE cases",
            "year": config.get("comparison_year", 2050), "tie_break": "Region code ascending",
            "selected_regions": data.partner_rankings().loc[lambda frame: frame.Selected, "Region"].tolist(),
            "complete_cases_required": 8,
        }
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (directory / "captions.md").write_text("# Consolidated report figure captions\n\n" + "\n".join(captions), encoding="utf-8")
    _write_methods(directory, config)
    _write_gallery(directory, records, config)
    print(f"Saved consolidated figures to {directory}", flush=True)


def _write_methods(directory, config):
    assumption = ("Missing additive production, energy, emissions and capture values and absent regional rows are provisionally treated as zero, following the existing report convention."
                  if config["missing_activity"] == "zero"
                  else "Missing inputs are retained. Global sums require all 28 regions, and an incomplete route also makes the metal total missing.")
    preserved = ", ".join(f"`{slug}`" for slug in config.get("preserve_individual_figures", []))
    preservation_note = (f"Configured preserved individual figures: {preserved}."
                         if preserved else
                         "No individual figures are reused in this configuration; all are exported from fresh raw-source calculations.")
    scenario = config.get("simplified_production_scenario")
    simplified = scenario is not None
    production_layout = (
        f"One row contains cement/clinker, iron and steel, and aluminium panels. It selects `{scenario}` "
        "once, retaining its source identifier in the figure CSV. The duplicate climate row and "
        "climate names in panel titles are omitted; quantities are not averaged between pathways."
        if simplified else
        "The top row uses `ndc_noce` (NDC, no CE), and the bottom row uses `baseline_noce` (NDC+LTT, no CE). Columns show cement/clinker, iron and steel, and aluminium.")
    production_scales = (
        "Panels a–c show the three sectors, with separate vertical scales."
        if simplified else
        "Panels a–c and d–f represent NDC and NDC+LTT respectively. Each sector has identical vertical limits in both rows, with different scales across sectors.")
    producer_ranking = (
        f"For each sector, the top {config.get('top_n', 12)} groups are selected and ordered by positive, available "
        f"**`{scenario}` output in {config.get('ranking_year', 2050)}**, with alphabetical tie breaking. "
        "Only this selected raw scenario is plotted, with one endpoint pair per group. Its ranking "
        "scenario, output and global shares are recorded in `producer_rankings.csv`."
        if simplified else
        f"For each sector, the top {config.get('top_n', 12)} groups are selected and ordered by positive, available "
        f"**NDC+LTT (`baseline_noce`) output in {config.get('ranking_year', 2050)}**, with alphabetical tie breaking. "
        "This one cohort and order applies to both pathways. Under preserved missingness it denotes the "
        "largest groups with available ranking-year totals. NDC is not ranked independently. "
        "`producer_rankings.csv` records both pathways' output and global shares for the common cohort.")
    producer_pairs = (
        f"Each producer group has one horizontal connector joining {min(config['years'])} and "
        f"{config.get('ranking_year', 2050)}, at its central row position. The legend distinguishes "
        "years only, with no duplicate climate pair."
        if simplified else
        f"Each producer group has two endpoint pairs, joining {min(config['years'])} and "
        f"{config.get('ranking_year', 2050)}: upper solid line for NDC and lower dashed line for NDC+LTT.")
    producer_coverage = (
        "The coverage annotation reports the selected cohort's share of global output for the chosen raw scenario."
        if simplified else
        "Coverage is calculated separately for each pathway as the common cohort's output divided by "
        "that pathway's global output, with displayed shares in NDC / NDC+LTT order.")
    coverage_methods = ""
    if {"material_policy_coverage", "energy_policy_coverage"} & set(config.get("figures", [])):
        coverage_methods = f"""## Figures 9–10: responses inside and outside CE policy areas

Both figures use a **2 × 3** layout with **NDC above NDC+LTT** and cement, iron and steel, and aluminium columns. Each panel compares medium/high CE in the EU, followed by medium/high CE in the EU plus international partners, in **{config.get('comparison_year', 2050)}**. Paired blue/orange bars show inside/outside the adoption area. Vertical limits match across climate rows within each sector, preserve both increases and decreases, and include zero. Each figure is 180 × 175 mm.

Geography comes directly from `{config['policy_regions_csv']}`. **EU-only cases:** inside is EUE, EUM and EUW (3 model regions); outside is their complement (25). **EU+ cases:** inside is the EU plus the 14 international partner regions (17); outside is the remaining 11. EU+ is not worldwide adoption. For each adoption scope the two groups are disjoint and exhaust all configured regions. `policy_coverage_membership.csv` records both complete partitions, their region descriptions and the inside flag. The five-region Europe producer group is not used here.

Every plotted percentage is **100 × (scenario group total − no-CE group total) / no-CE group total**. The reference matches climate pathway, sector, geographic group **and comparison year**. These are ratios of group sums, not averages of regional percentage changes. Inside and outside have different reference denominators, so their percentages cannot be added and are not shares of the global change. Missing/nonfinite quantities or nonpositive reference totals produce undefined percentages, retained as gaps. Under preserved missingness, each group requires all its constituent regions and each required route/carrier; incompleteness outside does not invalidate the inside group, or vice versa.

**Figure 9 measures material production, not consumption-based material demand.** The raw export contains no consumption or trade variables. Cement selects only `Production|Non-Metallic Minerals|Cement`, excluding clinker. Steel and aluminium sum their exact primary and secondary parents in `Mt/yr`. **Figure 10 measures final-energy demand** in `EJ/yr`, summing exact carrier parents: electricity, gases, liquids and solids for cement, plus hydrogen for steel/aluminium. Nested fuel subcategories and other units are excluded. All additive quantities follow the documented provisional missing-activity policy.

Each figure CSV retains 48 plotted percentage records (`Role=plotted`) and 48 signed absolute-change records (`Role=context`). Absolute changes equal the numerator, in `Mt/yr` or `EJ/yr`, and inside plus outside reconstruct the global sector change when complete. Raw scenario/reference group totals, their source regions/variables/coefficients, reference scenario/year, region counts, numerator units and denominator units are explicit. Percentage records use `ConversionFactor=100`, `Offset=0` and `Denominator=ReferenceValue`; absolute records use factor 1 and an empty denominator. Responses outside the adoption area describe model results and do not by themselves identify leakage, trade displacement or a rebound mechanism.

"""
    text = f"""# Consolidated report methods

All consolidated figures use direct extraction from `{config['input']}` through the unit-aware `Results` calculations. Existing exported figures and their CSVs are not calculation inputs. Future figure builders should use the same `ConsolidationData` source interface.

Every `baseline_*` scenario in this source denotes **NDC+LTT**; `ndc_*` denotes **NDC**. Raw source identifiers remain unchanged for traceability. This interpretation applies to every CE setting. Earlier figure collections retain their original labels and files.

## Figure 1: production

{production_layout} All 28 configured model regions enter each global sum exactly once. The `Mt/yr` unit must match exactly; detailed child variables and other units do not enter these sums.

Cement and clinker use `Production|Non-Metallic Minerals|Cement` and `Production|Non-Metallic Minerals|Cement Clinker`. These products are shown separately. Steel and aluminium use their exact primary and secondary production parents; total output is the sum of these two routes. The figure CSV records the original variable names for every plotted observation in `SourceVariables`.

{assumption} An entirely absent variable/unit is an error under either policy. `missing_inputs.csv` identifies every affected regional observation. Prices, costs and reported intensities retain missingness in the shared source-calculation module.

{production_scales} Markers show the supplied model years; straight lines connect those observations. No smoothing, extrapolation or uncertainty estimates are added.

## Figure 2: leading producers

Producer groups use the same partition as the original leading-producer report. **Europe combines ENE, ENW, EUE, EUM and EUW**; this statistical producer group differs from the three-region EU policy group used in the CE comparisons. Remaining groups preserve individual model regions, with readable model-group labels. Indonesia group includes Indonesia, Philippines and Viet Nam. `producer_membership.csv` records exact model-region membership, which partitions all 28 regions once.

{producer_ranking}

{producer_pairs} Open circles denote the earlier year and filled sector-coloured diamonds the later year. Endpoints use regional cement production (excluding clinker), or primary plus secondary steel/aluminium output. Thin connectors describe two endpoint observations. The Other model regions row sums the groups outside the selected cohort and is unranked; it completes global production for every plotted scenario and endpoint. {producer_coverage} The exported context data retain every supplied model year.

Logarithmic production axes retain the original figure convention. Zero or negative observed endpoints require `producer_axis_scale` set to `linear`; missing endpoints remain gaps. Axis ranges include every plotted endpoint within each sector. Figure 2 is 180 × 165 mm.

## Figure 3: final energy and derived intensity

Six panels align cement, iron and steel, and aluminium by column. The upper row shows global final energy (EJ/yr); the lower row shows derived global energy intensity (GJ/t). Every panel contains both no-CE pathways, presented in this order: NDC (`ndc_noce`; solid lines with open circles) and NDC+LTT (`baseline_noce`; dashed lines with filled squares). The legend follows this order. Each panel uses a common axis for the two pathways, with separate sector scales. Source years retain their exact positions; coincident curves are not displaced.

Final energy sums the exact carrier parents listed in `ResultsMetrics.docx`: electricity, gases, liquids and solids for cement, plus hydrogen for steel and aluminium. Nested variables such as Gases|Gas, Liquids|Oil and Solids|Coal are excluded. Each of the 28 model regions is counted once.

Derived intensity equals **1000 × global final energy (EJ/yr) / global material production (Mt/yr)**. Cement uses cement output, excluding clinker; steel and aluminium use primary plus secondary output. This is a ratio of global sums, not a mean of regional ratios and not the supplied regional intensity series. Missing or nonpositive production makes intensity undefined. Missingness follows the same explicit activity policy as production and energy totals. Each exported intensity row includes its numerator, denominator and unit-conversion factor. Figure 3 is 180 × 150 mm.

## Figure 4: emissions and capture

Four panels compare cement, iron and steel, and aluminium under NDC (`ndc_noce`) and NDC+LTT (`baseline_noce`), both without CE. Sector colours are consistent with the preceding figures; NDC uses solid lines with open circles and precedes NDC+LTT, which uses dashed lines with filled squares, in both the curve order and pathway legend. Each panel contains all six sector/pathway curves on one axis, without displacement of coincident observations. Figure 4 is 180 × 150 mm.

Panel a sums each sector's exact `Emissions|GHG|Industry|...` series in `MtCO2e/yr` across all 28 model regions and divides by 1,000 for `GtCO2e/yr`. Negative source emissions remain signed. Panel b divides those sector totals by the **same pathway's** `Emissions|GHG|Industry` global total and multiplies by 100. The three material sectors are not assumed to exhaust industrial emissions.

Panel c divides global sector GHG (`MtCO2e/yr`) by global material production (`Mt/yr`), giving `tCO2e/t` with a numerical conversion factor of one. Cement uses cement output excluding clinker, and metals use primary plus secondary output. These are ratios of global sums, not averages of regional intensities. Missing or nonpositive denominators make shares or intensities undefined.

Panel d sums the exact `Carbon Capture|Industry|...` series in `MtCO2/yr`. Capture is shown separately and is **not subtracted again** from the reported GHG series. GHG emissions include CO₂-equivalent units, while capture refers to CO₂. Small aluminium capture values retain their full source precision even when visually close to zero on the shared scale. All four metrics use exact variable/unit selections and the same explicit missing-activity policy. The source CSV retains raw numerators, denominators, their units and conversion factors for audit.

## Figure 5: material production across CE scenarios

The 2 × 3 layout places **NDC above NDC+LTT**, with cement/clinker, iron and steel, and aluminium columns. Each panel starts with a **{config.get('historical_year', 2019)} no-CE bar**, extracted from that row's own no-CE scenario. This historical observation is not averaged across scenarios. The five remaining categories show **{config.get('comparison_year', 2050)} production** in this fixed order: no CE, medium CE in the EU, high CE in the EU, medium CE in the EU plus partners, and high CE in the EU plus partners.

Cement and clinker use distinct, side-by-side bars and are never added. Metal bars stack exact primary and secondary production parents. Every component is a sum of all 28 model regions in `Mt/yr`, using the same variable and unit rules as Figure 1. Bar height shows production directly for each climate pathway; there are no NDC output diamonds. Vertical limits match across rows within each sector. Figure 5 is 180 × 175 mm.

Labels above the five future categories give **100 × future production / same-pathway future no-CE production − 100**. Cement changes exclude clinker, and metal changes use primary plus secondary totals. The future no-CE bar is the reference; the historical bar has no change label. Labels within sufficiently large metal secondary segments show **100 × secondary / total production**, including the historical bar. Missing or nonpositive denominators yield undefined percentages.

`M/H` denotes medium/high CE. `EU+` denotes the EU plus international partner regions (17 model regions), rather than worldwide CE adoption. The exported data identify each bar's year, order, type and policy, and retain component values, metal totals, percentage changes and secondary shares. Numerator, denominator, units, conversion factor, `Offset`, and reference scenario/year document every calculation. For percentage changes, `Offset` is −100; it is zero for quantities and shares. These fields distinguish the historical anchor from the future no-CE comparison reference.

## Figure 6: final-energy mix across CE scenarios

The 2 × 3 layout places **NDC above NDC+LTT**, with cement, iron and steel, and aluminium columns. Each panel starts with **{config.get('historical_year', 2019)} final energy from its own no-CE scenario**, followed by the same five **{config.get('comparison_year', 2050)} CE settings** as Figure 5. The historical bar is extracted separately for each pathway and is not averaged across scenarios.

Stacks retain the original carrier colours and order: electricity, gases, hydrogen, liquids and solids. Cement has four exact parents and excludes hydrogen; steel and aluminium include all five. Every carrier value selects its exact `Final Energy|Industry|sector|carrier` parent and `EJ/yr` unit, summing the 28 configured regions once. Nested fuel categories are excluded, and no additional aggregate final-energy variable is added. Total final energy sums all required carrier totals; a missing carrier under preserved missingness leaves the total undefined. The provisional zero policy follows the same documented activity convention as preceding figures.

Labels above future bars give **100 × scenario final energy / same-pathway future no-CE final energy − 100**; the historical bar has no change label. Missing or nonpositive reference energy yields an undefined percentage. Both rows use the same scale within each sector, including historical values. Figure 6 is 180 × 175 mm. The source CSV records all carrier quantities, stack totals, future changes, exact source parents, and the historical/future bar metadata. Change rows retain numerator, denominator, their `EJ/yr` units, factor 100, offset −100, and the future reference scenario and year.

## Figure 7: emissions response and geographic contributions

The **3 × 2** layout puts **NDC on the left and NDC+LTT on the right**. Panels a,b show global cement, steel and aluminium GHG trajectories for all five CE settings. Panels c,d decompose **{config.get('comparison_year', 2050)} total industrial GHG changes** into those three sectors and other industry. Panels e,f decompose **the three sectors' changes** geographically. Every change subtracts no CE under the same climate pathway in the same year. Row pairs use common vertical limits. Model-year markers retain their actual coordinates.

Sector data select exact `Emissions|GHG|Industry|sector` parents and the `MtCO2e/yr` unit. Total industry selects `Emissions|GHG|Industry`. Negative source values remain signed, and separately reported CO₂ capture is not subtracted again. **Other industry = reported total industry − cement − steel − aluminium**; its change is an accounting residual and does not establish a particular rebound mechanism. Signed stack components accumulate increases and decreases independently from zero. Black diamonds and signed labels give the net change: **all industry** in the middle row, **three sectors** in the last row.

Policy geography comes from `{config.get('policy_regions_csv', 'data/CE_policy_regions_261002.csv')}`. EU consists of EUE, EUM and EUW. Partners comprise the 14 additional participating regions, including ENE and ENW; the other 11 model regions are Rest of world. These memberships differ from the five-region Europe producer group in Figure 2.

The three main partner regions are selected **once for the whole figure**, by their **mean absolute {config.get('comparison_year', 2050)} change in three-sector GHG across the eight CE cases** (four CE settings under each of two climate pathways). Each case uses its own climate-matched no-CE reference. Ranking requires all eight changes to be available under preserved missingness; ties use region code ascending. The selected regions and their order are recalculated from the configured results CSV and recorded in `partner_rankings.csv`. IDN combines Indonesia, the Philippines and Viet Nam. Keeping the same cohort, ordering and colours supports comparisons between columns and CE levels. Remaining partner regions form **Other partners**. The six disjoint groups comprise EU (3 regions), three individual partner regions, Other partners (11), and Rest of world (11), covering all 28 source regions exactly once. Aggregating the four partner categories reconstructs the original Partners quantity.

`partner_rankings.csv` records every partner's mean absolute score, rank, selection, ranking scenarios, signed case contributions and source parents. `geographic_membership.csv` records the policy group and final plotted group for every model region. The figure CSV records `ScenarioValue`, `ReferenceValue`, reference scenario/year, source regions, variables and their arithmetic coefficients. Absolute changes preserve the meaning of near-zero or negative regional reference emissions; no regional percentage changes are introduced. Missingness follows the same explicit activity policy as previous figures.

## Figure 8: annualised sector-cost changes

The **2 × 3** layout puts **NDC in the top row and NDC+LTT in the bottom row**, with cement, iron and steel, and aluminium columns. All five CE settings appear in every panel, using the same policy colours, markers and line styles as Figure 7. No CE provides the zero reference. Vertical scales match between climate pathways within each sector, include every finite change and zero, and differ between sectors.

Global annualised sector costs select the exact `Total Annualised Cost|Industry|Non-Metallic Minerals|Cement`, `Total Annualised Cost|Industry|Iron and Steel`, and `Total Annualised Cost|Industry|Non-Ferrous Metals|Aluminum` parents and `Millions USD_2010/yr`. Each global cost sums the {len(config['regions'])} configured regions once, excluding nested children and aggregate system costs. **Costs always retain missingness**, regardless of the additive-activity policy; incomplete global coverage leaves costs undefined.

Every plotted percentage is **100 × (CE scenario cost − no-CE cost) / no-CE cost**, where the reference matches the sector, climate pathway **and model year**. This is a ratio of global sums, not an average of regional percentage changes. Finite negative scenario costs are retained; a missing, nonfinite or nonpositive no-CE denominator leaves the change undefined. No CE equals zero wherever its reference is valid. No additional inflation or currency conversion enters this dimensionless ratio.

As in the original main cost figure, plotted years start at **{config.get('main_cost_start_year', 2024)}**. All earlier supplied costs remain in the figure CSV with `Role=context`. Main-period records have `Role=plotted`. Markers show supplied model years joined by straight segments; no interpolation to extra years, smoothing or extrapolation is added. `ScenarioCost`, `ReferenceCost`, numerator and denominator retain raw millions of 2010 USD per year, while `ConversionFactor=100` and `Offset=-100` reproduce the percentage. Reference scenario and year are explicit for every observation.

{coverage_methods}## Exports and reproducibility

Individual figure CSVs contain the observations, original scenario identifiers, displayed climate labels, panel/sector/series identifiers, source variables, values and units. Figure 2 also identifies groups, constituent regions, display order and the ranking scenario/year. Figures 3–6 identify metrics, all constituent source regions, numerators/denominators and conversion factors. Figures 5 and 6 also record bar categories and calculation offsets. Figure 7 records absolute scenario/reference quantities, source coefficients and the policy/geographic breakdown. Figure 8 records raw sector costs, year-specific no-CE references and the percentage calculation. Figures 9–10 record matched inside/outside quantities, percentages and absolute changes. `source_data.csv` combines these tables and adds the figure identifier. `audit.json` records the source SHA-256 hash, configuration, software versions and scenario interpretation, plus policy-table hashes, coverage counts and Figure 7's partner-selection rule where applicable. `manifest.json` records captions, figure dimensions, panels and formats.

Figures use embedded TrueType fonts in PDF, editable SVG text and {config['dpi']} dpi PNGs. Figure 1 is 180 × {100 if simplified else 150} mm. The PDF font context remains active through finalisation, including the combined `{config.get('combined_pdf', 'report_figures.pdf')}`. Label bounds are checked before saving. This workflow writes into `{config['output']}` without deleting unrelated files.

`preserve_individual_figures` retains reviewed individual exports when newly calculated source data and a fresh PNG match them exactly. If they differ, regeneration stops rather than overwriting the reviewed version. The combined PDF always renders every configured figure directly from the raw source. {preservation_note} Remove an entry when intentionally revising a preserved figure. Earlier collections remain in their original output directories.

Regenerate from the repository root:

```powershell
python -m omnia_results --config {config.get('config_file', 'figures.consolidation.json')}
```
"""
    (directory / "methods.md").write_text(text, encoding="utf-8")


def _write_gallery(directory, records, config):
    cards = "".join(_figure_card(record) for record in records)
    geography_links = ('<a href="partner_rankings.csv">Partner rankings</a>'
                       '<a href="geographic_membership.csv">Geographic membership</a>'
                       if any(record["slug"] == "fig07_emissions_response" for record in records) else "")
    if any(record["slug"] in {"fig09_material_policy_coverage", "fig10_energy_policy_coverage"}
           for record in records):
        geography_links += '<a href="policy_coverage_membership.csv">Policy coverage membership</a>'
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Consolidated OMNIA report figures</title><style>
body{{margin:0;background:#f5f6f4;color:#233139;font-family:Arial,Helvetica,sans-serif;line-height:1.6}}main{{max-width:1120px;margin:auto;padding:40px 24px}}a{{color:#006887}}h1{{line-height:1.2;font-size:32px}}h3{{font-size:23px;margin:4px 0}}.intro,.purpose{{color:#52616a}}.links,.downloads{{display:flex;gap:18px;flex-wrap:wrap;margin:20px 0}}.figure-card{{background:white;border:1px solid #dce2e2;border-radius:5px;margin:30px 0;padding:24px}}.figure-number{{text-transform:uppercase;font-size:12px;color:#62747e}}figure{{margin:0}}.artwork img{{width:100%;height:auto;display:block}}figcaption{{font-size:13px;color:#52616a;border-top:1px solid #e5e9e9;padding-top:16px}}.downloads{{font-size:12px;font-weight:bold}}
@media(max-width:650px){{main{{padding:24px 12px}}.figure-card{{padding:14px}}h1{{font-size:27px}}h3{{font-size:20px}}}}
</style></head><body><main><h1>Consolidated OMNIA report figures</h1>
<p class="intro">Refined report figures calculated directly from {escape(str(config['input']))}. Source scenarios labelled baseline are interpreted as NDC+LTT.</p>
<div class="links"><a href="{escape(str(config.get('combined_pdf', 'report_figures.pdf')))}">Complete PDF</a><a href="captions.md">Captions</a><a href="methods.md">Methods</a><a href="source_data.csv">Source data</a><a href="audit.json">Audit</a><a href="missing_inputs.csv">Missing inputs</a>{geography_links}</div>
{cards}</main></body></html>'''
    (directory / "index.html").write_text(page, encoding="utf-8")
