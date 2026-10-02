"""Five reproducible cross-scenario figures for the circular-economy deliverable."""

from datetime import datetime, timezone
from html import escape
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from .comparison_common import COMPARISON_STYLE
from .comparison_data import ComparisonData, percent_change
from .comparison_materials import MaterialsEnergyFigures
from .comparison_impacts import ImpactFigures
from .metrics import SECTORS
from .report import check_figure_bounds
from .report_gallery import _figure_card


class ComparisonBuilder(MaterialsEnergyFigures, ImpactFigures):
    def figures(self):
        yield self.materials()
        yield self.energy_mix()
        yield self.emissions_response()
        yield self.geography()
        yield self.economics()


def _summary(data):
    quantities = {}
    for scenario in data.results:
        values = {}
        for sector in SECTORS:
            for component, series in data.global_production(scenario, sector).items():
                values[(f"{sector.label} production: {component}", "Mt/yr")] = series.loc[data.comparison_year]
            values[(f"{sector.label} final energy", "EJ/yr")] = data.energy_by_carrier(scenario, sector)[data.comparison_year].sum(min_count=len(sector.fuels))
        values[("Three-sector GHG emissions", "MtCO2e/yr")] = data.target_emissions(scenario).loc[data.comparison_year]
        values[("All-industry GHG emissions", "MtCO2e/yr")] = data.industry_emissions(scenario).loc[data.comparison_year]
        values[("Other-industry GHG emissions", "MtCO2e/yr")] = data.emissions_components(scenario).loc["Other industry", data.comparison_year]
        values[("Annualised system cost", "billion USD_2010/yr")] = data.system_cost(scenario).loc[data.comparison_year]
        values[("EU carbon-price index", "USD_2010/tCO2e")] = data.carbon_price(scenario).loc[data.comparison_year]
        quantities[scenario] = values
    rows = []
    for scenario, values in quantities.items():
        reference = data.reference(scenario)
        for (metric, unit), value in values.items():
            baseline = quantities[reference][(metric, unit)]
            rows.append({"Scenario": scenario, "ReferenceScenario": reference, "Year": data.comparison_year,
                         "Metric": metric, "Unit": unit, "Value": value, "ReferenceValue": baseline,
                         "Change": value - baseline, "Change_percent": percent_change(value, baseline)})
    return pd.DataFrame(rows)


def generate_comparison(config: dict, base: Path):
    data = ComparisonData(config, base)
    directory = (base / config["output"]).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    builder = ComparisonBuilder(data, config)
    records, plotted_data, captions = [], [], []
    # The font context must remain active until both individual and book PDFs close.
    with plt.rc_context(COMPARISON_STYLE), PdfPages(directory / "report_figures.pdf",
            metadata={"Title": "Circular economy across scenarios", "Subject": "Five Baseline/NDC and EU/partner comparisons"}) as book:
        for spec in builder.figures():
            check_figure_bounds(spec.figure, spec.slug)
            for fmt in config["formats"]:
                metadata = {"Title": spec.title, "Description": spec.caption, "Date": None} if fmt == "svg" else None
                spec.figure.savefig(directory / f"{spec.slug}.{fmt}", dpi=config["dpi"], metadata=metadata)
            book.savefig(spec.figure)
            size = spec.figure.get_size_inches() * 25.4
            plt.close(spec.figure)
            exported = spec.data.assign(Figure=spec.slug)
            exported.to_csv(directory / f"{spec.slug}.csv", index=False)
            plotted_data.append(exported)
            assumption = ("Missing additive activity is provisionally treated as zero; prices, costs and reported intensities retain missingness."
                          if config["missing_activity"] == "zero" else "Missing inputs remain missing and aggregates require complete coverage.")
            caption = f"Figure {spec.number}. {spec.title}. {spec.caption} {assumption} No uncertainty estimates were supplied."
            records.append({"number": spec.number, "slug": spec.slug, "title": spec.title, "question": spec.question,
                            "caption": caption, "section": "Main report", "width_mm": float(size[0]),
                            "height_mm": float(size[1]), "panel_count": spec.panel_count, "formats": config["formats"]})
            captions.append(f"## Figure {spec.number} · {spec.title}\n\n{caption}\n")
            print(f"Rendered comparison/{spec.slug}", flush=True)
    data.membership().to_csv(directory / "region_membership.csv", index=False)
    data.price_weights.to_csv(directory / "carbon_price_weights.csv", index=False)
    _summary(data).to_csv(directory / "comparison_summary.csv", index=False)
    pd.concat(plotted_data, ignore_index=True).to_csv(directory / "source_data.csv", index=False)
    missing = [entry for results in data.results.values() for entry in results.coverage]
    pd.DataFrame(missing, columns=["Scenario", "Region", "Variable", "Unit", "Year", "Issue", "Treatment"]).drop_duplicates().to_csv(directory / "missing_inputs.csv", index=False)
    hashes = {key: hashlib.sha256((base / config[key]).read_bytes()).hexdigest()
              for key in ("input", "policy_regions_csv", "policy_regions_document")}
    notes = [
        "Five CE settings are compared separately within Baseline and NDC. Each CE change uses no CE in the same climate pathway as reference.",
        "EU contains EUE, EUM and EUW. Partners contain 14 further regions, including ENE and ENW. Rest contains the other 11. The three groups partition all 28 source regions.",
        "The _gbl scenario suffix denotes CE applied in EU + international partners (17 model regions), according to the supplied table; it does not denote adoption in every model region.",
        "Baseline and NDC are neutral climate-pathway labels. The export has lower 2050 industrial GHG emissions under Baseline; no ordering of ambition is assumed.",
        "Global production values are nearly identical between climate pathways for matched CE settings. Figure 1 shows Baseline composition bars and NDC output diamonds; both branches are retained in source data.",
        "Other industry is the exact all-industry GHG total minus the three material sectors. Its change is an accounting residual; physical mechanisms require model documentation.",
        "Regional emissions changes use absolute units because several sector baselines are near zero or negative. Signed values are retained.",
        "The EU carbon-price index uses common baseline_noce 2019 industrial-GHG weights in every scenario. Prices with a positive weight must be available; weights never use zero-filled missing emissions.",
        "Cost comparisons use 2050 annualised modelled system costs, in constant 2010 USD. They do not use the anomalous 2019 cost values.",
        ("Missing additive activity is provisionally interpreted as zero. This export convention remains unconfirmed; every affected source input is recorded." if config["missing_activity"] == "zero"
         else "Missing inputs are preserved; incomplete quantities remain gaps."),
        "Supplied model years are connected directly. No uncertainty bands, annual interpolation, smoothing or extrapolation are added.",
    ]
    audit = {"generated_utc": datetime.now(timezone.utc).isoformat(), "source": config["input"],
             "source_sha256": hashes["input"], "input_hashes": hashes, "figure_count": len(records),
             "scenario_count": len(data.results), "scenarios": list(data.results), "config": config,
             "geographic_counts": {group: len(members) for group, members in data.groups.items()},
             "versions": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                          "numpy": np.__version__, "pandas": pd.__version__}, "notes": notes}
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (directory / "captions.md").write_text("# Circular economy comparison captions\n\n" + "\n".join(captions), encoding="utf-8")
    _write_methods(directory, notes)
    _write_gallery(directory, records, audit)
    print(f"Saved five comparison figures and combined PDF to {directory}", flush=True)


def _write_methods(directory, notes):
    text = "# Cross-scenario comparison methods\n\n" + "\n\n".join(notes)
    text += """\n\n## Calculations and source data

- Global additive values sum each of the 28 model regions once, with exact scenario, variable and unit matching.
- Steel and aluminium total production is primary plus secondary. Cement and clinker are distinct products and are never added together.
- Final energy uses the carrier parents listed in ResultsMetrics.docx, excluding overlapping child variables.
- Absolute changes subtract the matching climate pathway's no-CE values. Percent changes are 100 × (CE − no CE) / no CE and require a finite, positive reference.
- Signed decomposition bars stack increases and decreases separately. A net marker is the sum of all signed components.
- All-industry change equals the changes in cement, steel, aluminium and Other industry. Geographic changes partition the three-sector total into EU, partners and rest.
- Fixed price weights normalize baseline_noce 2019 industrial GHG emissions within the EU group. The same weights apply to all years and scenarios.
- System costs convert millions USD_2010/yr to billions by dividing by 1,000. No cumulative or discounted costs are calculated.

Each figure CSV and source_data.csv identify the figure, panel, scenario, climate pathway, policy setting, scope, metric, component, year, value, unit and reference scenario. Role=plotted denotes chart marks; Role=context retains supporting totals and annotations; Role=reference identifies comparison denominators. Empty values denote missingness. comparison_summary.csv provides full-precision 2050 quantities and matched-reference changes.

## Export and review

Figures are 180 mm wide and at most 170 mm tall. PDF embeds TrueType fonts, SVG retains editable text, and PNG uses 600 dpi. The combined five-page PDF contains the same figures in narrative order. Labels are checked against canvas bounds before saving. Source hashes, geographic definitions, configuration and software versions accompany the exports in audit.json. Baseline/NDC definitions and the missing-activity convention should accompany substantive interpretation.
"""
    (directory / "methods.md").write_text(text, encoding="utf-8")


def _write_gallery(directory, records, audit):
    cards = "".join(_figure_card(record) for record in records)
    notes = "".join(f"<li>{escape(note)}</li>" for note in audit["notes"])
    navigation = "".join(f'<a href="#figure{record["number"]}">{record["number"]}</a>' for record in records)
    assumption = ("Missing additive activity is provisionally treated as zero; prices and costs retain missing values."
                  if audit["config"]["missing_activity"] == "zero"
                  else "Missing inputs are preserved; incomplete aggregates remain gaps.")
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Circular economy across scenarios</title><style>
body{{margin:0;background:#f5f6f4;color:#233139;font-family:Arial,Helvetica,sans-serif;line-height:1.6}}main{{max-width:1120px;margin:auto;padding:40px 24px}}a{{color:#006887}}h1{{line-height:1.2;font-size:34px}}h3{{font-size:23px;margin:4px 0}}.intro,.purpose{{color:#52616a}}.links,.downloads,.nav{{display:flex;gap:18px;flex-wrap:wrap;margin:20px 0}}.primary{{background:#173f4b;color:white;padding:10px 16px;border-radius:4px;text-decoration:none}}.assumption{{background:#fff7e8;border-left:3px solid #ad7e2e;padding:12px 16px}}.figure-card{{background:white;border:1px solid #dce2e2;border-radius:5px;margin:30px 0;padding:24px;scroll-margin-top:16px}}.figure-number{{text-transform:uppercase;font-size:12px;color:#62747e}}figure{{margin:0}}.artwork img{{width:100%;height:auto;display:block}}figcaption{{font-size:13px;color:#52616a;border-top:1px solid #e5e9e9;padding-top:16px}}.downloads{{font-size:12px;font-weight:bold}}details{{padding:20px;background:#eaf0ee}}summary{{cursor:pointer;font-weight:bold}}li{{font-size:13px;margin:8px 0}}
@media(max-width:650px){{main{{padding:24px 12px}}.figure-card{{padding:14px}}h1{{font-size:28px}}h3{{font-size:20px}}}}
</style></head><body><main><h1>Circular economy across scenarios</h1>
<p class="intro">Five figures compare Baseline and NDC, medium and high circular economy, and measures applied in the EU or EU + international partners.</p>
<div class="links"><a class="primary" href="report_figures.pdf">Download all five figures · PDF</a><a href="captions.md">Captions</a><a href="methods.md">Methods</a><a href="comparison_summary.csv">2050 comparison data</a></div>
<p>EU = EUE, EUM and EUW. EU + partners covers 17 model regions. Climate-pathway labels are retained without imposing an ambition ordering.</p>
<p class="assumption">{assumption}</p>
<nav class="nav" aria-label="Jump to figure">Figures: {navigation}</nav>{cards}
<details><summary>Definitions and calculation checks</summary><ul>{notes}</ul>
<div class="links"><a href="source_data.csv">All plotted data</a><a href="region_membership.csv">EU and partner definitions</a><a href="carbon_price_weights.csv">Common price weights</a><a href="missing_inputs.csv">Missing inputs</a><a href="audit.json">Audit</a></div></details>
</main></body></html>'''
    (directory / "index.html").write_text(page, encoding="utf-8")
