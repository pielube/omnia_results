"""Command-line generation, source-data export and a local review gallery."""

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import platform
import re

import matplotlib
import numpy as np
import pandas as pd

from .metrics import Results
from .plots import build_panels, export_panel, panel_data


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not config["regions"] or not config["scenarios"]:
        raise ValueError("At least one region and scenario must be configured")
    if len(set(config["scenarios"])) != len(config["scenarios"]):
        raise ValueError("Scenario selections must be unique")
    if config.get("suite") == "report":
        if not config["formats"] or not set(config["formats"]) <= {"pdf", "svg", "png"}:
            raise ValueError("Formats must be chosen from pdf, svg and png")
        if config["dpi"] < 300:
            raise ValueError("Use at least 300 dpi for report figures")
        return config
    if len(set(config["europe_regions"])) != len(config["europe_regions"]):
        raise ValueError("European regions must be unique")
    if not set(config["europe_regions"]) <= set(config["regions"]):
        raise ValueError("European regions must be source region codes")
    groups = list(config["regional_groups"].values())
    if any(not group or len(group) > 7 for group in groups):
        raise ValueError("Each regional group must have between one and seven regions")
    flat = [region for group in groups for region in group]
    if sorted(flat) != sorted(config["regions"]):
        raise ValueError("Regional groups must cover every configured region exactly once")
    if not config["formats"] or not set(config["formats"]) <= {"pdf", "svg", "png"}:
        raise ValueError("Formats must be chosen from pdf, svg and png")
    if config["dpi"] < 300:
        raise ValueError("Use at least 300 dpi for report figures")
    return config


def write_gallery(directory: Path, records: list[dict], audit: dict, scenario_label: str):
    esc = html.escape
    cards = []
    for record in records:
        category = ("global" if record["scope"] == "Global" else
                    "europe" if record["family"] == "Production" else "regional")
        slug = record["slug"]
        links = " · ".join(f'<a href="{slug}.{fmt}">{fmt.upper()}</a>' for fmt in record["formats"])
        preview = next((fmt for fmt in ("svg", "png") if fmt in record["formats"]), None)
        image = (f'<img loading="lazy" src="{slug}.{preview}" alt="{esc(record["title"] + ", " + record["scope"])}">'
                 if preview else f'<p><a href="{slug}.pdf">Open figure PDF</a></p>')
        coverage = (f'<p class="incomplete">Incomplete inputs: {record["available_points"]} of {record["expected_points"]} plotted values available.</p>'
                    if record["available_points"] < record["expected_points"] else "")
        cards.append(f'''<article data-category="{category}" data-search="{esc((record['title'] + ' ' + record['scope']).lower())}">
<div class="figure">{image}</div><div class="details"><p class="scope">{esc(record['scope'])} · {esc(record['family'])}</p>
<h2>{esc(record['title'])}</h2>{coverage}<p>{esc(record['caption'])}</p>
<p class="downloads">{links} · <a href="{slug}.csv">DATA CSV</a></p></div></article>''')
    warnings = "".join(f"<li>{esc(item)}</li>" for item in audit["notes"])
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OMNIA · __SCENARIO__ figures</title><style>
:root{font-family:Arial,Helvetica,sans-serif;color:#233139;background:#f4f5f3;font-size:15px}
*{box-sizing:border-box}body{margin:0}main{max-width:1160px;margin:auto;padding:48px 32px}
.eyebrow{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:#546e7a}h1{font-size:34px;letter-spacing:-.03em;margin:12px 0}
.intro{color:#52616a;line-height:1.6;max-width:800px}a{color:#006b94;text-underline-offset:3px}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:28px 0 16px}
button,input{font:inherit;border:1px solid #bac7cb;border-radius:5px;padding:9px 13px;background:white;color:#233139}
button{cursor:pointer}button[aria-pressed=true]{background:#233e4d;color:white;border-color:#233e4d}input{margin-left:auto;min-width:230px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}article{background:white;border:1px solid #dbe1df;border-radius:6px;overflow:hidden}
.figure{display:flex;align-items:center;justify-content:center;padding:16px 22px 0;min-height:360px}.figure img{width:100%;max-height:400px;object-fit:contain}
.details{border-top:1px solid #edf0ef;padding:18px 22px}.scope{font-size:11px;color:#667983;text-transform:uppercase;letter-spacing:.06em}
h2{font-size:18px;margin:8px 0}.details p{font-size:13px;line-height:1.55}.downloads{font-weight:600;font-size:11px!important;letter-spacing:.03em}
.incomplete{color:#865c10;background:#faf2df;padding:8px 10px;border-radius:3px}
details{margin:26px 0;padding:18px 22px;background:#eaf0f0;border:1px solid #d2dede;border-radius:5px}summary{cursor:pointer;font-weight:600}
li{line-height:1.6;margin:7px 0}#count{font-size:12px;color:#657781;margin:12px 0 20px}footer{font-size:12px;color:#62747e;margin-top:30px;line-height:1.7}
[hidden]{display:none!important}@media(max-width:720px){main{padding:25px 16px}.grid{grid-template-columns:1fr}input{margin-left:0;width:100%}.figure{min-height:0}}
</style></head><body><main><p class="eyebrow">OMNIA / report figures</p>
<h1>Industrial transitions</h1><p class="intro">__SCENARIO__ · 2019–2050. Standalone publication figures with editable vector artwork and source data. Select a scope to review the full set.</p>
<details><summary>Definitions and data checks</summary><ul>__NOTES__</ul>
<p><a href="audit.json">Audit</a> · <a href="missing_inputs.csv">Missing-input ledger</a> · <a href="intensity_comparison.csv">Intensity comparison</a> · <a href="captions.md">Figure captions</a> · <a href="source_data.csv">All plotted data</a></p></details>
<nav class="toolbar" aria-label="Figure filters"><button aria-pressed="true" data-filter="global">Global</button><button aria-pressed="false" data-filter="europe">European production</button><button aria-pressed="false" data-filter="regional">Regional metrics</button><button aria-pressed="false" data-filter="all">All figures</button><input id="search" type="search" aria-label="Search figures" placeholder="Search metric or region"></nav>
<p id="count" aria-live="polite"></p><div class="grid">__CARDS__</div>
<footer>Markers show supplied model years; segments connect points without smoothing. No uncertainty intervals are available in this export.<br>
PDF/SVG preserve vector lines and editable text. PNG exports use __DPI__ dpi. Currency is expressed in constant 2010 USD.</footer></main>
<script>let active='global';const cards=[...document.querySelectorAll('article')],buttons=[...document.querySelectorAll('[data-filter]')];function filter(){const q=document.querySelector('#search').value.toLowerCase();let count=0;cards.forEach(c=>{c.hidden=!(active==='all'||c.dataset.category===active)||!c.dataset.search.includes(q);if(!c.hidden)count++});document.querySelector('#count').textContent=count+' of '+cards.length+' figures';}buttons.forEach(b=>b.addEventListener('click',()=>{active=b.dataset.filter;buttons.forEach(x=>x.setAttribute('aria-pressed',String(x===b)));filter()}));document.querySelector('#search').addEventListener('input',filter);filter();</script></body></html>'''
    page = page.replace("__SCENARIO__", esc(scenario_label)).replace("__NOTES__", warnings)
    page = page.replace("__CARDS__", "\n".join(cards)).replace("__DPI__", str(audit["config"]["dpi"]))
    (directory / "index.html").write_text(page, encoding="utf-8")


def generate(config: dict, base: Path):
    if config.get("suite") == "report":
        from .report import generate_report
        return generate_report(config, base)
    input_path = (base / config["input"]).resolve()
    output_path = (base / config["output"]).resolve()
    for scenario in config["scenarios"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", scenario):
            raise ValueError("Scenario names used for output folders must be simple identifiers")
        results = Results(input_path, scenario, config["regions"], config["years"], config["missing_activity"])
        panels = build_panels(results, config)
        comparison = results.intensity_comparison()
        directory = output_path / scenario
        directory.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(directory / "intensity_comparison.csv", index=False)
        costs = results.raw.loc[results.raw.Variable.str.startswith("Total Annualised Cost")]
        negative = costs.melt(id_vars=["Scenario", "Region", "Variable", "Unit"], var_name="Year", value_name="Value")
        negative = negative.loc[negative.Value < 0]
        negative.to_csv(directory / "negative_costs.csv", index=False)
        deviation = comparison.Difference_GJ_per_t.abs()
        finite_comparisons = comparison[["Reported_GJ_per_t", "Derived_GJ_per_t"]].notna().all(axis=1)
        mismatched = finite_comparisons & ~np.isclose(comparison.Reported_GJ_per_t, comparison.Derived_GJ_per_t,
                                                    rtol=0.01, atol=1e-6)
        cost_note = f"The source contains {len(negative)} negative regional cost observations. All are preserved (negative_costs.csv)."
        system_cost = results.series("Total Annualised Cost", "Millions USD_2010/yr", activity=False)
        steel_cost = results.series("Total Annualised Cost|Industry|Iron and Steel", "Millions USD_2010/yr", activity=False)
        exceeded = [str(year) for year in results.years if steel_cost.loc[year] > system_cost.loc[year]]
        if exceeded:
            cost_note += f" Global steel cost exceeds the global system total in {', '.join(exceeded)}; check the source cost accounting before interpreting this comparison."
        notes = [
            f"Only {scenario} is plotted. Source scenarios: {', '.join(results.available_scenarios)}.",
            f"Global quantities sum the {len(results.regions)} configured source regions, assumed to form a disjoint world partition. No global aggregate is supplied in the CSV.",
            ("Blank cells and absent regional rows for additive activity variables are treated as zero, under the configured zero-activity interpretation. "
             if config["missing_activity"] == "zero" else
             "Blank cells and absent regional rows are kept missing. Global sums require every region; incomplete totals appear as gaps. ")
            + "Prices, costs and reported intensities always retain missing values. The missing-input ledger records every affected input.",
            "European production is shown separately for " + ", ".join(config["europe_regions"]) + ". Region names are not expanded without a model codebook.",
            "Regional sets contain at most seven regions in source-code order for legibility; these sets do not represent geographic aggregates.",
            "Cement production is the cement-sector intensity denominator. Steel and aluminium production are primary plus secondary; cement and clinker are not summed.",
            "Global energy and emissions intensities are ratios of summed quantities, never unweighted averages of regional intensities.",
            f"Reported regional energy intensities differ from the document's energy/production calculation by more than 1% in {int(mismatched.sum())} of {int(finite_comparisons.sum())} comparable region-sector-years. Both definitions are retained and exported in intensity_comparison.csv.",
            cost_note,
            "No smoothing, extrapolation, uncertainty bands, clipping of negative data, or normalization to an arbitrary base year is applied.",
        ]
        audit = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source": input_path.name,
            "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "scenario": scenario,
            "source_rows_selected": len(results.raw),
            "figure_count": len(panels),
            "missing_input_count": len(results.coverage),
            "negative_cost_count": len(negative),
            "intensity_comparable_count": int(finite_comparisons.sum()),
            "intensity_mismatch_count": int(mismatched.sum()),
            "max_intensity_difference_GJ_per_t": float(deviation.max()) if deviation.notna().any() else None,
            "config": config,
            "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                         "numpy": np.__version__, "matplotlib": matplotlib.__version__},
            "notes": notes,
        }
        pd.DataFrame(results.coverage, columns=["Scenario", "Region", "Variable", "Unit", "Year", "Issue", "Treatment"]).to_csv(directory / "missing_inputs.csv", index=False)
        scenario_label = config.get("scenario_labels", {}).get(scenario, scenario)
        records, all_data, captions = [], [], []
        for panel in panels:
            dimensions = export_panel(panel, directory, scenario_label, config["formats"], config["dpi"])
            data = panel_data(panel, scenario)
            data.to_csv(directory / f"{panel.slug}.csv", index=False)
            all_data.append(data.assign(Figure=panel.slug))
            caption = f"{panel.title}. {scenario_label}; {panel.scope}; {min(results.years)}–{max(results.years)}. {panel.caption}"
            if config["missing_activity"] == "zero":
                caption += " Missing additive activity values are treated as zero; missing prices, costs and reported intensities remain gaps."
            else:
                caption += " Missing inputs remain gaps; global totals require complete regional coverage."
            caption += " Markers denote supplied model years, joined by straight line segments; no uncertainty estimates were supplied."
            records.append({"slug": panel.slug, "title": panel.title, "scope": panel.scope,
                            "family": panel.family, "unit": panel.unit, "caption": caption,
                            "available_points": int(data.Value.notna().sum()), "expected_points": len(data),
                            "formats": config["formats"], **dimensions})
            captions.append(f"## {panel.slug}\n\n{caption}\n")
            print(f"Rendered {scenario}/{panel.slug}", flush=True)
        pd.concat(all_data, ignore_index=True).to_csv(directory / "source_data.csv", index=False)
        (directory / "captions.md").write_text("# Figure captions\n\n" + "\n".join(captions), encoding="utf-8")
        (directory / "manifest.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (directory / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        write_gallery(directory, records, audit, scenario_label)
        print(f"Saved {len(panels)} figures and audit to {directory}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("figures.json"))
    parser.add_argument("--scenario", action="append", help="Override scenario selection; repeat for multiple scenarios")
    parser.add_argument("--output", type=Path, help="Override output folder")
    parser.add_argument("--missing-activity", choices=["zero", "preserve"], help="Override treatment of missing additive activity values")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.scenario:
        config["scenarios"] = args.scenario
    if args.output:
        config["output"] = str(args.output.resolve())
    if args.missing_activity:
        config["missing_activity"] = args.missing_activity
    generate(config, args.config.resolve().parent)


if __name__ == "__main__":
    main()
