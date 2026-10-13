"""One macroregional carbon-price comparison for all ten climate/CE scenarios."""

from datetime import datetime, timezone
from html import escape
import hashlib
import json
from pathlib import Path
import platform
import re

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from .comparison_common import COMPARISON_STYLE, POLICY_COLORS, POLICY_MARKERS, POLICY_SHORT
from .consolidation_data import CE_POLICIES, CLIMATE_LABELS, MATERIAL_CLIMATES
from .metrics import Results
from .report import ReportFigure, check_figure_bounds
from .report_data import ReportData
from .report_gallery import _figure_card


PRICE_VARIABLE = "Price|Carbon"
PRICE_UNIT = "USD_2010/t CO2e"
DISPLAY_UNIT = "USD_2010/tCO2e"
CLIMATE_STYLES = {"ndc": "-", "baseline": "--"}


class CarbonPriceData:
    """Keep the reference macroregions and use common fixed emissions weights."""

    def __init__(self, config: dict, base: Path):
        self.config = config
        self.source = (Path(base) / config["input"]).resolve()
        self.years = list(config["years"])
        expected = [f"{climate}_{policy}" for climate in MATERIAL_CLIMATES for policy in CE_POLICIES]
        selected = config["scenarios"]
        if len(selected) != len(expected) or set(selected) != set(expected):
            raise ValueError("Carbon-price comparisons require all ten climate/CE scenarios exactly once")
        self.results = {scenario: Results(self.source, scenario, config["regions"], self.years,
                                          config.get("missing_activity", "preserve"))
                        for scenario in expected}
        if set(next(iter(self.results.values())).available_scenarios) != set(expected):
            raise ValueError("Carbon-price source must contain exactly the ten selected scenarios")
        self.weight_scenario = config.get("price_weight_scenario", "baseline_noce")
        if self.weight_scenario not in self.results:
            raise ValueError("price_weight_scenario must be a selected scenario")
        # Reuse the reference figure's partition and base-emissions validation.
        reference = ReportData(self.results[self.weight_scenario],
                               {**config, "producer_groups": config.get("macroregions"),
                                "ranking_year": max(self.years)})
        self.groups = reference.macroregions
        self.weight_year = reference.price_weight_year
        self.weights = reference.carbon_price_weights().assign(WeightScenario=self.weight_scenario)
        base = self.weights.set_index("Region").BaseEmissions_MtCO2e.reindex(config["regions"])
        self.weight_emissions_identical = all(
            np.array_equal(base.to_numpy(),
                           results.raw.loc[(results.raw.Variable == "Emissions|GHG|Industry") &
                                           (results.raw.Unit == "MtCO2e/yr")]
                           .set_index("Region")[str(self.weight_year)].reindex(config["regions"]).to_numpy())
            for results in self.results.values())
        self._prices = None
        self._components = None

    def _calculate(self):
        rows, components = [], []
        for scenario, results in self.results.items():
            climate, policy = scenario.split("_", 1)
            prices = results.values(PRICE_VARIABLE, PRICE_UNIT, activity=False)
            for group, members in self.groups.items():
                weights = self.weights.loc[self.weights.Group == group].set_index("Region").Weight
                weighted = prices.loc[members].mul(weights, axis=0)
                # A zero-weight region cannot invalidate the group's index.
                weighted.loc[weights.index[weights.eq(0)]] = 0.
                values = weighted.sum(axis=0, min_count=len(members))
                for year in self.years:
                    rows.append({"Scenario": scenario, "Climate": CLIMATE_LABELS[climate],
                                 "Policy": policy, "Group": group, "Year": year,
                                 "Value": values.loc[year], "Unit": DISPLAY_UNIT,
                                 "SourceVariable": PRICE_VARIABLE, "SourceUnit": PRICE_UNIT,
                                 "SourceRegions": json.dumps(members),
                                 "WeightScenario": self.weight_scenario, "WeightYear": self.weight_year,
                                 "Role": "plotted"})
                    for region in members:
                        components.append({"Scenario": scenario, "Group": group, "Region": region,
                                           "Year": year, "Price": prices.loc[region, year],
                                           "Weight": weights.loc[region],
                                           "WeightedPrice": weighted.loc[region, year],
                                           "SourceVariable": PRICE_VARIABLE, "SourceUnit": PRICE_UNIT,
                                           "WeightScenario": self.weight_scenario,
                                           "WeightYear": self.weight_year})
        self._prices, self._components = pd.DataFrame(rows), pd.DataFrame(components)

    def prices(self) -> pd.DataFrame:
        if self._prices is None:
            self._calculate()
        return self._prices.copy()

    def components(self) -> pd.DataFrame:
        if self._components is None:
            self._calculate()
        return self._components.copy()


class CarbonPriceBuilder:
    """One axis per reference macroregion, with ten unshifted scenario curves."""

    def __init__(self, data: CarbonPriceData, config: dict):
        self.data, self.config = data, config

    def figures(self):
        data = self.data.prices()
        years = self.data.years
        first, last = min(years), max(years)
        ticks = sorted({first, last, *[year for year in years if year % 10 == 0]})
        for number, (group, members) in enumerate(self.data.groups.items(), start=1):
            selected = data.loc[data.Group == group].copy()
            title = f"Carbon-price index | {group}"
            fig, ax = plt.subplots(figsize=(183 / 25.4, 108 / 25.4))
            fig.subplots_adjust(left=.105, right=.69, bottom=.20, top=.80)
            fig.text(.035, .977, title, fontsize=8, weight="bold", va="top")
            fig.text(.035, .925, f"All ten climate/CE scenarios | {first}\N{EN DASH}{last}",
                     fontsize=6.7, color="#52616A", va="top")
            fig.text(.105, .86, f"Fixed {self.data.weight_year} industrial-emissions weights",
                     fontsize=6.5, color="#52616A", va="top")
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#DDE3E6", linewidth=.35)
            ax.tick_params(labelsize=6.3, pad=3)
            ax.set_xticks(ticks)
            ax.set_xlim(first - .6, last + .6)
            ax.set_xlabel("Year", fontsize=7, labelpad=4)
            ax.set_ylabel("Carbon price (USD$_{2010}$ per t CO$_2$e)", fontsize=7, labelpad=4)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4, min_n_ticks=3))
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.4g}"))
            handles, labels = [], []
            for scenario in self.data.results:
                climate, policy = scenario.split("_", 1)
                values = selected.loc[selected.Scenario == scenario].set_index("Year").Value.reindex(years)
                line, = ax.plot(years, values.to_numpy(dtype=float), color=POLICY_COLORS[policy],
                                linestyle=CLIMATE_STYLES[climate], marker=POLICY_MARKERS[policy],
                                markerfacecolor="white" if policy.endswith("_eu") else POLICY_COLORS[policy],
                                linewidth=1.05, markersize=3, markeredgewidth=.65, label=scenario)
                handles.append(line)
                labels.append(f"{CLIMATE_LABELS[climate]} | {POLICY_SHORT[policy]}")
            finite = selected.Value.to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            low = min(0., float(finite.min())) if finite.size else 0.
            high = max(0., float(finite.max())) if finite.size else 1.
            span = max(high - low, 1.)
            ax.set_ylim(low - .04 * span if low < 0 else 0., high + .08 * span)
            fig.legend(handles=handles, labels=labels, title="Scenarios", title_fontsize=6.7,
                       loc="upper left", bbox_to_anchor=(.72, .81), fontsize=6.2,
                       handlelength=2.5, handletextpad=.6, labelspacing=.85, frameon=False)
            fig.text(.035, .083, "Colour = CE setting; solid = NDC; dashed = NDC+LTT.",
                     fontsize=5.7, color="#52616A")
            fig.text(.035, .051,
                     "M/H = medium/high CE; EU+ = EU + international partners (17 model regions).",
                     fontsize=5.5, color="#52616A")
            fig.text(.035, .020, "Prices retain missingness. Curves can overlap where scenario prices coincide.",
                     fontsize=5.5, color="#63717A")
            caption = (
                f"Carbon-price indices for {group}, comprising {', '.join(members)}, in all ten "
                f"climate/CE scenarios from {self.config['input']}. Each curve is the weighted mean "
                f"of exact {PRICE_VARIABLE} [{PRICE_UNIT}] source prices. Weights use "
                f"{self.data.weight_scenario} reported industrial GHG emissions in {self.data.weight_year}, "
                "normalized within this group and held fixed across every scenario and year. "
                "Colours identify the five CE settings using the shared report palette; solid lines "
                "denote NDC (ndc_*) and dashed lines NDC+LTT (baseline_*). Squares denote no CE, "
                "open circles EU-only CE and filled diamonds EU-plus-partners CE. EU+ covers "
                "17 participating model regions, rather than worldwide adoption. "
                "Prices are in constant 2010 USD per tonne CO2-equivalent. Source-year coordinates "
                "and signs are preserved; straight segments connect the supplied observations. "
                "Missing prices with positive weights yield gaps; zero-weight prices do not "
                "invalidate an index. Coincident curves are not displaced. Each region has its "
                "own linear price scale, including zero. Weights and raw regional contributions "
                "are exported alongside the figure data.")
            slug = f"fig{number:02}_carbon_prices_" + re.sub(r"[^a-z0-9]+", "_", group.lower()).strip("_")
            yield ReportFigure(slug, str(number), title,
                               f"How do carbon prices in {group} differ across climate and CE settings?",
                               caption, fig, selected, 1)


def generate_carbon_prices(config: dict, base: Path):
    data = CarbonPriceData(config, base)
    directory = (base / config["output"]).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    records, exported, captions = [], [], []
    with plt.rc_context(COMPARISON_STYLE), PdfPages(directory / "report_figures.pdf",
            metadata={"Title": "Regional carbon prices across all CE scenarios"}) as book:
        for spec in CarbonPriceBuilder(data, config).figures():
            check_figure_bounds(spec.figure, spec.slug)
            for fmt in config["formats"]:
                metadata = {"Title": spec.title, "Description": spec.caption, "Date": None} if fmt == "svg" else None
                spec.figure.savefig(directory / f"{spec.slug}.{fmt}", dpi=config["dpi"], metadata=metadata)
            book.savefig(spec.figure)
            width, height = spec.figure.get_size_inches() * 25.4
            plt.close(spec.figure)
            frame = spec.data.assign(Figure=spec.slug)
            frame.to_csv(directory / f"{spec.slug}.csv", index=False)
            exported.append(frame)
            records.append({"number": spec.number, "slug": spec.slug, "title": spec.title,
                            "question": spec.question, "caption": spec.caption, "section": "Main report",
                            "group": spec.data.Group.iloc[0], "panel_count": spec.panel_count,
                            "scenario_count": len(data.results), "width_mm": float(width),
                            "height_mm": float(height), "formats": config["formats"]})
            captions.append(f"## Figure {spec.number}: {spec.title}\n\n{spec.caption}\n")
            print(f"Rendered carbon_prices/{spec.slug}", flush=True)
    pd.concat(exported, ignore_index=True).to_csv(directory / "source_data.csv", index=False)
    data.components().to_csv(directory / "regional_prices.csv", index=False)
    data.weights.to_csv(directory / "carbon_price_weights.csv", index=False)
    data.weights[["Group", "Region"]].to_csv(directory / "region_membership.csv", index=False)
    missing = [entry for results in data.results.values() for entry in results.coverage]
    pd.DataFrame(missing, columns=["Scenario", "Region", "Variable", "Unit", "Year", "Issue", "Treatment"]).to_csv(
        directory / "missing_inputs.csv", index=False)
    audit = {"generated_utc": datetime.now(timezone.utc).isoformat(), "source": config["input"],
             "source_sha256": hashlib.sha256(data.source.read_bytes()).hexdigest(),
             "figure_count": len(records), "scenario_count": len(data.results),
             "plotted_point_count": len(data.prices()), "missing_plotted_point_count": int(data.prices().Value.isna().sum()),
             "missing_input_count": len(missing), "scenario_interpretation": CLIMATE_LABELS,
             "weight_year_emissions_identical_across_scenarios": data.weight_emissions_identical,
             "scenario_order": list(data.results), "config": config,
             "style": {"policy_colors": POLICY_COLORS, "climate_line_styles": CLIMATE_STYLES,
                       "policy_markers": POLICY_MARKERS},
             "versions": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                          "numpy": np.__version__, "pandas": pd.__version__},
             "notes": ["One single-axis figure per reference macroregion; every figure contains all ten scenarios.",
                       "Europe retains the five-region ENE/ENW/EUE/EUM/EUW definition from the reference figure, not the three-region EU policy area.",
                       f"All indices use {data.weight_scenario} {data.weight_year} industrial-GHG weights, fixed across all years and scenarios.",
                       "Carbon prices and weighting emissions always retain missingness; prices are never zero-filled.",
                       "Figures calculate directly from the configured raw CSV. Existing figure collections are preserved."]}
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (directory / "captions.md").write_text("# Regional carbon-price figure captions\n\n" + "\n".join(captions), encoding="utf-8")
    _write_methods(directory, data, config)
    _write_gallery(directory, records, config)
    print(f"Saved {len(records)} regional carbon-price figures to {directory}", flush=True)


def _write_methods(directory, data, config):
    groups = "\n".join(f"- **{group}:** {', '.join(members)}" for group, members in data.groups.items())
    weight_note = (
        f"Regional industrial emissions in {data.weight_year} are identical across all ten scenarios "
        "in this source, so the common weights also reproduce the original scenario-specific Figure 3 definition."
        if data.weight_emissions_identical else
        "The same reference weights apply throughout, avoiding changes in the index caused by scenario-specific weighting.")
    text = f"""# Regional carbon prices across scenarios

The collection uses `{config['input']}` and {len(data.groups)} configured macroregions. The delivered configuration retains the nine groups of the reference Figure 3. Each macroregion has its own one-axis figure, with a separate curve for each of the ten climate/CE scenarios. `baseline_*` denotes NDC+LTT and `ndc_*` denotes NDC. All supplied years ({', '.join(map(str, data.years))}) retain their source coordinates.

## Geographic definitions

{groups}

These groups partition all {len(config['regions'])} source model regions exactly once. Europe contains five source regions; it differs from the three-region EU policy area. CE scope EU+ refers to EU plus international partners (17 regions), rather than worldwide adoption.

## Price calculation

Select exactly `{PRICE_VARIABLE}` in `{PRICE_UNIT}`. For each group, normalize `{data.weight_scenario}` `{data.weight_year}` `Emissions|GHG|Industry` values in `MtCO2e/yr` to sum to one. The group price in every scenario/year is the sum of each member's price times that fixed weight. {weight_note}

Weighting emissions must be finite, nonnegative and complete, with a positive sum in each group. Missing prices with positive weights invalidate an index; zero-weight regions contribute zero even when their prices are missing. Neither prices nor weighting emissions are zero-filled. Negative values, including tiny source numerical residuals, are retained. Prices are weighted model indices, not sums or uniform country-wide taxes.

## Plot interpretation and exports

The five CE settings use the same colours as current report Figures 7 and 8. Solid lines show NDC; dashed lines show NDC+LTT. Square markers show no CE, open circles EU-only CE and filled diamonds EU-plus-partners CE. Each region uses a linear vertical scale including zero; different regions have different price ranges. Coincident observations can overlap and are not displaced. Straight segments connect source observations without smoothing or extrapolation.

Every individual PDF/SVG/{config['dpi']} dpi PNG has a CSV of its plotted values. `report_figures.pdf` combines all {len(data.groups)} figures in region order. `source_data.csv` collects the plotted observations, `carbon_price_weights.csv` records the fixed weighting data, and `regional_prices.csv` records every raw regional price and weighted contribution. The combined CSVs retain exact variables, units, group membership and weighting scenario/year. `region_membership.csv`, `manifest.json`, `audit.json` and `missing_inputs.csv` document geography, dimensions, source hash, scenario/style mapping and missingness. PDF fonts are embedded and SVG text remains editable. The font context stays active until the combined PDF closes; figure text bounds are checked before export.

Regenerate from the repository root:

```powershell
python -m omnia_results --config {config.get('config_file', 'figures.carbon_prices.261008.json')}
```
"""
    (directory / "methods.md").write_text(text, encoding="utf-8")


def _write_gallery(directory, records, config):
    cards = "".join(_figure_card(record) for record in records)
    links = "".join(f'<a href="{name}">{label}</a>' for name, label in (
        ("report_figures.pdf", "Complete PDF"), ("source_data.csv", "Plotted data"),
        ("carbon_price_weights.csv", "Fixed weights"), ("regional_prices.csv", "Regional prices"),
        ("methods.md", "Methods"), ("captions.md", "Captions"), ("audit.json", "Audit")))
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Regional carbon prices across scenarios</title><style>
body{{margin:0;background:#f5f6f4;color:#233139;font-family:Arial,Helvetica,sans-serif;line-height:1.6}}main{{max-width:1120px;margin:auto;padding:40px 24px}}a{{color:#006887}}h1{{font-size:32px;line-height:1.2}}h3{{font-size:23px;margin:4px 0}}.links,.downloads{{display:flex;gap:18px;flex-wrap:wrap;margin:20px 0}}.figure-card{{background:white;border:1px solid #dce2e2;border-radius:5px;margin:30px 0;padding:24px}}.figure-number{{font-size:12px;text-transform:uppercase;color:#62747e}}figure{{margin:0}}.artwork img{{width:100%;height:auto;display:block}}figcaption{{font-size:13px;color:#52616a;border-top:1px solid #e5e9e9;padding-top:16px}}.purpose{{color:#52616a}}.downloads{{font-size:12px;font-weight:bold}}
@media(max-width:650px){{main{{padding:24px 12px}}.figure-card{{padding:14px}}h1{{font-size:27px}}h3{{font-size:20px}}}}
</style></head><body><main><h1>Regional carbon prices across scenarios</h1>
<p>{len(records)} regional figures from {escape(config['input'])}, each with all ten scenarios. Colours identify CE settings; solid lines show NDC and dashed lines NDC+LTT. Indices retain the reference figure's fixed 2019 industrial-emissions weighting.</p>
<div class="links">{links}</div>{cards}</main></body></html>'''
    (directory / "index.html").write_text(page, encoding="utf-8")
