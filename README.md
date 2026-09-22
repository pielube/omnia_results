# OMNIA report figures

Reproducible, standalone trend plots from `results_NoCE_260915.csv`, following the metric definitions in `ResultsMetrics.docx`. The default run selects **`baseline_noce` only**, even though the input also contains `ndc_noce`.

Open [the figure gallery](figures/baseline_noce/index.html) in a browser. It starts with the 11 global figures; filters reveal European production and regional metrics. Each figure has an editable PDF, an SVG with editable text, a 600 dpi PNG, a source-data CSV, and a caption.

## Generate the figures

Use Python 3.11 or newer. In the existing `dataviz-env`, the dependencies are already available:

```powershell
python -m omnia_results
python -m unittest discover -s tests -v
```

For a fresh environment:

```powershell
python -m pip install -e .
python -m omnia_results
```

Paths in [figures.json](figures.json) resolve relative to that configuration file. Re-running replaces the generated files with matching names; it does not delete unrelated files. If changing region lists, use a fresh output directory to avoid keeping old figures alongside the new set.

```powershell
python -m omnia_results --config figures.json --output figures_review
```

## Missing values and source checks

**The initial configuration preserves missing data.** The CSV and metrics document do not establish whether blank cells and absent regional rows mean zero activity. Consequently, a global total is available only when all 28 regional inputs are available. Incomplete series have gaps and a figure note; the gallery reports their coverage. Global aluminium production and carbon capture have no complete points under this policy.

If the model's export convention is confirmed to mean zero activity, set `"missing_activity": "zero"` in `figures.json` and regenerate, or use:

```powershell
python -m omnia_results --missing-activity zero
```

That policy applies only to additive production, fuel use, emissions and capture. It never fills missing prices, costs or reported intensities, and never turns an entirely absent variable/unit into zero. Every missing input is recorded with its treatment in `missing_inputs.csv`. A zero denominator produces an undefined intensity, not zero.

Further source details:

- There is no `Global` or `World` row. Global quantities sum the 28 explicitly configured regions, assumed to be a disjoint world partition. The loader rejects unexpected regions so an aggregate cannot accidentally be summed with its components.
- European production currently uses **ENE, ENW, EUE, EUM and EUW**, separately. This selection is configurable. Region codes are kept unchanged because no region codebook was supplied.
- The emissions variables occur under both `MtCO2e/yr` and `MtCO2e/Mt`. The latter rows are blank in this export. Calculations select the full variable **and** unit.
- Fuel parent totals overlap their detailed subcategories. The pipeline sums only the exact carriers in the document, never `Gases|Gas`, `Liquids|Oil` or `Solids|Coal` alongside their parents.
- Many reported regional energy intensities differ from the document's final-energy/production formula. Regional charts show the supplied intensity, while global charts use the requested derived formula. `intensity_comparison.csv` preserves both definitions without altering either.
- There are 14 negative regional cost observations in `baseline_noce`, all in 2019. Global steel annualised cost in 2019 exceeds the global system total. Values are retained in the plots and listed in `negative_costs.csv`; their accounting needs interpretation before making claims about costs.

`audit.json` records the configuration, selected scenario, input SHA-256, software versions, coverage and source checks. `manifest.json` lists figure dimensions, filenames and captions. `source_data.csv` contains every plotted value; an empty value denotes missingness.

## Figure coverage

| Metric in the document | Output | Calculation / display unit |
| --- | --- | --- |
| Production | Three sector plots globally and for each of five European regions (18 figures) | Cement and clinker separately; steel and aluminium total, primary and secondary; Mt/yr |
| Production (derived) | Total curves in steel and aluminium production plots | Primary + secondary; Mt/yr |
| Sector energy use | `energy` | Sum the listed carriers; EJ/yr. Carrier inputs are not plotted separately. |
| Global energy intensity | `energy_intensity` | Global EJ/yr ÷ global Mt/yr × 1,000 = GJ/t |
| Regional energy intensity | Three sectors × four regional sets (12 figures) | Supplied EJ/Mt × 1,000 = GJ/t |
| Regional carbon price | Four regional sets | Supplied USD_2010/t CO2e |
| Sectoral emissions | `emissions` | Global MtCO2e/yr ÷ 1,000 = GtCO2e/yr |
| Share of industry emissions | `emissions_share` | Global sector emissions ÷ global industry emissions × 100 |
| Emissions intensity | `emissions_intensity` | Global MtCO2e/yr ÷ global Mt/yr = tCO2e/t |
| CO₂ capture | `capture` | Global MtCO2/yr |
| System costs | `system_cost`, `sector_costs` | Trillion and billion USD_2010/yr respectively; separate axes/scales |

The full default suite contains **42 single-panel figures**: 11 global, 15 European production, and 16 regional. Regional sets partition the source-code order into groups of seven to keep individual curves legible; these sets have no geographic or aggregation meaning. Edit `regional_groups` to arrange the codes differently, preserving each region exactly once.

Cement production, rather than clinker production, is the cement-sector intensity denominator. Cement and clinker are never added together. The duplicated steel/aluminium total variable names in the document's “Production (derived)” row are interpreted as one total per sector. Global intensities and emissions shares are ratios of global sums, not unweighted averages of regional ratios.

## Visual conventions

Figures use white backgrounds, restrained horizontal guides, 1 pt lines, 5–7 pt sans-serif text, and distinct colours with redundant markers and line patterns. The portable DejaVu Sans font is embedded in PDFs; SVG text remains editable and includes fallback font families. Single-column plots are 89 × 78 mm; the regional plots are 120 × 85 mm. The image canvas is not tightly cropped on export, preserving the physical dimensions.

These choices follow [Nature's figure preparation guidance](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/) on figure sizes, legible text and editable artwork. They provide a report-ready visual style, rather than asserting compliance with every journal's submission requirements. Use PDF or SVG for layout, and retain the native physical size when judging label legibility. PNG files are provided for report software that needs raster images.

Markers show the six supplied years, positioned on a numerical time axis. Straight segments connect the supplied values; there is no smoothing, extrapolation or invented uncertainty. All plots include zero unless negative source values require a lower bound. Sector colours and line patterns are consistent across the global metric plots. Production-route plots distinguish totals, primary and secondary using separate markers and dashes.

## Extend to later scenarios and panel figures

Add scenario names to `scenarios` in the configuration, or repeat the command-line option:

```powershell
python -m omnia_results --scenario baseline_noce --scenario ndc_noce
```

Each scenario receives its own directory and audit. This does not combine scenarios into a single chart yet. No `ndc_noce` figures are generated by default.

The code separates calculation from rendering:

- [metrics.py](omnia_results/metrics.py): source validation, exact variable/unit selection, aggregation and unit conversions.
- [plots.py](omnia_results/plots.py): curve/panel definitions, style, axes-level rendering and export.
- [pipeline.py](omnia_results/pipeline.py): configuration, generation, provenance and review gallery.

`draw_panel(ax, panel)` accepts an existing Matplotlib `Axes`, so later panel figures can compose these same plots without rewriting the metric calculations. Shared scales and scenario encodings can be chosen when the comparison design is known.
