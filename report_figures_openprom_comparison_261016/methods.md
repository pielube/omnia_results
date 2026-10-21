# Global steel: OMNIA-ENGAGE and OPEN-PROM

Regenerate from the raw files with Python 3.11+ and the project dependencies:

```powershell
python -m omnia_results --config figures.openprom_comparison.261016.json
```

## Purpose and source selection

The collection compares global final energy and source-defined steel emissions under NDC and NDC+LTT. Absolute-level Figures 1 and 3 and the summary tables retain both EU and EU-plus-partner OMNIA CE adoption areas. Response Figures 2 and 4 show only OMNIA EU+ alongside OPEN-PROM; EU-only cases remain unplotted context. Figure 2 retains four time-series panels. Figure 4 uses two paired-bar panels at 2050, styled like the European three-sector comparison. The four sheets of `open_prom_results/ENGAGE Medium and High Ambition Scenarios.xlsx` supply OPEN-PROM medium/high cases. Its two configured CSVs supply the matching climate-pathway references. OMNIA reads `results_allCE_261008.csv` directly; `ndc_*` is NDC and `baseline_*` is NDC+LTT. Exact identifiers, sheets and reference scenarios are retained in `scenario_mapping.csv`. No previous figure data or temporary exploration cache is a calculation input.

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

OMNIA follows the configured `zero` missing-activity treatment. The default provisionally treats missing additive activity as zero; blank emissions and absent regional carrier rows are documented in `missing_inputs.csv`. This remains an assumption about the OMNIA export convention. Entirely absent required variable/unit combinations fail. OPEN-PROM required World totals must be complete; blanks are never filled with zero. Infinite inputs, duplicate required source keys and mismatched scenarios/units are rejected. Missing or nonfinite reference quantities never produce a defined percentage.

The XLSX reader decodes shared/inline strings and converts numeric strings before calculation. It streams actual data rows rather than trusting trailing formatted worksheet ranges. `aggregation_checks.csv` records source accounting checks, including OPEN-PROM fuel totals, gross/net/capture and regional reconciliation. Checks use tolerances for the supplied four-decimal source rounding; no source values are corrected to force closure.

## Artifacts

Figures 1/2 show final-energy levels/responses; Figures 3/4 show source-defined emissions levels/responses. Each has PDF, editable SVG, 600 dpi PNG and figure CSV. The four-page PDF collects them in that order. `global_steel_metrics.csv` contains one row per metric/model/scenario/year; `source_data.csv` also records figure/panel, plotted value/unit and context roles. `comparison_2050.csv` retains levels, absolute changes, references and percentages at full precision. The gallery presents 2050 totals and responses together. Captions, scenario mapping, diagnostics, missing inputs, checks and source SHA-256 hashes accompany the exports. The configured preserved Figures 1 and 3 retain their reviewed individual files only after fresh source CSV and PNG verification.

The same command also exports a separate four-panel emissions figure in `report_figures_emissions_comparison_261019`. Its top row compares the combined European cement, iron-and-steel and aluminium response under EU-only CE in both OMNIA and GCAM; other industry is excluded from the bars. OMNIA sums reported sector GHG, while GCAM sums the available sector CO2: cement energy and processes, steel energy and aluminium-specific energy. GCAM's non-metallic-minerals energy category is a cement proxy, supported by the supplied final-energy parent equalling its cement child; it is not documented as a cement-exclusive emissions inventory. Sector quantities are summed before normalising to each model's own combined three-sector no-CE reference. OMNIA NDC+LTT aluminium blanks retain the configured provisional-zero assumption. The GCAM input is selected by `figures.europe_relative.261016.json`. The lower row retains the global-steel/OMNIA-EU+ comparison with OPEN-PROM. Scales match within rows; geography, sectors and emissions inventories remain different between rows. Both rows are calculated from raw inputs, and the combined figure includes source data, methods and an audit. This export writes only to its separate combined-output directory.
