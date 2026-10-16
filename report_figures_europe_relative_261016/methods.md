# European CE percentage-response methods

Regenerate the collection:

```powershell
python -m omnia_results --config figures.europe_relative.261016.json
```

## What the normalisation compares

All three figures compare EU-only medium/high CE with **the same model's own 2050 no-CE result**, matching climate pathway and emissions/sector scope. The formula is **100 × (CE − no CE) / no CE**. A model's reference is never taken from the other model or the other climate pathway.

No CE is 0% by construction. Equal zero references do not establish agreement in absolute production or energy. These figures focus on CE responses; the [companion absolute collection](../report_figures_europe_comparison_261014/index.html) documents the underlying level differences. Normalisation also does not establish identical process boundaries, CE assumptions or calibration. Historical observations are omitted because these are CE-versus-no-CE comparisons at one future year.

The sources remain `results_allCE_261008.csv` and `DIAMOND_GCAMEurope8p7p3_circularity_13102026_standardized.xlsx`, sheet `Sheet1`. Both are read directly; earlier figure CSVs and artwork are not calculation inputs. Input and regional-mapping SHA-256 hashes, software versions and the complete configuration are in `audit.json`.

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

Figure 3 uses a **1 × 2** layout with NDC on the left and NDC+LTT on the right. **Panels a,b** reproduce the earlier emissions figure's industry-level percentage comparison: paired OMNIA and GCAM bars, each divided by its own 2050 no-CE industry total. **OMNIA selects the reported all-industry parent `Emissions|GHG|Industry` directly**, in `MtCO2e/yr`. It includes sectors beyond cement, iron and steel, and aluminium; it is not formed by adding those three sectors. GCAM sums `Emissions|Kyoto Gases|Energy|Demand|Industry` and `Emissions|Kyoto Gases|Industrial Processes`, both in `Mt CO2-equiv/yr`. The two panels share a common percentage scale. This compares source-defined industrial responses. Their gas coverage, GWP conventions and process boundaries are not documented as harmonised.

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

OMNIA follows the configured `zero` activity policy; the default provisional zero treatment is retained from the existing report and identified where plotted inputs use it. In Figure 3, missing sector inputs affect unplotted decomposition context, while the bars use independently reported industry-parent inputs. GCAM missing aggregate values remain gaps. Required variables/units and GCAM aggregate rows cannot be fabricated. Nonfinite or nonpositive total references produce undefined contributions and net responses. `missing_inputs.csv` records affected selected inputs and their treatment.

Each source-data row retains its CE and no-CE absolute quantities, signed numerator, total reference denominator, conversion factor, original scenarios, year, variables, units, coefficients and regions. `comparison_summary.csv` collects the net responses. Components use `Unit=percentage points`; net rows use `Unit=%`. The `Plotted` flag distinguishes chart marks from context. Emissions records additionally identify `PanelKind`, `EmissionsScope` and `EmissionsCoverage`.

Figures 1–2 are 180 × 175 mm; the single-row Figure 3 is 180 × 105 mm. All include vector PDF, editable SVG text and 600 dpi PNG exports, plus a combined three-page PDF (`report_figures_with_emissions.pdf`). The reviewed individual Figures 1–2 are retained byte-for-byte after their freshly calculated CSV and PNG match the existing exports. Automated bounds checks and visual review check label placement. No values are rescaled between models, displaced, clipped, smoothed or extrapolated to improve apparent agreement.
