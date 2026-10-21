# European CE percentage-response methods

Regenerate the collection:

```powershell
python -m omnia_results --config figures.europe_relative.261016.json
```

## What the normalisation compares

All figures compare EU-only medium/high CE with **the same model's own 2050 no-CE result**, matching climate pathway and production/energy/emissions scope. The formula is **100 × (CE − no CE) / no CE**. A model's reference is never taken from the other model or the other climate pathway. Separate steel routes, when shown, each use their own route's reference.

No CE is 0% by construction. Equal zero references do not establish agreement in absolute production or energy. These figures focus on CE responses; the [companion absolute collection](../report_figures_europe_comparison_261014/index.html) documents the underlying level differences. Normalisation also does not establish identical process boundaries, CE assumptions or calibration. Historical observations are omitted because these are CE-versus-no-CE comparisons at one future year.

The sources remain `results_allCE_261008.csv` and `DIAMOND_8p7p3_industry_19102026_standardized_v2.xlsx`, sheet `Sheet1`. Both are read directly; earlier figure CSVs and artwork are not calculation inputs. Input and regional-mapping SHA-256 hashes, software versions and the complete configuration are in `audit.json`.

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

Figure 3 retains its **1 × 2** layout, paired model bars, colours and endpoint markers, with NDC on the left and NDC+LTT on the right. Each bar now shows **100 × (sum of three-sector CE emissions − sum of three-sector no-CE emissions) / sum of three-sector no-CE emissions** within the same model, pathway, geography and 2050 year. The sum covers cement, iron and steel, and aluminium only. Other industry is excluded from both numerator and denominator. Individual-sector percentages are not averaged. The two panels share one percentage scale.

**OMNIA reports sector GHG in MtCO2e/yr; GCAM reports sector CO2 in Mt CO2/yr.** The updated workbook adds useful CO2 disaggregation, but still supplies no complete three-sector Kyoto/GHG inventory. This is a comparison of responses within the reported quantities, with gas coverage, GWP conventions and process boundaries explicitly source-defined.

| Model/sector | Exact source selection |
| --- | --- |
| OMNIA cement | Emissions\|GHG\|Industry\|Non-Metallic Minerals\|Cement |
| OMNIA steel | Emissions\|GHG\|Industry\|Iron and Steel |
| OMNIA aluminium | Emissions\|GHG\|Industry\|Non-Ferrous Metals\|Aluminum |
| GCAM cement energy | Emissions\|CO2\|Energy\|Demand\|Industry\|Non-Metallic Minerals |
| GCAM cement processes | Emissions\|CO2\|Industrial Processes\|Non-Metallic Minerals\|Cement |
| GCAM steel | Emissions\|CO2\|Energy\|Demand\|Industry\|Iron and Steel |
| GCAM aluminium | Emissions\|CO2\|Energy\|Demand\|Industry\|Non-Ferrous Metals\|Aluminum |

GCAM's non-metallic-minerals energy category is assigned to cement: the reported final-energy parent equals its cement child in this workbook. A runtime check verifies this identity for each selected scenario/year and exports the values in `gcam_sector_scope_checks.csv`. The cement-process parent overlaps the selected cement child; the non-ferrous-metals energy-CO2 parent overlaps the selected aluminium child. These parents are not added again. No additional steel/aluminium process-CO2 child is exported, so none is fabricated. Capture is not subtracted again.

Only combined three-sector net responses are displayed: `PanelKind=three_sector`, `Role=net`, `Plotted=True`. Sector contributions remain context, each divided by its own model's combined three-sector no-CE total; they reconstruct the net response. All-industry GHG responses are also retained as unplotted context, with their separate industry denominators. `ContextOnly=True`, `DisplayRole=context` and `Plotted=False` identify unplotted rows, which have no displayed panel assignment. GCAM's unplotted contributions are also exported in `gcam_emissions_context.csv`, and `gcam_emissions_availability.csv` inventories the available source quantities.

Negative reported emissions are retained. The NDC+LTT no-CE steel GHG value in OMNIA is negative, so individual-sector percentages are not used. Under the configured zero policy, **NDC+LTT aluminium GHG blanks in all three OMNIA EU regions, in no CE and both CE cases, are provisionally treated as zero**. This affects the displayed combined sum and does not establish physical zero emissions. `UsesProvisionalZeros`, `AllEUAluminiumMissingAsZero`, raw missing-input lists and scenario flags retain this assumption in the CSV. Under `preserve`, incomplete sector sums remain gaps.

## Figure 4: material production with separate steel routes

The new **2 × 4** figure retains cement and aluminium from Figure 1, with separate primary and secondary iron-and-steel columns between them. NDC appears above NDC+LTT. Paired medium/high EU-only CE bars retain the model colours, hatch and endpoint markers. All eight panels share one percentage scale.

For steel, the formula is **100 × (CE route output − no-CE route output) / no-CE route output**. Each route uses its own same-model, same-pathway, same-year no-CE production. These are production changes, not route shares or changes in shares. The two percentages are not additive and do not use the total-steel denominator from Figure 1's unplotted contribution rows. Cement and aluminium responses equal Figure 1 exactly.

OMNIA selects `Production|Iron and Steel|Primary` and `Production|Iron and Steel|Secondary`; GCAM selects `Production|Iron and Steel|Steel|Primary` and `Production|Iron and Steel|Steel|Secondary`, all in Mt/yr. GCAM's reported primary-plus-secondary sum is checked against its reported steel total for each selected source scenario/year in `steel_route_checks.csv`. Incomplete checks remain flagged; no missing route output is fabricated. These checks are distinct from percentage-point contribution closure.

**OMNIA NDC+LTT no-CE secondary steel is blank in all three EU regions in 2050.** Under the configured zero policy its route reference becomes provisional zero, so the percentage remains undefined. Preserving the blanks also leaves it undefined. Panel g therefore contains only GCAM bars and explicitly identifies the unavailable OMNIA reference; it does not substitute a total-steel denominator or infer a production value. Unavailable, nonfinite and nonpositive aggregate route references produce gaps, while observed zero changes retain endpoint markers. Partly missing OMNIA inputs follow the configured activity policy and retain explicit completeness and provisional-zero flags. `ResponseDefined`, `ReferenceComplete`, `ReferenceIssue`, `ReferenceMissingInputCount` and the full source-input lists distinguish missing references from reported physical zero production.

## Missing inputs and provenance

OMNIA follows the configured `zero` activity policy; the default provisional zero treatment is retained from the existing report and identified where plotted inputs use it. In Figure 3, missing sector inputs affect the displayed combined three-sector response and are flagged explicitly. GCAM missing aggregate values remain gaps. Required variables/units and GCAM aggregate rows cannot be fabricated. Nonfinite or nonpositive total references produce undefined contributions and net responses. `missing_inputs.csv` records affected selected inputs and their treatment.

Each source-data row retains its CE and no-CE absolute quantities, signed numerator, reference denominator, conversion factor, original scenarios, year, variables, units, coefficients and regions. `comparison_summary.csv` collects the net responses. Components use `Unit=percentage points`; net rows use `Unit=%`. The `Plotted` flag identifies series selected for display; an undefined response still has no mark. Emissions records additionally identify `PanelKind`, `EmissionsScope` and `EmissionsCoverage`. Separate steel-route records identify `ProductionGroup`, `Route` and their own-route `DenominatorScope`.

Figures 1–2 are 180 × 175 mm; the single-row Figure 3 is 180 × 105 mm. All include vector PDF, editable SVG text and 600 dpi PNG exports. Figure 4 is 240 × 175 mm. The four-page combined PDF is `report_figures_with_steel_routes.pdf`; the earlier three-page collection remains available. Existing Figures 1–3 individual artwork and CSVs are preserved after equality checks; the new figure and collection metadata are generated from the raw sources. The updated GCAM production values retain their supplied precision. Automated bounds checks and visual review check label placement. No values are rescaled between models, displaced, clipped, smoothed or extrapolated to improve apparent agreement.
