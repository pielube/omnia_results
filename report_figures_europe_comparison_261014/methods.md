# European OMNIA-ENGAGE / GCAM comparison methods

Regenerate all three figures with Python 3.11+ and the project dependencies:

```powershell
python -m omnia_results --config figures.europe_comparison.261014.json
```

## Sources and selection

Calculations read `results_allCE_261008.csv` and `DIAMOND_GCAMEurope8p7p3_circularity_13102026_standardized.xlsx` (`Sheet1`) directly. Previously exported figures and CSVs are not calculation inputs. `audit.json` records source and regional-mapping hashes, software versions and the complete configuration.

Six scenarios are selected from each model: NDC and NDC+LTT, each with no, medium and high CE applied in the EU only. No EU+ scenarios enter the collection. OMNIA `baseline_*` and GCAM `LTT_*` are paired under the display label NDC+LTT; OMNIA `ndc_*` and GCAM `NDC_*` are paired as NDC. These source labels do not establish identical policy assumptions. `scenario_mapping.csv` retains every original identifier.

## Geographic scope

OMNIA sums EUE, EUM and EUW once. This is the repository's EU policy region, rather than the original report's five-region Europe producer group. The supplied country mapping assigns all EU-27 countries and these additional countries/territories to the three inseparable OMNIA regions: Greenland, Faroe Islands, Andorra, Monaco, San Marino. They cannot be subtracted using the available regional results, so OMNIA is a geographic approximation to GCAM's EU-27.

GCAM selects the supplied **EU-27 aggregate once**, without adding overlapping country rows. Its aggregate is checked against the sum of available country observations for every selected source variable and year. Some production and process-emissions variables have fewer than 27 country rows; an absent country observation is not silently assigned zero. The comparison uses the authoritative reported aggregate, while `gcam_aggregation_checks.csv` documents available-country coverage and reconciliation. `geography_membership.csv` records both model boundaries.

## Historical observations and references

Historical bars use each model's own no-CE scenario: 2019 for OMNIA and 2021 for GCAM. Future bars use 2050. No historical-year interpolation or calibration is introduced. Emissions trajectories retain all supplied observations from each model's historical reference to the comparison year; markers use their actual numerical years.

Every future CE percentage uses **the same model, climate pathway, sector and future year** as its no-CE denominator: `100 × (scenario / no_CE − 1)`. It is not a change from the historical bar. Positive, finite denominators are required; unavailable references produce gaps. All numerator/denominator values and source scenarios remain in the CSVs.

## Figure 1: material production

The 2 × 3 layout has NDC above NDC+LTT and cement, iron and steel, and aluminium columns. Within each historical/future category, OMNIA is on the left and hatched GCAM on the right. Scales match across models and climate rows within each sector.

OMNIA cement selects `Production|Non-Metallic Minerals|Cement`; metal totals sum exact primary and secondary production parents. GCAM selects the exact reported cement parent, `Production|Iron and Steel|Steel`, and `Production|Non-Ferrous Metals|Aluminum`, in `Mt/yr`. GCAM has no exported production-route split, so its reported metal output is shown as grey hatched bars without assigning it to a route. **GCAM aluminium's primary/secondary coverage is unconfirmed**; the export is not assumed to represent the same production boundary as OMNIA's primary-plus-secondary total. OMNIA secondary shares are annotated where legible. Clinker has no matching GCAM series and is omitted from the figure; any exported OMNIA clinker values are explicitly context.

## Figure 2: final-energy mix

The layout and bar categories match Figure 1. Exact carrier parents select electricity, gases, hydrogen, liquids and solids without adding overlapping detailed fuel children. OMNIA totals sum its documented carrier parents; its cement export has no hydrogen parent. GCAM totals select the reported sector final-energy parent in `EJ/yr`. GCAM cement and steel report five carrier parents; aluminium reports electricity, liquids and solids, without gas or hydrogen parents.

**Unallocated energy** is GCAM's reported sector total minus the sum of its supplied carrier parents. The residual is positive for aluminium in the supplied export and is shown as a separate stack component. No unreported carrier is invented and no residual is assigned to gases or hydrogen. Near-zero arithmetic residue below `1e-12 EJ/yr` is recorded as zero. The reported totals and exact residual coefficients remain in the source CSV. All stack components reproduce the totals used for percentage annotations.

### Aluminium scope remains unconfirmed

The workbook does not define whether its aluminium production includes primary only or primary plus secondary, nor whether its aluminium final-energy parent includes alumina refining. These definitions remain unconfirmed. The figures retain the source-defined output and energy quantities, with explicit caveats on both charts.

[Public GCAM documentation](https://jgcri.github.io/gcam-doc/demand_energy.html) describes alumina refining and aluminium smelting, with historical calibration using electrolytic aluminium production. This supports a possible primary-oriented process-scope explanation for the cross-model difference. It does **not establish the custom GCAM-Europe circularity export's definitions**, especially in future CE scenarios. The unidentified energy residual cannot be attributed to alumina refining, a fuel, feedstocks or duplicate accounting from the workbook alone.

The historical plotted quantities imply apparent energy/output ratios of approximately 25.2 GJ/t for OMNIA (7.76 Mt and 0.1953 EJ in 2019) and 55.5 GJ/t for GCAM (4.28 Mt and 0.2377 EJ in 2021). These are **diagnostic apparent ratios, not harmonised efficiency comparisons**. OMNIA's historical output includes 2.25 Mt primary and 5.51 Mt secondary aluminium. In 2050 NDC, its secondary share is approximately 76.9% with no CE, versus 57% and 67% with medium/high CE; the changing route mix helps explain energy growth despite lower total output. The scenario inputs are needed to identify the constraints behind those shares.

The outstanding GCAM information is: the production query's technology/route membership, the energy query's process and fuel membership, and how CE changes those routes and inputs. These definitions are needed before harmonising output or comparing energy intensity across models.

## Figure 3: reported industrial GHG response

This adaptation retains Figure 7's trajectory and signed-response approach. Columns show NDC and NDC+LTT. The top row compares industrial GHG trajectories; the middle row decomposes 2050 changes into each model's own reported components; the bottom row compares percentage changes against each model's own no-CE industrial total. Regional/partner contributions are omitted because the requested scope is EU-only and GCAM has no matched sector GHG decomposition.

OMNIA selects `Emissions|GHG|Industry` in `MtCO2e/yr`. GCAM sums `Emissions|Kyoto Gases|Energy|Demand|Industry` and `Emissions|Kyoto Gases|Industrial Processes`, both reported in `Mt CO2-equiv/yr`. The unit scale is the same; **the accounting scope, gas coverage and GWP conventions are not documented as harmonised**. GCAM's industrial-process aggregate is source-defined and is not presented as an all-sector GHG decomposition. GCAM sector CO2 variables cannot supply the missing sector GHG comparison, and no CO2/GHG substitution is made.

OMNIA's components are cement, iron and steel, aluminium and Other industry (reported industry minus the three sectors). GCAM's components are industrial energy demand and industrial processes. The component colours and legend distinguish these model-specific accounts. Signed components stack separately above and below zero; their sum equals the net change shown by the black diamond. Carbon capture is never subtracted again. The relative-response row compares `100 × (CE − no_CE) / no_CE` within each model, rather than treating the two source totals as identical inventories.

## Missing values and export quality

OMNIA additive activity follows the configured `zero` convention, preserving the existing report's provisional zero assumption by default. GCAM blank aggregate values remain missing and missing required aggregate rows/variables fail validation. Prices and costs are not used. `missing_inputs.csv` lists affected selected observations and their treatment.

Figures retain 180 mm width, embedded PDF fonts, editable SVG text, and 600 dpi PNG export. The first two figures are 185 mm high; Figure 3 is 235 mm high. Automated canvas bounds checks prevent cropped labels. Common sector/row limits preserve model comparisons; lines connect actual observations without smoothing, uncertainty estimates or extrapolation. The collection includes a three-page PDF, individual artwork, full-precision source CSVs, captions, this methods document, scenario/geography/variable mappings and an audit.
