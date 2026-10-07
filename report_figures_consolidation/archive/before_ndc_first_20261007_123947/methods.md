# Consolidated report methods

All consolidated figures use direct extraction from `results_allCE_261002.csv` through the unit-aware `Results` calculations. Existing exported figures and their CSVs are not calculation inputs. Future figure builders should use the same `ConsolidationData` source interface.

Every `baseline_*` scenario in this source denotes **NDC+LTT**; `ndc_*` denotes **NDC**. Raw source identifiers remain unchanged for traceability. This interpretation applies to every CE setting. Earlier figure collections retain their original labels and files.

## Figure 1: production

The top row uses `baseline_noce` (NDC+LTT, no CE), and the bottom row uses `ndc_noce` (NDC, no CE). Columns show cement/clinker, iron and steel, and aluminium. All 28 configured model regions enter each global sum exactly once. The `Mt/yr` unit must match exactly; detailed child variables and other units do not enter these sums.

Cement and clinker use `Production|Non-Metallic Minerals|Cement` and `Production|Non-Metallic Minerals|Cement Clinker`. These products are shown separately. Steel and aluminium use their exact primary and secondary production parents; total output is the sum of these two routes. The figure CSV records the original variable names for every plotted observation in `SourceVariables`.

Missing additive production, energy, emissions and capture values and absent regional rows are provisionally treated as zero, following the existing report convention. An entirely absent variable/unit is an error under either policy. `missing_inputs.csv` identifies every affected regional observation. Prices, costs and reported intensities retain missingness in the shared source-calculation module.

Panels a–c and d–f represent NDC+LTT and NDC respectively. Each sector has identical vertical limits in both rows, with different scales across sectors. Markers show the supplied model years; straight lines connect those observations. No smoothing, extrapolation or uncertainty estimates are added.

## Figure 2: paired leading producers

Producer groups use the same partition as the original leading-producer report. **Europe combines ENE, ENW, EUE, EUM and EUW**; this statistical producer group differs from the three-region EU policy group used in the CE comparisons. Remaining groups preserve individual model regions, with readable model-group labels. Indonesia group includes Indonesia, Philippines and Viet Nam. `producer_membership.csv` records exact model-region membership, which partitions all 28 regions once.

For each sector, the top 12 groups are selected and ordered by positive, available **NDC+LTT (`baseline_noce`) output in 2050**, with alphabetical tie breaking. This one cohort and order applies to both pathways. Under preserved missingness it denotes the largest groups with available ranking-year totals. NDC is not ranked independently. `producer_rankings.csv` records both pathways' output and global shares for the common cohort.

Each producer group has two endpoint pairs, joining 2019 and 2050: upper solid line for NDC+LTT and lower dashed line for NDC. Open circles denote the earlier year and filled sector-coloured diamonds the later year. Endpoints use regional cement production (excluding clinker), or primary plus secondary steel/aluminium output. Thin connectors describe two endpoint observations. The Other model regions row sums the groups outside the fixed cohort and is unranked; it completes global production for each pathway and endpoint. Coverage is calculated separately for each pathway as the common cohort's output divided by that pathway's global output; the exported context data retain every supplied model year.

Logarithmic production axes retain the original figure convention. Zero or negative observed endpoints require `producer_axis_scale` set to `linear`; missing endpoints remain gaps. Axis ranges include both pathways within each sector. Figure 2 is 180 × 165 mm.

## Figure 3: final energy and derived intensity

Six panels align cement, iron and steel, and aluminium by column. The upper row shows global final energy (EJ/yr); the lower row shows derived global energy intensity (GJ/t). Every panel contains both no-CE pathways: NDC+LTT (`baseline_noce`) and NDC (`ndc_noce`). The legend distinguishes line styles and markers. Each panel uses a common axis for the two pathways, with separate sector scales. Source years retain their exact positions; coincident curves are not displaced.

Final energy sums the exact carrier parents listed in `ResultsMetrics.docx`: electricity, gases, liquids and solids for cement, plus hydrogen for steel and aluminium. Nested variables such as Gases|Gas, Liquids|Oil and Solids|Coal are excluded. Each of the 28 model regions is counted once.

Derived intensity equals **1000 × global final energy (EJ/yr) / global material production (Mt/yr)**. Cement uses cement output, excluding clinker; steel and aluminium use primary plus secondary output. This is a ratio of global sums, not a mean of regional ratios and not the supplied regional intensity series. Missing or nonpositive production makes intensity undefined. Missingness follows the same explicit activity policy as production and energy totals. Each exported intensity row includes its numerator, denominator and unit-conversion factor. Figure 3 is 180 × 150 mm.

## Figure 4: emissions and capture

Four panels compare cement, iron and steel, and aluminium under NDC+LTT (`baseline_noce`) and NDC (`ndc_noce`), both without CE. Sector colours are consistent with the preceding figures; line styles and markers distinguish pathways. Each panel contains all six sector/pathway curves on one axis, without displacement of coincident observations. Figure 4 is 180 × 150 mm.

Panel a sums each sector's exact `Emissions|GHG|Industry|...` series in `MtCO2e/yr` across all 28 model regions and divides by 1,000 for `GtCO2e/yr`. Negative source emissions remain signed. Panel b divides those sector totals by the **same pathway's** `Emissions|GHG|Industry` global total and multiplies by 100. The three material sectors are not assumed to exhaust industrial emissions.

Panel c divides global sector GHG (`MtCO2e/yr`) by global material production (`Mt/yr`), giving `tCO2e/t` with a numerical conversion factor of one. Cement uses cement output excluding clinker, and metals use primary plus secondary output. These are ratios of global sums, not averages of regional intensities. Missing or nonpositive denominators make shares or intensities undefined.

Panel d sums the exact `Carbon Capture|Industry|...` series in `MtCO2/yr`. Capture is shown separately and is **not subtracted again** from the reported GHG series. GHG emissions include CO₂-equivalent units, while capture refers to CO₂. Small aluminium capture values retain their full source precision even when visually close to zero on the shared scale. All four metrics use exact variable/unit selections and the same explicit missing-activity policy. The source CSV retains raw numerators, denominators, their units and conversion factors for audit.

## Figure 5: material production across CE scenarios

The 2 × 3 layout places **NDC above NDC+LTT**, with cement/clinker, iron and steel, and aluminium columns. Each panel starts with a **2019 no-CE bar**, extracted from that row's own no-CE scenario. This historical observation is not averaged across scenarios. The five remaining categories show **2050 production** in this fixed order: no CE, medium CE in the EU, high CE in the EU, medium CE in the EU plus partners, and high CE in the EU plus partners. The supplied source has identical historical regional component values and missingness across all ten scenarios, including both climate pathways.

Cement and clinker use distinct, side-by-side bars and are never added. Metal bars stack exact primary and secondary production parents. Every component is a sum of all 28 model regions in `Mt/yr`, using the same variable and unit rules as Figure 1. Bar height shows production directly for each climate pathway; there are no NDC output diamonds. Vertical limits match across rows within each sector. Figure 5 is 180 × 175 mm.

Labels above the five future categories give **100 × future production / same-pathway future no-CE production − 100**. Cement changes exclude clinker, and metal changes use primary plus secondary totals. The future no-CE bar is the reference; the historical bar has no change label. Labels within sufficiently large metal secondary segments show **100 × secondary / total production**, including the historical bar. Missing or nonpositive denominators yield undefined percentages.

`M/H` denotes medium/high CE. `EU+` denotes the EU plus international partner regions (17 model regions), rather than worldwide CE adoption. The exported data identify each bar's year, order, type and policy, and retain component values, metal totals, percentage changes and secondary shares. Numerator, denominator, units, conversion factor, `Offset`, and reference scenario/year document every calculation. For percentage changes, `Offset` is −100; it is zero for quantities and shares. These fields distinguish the historical anchor from the future no-CE comparison reference.

## Figure 6: final-energy mix across CE scenarios

The 2 × 3 layout places **NDC above NDC+LTT**, with cement, iron and steel, and aluminium columns. Each panel starts with **2019 final energy from its own no-CE scenario**, followed by the same five **2050 CE settings** as Figure 5. The historical bar is extracted separately for each pathway and is not averaged across scenarios. Historical carrier quantities and missing-cell patterns agree across all ten scenarios to numerical precision (maximum global carrier discrepancy approximately 1.6 × 10⁻¹⁵ EJ/yr).

Stacks retain the original carrier colours and order: electricity, gases, hydrogen, liquids and solids. Cement has four exact parents and excludes hydrogen; steel and aluminium include all five. Every carrier value selects its exact `Final Energy|Industry|sector|carrier` parent and `EJ/yr` unit, summing the 28 configured regions once. Nested fuel categories are excluded, and no additional aggregate final-energy variable is added. Total final energy sums all required carrier totals; a missing carrier under preserved missingness leaves the total undefined. The provisional zero policy follows the same documented activity convention as preceding figures.

Labels above future bars give **100 × scenario final energy / same-pathway future no-CE final energy − 100**; the historical bar has no change label. Missing or nonpositive reference energy yields an undefined percentage. Both rows use the same scale within each sector, including historical values. Figure 6 is 180 × 175 mm. The source CSV records all carrier quantities, stack totals, future changes, exact source parents, and the historical/future bar metadata. Change rows retain numerator, denominator, their `EJ/yr` units, factor 100, offset −100, and the future reference scenario and year.

## Figure 7: emissions response and geographic contributions

The **3 × 2** layout puts **NDC on the left and NDC+LTT on the right**. Panels a,b show global cement, steel and aluminium GHG trajectories for all five CE settings. Panels c,d decompose **2050 total industrial GHG changes** into those three sectors and other industry. Panels e,f decompose **the three sectors' changes** geographically. Every change subtracts no CE under the same climate pathway in the same year. Row pairs use common vertical limits. Model-year markers retain their actual coordinates.

Sector data select exact `Emissions|GHG|Industry|sector` parents and the `MtCO2e/yr` unit. Total industry selects `Emissions|GHG|Industry`. Negative source values remain signed, and separately reported CO₂ capture is not subtracted again. **Other industry = reported total industry − cement − steel − aluminium**; its change is an accounting residual and does not establish a particular rebound mechanism. Signed stack components accumulate increases and decreases independently from zero. Black diamonds and signed labels give the net change: **all industry** in the middle row, **three sectors** in the last row.

Policy geography comes from `data/CE_policy_regions_261002.csv`. EU consists of EUE, EUM and EUW. Partners comprise the 14 additional participating regions, including ENE and ENW; the other 11 model regions are Rest of world. These memberships differ from the five-region Europe producer group in Figure 2.

The three main partner regions are selected **once for the whole figure**, by their **mean absolute 2050 change in three-sector GHG across the eight CE cases** (four CE settings under each of two climate pathways). Each case uses its own climate-matched no-CE reference. Ranking requires all eight changes to be available under preserved missingness; ties use region code ascending. The supplied source selects **India (IND), China (CHN), and Indonesia group (IDN)** in that order. IDN combines Indonesia, the Philippines and Viet Nam. Keeping the same cohort, ordering and colours supports comparisons between columns and CE levels. Remaining partner regions form **Other partners**. The six disjoint groups comprise EU (3 regions), three individual partner regions, Other partners (11), and Rest of world (11), covering all 28 source regions exactly once. Aggregating the four partner categories reconstructs the original Partners quantity.

`partner_rankings.csv` records every partner's mean absolute score, rank, selection, ranking scenarios, signed case contributions and source parents. `geographic_membership.csv` records the policy group and final plotted group for every model region. The figure CSV records `ScenarioValue`, `ReferenceValue`, reference scenario/year, source regions, variables and their arithmetic coefficients. Absolute changes preserve the meaning of near-zero or negative regional reference emissions; no regional percentage changes are introduced. Missingness follows the same explicit activity policy as previous figures.

## Figure 8: annualised sector-cost changes

The **2 × 3** layout puts **NDC in the top row and NDC+LTT in the bottom row**, with cement, iron and steel, and aluminium columns. All five CE settings appear in every panel, using the same policy colours, markers and line styles as Figure 7. No CE provides the zero reference. Vertical scales match between climate pathways within each sector, include every finite change and zero, and differ between sectors.

Global annualised sector costs select the exact `Total Annualised Cost|Industry|Non-Metallic Minerals|Cement`, `Total Annualised Cost|Industry|Iron and Steel`, and `Total Annualised Cost|Industry|Non-Ferrous Metals|Aluminum` parents and `Millions USD_2010/yr`. Each global cost sums the 28 configured regions once, excluding nested children and aggregate system costs. **Costs always retain missingness**, regardless of the additive-activity policy; incomplete global coverage leaves costs undefined.

Every plotted percentage is **100 × (CE scenario cost − no-CE cost) / no-CE cost**, where the reference matches the sector, climate pathway **and model year**. This is a ratio of global sums, not an average of regional percentage changes. Finite negative scenario costs are retained; a missing, nonfinite or nonpositive no-CE denominator leaves the change undefined. No CE equals zero wherever its reference is valid. No additional inflation or currency conversion enters this dimensionless ratio.

As in the original main cost figure, plotted years start at **2024**. All earlier supplied costs, including the anomalous 2019 steel costs, remain in the figure CSV with `Role=context`. Main-period records have `Role=plotted`. Markers show supplied model years joined by straight segments; no interpolation to extra years, smoothing or extrapolation is added. `ScenarioCost`, `ReferenceCost`, numerator and denominator retain raw millions of 2010 USD per year, while `ConversionFactor=100` and `Offset=-100` reproduce the percentage. Reference scenario and year are explicit for every observation.

## Exports and reproducibility

Individual figure CSVs contain the observations, original scenario identifiers, displayed climate labels, panel/sector/series identifiers, source variables, values and units. Figure 2 also identifies groups, constituent regions, display order and the ranking scenario/year. Figures 3–6 identify metrics, all constituent source regions, numerators/denominators and conversion factors. Figures 5 and 6 also record bar categories and calculation offsets. Figure 7 records absolute scenario/reference quantities, source coefficients and the policy/geographic breakdown. Figure 8 records raw sector costs, year-specific no-CE references and the percentage calculation. `source_data.csv` combines these tables and adds the figure identifier. `audit.json` records the source SHA-256 hash, configuration, software versions and scenario interpretation, plus Figure 7's policy-table hash and partner-selection rule. `manifest.json` records captions, figure dimensions, panels and formats.

Figures use embedded TrueType fonts in PDF, editable SVG text and 600 dpi PNGs. Figure 1 is 180 × 150 mm. The PDF font context remains active through finalisation, including the combined `report_figures.pdf`. Label bounds are checked before saving. This workflow writes into `report_figures_consolidation` without deleting unrelated files.

`preserve_individual_figures` retains reviewed individual exports when newly calculated source data and a fresh PNG match them exactly. If they differ, regeneration stops rather than overwriting the reviewed version. The combined PDF always renders every configured figure directly from the raw source. The current configuration preserves Figures 1–7 while adding Figure 8; remove an entry from this list when intentionally revising that figure.

Regenerate from the repository root:

```powershell
python -m omnia_results --config figures.consolidation.json
```
