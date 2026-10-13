# Regional carbon prices across scenarios

The collection uses `results_allCE_261008.csv` and 9 configured macroregions. The delivered configuration retains the nine groups of the reference Figure 3. Each macroregion has its own one-axis figure, with a separate curve for each of the ten climate/CE scenarios. `baseline_*` denotes NDC+LTT and `ndc_*` denotes NDC. All supplied years (2019, 2024, 2030, 2035, 2040, 2050) retain their source coordinates.

## Geographic definitions

- **Europe:** ENE, ENW, EUE, EUM, EUW
- **China region:** CHN
- **India:** IND
- **North America:** USA, CAN, MEX
- **Latin America & Caribbean:** BRA, CHL, LAM
- **Other Asia–Pacific:** ANZ, ASE, ASO, IDN, JPN, SKT
- **Russia & Central Asia:** RUS, ASC
- **Middle East:** MDA, MEA
- **Africa:** AFE, AFN, AFW, AFZ, NIG

These groups partition all 28 source model regions exactly once. Europe contains five source regions; it differs from the three-region EU policy area. CE scope EU+ refers to EU plus international partners (17 regions), rather than worldwide adoption.

## Price calculation

Select exactly `Price|Carbon` in `USD_2010/t CO2e`. For each group, normalize `baseline_noce` `2019` `Emissions|GHG|Industry` values in `MtCO2e/yr` to sum to one. The group price in every scenario/year is the sum of each member's price times that fixed weight. Regional industrial emissions in 2019 are identical across all ten scenarios in this source, so the common weights also reproduce the original scenario-specific Figure 3 definition.

Weighting emissions must be finite, nonnegative and complete, with a positive sum in each group. Missing prices with positive weights invalidate an index; zero-weight regions contribute zero even when their prices are missing. Neither prices nor weighting emissions are zero-filled. Negative values, including tiny source numerical residuals, are retained. Prices are weighted model indices, not sums or uniform country-wide taxes.

## Plot interpretation and exports

The five CE settings use the same colours as current report Figures 7 and 8. Solid lines show NDC; dashed lines show NDC+LTT. Square markers show no CE, open circles EU-only CE and filled diamonds EU-plus-partners CE. Each region uses a linear vertical scale including zero; different regions have different price ranges. Coincident observations can overlap and are not displaced. Straight segments connect source observations without smoothing or extrapolation.

Every individual PDF/SVG/600 dpi PNG has a CSV of its plotted values. `report_figures.pdf` combines all 9 figures in region order. `source_data.csv` collects the plotted observations, `carbon_price_weights.csv` records the fixed weighting data, and `regional_prices.csv` records every raw regional price and weighted contribution. The combined CSVs retain exact variables, units, group membership and weighting scenario/year. `region_membership.csv`, `manifest.json`, `audit.json` and `missing_inputs.csv` document geography, dimensions, source hash, scenario/style mapping and missingness. PDF fonts are embedded and SVG text remains editable. The font context stays active until the combined PDF closes; figure text bounds are checked before export.

Regenerate from the repository root:

```powershell
python -m omnia_results --config figures.carbon_prices.261008.json
```
