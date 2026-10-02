# Cross-scenario comparison methods

Five CE settings are compared separately within Baseline and NDC. Each CE change uses no CE in the same climate pathway as reference.

EU contains EUE, EUM and EUW. Partners contain 14 further regions, including ENE and ENW. Rest contains the other 11. The three groups partition all 28 source regions.

The _gbl scenario suffix denotes CE applied in EU + international partners (17 model regions), according to the supplied table; it does not denote adoption in every model region.

Baseline and NDC are neutral climate-pathway labels. The export has lower 2050 industrial GHG emissions under Baseline; no ordering of ambition is assumed.

Global production values are nearly identical between climate pathways for matched CE settings. Figure 1 shows Baseline composition bars and NDC output diamonds; both branches are retained in source data.

Other industry is the exact all-industry GHG total minus the three material sectors. Its change is an accounting residual; physical mechanisms require model documentation.

Regional emissions changes use absolute units because several sector baselines are near zero or negative. Signed values are retained.

The EU carbon-price index uses common baseline_noce 2019 industrial-GHG weights in every scenario. Prices with a positive weight must be available; weights never use zero-filled missing emissions.

Cost comparisons use 2050 annualised modelled system costs, in constant 2010 USD. They do not use the anomalous 2019 cost values.

Missing additive activity is provisionally interpreted as zero. This export convention remains unconfirmed; every affected source input is recorded.

Supplied model years are connected directly. No uncertainty bands, annual interpolation, smoothing or extrapolation are added.

## Calculations and source data

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
