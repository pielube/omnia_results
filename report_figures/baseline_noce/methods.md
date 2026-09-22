# Report design and aggregation methods

Six main figures tell the report story; Figures S1 and S2 hold regional-intensity detail and full-period costs.

Blank activity values are provisionally treated as zero. This is a draft assumption, not a confirmed export convention. Prices, costs and reported intensities are never zero-filled.

Carbon-price indices use fixed 2019 industrial-GHG-emissions weights. The nine geographic groups partition the 28 model regions without overlap.

Leading producers are selected independently by sector using 2050 total production. The same top-12 cohort is shown at both endpoints. Other model regions preserve the global remainder.

Europe combines ENE, ENW, EUE, EUM and EUW. North America contains USA, CAN and MEX. Regional definitions follow whole model regions, including the territories assigned to them in the mapping.

China is the CHN model region, including Hong Kong, Macao and Taiwan. Indonesia group includes Indonesia, Philippines and Viet Nam. The mapping's ZijieRegion column crosses model boundaries and is not used for aggregation.

Global intensities are ratios of global sums. Reported producer-region energy intensities retain the source definition; Europe's regional intensities are production-weighted. These definitions do not agree in this export.

There are 14 negative regional cost entries. Main costs start in 2024; all source years, including anomalous initial costs, remain in Figure S2 and its data.

## Geographic definitions

| Carbon-price group | Model regions |
| --- | --- |
| Europe | ENE, ENW, EUE, EUM, EUW |
| China region | CHN |
| India | IND |
| North America | USA, CAN, MEX |
| Latin America & Caribbean | BRA, CHL, LAM |
| Other Asia–Pacific | ANZ, ASE, ASO, IDN, JPN, SKT |
| Russia & Central Asia | RUS, ASC |
| Middle East | MDA, MEA |
| Africa | AFE, AFN, AFW, AFZ, NIG |

The country mapping is included as a local snapshot under data/. Country rows are used to describe model-region membership, not as weights or replicated result rows. Duplicate country rows do not change any aggregate. country_membership.csv lists the exact countries and territories for both grouping systems. Labels describe model regions rather than strict political borders. For example, Central Asia group is ASC (including the Caucasus, Mongolia and Afghanistan); Other Asia–Pacific in producer comparisons is ASE; Gulf & Iran is MEA, which also includes Iraq and Yemen.

## Formulas

- Global additive quantities: sum once across the 28 source model regions.
- Producer output: cement production; primary + secondary for steel/aluminium. Europe is summed before ranking.
- Carbon-price index: sum(price in year t × industry GHG in weight year) / sum(industry GHG in weight year), within each group. Weights are fixed for all plotted years.
- Global energy intensity: sum(final energy, EJ/yr) / sum(production, Mt/yr) × 1,000 = GJ/t.
- Reported producer-group energy intensity: sum(reported regional GJ/t × current production) / sum(current production). A missing intensity with positive production invalidates the weighted mean.
- Emissions intensity: sum(MtCO2e/yr) / sum(Mt/yr) = tCO2e/t.
- Emissions share: sector GHG / all-industry GHG × 100.
- Costs: millions USD_2010 divided by 1,000 for billions or 1,000,000 for trillions. Negative values are retained.

Each figure has an accompanying CSV. Role=plotted identifies points actually drawn; Role=context records top-producer coverage, which is annotated in Figure 2. Caption text and axis labels specify when quantities or scales differ. All lines connect supplied years directly; no annual interpolation, smoothing, extrapolation or uncertainty is invented.

## Figure choices

Figures 1–2 establish global output and geography; Figure 3 presents the regional price context; Figure 4 connects energy demand to intensity; Figure 5 shows emissions and capture; Figure 6 shows costs. Figure S1 uses the same leading-producer cohort for reported energy intensities, and Figure S2 preserves the entire cost period. The main cost view deliberately excludes the anomalous first year, with that choice stated on the figure.

All figures are 183 mm wide; the tallest is 149 mm. PDF/SVG preserve editable text and vector lines; PNG uses the configured resolution. Titles, panel letters, sector colours, marker styles and typography are consistent across the report.
