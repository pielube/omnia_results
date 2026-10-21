"""Aggregated cement, steel and aluminium responses in the European comparison."""

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

from .europe_comparison import CLIMATES, CLIMATE_LABELS, HATCH, MODELS, _clean_axes
from .report import ReportFigure


class RelativeEmissionsFigures:
    def emissions_response(self):
        from .europe_relative import (MODEL_COLORS, OFFSETS, POLICIES, POSITIONS, WIDTH,
                                      _model_legend, _net_labels)

        data = self.data.emissions_response().copy()
        data["Plotted"] = data.PanelKind.eq("three_sector") & data.Role.eq("net")
        data["ContextOnly"] = ~data.Plotted
        data["DisplayRole"] = np.where(data.Plotted, "plotted", "context")
        data.loc[data.PanelKind.ne("three_sector"), "Panel"] = ""
        title = "Circular economy effects on European material-sector emissions"
        year = self.config["comparison_year"]
        fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 105 / 25.4), squeeze=False)
        for column, ax in enumerate(axes[0]):
            ax.set_position([.105 if column == 0 else .610, .290, .365, .380])
        fig.text(.035, .970, title, fontsize=8, weight="bold", va="top")
        fig.text(.035, .900, f"EU-only CE in {year} | cement + steel + aluminium | change from own no-CE case",
                 fontsize=6.7, color="#52616A", va="top")
        _model_legend(fig, production=True)
        fig.legends[0].set_bbox_to_anchor((.035, .835))
        plotted_values = []
        for column, climate in enumerate(CLIMATES):
            ax = axes[0, column]
            _clean_axes(ax, chr(97 + column), CLIMATE_LABELS[climate],
                        "Three-sector emissions change\nfrom no CE (%)")
            ax.set_xlim(-.65, 2.0)
            ax.set_xticks(POSITIONS, ["Medium CE\nEU", "High CE\nEU"], fontsize=6.3)
            ax.tick_params(axis="x", length=0, pad=5)
            ax.axhline(0, color="#53626B", linewidth=.7, zorder=2)
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
            for index, model in enumerate(MODELS):
                selected = data.loc[(data.Model == model) & (data.Climate == climate) & data.Plotted]
                values = selected.set_index("Policy").Value.reindex(POLICIES).to_numpy(dtype=float)
                plotted_values.extend(values[np.isfinite(values)])
                x = POSITIONS + OFFSETS[index]
                ax.bar(x, values, WIDTH, color=MODEL_COLORS[model],
                       edgecolor="#53626B" if index else "white", linewidth=.45,
                       hatch=HATCH if index else None, zorder=3)
                present = np.isfinite(values)
                ax.scatter(x[present], values[present], marker="D" if index else "o", s=13,
                           color=MODEL_COLORS[model], edgecolors="white", linewidths=.4, zorder=5)
                _net_labels(ax, x, values, index, stagger=False)
        plotted_values = np.asarray([0., *plotted_values], dtype=float)
        lo, hi = float(plotted_values.min()), float(plotted_values.max())
        span = max(hi - lo, 5.)
        limits = (5 * np.floor((lo - .20 * span) / 5), 5 * np.ceil((hi + .15 * span) / 5))
        for ax in axes[0]:
            ax.set_ylim(*limits)
        notes = [
            "Combined cement, steel and aluminium: OMNIA GHG; GCAM CO2. Gas and process boundaries differ.",
            f"GCAM: cement energy + processes; steel/aluminium energy CO2. Reference: own {year} three-sector no CE.",
            ("NDC+LTT aluminium blanks in OMNIA = zero (provisional). Capture is not subtracted again."
             if self.config["missing_activity"] == "zero" else
             "Missing sector inputs remain gaps. Negative emissions are retained; capture is not subtracted again."),
        ]
        for y, note in zip((.130, .085, .040), notes):
            fig.text(.035, y, note, fontsize=5.4, color="#52616A", va="bottom")
        caption = (
            f"EU-only CE effects on combined European cement, iron-and-steel and aluminium emissions in {year}, "
            "with NDC on the left and NDC+LTT on the right. Panels a,b retain paired OMNIA-ENGAGE solid bars "
            "on the left and GCAM hatched bars on the right. Each bar is "
            "100 × (sum of three-sector CE emissions − sum of three-sector no-CE emissions) / "
            "sum of three-sector no-CE emissions, within the same model, pathway, geography and year. "
            "Sector percentages are not averaged. Other industry is excluded from numerator and denominator. "
            "OMNIA sums its three reported Emissions|GHG|Industry sector children in MtCO2e/yr. "
            "GCAM sums reported CO2 in Mt CO2/yr from "
            "Emissions|CO2|Energy|Demand|Industry|Non-Metallic Minerals and "
            "Emissions|CO2|Industrial Processes|Non-Metallic Minerals|Cement for cement, "
            "Emissions|CO2|Energy|Demand|Industry|Iron and Steel, and "
            "Emissions|CO2|Energy|Demand|Industry|Non-Ferrous Metals|Aluminum. "
            "The non-metallic-minerals energy category is assigned to cement: its final-energy parent equals "
            "the cement child throughout the supplied workbook. Overlapping cement-process and non-ferrous "
            "parents are not added. No additional steel/aluminium process-CO2 child or complete sector Kyoto/GHG "
            "inventory is supplied. This therefore compares OMNIA sector GHG responses with GCAM's reported "
            "sector CO2 responses; gas coverage, GWP conventions and process boundaries remain source-defined. "
            "Scales match across climate columns. Reported negative emissions are retained, and carbon capture "
            "is not subtracted again. "
            + (f"NDC+LTT aluminium GHG is blank in all three OMNIA EU regions for no CE and both CE cases in {year}; "
               "the configured provisional zero treatment is retained and flagged, without establishing physical "
               "zero emissions. " if self.config["missing_activity"] == "zero" else
               "Missing sector inputs are preserved; incomplete three-sector totals remain gaps. ")
            + "All-industry responses and sector contributions remain unplotted CSV context with Plotted=False. "
            "Source quantities, signed changes, reference totals, variables, units, coefficients, missing treatments "
            "and regions accompany the CSVs. Only medium/high EU-only CE enters. OMNIA sums EUE, EUM and EUW; "
            "GCAM selects EU-27 once. These boundaries differ slightly. No CE is zero by construction, which does "
            "not establish absolute-level alignment or harmonise the source inventories.")
        return ReportFigure("fig03_emissions_response", "3", title,
                            "How do combined cement, steel and aluminium emissions respond to circular economy?",
                            caption, fig, data, 2)
