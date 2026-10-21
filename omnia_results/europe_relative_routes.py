"""Material-production responses with separate primary and secondary steel."""

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

from .europe_comparison import CLIMATES, CLIMATE_LABELS, HATCH, MODELS, _clean_axes
from .report import ReportFigure


class RelativeRouteProductionFigures:
    def material_response_by_steel_route(self):
        from .europe_relative import (MODEL_COLORS, OFFSETS, POLICIES, POSITIONS, WIDTH,
                                      _model_legend, _net_labels)

        data = self.data.material_response_by_steel_route().copy()
        data["Plotted"] = True
        groups = ("Cement", "Primary steel", "Secondary steel", "Aluminium")
        title = "Circular-economy effects on European production: separate steel routes"
        year = self.config["comparison_year"]
        fig, axes = plt.subplots(2, 4, figsize=(240 / 25.4, 175 / 25.4), squeeze=False)
        fig.subplots_adjust(left=.075, right=.985, bottom=.215, top=.745,
                            wspace=.45, hspace=.63)
        fig.text(.035, .977, title, fontsize=8, weight="bold", va="top")
        fig.text(.035, .941, f"EU-only CE in {year} | change from each model's own {year} no-CE case",
                 fontsize=6.7, color="#52616A", va="top")
        _model_legend(fig, production=True)
        finite = data.Value.to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        low = min(0., float(finite.min())) if finite.size else 0.
        high = max(0., float(finite.max())) if finite.size else 0.
        span = max(high - low, 5.)
        step = 20. if span > 100. else 10.
        limits = (step * np.floor((low - .12 * span) / step),
                  step * np.ceil((high + .12 * span) / step))
        for row, climate in enumerate(CLIMATES):
            for column, group in enumerate(groups):
                ax = axes[row, column]
                _clean_axes(ax, chr(97 + 4 * row + column), f"{group} | {CLIMATE_LABELS[climate]}",
                            "Change from no CE (%)")
                ax.set_ylim(*limits)
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
                ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
                ax.tick_params(labelsize=6.1)
                ax.set_xlim(-.65, 2.)
                ax.set_xticks(POSITIONS, ["Medium CE\nEU", "High CE\nEU"], fontsize=6.3)
                ax.tick_params(axis="x", length=0, pad=5)
                ax.axhline(0, color="#53626B", linewidth=.7, zorder=2)
                panel = data.loc[data.Climate.eq(climate) & data.ProductionGroup.eq(group)]
                paired = panel.pivot(index="Model", columns="Policy", values="Value").reindex(
                    index=MODELS, columns=POLICIES).to_numpy(dtype=float)
                height_pt = ax.get_position().height * fig.get_figheight() * 72
                label_gap_pt = np.abs(paired[0] - paired[1]) * height_pt / (limits[1] - limits[0])
                # Separate endpoints already provide room for labels. Stagger only
                # close responses, where the model labels would otherwise collide.
                stagger = not (np.isfinite(label_gap_pt).all() and (label_gap_pt >= 10).all())
                for index, model in enumerate(MODELS):
                    selected = data.loc[data.Model.eq(model) & data.Climate.eq(climate)
                                        & data.ProductionGroup.eq(group)]
                    values = selected.set_index("Policy").Value.reindex(POLICIES).to_numpy(dtype=float)
                    x = POSITIONS + OFFSETS[index]
                    ax.bar(x, values, WIDTH, color=MODEL_COLORS[model],
                           edgecolor="#53626B" if index else "white", linewidth=.45,
                           hatch=HATCH if index else None, zorder=3)
                    present = np.isfinite(values)
                    ax.scatter(x[present], values[present], marker="o" if index == 0 else "D",
                               s=12, color=MODEL_COLORS[model], edgecolors="white", linewidths=.4, zorder=5)
                    _net_labels(ax, x, values, index, stagger=stagger)
                    if not present.all():
                        references = selected.set_index("Policy").ReferenceValue.reindex(POLICIES)
                        missing_reference = ("ReferenceMissingInputCount" in selected
                                             and selected.ReferenceMissingInputCount.gt(0).any())
                        if missing_reference or references.isna().any():
                            reason = "no-CE reference unavailable"
                        elif references.le(0).any():
                            reason = "no-CE reference nonpositive"
                        else:
                            reason = "CE output unavailable"
                        ax.text(.025, .045 + index * .12, f"{model}:\n{reason}",
                                transform=ax.transAxes, fontsize=5.3, color="#52616A", va="bottom",
                                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.})
        missing_ltt_secondary = ("ReferenceMissingInputCount" in data and
                                 data.loc[data.Model.eq("OMNIA-ENGAGE") & data.Climate.eq("baseline")
                                          & data.ProductionGroup.eq("Secondary steel"),
                                          "ReferenceMissingInputCount"].gt(0).any())
        notes = [
            f"Reference: each model's own {year} no-CE output in the same pathway and production category (0%).",
            "Primary and secondary steel each use their own route reference. These percentage changes are not additive.",
            ("NDC+LTT secondary steel: OMNIA no-CE inputs are missing; its percentage response is undefined."
             if missing_ltt_secondary else
             "Unavailable, nonfinite or nonpositive no-CE route references give gaps; no route percentage is fabricated."),
            "Cement and aluminium retain Figure 1's total-output definitions. GCAM aluminium route coverage remains unconfirmed.",
            "All eight panels share one percentage scale. Labels rounded to 0.1%; EU-region boundaries differ slightly.",
            ("OMNIA missing additive activity = zero (provisional); GCAM missing values remain gaps."
             if self.config["missing_activity"] == "zero" else
             "Missing inputs retained in both models; unavailable or nonpositive references give gaps."),
        ]
        for y, note in zip((.140, .117, .094, .071, .048, .025), notes):
            fig.text(.035, y, note, fontsize=5.4, color="#52616A", va="bottom")
        caption = (
            f"European production responses to medium/high EU-only CE in {year}, with NDC above NDC+LTT. "
            "Columns show cement, primary iron and steel, secondary iron and steel, and aluminium. Each category "
            "pairs OMNIA-ENGAGE solid bars on the left with GCAM hatched bars on the right. Every value is "
            "100 × (CE output − no-CE output) / no-CE output for the same model, climate pathway, geography, "
            "production category and year. Primary and secondary steel use their own route-specific no-CE "
            "production in the denominator, not total steel production. These are output changes, not changes "
            "in route shares or percentage-point contributions, and the two route percentages are not additive. "
            "OMNIA selects Production|Iron and Steel|Primary and Production|Iron and Steel|Secondary; GCAM "
            "selects Production|Iron and Steel|Steel|Primary and Production|Iron and Steel|Steel|Secondary. "
            "All production source units are Mt/yr. A source check compares GCAM's primary-plus-secondary "
            "sum with its reported steel parent for each selected scenario and year, without filling missing "
            "GCAM values with zero. Cement excludes clinker. Aluminium retains OMNIA primary-plus-secondary "
            "output and GCAM's reported aluminium output; GCAM aluminium route coverage remains unconfirmed. "
            "Cement and aluminium responses equal those in Figure 1. All eight panels share a percentage scale; "
            "signed increases and decreases are retained. Markers identify observed values, including zero; "
            "labels round to 0.1% while bar coordinates and CSV values retain full precision. Unavailable, nonfinite "
            "or nonpositive no-CE references give gaps. "
            + ("OMNIA no-CE secondary steel is blank in all three EU regions under NDC+LTT. Its configured "
               "provisional zero treatment yields a zero route reference; the percentage remains undefined, "
               "as it does when the blanks are preserved, and no OMNIA response is drawn in panel g. "
               if missing_ltt_secondary else "")
            + self._common_caption())
        return ReportFigure("fig04_material_production_by_steel_route", "4", title,
                            "How do primary and secondary steel production respond alongside cement and aluminium?",
                            caption, fig, data, 8)
