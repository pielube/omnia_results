"""Production and final-energy changes within and outside CE policy areas."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Patch
import numpy as np

from .consolidation_data import CE_POLICIES, MATERIAL_CLIMATES, climate_label
from .metrics import SECTORS
from .report import ReportFigure


COVERAGE_AREAS = ("Inside policy area", "Outside policy area")
COVERAGE_COLORS = {"Inside policy area": "#0072B2",
                   "Outside policy area": "#D55E00"}
COMPARISON_POLICIES = tuple(policy for policy in CE_POLICIES if policy != "noce")
POLICY_TICKS = ("M\nEU", "H\nEU", "M\nEU+", "H\nEU+")


def _region_counts(data):
    """Read geographic complements from the unchanged source metadata."""
    counts = {}
    for policy_area in ("EU", "EU + partners"):
        counts[policy_area] = {}
        for scope in COVERAGE_AREAS:
            values = data.loc[(data.PolicyArea == policy_area) & (data.Scope == scope),
                              "SourceRegions"]
            counts[policy_area][scope] = len(json.loads(values.iloc[0]))
    return counts


class PolicyCoverageFigures:
    """Compare matched regional changes across four CE coverage settings."""

    def material_policy_coverage(self) -> ReportFigure:
        return self._policy_coverage_figure(
            self.data.material_policy_coverage(),
            metric="Material production percentage change",
            title="Material production inside and outside CE policy areas",
            axis_label="Production change (%)", quantity="material production",
            raw_unit="Mt/yr", number="9", slug="fig09_material_policy_coverage",
            definition="Production = cement output or primary + secondary metal output.",
            question="How does material production change within and outside the CE policy area?",
        )

    def energy_policy_coverage(self) -> ReportFigure:
        return self._policy_coverage_figure(
            self.data.energy_policy_coverage(),
            metric="Final energy percentage change",
            title="Final energy demand inside and outside CE policy areas",
            axis_label="Final energy change (%)", quantity="final energy demand",
            raw_unit="EJ/yr", number="10", slug="fig10_energy_policy_coverage",
            definition="Final energy sums exact parent carriers, excluding nested fuel subcategories.",
            question="How does final energy demand change within and outside the CE policy area?",
        )

    def _policy_coverage_figure(self, data, *, metric, title, axis_label, quantity,
                              raw_unit, number, slug, definition, question):
        future = self.config.get("comparison_year", 2050)
        plotted = data.loc[(data.Role == "plotted") & (data.Metric == metric) &
                           (data.Year == future)]
        fig, axes = plt.subplots(2, 3, figsize=(180 / 25.4, 175 / 25.4), squeeze=False)
        fig.subplots_adjust(left=.09, right=.985, bottom=.20, top=.78,
                            wspace=.51, hspace=.60)
        fig.text(.035, .977, title, fontsize=8, weight="bold", va="top")
        fig.text(.035, .941, f"Percentage change from matched no CE | {future}",
                 fontsize=6.7, color="#52616A", va="top")
        positions = np.arange(len(COMPARISON_POLICIES), dtype=float)
        width = .34

        for column, sector in enumerate(SECTORS):
            sector_data = plotted.loc[plotted.Sector == sector.label]
            finite = sector_data.Value.to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            low = min(0., float(finite.min())) if finite.size else 0.
            high = max(0., float(finite.max())) if finite.size else 0.
            span = high - low
            if span == 0:
                span = 1.
            limits = (low - .10 * span, high + .12 * span)
            for row, climate in enumerate(MATERIAL_CLIMATES):
                ax, panel = axes[row, column], chr(97 + row * 3 + column)
                selected = sector_data.loc[sector_data.Panel == panel]
                ax.spines[["top", "right"]].set_visible(False)
                ax.set_axisbelow(True)
                ax.grid(axis="y", color="#DDE3E6", linewidth=.35)
                ax.tick_params(labelsize=6.1, pad=3)
                ax.set_xlim(-.58, len(positions) - .42)
                ax.set_ylim(*limits)
                ax.yaxis.set_major_locator(ticker.MaxNLocator(4, min_n_ticks=3))
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.3g}"))
                ax.set_ylabel(axis_label, fontsize=6.8, labelpad=3)
                ax.set_xticks(positions, labels=POLICY_TICKS, fontsize=6)
                ax.tick_params(axis="x", length=0, pad=4)
                ax.set_title(f"{sector.label} | {climate_label(climate + '_noce')}",
                             loc="left", fontsize=7, pad=9)
                ax.text(-.14, 1.04, panel, transform=ax.transAxes, fontsize=8,
                        weight="bold", va="bottom", ha="left")
                ax.axhline(0, color="#64737C", linewidth=.65, zorder=2,
                           label="_zero_axis")
                for offset, scope in zip((-.18, .18), COVERAGE_AREAS):
                    values = (selected.loc[selected.Scope == scope]
                              .set_index("Policy")["Value"].reindex(COMPARISON_POLICIES)
                              .to_numpy(dtype=float))
                    ax.bar(positions + offset, values, width,
                           color=COVERAGE_COLORS[scope], edgecolor="white", linewidth=.4,
                           label=scope, zorder=3)

        handles = [Patch(facecolor=COVERAGE_COLORS[scope], label=scope)
                   for scope in COVERAGE_AREAS]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.035, .882), ncol=2,
                   fontsize=6.2, columnspacing=2.2, handlelength=1.5,
                   handletextpad=.5, frameon=False)
        counts = _region_counts(data)
        eu_inside, eu_outside = (counts["EU"][scope] for scope in COVERAGE_AREAS)
        partner_inside, partner_outside = (counts["EU + partners"][scope]
                                           for scope in COVERAGE_AREAS)
        fig.text(.035, .122,
                 "M/H = medium/high CE; EU+ = EU + international partners. Zero = matched no CE.",
                 fontsize=5.5, color="#52616A")
        fig.text(.035, .091,
                 f"EU: {eu_inside} inside / {eu_outside} outside; EU+: {partner_inside} inside / {partner_outside} outside model regions.",
                 fontsize=5.5, color="#52616A")
        fig.text(.035, .060, definition, fontsize=5.5, color="#52616A")
        assumption = ("Draft assumption: missing additive activity = zero. Undefined percentages appear as gaps."
                      if self.config.get("missing_activity", "preserve") == "zero" else
                      "Missing activity retained; undefined percentages appear as gaps.")
        fig.text(.035, .024, assumption, fontsize=5.5, color="#63717A", va="bottom")
        details = (
            "Material production is output. Cement uses cement output alone, excluding "
            "clinker; steel and aluminium sum primary and secondary production. These "
            "quantities do not measure consumption or trade. "
            if quantity == "material production" else
            "Final energy demand sums the exact carrier parents Electricity, Gases, "
            "Liquids and Solids for cement, with Hydrogen additionally included for steel "
            "and aluminium. Nested carrier subcategories are excluded to avoid double counting. "
        )
        caption = (
            f"Percentage changes in {quantity} in {future}, with NDC in the upper row and "
            "NDC+LTT in the lower row. Columns show cement, iron and steel, and aluminium. "
            "Each panel compares medium CE in the EU, high CE in the EU, medium CE in the "
            "EU plus international partners, and high CE in the EU plus partners. Blue "
            "bars show regions inside the respective policy area; orange bars show its "
            "geographic complement. The EU-only policy area contains "
            f"{eu_inside} model regions and its complement contains {eu_outside}; the EU-plus-partners "
            f"policy area contains {partner_inside} and its complement contains {partner_outside}. "
            "These groups are defined by CE adoption scope, with membership changing between "
            "EU-only and EU-plus-partners cases. Each bar gives 100 × (scenario value − no-CE "
            "value) / no-CE value, using the same climate pathway, sector, geographic group "
            f"and {future} reference year. Inside and outside percentages use their own "
            "group denominators and cannot be added. The no-CE reference is shown by the "
            "zero line. Signed changes are shown at their model-derived values. "
            + details +
            f"Values are extracted directly from {self.config['input']} using exact "
            f"parent-variable and {raw_unit} selections, summing each constituent model "
            "region once. Both the percentages plotted here and the corresponding absolute "
            f"changes in {raw_unit} are retained in the source data, together with scenario "
            "and reference values, reference scenario and year, exact source variables and "
            "regions, numerators, denominators, conversion factors and offsets. Missing "
            "values or nonpositive no-CE group totals yield undefined percentages and gaps. "
            "Vertical limits match within each sector across climate rows and include all "
            "finite changes and zero; each sector has its own scale. " + assumption
        )
        return ReportFigure(slug, number, title, question, caption, fig, data, 6)
