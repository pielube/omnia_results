"""Material and final-energy comparisons across the circular-economy cases."""

from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .comparison_common import ComparisonBuilderBase, CLIMATE_LABELS
from .metrics import SECTORS
from .report import ReportFigure


# Composition colours are independent of the policy palette: every segment is
# a physical product or energy carrier, with the same colour in every scenario.
PRODUCT_COLORS = {"Cement": "#0072B2", "Clinker": "#A7B3BA",
                  "Primary": "#0072B2", "Secondary": "#72C4AF"}
FUEL_COLORS = {"Electricity": "#0072B2", "Gases": "#E69F00",
               "Hydrogen": "#009E73", "Liquids": "#CC79A7",
               "Solids": "#66717A"}


def _change(value: float, reference: float) -> float:
    """Percentage changes require a strictly positive baseline quantity."""
    return (value / reference - 1) * 100 if reference > 0 else float("nan")


def _change_label(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if abs(value) < .05:
        return "0%"
    # A Unicode minus preserves consistent typography with axis ticks.
    return f"{value:+.1f}%".replace("-", "\N{MINUS SIGN}")


class MaterialsEnergyFigures(ComparisonBuilderBase):
    """Draw the first two figures of the scenario-comparison report."""

    def materials(self) -> ReportFigure:
        title = "Circular economy changes material demand and primary supply"
        fig, axes = self.canvas(title, 110, cols=3, top=.70, bottom=.235,
                                left=.085, right=.975, wspace=.46,
                                subtitle=f"Global production in {self.year}; all ten scenarios")
        positions = np.arange(len(self.data.policies), dtype=float)
        ndc_marker = dict(marker="D", s=13, facecolors="white",
                          edgecolors="#202D35", linewidths=.65, zorder=4)
        for column, sector in enumerate(SECTORS):
            ax, panel = axes[0, column], chr(97 + column)
            self.clean_axes(ax, time=False)
            # Comparisons use vertical bars, so the reference grid is horizontal.
            ax.grid(False)
            ax.grid(axis="y", color="#DDE3E6", linewidth=.35)
            self.panel_title(ax, panel, sector.label)
            ax.set_ylabel("Production (Mt yr$^{-1}$)", fontsize=6.5)
            self.policy_ticks(ax)
            ax.set_xlim(-.58, len(positions) - .42)

            products = {
                climate: [self.data.global_production(self.data.scenario(climate, policy), sector)
                          for policy in self.data.policies]
                for climate in self.data.climates
            }
            for climate in self.data.climates:
                for policy, values in zip(self.data.policies, products[climate]):
                    scenario = self.data.scenario(climate, policy)
                    for component, series in values.items():
                        role = ("plotted" if sector.key == "cement" or
                                (climate == "baseline" and component != "Total") or
                                (climate == "ndc" and component == "Total") else "context")
                        self.record(panel, scenario, "Material production", component,
                                    float(series.loc[self.year]), "Mt/yr", role=role)

            if sector.key == "cement":
                width = .30
                tops = []
                for component, offset in (("Cement", -.17), ("Clinker", .17)):
                    baseline = np.array([float(values[component].loc[self.year])
                                         for values in products["baseline"]])
                    ndc = np.array([float(values[component].loc[self.year])
                                   for values in products["ndc"]])
                    ax.bar(positions + offset, baseline, width,
                           color=PRODUCT_COLORS[component], edgecolor="white", linewidth=.45)
                    ax.scatter(positions + offset, ndc, **ndc_marker)
                    tops.append(np.maximum(baseline, ndc))
                high = float(np.max(tops))
                ax.set_ylim(0, high * 1.23)
                # Cement and clinker are intermediate/final products, never added.
                ax.text(.02, .98, "Cement and clinker shown separately", transform=ax.transAxes,
                        fontsize=5.5, color="#52616A", va="top")
            else:
                primary = np.array([float(values["Primary"].loc[self.year])
                                    for values in products["baseline"]])
                secondary = np.array([float(values["Secondary"].loc[self.year])
                                      for values in products["baseline"]])
                ndc_totals = np.array([float(values["Total"].loc[self.year])
                                       for values in products["ndc"]])
                ax.bar(positions, primary, .62, color=PRODUCT_COLORS["Primary"],
                       edgecolor="white", linewidth=.45)
                ax.bar(positions, secondary, .62, bottom=primary,
                       color=PRODUCT_COLORS["Secondary"], edgecolor="white", linewidth=.45)
                ax.scatter(positions, ndc_totals, **ndc_marker)
                high = float(np.max(np.maximum(primary + secondary, ndc_totals)))
                ax.set_ylim(0, high * 1.23)
                for x, policy, p, s in zip(positions, self.data.policies, primary, secondary):
                    total = p + s
                    share = 100 * s / total if total > 0 else float("nan")
                    if np.isfinite(share) and s / high > .13:
                        ax.text(x, p + s / 2, f"{share:.0f}%", fontsize=5.4,
                                color="#173F39", ha="center", va="center")
                    self.record(panel, self.data.scenario("baseline", policy),
                                "Secondary production share", "Secondary", share, "%",
                                role="plotted" if np.isfinite(share) and s / high > .13 else "context")

            # The labels describe total metal or cement production, not clinker.
            total_key = "Cement" if sector.key == "cement" else "Total"
            baseline_reference = float(products["baseline"][0][total_key].loc[self.year])
            for x, policy, baseline, ndc in zip(positions, self.data.policies,
                                               products["baseline"], products["ndc"]):
                value = float(baseline[total_key].loc[self.year])
                delta = _change(value, baseline_reference)
                marker_top = max(value, float(ndc[total_key].loc[self.year]))
                ax.text(x - (.17 if sector.key == "cement" else 0), marker_top + high * .036,
                        "ref." if policy == "noce" else _change_label(delta),
                        ha="center", va="bottom", fontsize=5.6, color="#202D35")
                self.record(panel, self.data.scenario("baseline", policy),
                            "Production change from no CE", total_key, delta, "%",
                            reference=self.data.scenario("baseline", "noce"))

        handles = [Patch(facecolor=PRODUCT_COLORS["Cement"], label="Cement / primary metal"),
                   Patch(facecolor=PRODUCT_COLORS["Clinker"], label="Clinker"),
                   Patch(facecolor=PRODUCT_COLORS["Secondary"], label="Secondary metal"),
                   Line2D([], [], color="#202D35", marker="D", markerfacecolor="white",
                          markeredgewidth=.65, markersize=3.5, linestyle="none", label="NDC output")]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.035, .845),
                   ncol=4, fontsize=5.8, frameon=False, columnspacing=1.1,
                   handlelength=1.2, handletextpad=.45)
        fig.text(.035, .170, "Bars: Baseline. Diamonds: NDC. Labels above bars: Baseline change from no CE.",
                 fontsize=5.6, color="#52616A")
        fig.text(.035, .130, "Labels within metal stacks: share of secondary production.",
                 fontsize=5.6, color="#52616A")
        return self.finish(
            fig, "1", "fig01_material_production", title,
            "How does circular economy change material production and primary supply?",
            f"Global production in {self.year} across five circular-economy settings. "
            "Bars show Baseline climate-policy results; open diamonds show NDC results at the same "
            "positions, retaining both pathways when production differences are small. "
            "Cement and clinker are distinct products and are plotted side by side, never summed. "
            "Steel and aluminium totals are primary plus secondary output. Labels within metal "
            "stacks give the Baseline secondary-production share; labels above bars give the "
            "Baseline change in cement or total metal production relative to Baseline no CE. "
            "Each panel has its own vertical scale. EU+ denotes CE adoption in the EU and the "
            "international partner regions, rather than global adoption.", 3)

    def energy_mix(self) -> ReportFigure:
        title = "Circular economy reduces final energy demand and changes the fuel mix"
        fig, axes = self.canvas(title, 150, rows=2, cols=3, top=.75, bottom=.155,
                                left=.085, right=.975, wspace=.46, hspace=.62,
                                subtitle=f"Global final energy in {self.year}; Baseline and NDC")
        positions = np.arange(len(self.data.policies), dtype=float)
        fuel_order = tuple(FUEL_COLORS)
        # Keep the same sector scale across climate pathways so row comparisons
        # are direct, while allowing different scales among sectors.
        quantities = {}
        maxima = {}
        for sector in SECTORS:
            maximum = 0.0
            for climate in self.data.climates:
                for policy in self.data.policies:
                    scenario = self.data.scenario(climate, policy)
                    energy = self.data.energy_by_carrier(scenario, sector)
                    quantities[(climate, policy, sector.key)] = energy
                    maximum = max(maximum, float(energy.loc[:, self.year].sum(min_count=len(energy))))
            maxima[sector.key] = maximum

        for row, climate in enumerate(self.data.climates):
            for column, sector in enumerate(SECTORS):
                ax, panel = axes[row, column], chr(97 + 3 * row + column)
                self.clean_axes(ax, time=False)
                ax.grid(False)
                ax.grid(axis="y", color="#DDE3E6", linewidth=.35)
                self.panel_title(ax, panel, f"{sector.label} | {CLIMATE_LABELS[climate]}")
                self.policy_ticks(ax)
                ax.set_xlim(-.58, len(positions) - .42)
                ax.set_ylim(0, maxima[sector.key] * 1.24)
                ax.set_ylabel("Final energy (EJ yr$^{-1}$)", fontsize=6.5)
                bottom = np.zeros(len(positions))
                for fuel in fuel_order:
                    # Cement has no Hydrogen parent in this source. Skip it;
                    # do not infer a missing carrier from another sector.
                    if fuel not in sector.fuels:
                        continue
                    values = np.array([float(quantities[(climate, policy, sector.key)].loc[fuel, self.year])
                                       for policy in self.data.policies])
                    ax.bar(positions, values, .65, bottom=bottom, color=FUEL_COLORS[fuel],
                           edgecolor="white", linewidth=.35)
                    bottom += values
                    for policy, value in zip(self.data.policies, values):
                        self.record(panel, self.data.scenario(climate, policy),
                                    "Final energy", fuel, value, "EJ/yr")
                reference = bottom[0]
                for x, policy, value in zip(positions, self.data.policies, bottom):
                    scenario = self.data.scenario(climate, policy)
                    change = _change(float(value), float(reference))
                    ax.text(x, value + maxima[sector.key] * .035,
                            "ref." if policy == "noce" else _change_label(change),
                            ha="center", va="bottom", fontsize=5.8, color="#202D35")
                    self.record(panel, scenario, "Final energy", "Total", float(value),
                                "EJ/yr", role="context")
                    self.record(panel, scenario, "Final energy change from no CE", "Total",
                                change, "%", reference=self.data.scenario(climate, "noce"))

        fig.legend(handles=[Patch(facecolor=FUEL_COLORS[fuel], label=fuel) for fuel in fuel_order],
                   loc="upper left", bbox_to_anchor=(.035, .879), ncol=5, frameon=False,
                   fontsize=6, columnspacing=1.7, handlelength=1.3, handletextpad=.5)
        fig.text(.035, .097, "Labels: change from no CE under the same climate pathway. Sector scales are identical across rows.",
                 fontsize=5.5, color="#52616A")
        return self.finish(
            fig, "2", "fig02_energy_mix", title,
            "How does circular economy change industrial final energy demand and its fuel mix?",
            f"Global final energy demand in {self.year} for cement, iron and steel, and aluminium. "
            "Rows distinguish Baseline and NDC climate-policy cases; bars distinguish the five CE "
            "settings. Segment colours identify exact parent energy-carrier categories. Nested "
            "fuel subcategories are excluded to prevent double counting. Cement has four reported "
            "carrier categories; steel and aluminium also include hydrogen. Labels give the "
            "percentage change in each sector's total final energy relative to no CE under the "
            "same climate pathway. The same y-axis scale is used within each sector across both "
            "rows, while sectors have different scales. EU+ means EU plus international partners.", 6)
