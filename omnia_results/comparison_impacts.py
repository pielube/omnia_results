"""Emissions, policy geography and economic comparisons across CE scenarios."""

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import ticker
import numpy as np

from .comparison_common import (
    CLIMATE_LABELS,
    POLICY_COLORS,
    POLICY_MARKERS,
    POLICY_SHORT,
    POLICY_STYLES,
    ComparisonBuilderBase,
)
from .comparison_data import percent_change


EMISSION_COLORS = {"Cement": "#0072B2", "Iron and steel": "#D55E00",
                   "Aluminium": "#009E73", "Other industry": "#ADB8BE"}
GEOGRAPHY_COLORS = {"EU": "#0072B2", "Partners": "#E69F00", "Rest": "#ADB8BE"}
EMISSION_UNIT = "MtCO2e/yr"


def _signed_stacks(ax, positions, components, colors, *, width=.59):
    """Accumulate increases and decreases from zero independently."""
    positive = np.zeros(len(positions))
    negative = np.zeros(len(positions))
    for label, values in components.items():
        values = np.asarray(values, dtype=float)
        bottom = np.where(values >= 0, positive, negative)
        ax.bar(positions, values, bottom=bottom, width=width, color=colors[label],
               edgecolor="white", linewidth=.3, label=label, zorder=3)
        positive += np.where(np.isfinite(values) & (values >= 0), values, 0)
        negative += np.where(np.isfinite(values) & (values < 0), values, 0)
    ax.axhline(0, color="#56636B", linewidth=.65, zorder=2)
    return negative, positive


def _net_markers(ax, positions, values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    magnitude = max(float(np.max(np.abs(finite))) if finite.size else 0., 1.)
    ax.scatter(positions, values, marker="D", s=16, color="#1D262C",
               edgecolors="white", linewidths=.45, zorder=5)
    for position, value in zip(positions, values):
        if np.isfinite(value):
            above = value < -.65 * magnitude
            ax.annotate(f"{value:+,.0f}".replace("-", "\N{MINUS SIGN}"),
                        (position, value), xytext=(0, 7 if above else -12), textcoords="offset points",
                        ha="center", va="bottom" if above else "top", fontsize=5.6, color="#1D262C",
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": .35},
                        zorder=6)


def _stack_limits(bounds):
    low = min(0., *(float(np.min(a)) for a, _ in bounds))
    high = max(0., *(float(np.max(b)) for _, b in bounds))
    span = max(high - low, 1.)
    return low - .14 * span, high + .13 * span


class ImpactFigures(ComparisonBuilderBase):
    def emissions_response(self):
        title = "CE emissions savings and the wider industrial response"
        fig, axes = self.canvas(title, 158, rows=2, cols=2, left=.095, right=.975,
                                top=.78, bottom=.22, wspace=.37, hspace=.69,
                                subtitle="Global greenhouse-gas emissions \N{MIDDLE DOT} trajectories and matched 2050 changes")
        self.scenario_legend(fig, bbox_to_anchor=(.09, .865), columnspacing=1.3)
        trajectory_values = []
        bounds = []
        ce_policies = self.data.policies[1:]
        for column, climate in enumerate(self.data.climates):
            ax, panel = axes[0, column], chr(97 + column)
            self.clean_axes(ax, time=True, years=self.data.years)
            self.panel_title(ax, panel, f"{CLIMATE_LABELS[climate]} \N{MIDDLE DOT} three sectors")
            for policy in self.data.policies:
                scenario = self.data.scenario(climate, policy)
                values = self.data.target_emissions(scenario)
                trajectory_values.extend(values.to_numpy(dtype=float))
                ax.plot(values.index, values.to_numpy(dtype=float), color=POLICY_COLORS[policy],
                        linestyle=POLICY_STYLES[policy], marker=POLICY_MARKERS[policy],
                        markerfacecolor="white" if policy.endswith("_eu") else POLICY_COLORS[policy],
                        markeredgewidth=.65, markersize=2.6, linewidth=1.0,
                        label=POLICY_SHORT[policy], zorder=4 if policy.endswith("gbl") else 3)
                self.record_series(panel, scenario, "GHG emissions", "Three sectors", values,
                                   EMISSION_UNIT)
            ax.set_ylabel("GHG emissions\n(MtCO$_2$e yr$^{-1}$)", fontsize=6.3, labelpad=3)

            ax, panel = axes[1, column], chr(99 + column)
            self.clean_axes(ax)
            self.panel_title(ax, panel, f"{CLIMATE_LABELS[climate]} \N{MIDDLE DOT} industrial change, {self.year}")
            scenarios = [self.data.scenario(climate, policy) for policy in ce_policies]
            components = {label: [] for label in EMISSION_COLORS}
            net_values = []
            for scenario in scenarios:
                reference = self.data.reference(scenario)
                changes = self.data.emissions_delta(scenario)[self.year]
                absolute = self.data.emissions_components(scenario)[self.year]
                base = self.data.emissions_components(reference)[self.year]
                for label in components:
                    components[label].append(changes.loc[label])
                    self.record(panel, scenario, "GHG emissions change", label, changes.loc[label],
                                EMISSION_UNIT, reference=reference)
                    self.record(panel, scenario, "GHG emissions", label, absolute.loc[label],
                                EMISSION_UNIT, role="context", reference=reference)
                    self.record(panel, reference, "GHG emissions", label, base.loc[label],
                                EMISSION_UNIT, role="reference")
                net = (self.data.industry_emissions(scenario).loc[self.year]
                       - self.data.industry_emissions(reference).loc[self.year])
                net_values.append(net)
                self.record(panel, scenario, "GHG emissions change", "Total industry", net,
                            EMISSION_UNIT, reference=reference)
            positions = np.arange(len(ce_policies))
            bounds.append(_signed_stacks(ax, positions, components, EMISSION_COLORS))
            _net_markers(ax, positions, net_values)
            ax.set_xticks(positions, [POLICY_SHORT[p] for p in ce_policies], fontsize=5.8)
            ax.tick_params(axis="x", length=0, pad=4)
            ax.set_xlim(-.6, len(ce_policies) - .4)
            ax.set_ylabel("Change vs no CE\n(MtCO$_2$e yr$^{-1}$)", fontsize=6.3, labelpad=3)

        finite = np.asarray(trajectory_values)[np.isfinite(trajectory_values)]
        if finite.size:
            high = max(float(finite.max()), 1.)
            for ax in axes[0]:
                ax.set_ylim(min(0., float(finite.min())) - .02 * high, high * 1.08)
        for ax in axes[1]:
            ax.set_ylim(*_stack_limits(bounds))
        handles = [Patch(facecolor=color, edgecolor="none", label=label)
                   for label, color in EMISSION_COLORS.items()]
        handles.append(Line2D([], [], color="#1D262C", marker="D", linestyle="none",
                              markersize=3.5, label="Total industry"))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .105),
                   ncol=5, fontsize=5.8, frameon=False, handlelength=1.3,
                   columnspacing=1.3, handletextpad=.45)
        caption = (
            "Global greenhouse-gas emissions in cement, iron and steel, and aluminium under "
            "(a) Baseline and (b) NDC. Markers show the model years. (c,d) Signed 2050 changes "
            "relative to no CE under the same climate pathway, decomposed into the three sectors "
            "and other industry. Increases and decreases stack separately from zero. Black "
            "diamonds and signed labels give the net change in total industrial emissions, "
            "including other industry. Other industry is calculated as reported total industrial "
            "GHG emissions minus the three sectors; its response is an accounting residual, "
            "not evidence for an identified physical rebound mechanism. Panel pairs use common "
            "vertical scales. EU+ denotes EU plus international partners, rather than worldwide "
            "CE adoption. Missing additive activity is provisionally treated as zero."
        )
        return self.finish(fig, 3, "fig03_emissions_response", title,
                           "How do savings in the three sectors translate into total industrial emissions?",
                           caption, 4)

    def geography(self):
        title = "Where CE emissions benefits occur, and what partners add"
        fig, axes = self.canvas(title, 113, cols=2, left=.10, right=.975,
                                top=.77, bottom=.265, wspace=.38,
                                subtitle=f"Three sectors \N{MIDDLE DOT} signed regional changes vs matching no CE \N{MIDDLE DOT} {self.year}")
        ce_policies = self.data.policies[1:]
        positions = np.arange(len(ce_policies))
        bounds = []
        for column, climate in enumerate(self.data.climates):
            ax, panel = axes[0, column], chr(97 + column)
            self.clean_axes(ax)
            self.panel_title(ax, panel, CLIMATE_LABELS[climate])
            components = {group: [] for group in GEOGRAPHY_COLORS}
            net_values = []
            for policy in ce_policies:
                scenario = self.data.scenario(climate, policy)
                reference = self.data.reference(scenario)
                changes = self.data.regional_emissions_delta(scenario)[self.year]
                for group in components:
                    components[group].append(changes.loc[group])
                    self.record(panel, scenario, "GHG emissions change", "Three sectors",
                                changes.loc[group], EMISSION_UNIT, scope=group, reference=reference)
                absolute = self.data.target_emissions(scenario).loc[self.year]
                baseline = self.data.target_emissions(reference).loc[self.year]
                net = absolute - baseline
                net_values.append(net)
                self.record(panel, scenario, "GHG emissions change", "Three sectors", net,
                            EMISSION_UNIT, reference=reference)
                self.record(panel, scenario, "GHG emissions", "Three sectors", absolute,
                            EMISSION_UNIT, role="context", reference=reference)
                self.record(panel, reference, "GHG emissions", "Three sectors", baseline,
                            EMISSION_UNIT, role="reference")
            bounds.append(_signed_stacks(ax, positions, components, GEOGRAPHY_COLORS))
            _net_markers(ax, positions, net_values)
            ax.set_xticks(positions, [POLICY_SHORT[p] for p in ce_policies], fontsize=6)
            ax.tick_params(axis="x", length=0, pad=4)
            ax.set_xlim(-.6, len(ce_policies) - .4)
            ax.set_ylabel("GHG emissions change\n(MtCO$_2$e yr$^{-1}$)", fontsize=6.5, labelpad=3)
        for ax in axes.flat:
            ax.set_ylim(*_stack_limits(bounds))
        handles = [Patch(facecolor=color, edgecolor="none",
                         label="Rest of world" if group == "Rest" else group)
                   for group, color in GEOGRAPHY_COLORS.items()]
        handles.append(Line2D([], [], color="#1D262C", marker="D", linestyle="none",
                              markersize=3.5, label="Global net"))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .105),
                   ncol=4, fontsize=6, frameon=False, handlelength=1.5, columnspacing=1.7)
        caption = (
            "Regional contributions to the 2050 change in cement, iron and steel, and aluminium "
            "GHG emissions relative to no CE within (a) Baseline and (b) NDC. Negative values "
            "indicate lower emissions; positive values indicate higher emissions. Increases and "
            "decreases stack separately from zero. Black diamonds and signed labels show the "
            "global net change. EU comprises EUE, EUM and EUW. Partners comprise the 14 additional "
            "model regions designated in the supplied policy table, including ENE and ENW. The "
            "remaining 11 regions form Rest of world. These groups are disjoint and exhaust all "
            "28 model regions. EU+ denotes CE implementation in the EU plus partners (17 regions), "
            "not worldwide implementation. Absolute changes avoid misleading percentages where "
            "regional reference emissions are near zero or negative. Both panels use the same "
            "vertical scale. Missing additive activity is provisionally treated as zero."
        )
        return self.finish(fig, 4, "fig04_geographic_effects", title,
                           "Where do emissions benefits occur, and what does international cooperation add?",
                           caption, 2)

    def economics(self):
        title = "CE, annualised system costs and EU carbon-price requirements"
        fig, axes = self.canvas(title, 111, cols=3, left=.087, right=.975,
                                top=.755, bottom=.30, wspace=.68,
                                subtitle=f"Changes relative to climate-matched no CE \N{MIDDLE DOT} {self.year}")
        all_points = []
        for column, climate in enumerate(self.data.climates):
            ax, panel = axes[0, column], chr(97 + column)
            self.clean_axes(ax)
            self.panel_title(ax, panel, CLIMATE_LABELS[climate])
            reference = self.data.scenario(climate, "noce")
            base_emissions = self.data.target_emissions(reference).loc[self.year]
            base_cost = self.data.system_cost(reference).loc[self.year]
            for policy in self.data.policies:
                scenario = self.data.scenario(climate, policy)
                emissions = self.data.target_emissions(scenario).loc[self.year]
                cost = self.data.system_cost(scenario).loc[self.year]
                x, y = percent_change(emissions, base_emissions), percent_change(cost, base_cost)
                all_points.append((x, y))
                self.record(panel, scenario, "GHG emissions change", "Three sectors", x,
                            "%", reference=reference)
                self.record(panel, scenario, "Annualised system-cost change", "Total system", y,
                            "%", reference=reference)
                self.record(panel, scenario, "GHG emissions", "Three sectors", emissions,
                            EMISSION_UNIT, role="context", reference=reference)
                self.record(panel, scenario, "Annualised system cost", "Total system", cost,
                            "billion USD_2010/yr", role="context", reference=reference)
                if np.isfinite(x) and np.isfinite(y):
                    ax.scatter(x, y, marker=POLICY_MARKERS[policy], s=24,
                               facecolors="white" if policy.endswith("_eu") else POLICY_COLORS[policy],
                               edgecolors=POLICY_COLORS[policy], linewidths=.9, zorder=5)
                    offsets = {"noce": (-5, 17), "medce_eu": (-34, 5), "highce_eu": (5, -14),
                               "medce_gbl": (7, 5), "highce_gbl": (7, -12)}
                    offset = offsets[policy]
                    ax.annotate(POLICY_SHORT[policy], (x, y), xytext=offset,
                                textcoords="offset points", fontsize=5.8,
                                ha="left" if offset[0] > 0 else "right", va="center",
                                color="#263238", annotation_clip=False,
                                arrowprops={"arrowstyle": "-", "color": "#9AA6AC", "lw": .45,
                                            "shrinkA": 2, "shrinkB": 4})
            ax.axhline(0, color="#839097", linewidth=.55, zorder=2)
            ax.axvline(0, color="#839097", linewidth=.55, zorder=2)
            ax.set_xlabel("Three-sector GHG change (%)", fontsize=6.3, labelpad=5)
            ax.set_ylabel("Annualised system-cost\nchange (%)", fontsize=6.3, labelpad=3)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(3, min_n_ticks=3))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4, min_n_ticks=3))
        finite_points = np.asarray(all_points, dtype=float)
        if np.isfinite(finite_points).all(axis=1).any():
            finite_points = finite_points[np.isfinite(finite_points).all(axis=1)]
            xmin, ymin = np.minimum(finite_points.min(axis=0), 0.)
            xmax, ymax = np.maximum(finite_points.max(axis=0), 0.)
            xspan, yspan = max(xmax - xmin, 1.), max(ymax - ymin, 1.)
            for ax in axes[0, :2]:
                ax.set_xlim(xmin - .19 * xspan, xmax + .13 * xspan)
                ax.set_ylim(ymin - .23 * yspan, ymax + .20 * yspan)

        ax, panel = axes[0, 2], "c"
        self.clean_axes(ax)
        self.panel_title(ax, panel, "EU carbon-price index")
        values_by_climate = {}
        for climate in self.data.climates:
            values = []
            for policy in self.data.policies:
                scenario = self.data.scenario(climate, policy)
                price = self.data.carbon_price(scenario, "EU").loc[self.year]
                values.append(price)
                self.record(panel, scenario, "Emissions-weighted carbon price", "Carbon-price index",
                            price, "USD_2010/t CO2e", scope="EU")
            values_by_climate[climate] = np.asarray(values, dtype=float)
        positions = np.arange(len(self.data.policies))
        for i in positions:
            a, b = (values_by_climate[climate][i] for climate in self.data.climates)
            if np.isfinite(a) and np.isfinite(b):
                ax.plot([a, b], [i, i], color="#AAB5BB", linewidth=.9, zorder=2)
        for climate, marker, color in (("baseline", "o", "#263F52"), ("ndc", "s", "#7D567E")):
            ax.scatter(values_by_climate[climate], positions, marker=marker, s=18,
                       facecolors="white" if climate == "baseline" else color,
                       edgecolors=color, linewidths=.8, zorder=4, label=CLIMATE_LABELS[climate])
        ax.set_yticks(positions, [POLICY_SHORT[p] for p in self.data.policies], fontsize=5.8)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.set_ylim(len(positions) - .5, -.5)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color="#DFE4E6", linewidth=.35)
        ax.set_xlabel("USD$_{2010}$ tCO$_2$e$^{-1}$", fontsize=6.3, labelpad=5)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(3, min_n_ticks=3))
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.31), ncol=2, fontsize=5.8,
                  frameon=False, handlelength=.8, columnspacing=1.1, handletextpad=.5)
        caption = (
            "(a,b) 2050 percent changes in global GHG emissions from cement, iron and steel, "
            "and aluminium (horizontal axis), and in reported global annualised total system "
            "costs (vertical axis), relative to no CE under the same climate pathway. The grey "
            "square at the origin is no CE. Colour distinguishes medium and high CE; open circles "
            "denote EU-only implementation and filled diamonds denote EU plus partners. Annualised "
            "system costs are sums of the source Total Annualised Cost series in constant 2010 "
            "USD; they are neither cumulative expenditures nor a measure of welfare or investment. "
            "Percent changes require positive reference values. Both scatter panels use identical "
            "axis limits. (c) EU carbon-price index in constant 2010 USD per tonne CO2e, calculated "
            "from regional carbon prices with fixed weights proportional to 2019 industrial GHG "
            "emissions in baseline_noce. The same weights apply to all ten scenarios; this is a "
            "weighted price, not a ratio normalised to no CE. Lines connect the two model climate "
            "pathways for each CE setting. EU consists of EUE, EUM and EUW. Source costs and prices "
            "retain missingness; missing additive activity is provisionally treated as zero."
        )
        return self.finish(fig, 5, "fig05_costs_and_carbon_prices", title,
                           "How does CE affect annualised system costs and EU carbon-price requirements?",
                           caption, 3)
