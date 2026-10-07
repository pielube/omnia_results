"""Three-sector trajectories and signed industrial/geographic CE responses."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from .comparison_common import POLICY_COLORS, POLICY_MARKERS, POLICY_SHORT, POLICY_STYLES
from .consolidation_data import CE_POLICIES, MATERIAL_CLIMATES, climate_label
from .report import ReportFigure


INDUSTRY_COLORS = {"Cement": "#0072B2", "Iron and steel": "#D55E00",
                   "Aluminium": "#009E73", "Other industry": "#ADB8BE"}
REGIONAL_PALETTE = ("#0072B2", "#E69F00", "#D55E00", "#009E73", "#CC79A7", "#ADB8BE")
NET_COLOR = "#1D262C"


def _signed_stacks(ax, positions, components, colors):
    """Keep positive and negative contributions on separate sides of zero."""
    positive = np.zeros(len(positions), dtype=float)
    negative = np.zeros(len(positions), dtype=float)
    positive_extent = positive.copy()
    negative_extent = negative.copy()
    for label, observations in components.items():
        values = np.asarray(observations, dtype=float)
        bottom = np.where(values >= 0, positive, negative)
        ax.bar(positions, values, .59, bottom=bottom, color=colors[label],
               edgecolor="white", linewidth=.3, zorder=3, label=label)
        present = np.isfinite(values)
        positive += np.where(present & (values >= 0), values, 0.)
        negative += np.where(present & (values < 0), values, 0.)
        # Keep the bounds of known segments even when a later component is
        # missing, so incomplete stacks are not clipped by a fallback scale.
        positive_extent = np.fmax(positive_extent, positive)
        negative_extent = np.fmin(negative_extent, negative)
        # With an unknown sign/quantity, neither cumulative baseline is known.
        positive[~present], negative[~present] = np.nan, np.nan
    ax.axhline(0, color="#56636B", linewidth=.65, zorder=2)
    return negative_extent, positive_extent


def _row_limits(bounds, nets):
    negative = np.concatenate([np.asarray(low, dtype=float) for low, _ in bounds])
    positive = np.concatenate([np.asarray(high, dtype=float) for _, high in bounds])
    net = np.concatenate([np.asarray(values, dtype=float) for values in nets])
    negative, positive, net = (values[np.isfinite(values)] for values in (negative, positive, net))
    low = min(0., float(negative.min()) if negative.size else 0., float(net.min()) if net.size else 0.)
    high = max(0., float(positive.max()) if positive.size else 0., float(net.max()) if net.size else 0.)
    span = max(high - low, 1.)
    return low - .15 * span, high + .15 * span


def _net_markers(ax, positions, observations):
    values = np.asarray(observations, dtype=float)
    present = np.isfinite(values)
    ax.scatter(np.asarray(positions)[present], values[present], marker="D", s=16,
               color=NET_COLOR, edgecolors="white", linewidths=.45, zorder=5,
               label="Net change")
    finite = values[present]
    magnitude = max(float(np.max(np.abs(finite))) if finite.size else 0., 1.)
    for position, value in zip(positions, values):
        if not np.isfinite(value):
            continue
        # Labels on the larger savings bars sit inside the plot above the marker;
        # near-zero labels sit below, leaving the zero baseline easy to follow.
        above = value < -.65 * magnitude
        ax.annotate(f"{value:+,.0f}".replace("-", "\N{MINUS SIGN}"),
                    (position, value), xytext=(0, 7 if above else -9),
                    textcoords="offset points", ha="center",
                    va="bottom" if above else "top", fontsize=5.7,
                    color=NET_COLOR, zorder=6,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": .35})


class EmissionsResponseFigures:
    """Connect CE trajectories to industrial and regional accounting changes."""

    def emissions_response(self) -> ReportFigure:
        data = self.data.emissions_response()
        future = self.config.get("comparison_year", 2050)
        years = self.config["years"]
        first, last = min(years), max(years)
        policies = tuple(CE_POLICIES)
        ce_policies = policies[1:]
        positions = np.arange(len(ce_policies), dtype=float)
        geographic_labels = (data.loc[(data.PanelKind == "geography") &
                                      (data.Role == "plotted"), "Series"]
                             .drop_duplicates().tolist())
        geography_colors = {label: REGIONAL_PALETTE[index]
                            for index, label in enumerate(geographic_labels)}
        partner_labels = [label for label in geographic_labels
                          if label not in {"EU", "Other partners", "Rest of world"}]
        title = "CE emissions savings, industrial responses and international cooperation"
        fig, axes = plt.subplots(3, 2, figsize=(180 / 25.4, 235 / 25.4), squeeze=False)
        # Two header-legend rows and a two-row geographic legend have reserved
        # bands, keeping the six plotting areas at the same physical height.
        bottoms = (.650, .375, .100)
        for row in range(3):
            for column in range(2):
                axes[row, column].set_position([.095 if column == 0 else .605,
                                                bottoms[row], .370, .175])
        fig.text(.035, .977, title, fontsize=8, weight="bold", va="top")
        fig.text(.035, .941, f"Global GHG trajectories, {first}–{last} | matched {future} changes from no CE",
                 fontsize=6.7, color="#52616A", va="top")
        ticks = sorted({first, last, *[year for year in years if year % 10 == 0]})
        trajectory_values = []
        signed_bounds = {1: [], 2: []}
        signed_nets = {1: [], 2: []}

        for column, climate in enumerate(MATERIAL_CLIMATES):
            climate_name = climate_label(climate + "_noce")
            for row in range(3):
                ax, panel = axes[row, column], chr(97 + row * 2 + column)
                ax.spines[["top", "right"]].set_visible(False)
                ax.set_axisbelow(True)
                ax.grid(axis="y", color="#DDE3E6", linewidth=.35)
                ax.tick_params(labelsize=6.1, pad=3)
                ax.yaxis.set_major_locator(ticker.MaxNLocator(4, min_n_ticks=3))
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.4g}"))
                ax.text(-.14, 1.04, panel, transform=ax.transAxes, fontsize=8,
                        weight="bold", va="bottom", ha="left")
                selected = data.loc[data.Panel == panel]
                if row == 0:
                    ax.set_title(f"{climate_name} | three-sector emissions", loc="left", fontsize=7, pad=9)
                    ax.set_ylabel("Three-sector GHG\n(Mt CO$_2$e yr$^{-1}$)", fontsize=6.5, labelpad=3)
                    ax.set_xlabel("Year", fontsize=6.5, labelpad=3)
                    ax.set_xlim(first - .6, last + .6)
                    ax.set_xticks(ticks)
                    for policy in policies:
                        scenario = f"{climate}_{policy}"
                        values = (selected.loc[(selected.Scenario == scenario) &
                                               (selected.Role == "plotted")]
                                  .set_index("Year")["Value"].reindex(years))
                        observations = values.to_numpy(dtype=float)
                        trajectory_values.extend(observations)
                        ax.plot(values.index, observations, color=POLICY_COLORS[policy],
                                linestyle=POLICY_STYLES[policy], marker=POLICY_MARKERS[policy],
                                markerfacecolor="white" if policy.endswith("_eu") else POLICY_COLORS[policy],
                                markeredgewidth=.65, markersize=2.6, linewidth=1.0,
                                label=scenario, zorder=4 if policy.endswith("gbl") else 3)
                else:
                    if row == 1:
                        component_colors = INDUSTRY_COLORS
                        labels = list(INDUSTRY_COLORS)
                        ax.set_title(f"{climate_name} | industrial change, {future}",
                                     loc="left", fontsize=7, pad=9)
                        ax.set_ylabel("Industry GHG change\n(Mt CO$_2$e yr$^{-1}$)", fontsize=6.5, labelpad=3)
                    else:
                        component_colors = geography_colors
                        labels = geographic_labels
                        ax.set_title(f"{climate_name} | regional change, {future}",
                                     loc="left", fontsize=7, pad=9)
                        ax.set_ylabel("Three-sector GHG change\n(Mt CO$_2$e yr$^{-1}$)", fontsize=6.5, labelpad=3)
                    components = {}
                    for label in labels:
                        components[label] = (selected.loc[(selected.Series == label) &
                                                          (selected.Role == "plotted")]
                                             .set_index("Policy")["Value"].reindex(ce_policies)
                                             .to_numpy(dtype=float))
                    net = (selected.loc[selected.Role == "net"].set_index("Policy")["Value"]
                           .reindex(ce_policies).to_numpy(dtype=float))
                    signed_bounds[row].append(_signed_stacks(ax, positions, components, component_colors))
                    signed_nets[row].append(net)
                    ax.set_xlim(-.6, len(ce_policies) - .4)
                    ax.set_xticks(positions, [POLICY_SHORT[policy] for policy in ce_policies], fontsize=6)
                    ax.tick_params(axis="x", length=0, pad=4)

        finite = np.asarray(trajectory_values, dtype=float)
        finite = finite[np.isfinite(finite)]
        low = min(0., float(finite.min())) if finite.size else 0.
        high = max(0., float(finite.max())) if finite.size else 1.
        span = max(high - low, 1.)
        for ax in axes[0]:
            ax.set_ylim(low - .03 * span if low < 0 else 0., high + .08 * span)
        for row in (1, 2):
            limits = _row_limits(signed_bounds[row], signed_nets[row])
            for column in range(2):
                ax = axes[row, column]
                ax.set_ylim(*limits)
                _net_markers(ax, positions, signed_nets[row][column])

        policy_handles = [Line2D([], [], color=POLICY_COLORS[policy], marker=POLICY_MARKERS[policy],
                                  linestyle=POLICY_STYLES[policy], linewidth=1, markersize=3,
                                  markerfacecolor="white" if policy.endswith("_eu") else POLICY_COLORS[policy],
                                  label=POLICY_SHORT[policy]) for policy in policies]
        industry_handles = [Patch(facecolor=color, label=label)
                            for label, color in INDUSTRY_COLORS.items()]
        industry_handles.append(Line2D([], [], color=NET_COLOR, marker="D", linestyle="none",
                                       markersize=3.5, label="Total industry net"))
        # Matplotlib fills legend columns before rows: interleave the two bands
        # so their visual row order is policies, then industrial components/net.
        header_handles = [handle for pair in zip(policy_handles, industry_handles) for handle in pair]
        fig.legend(handles=header_handles, loc="upper center", bbox_to_anchor=(.535, .905),
                   ncol=5, fontsize=6, columnspacing=1.3, handlelength=2,
                   handletextpad=.45, labelspacing=.85, frameon=False)
        geography_handles = [Patch(facecolor=geography_colors[label], label=label)
                             for label in geographic_labels]
        geography_handles.append(Line2D([], [], color=NET_COLOR, marker="D", linestyle="none",
                                        markersize=3.5, label="Global three-sector net"))
        # Preserve the geographic stack/rank order when the legend is read
        # left to right, despite Matplotlib's column-first legend layout.
        geography_handles = [geography_handles[index] for column in range(4)
                             for index in range(column, len(geography_handles), 4)]
        fig.legend(handles=geography_handles, loc="upper center", bbox_to_anchor=(.535, .354),
                   ncol=4, fontsize=5.8, columnspacing=1.4, handlelength=1.4,
                   handletextpad=.45, labelspacing=.75, frameon=False)
        fig.text(.035, .057,
                 "Diamonds: middle = all-industry net change; bottom = global three-sector net change.",
                 fontsize=5.5, color="#52616A")
        fig.text(.035, .035,
                 "M/H = medium/high CE; EU+ = EU + international partners (17 model regions).",
                 fontsize=5.5, color="#52616A")
        assumption = ("Draft assumption: missing additive activity = zero."
                      if self.config["missing_activity"] == "zero" else
                      "Missing activity retained; incomplete stacks, trajectories and net changes appear as gaps.")
        fig.text(.035, .013, assumption, fontsize=5.5, color="#63717A", va="bottom")
        partner_list = (", ".join(partner_labels[:-1]) + " and " + partner_labels[-1]
                        if len(partner_labels) > 1 else "".join(partner_labels))
        caption = (
            "CE-related global emissions trajectories and signed industrial/geographic changes "
            "with NDC in the left column and NDC+LTT in the right. (a,b) Sum of reported GHG "
            "emissions in cement, iron and steel, and aluminium for all five CE settings; "
            "markers show the supplied model years joined by straight segments. "
            f"(c,d) Changes in {future} from no CE under the same climate pathway, decomposed "
            "into the three sectors and Other industry. Other industry is reported all-industry "
            "GHG minus the three plotted sectors and is an accounting residual. Black diamonds "
            "and signed labels show the net change in all-industry emissions, including that "
            f"residual. (e,f) Changes in the three sectors' {future} GHG emissions by disjoint "
            "regional groups; diamonds and signed labels show the global three-sector net, "
            "rather than the all-industry net used in the middle row. Increases and decreases "
            "stack independently from zero in both decomposition rows. The same vertical "
            "limits are used within each row across pathways. EU comprises EUE, EUM and EUW. "
            f"The fixed leading partner groups are {partner_list}, selected once by mean "
            f"absolute {future} three-sector emissions change across the eight CE cases "
            "(four CE variants and two climate pathways), each relative to its own no-CE "
            "reference. Their ranking, names and colours are reused in both regional panels. "
            "Indonesia group represents Indonesia, Philippines and Viet Nam. Other partners "
            "combines the 11 partner model regions outside that three-group cohort; Rest of "
            "world combines the 11 model regions outside EU plus partners. EU, the three "
            "selected partner groups, Other partners and Rest of world partition all 28 model "
            "regions once. EU+ denotes CE adoption in the EU plus international partners, "
            "covering 17 model regions rather than worldwide adoption. Every change is the "
            "scenario value minus the matching no-CE value at the same year, retaining the "
            "original signs. Reported GHG is not reduced again by carbon capture. "
            f"All quantities are extracted directly from {self.config['input']} using exact "
            "GHG variable and MtCO2e/yr selections. Source data retains constituent regions, "
            "variables and coefficients, scenario/reference values, and reference scenario "
            "and year. " + assumption)
        return ReportFigure("fig07_emissions_response", "7", title,
                            "How do CE emissions savings translate into wider industrial change, and where do those savings occur?",
                            caption, fig, data, 6)
