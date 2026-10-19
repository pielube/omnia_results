"""Four-panel emissions responses with independently defined comparison rows."""

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .europe_comparison import CLIMATES, CLIMATE_LABELS, HATCH, MODELS, _clean_axes
from .europe_relative import MODEL_COLORS, OFFSETS, POLICIES, POSITIONS, WIDTH, _net_labels
from .report import ReportFigure


def _row_legend(fig, external, y):
    handles = [
        Patch(facecolor=MODEL_COLORS["OMNIA-ENGAGE"], edgecolor="#53626B",
              linewidth=.45, label="OMNIA-ENGAGE · left"),
        Patch(facecolor=MODEL_COLORS["GCAM"], edgecolor="#53626B", linewidth=.45,
              hatch=HATCH, label=f"{external} · right"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.035, y),
               ncol=2, fontsize=6.2, handlelength=1.8, handletextpad=.6,
               columnspacing=2, frameon=False)


def _draw_europe_panels(axes, frame, year):
    data = frame.loc[frame.Year.eq(year)].copy()
    data["SourcePanel"] = data.get("Panel", "")
    data["ComparisonRow"] = "European all industry"
    data["Plotted"] = data.PanelKind.eq("industry") & data.Role.eq("net")
    data["ContextOnly"] = ~data.Plotted
    data["PlotRole"] = np.where(data.Plotted, "plotted", "context")
    data["DisplayRole"] = data.PlotRole
    data["PlotValue"] = data.Value
    data["PlotUnit"] = data.Unit
    data["Panel"] = ""
    all_values = []
    for column, climate in enumerate(CLIMATES):
        ax, letter = axes[column], chr(97 + column)
        data.loc[data.Plotted & data.Climate.eq(climate), "Panel"] = letter
        _clean_axes(ax, letter, CLIMATE_LABELS[climate], "Industry GHG change\nfrom no CE (%)")
        ax.set_xlim(-.65, 2.)
        ax.set_xticks(POSITIONS, ["Medium CE\nEU", "High CE\nEU"], fontsize=6.3)
        ax.tick_params(axis="x", length=0, pad=5)
        ax.axhline(0, color="#53626B", linewidth=.7, zorder=2)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        for index, model in enumerate(MODELS):
            selected = data.loc[data.Plotted & data.Model.eq(model) & data.Climate.eq(climate)]
            values = selected.set_index("Policy").Value.reindex(POLICIES).to_numpy(dtype=float)
            all_values.extend(values[np.isfinite(values)])
            x = POSITIONS + OFFSETS[index]
            ax.bar(x, values, WIDTH, color=MODEL_COLORS[model],
                   edgecolor="#53626B" if index else "white", linewidth=.45,
                   hatch=HATCH if index else None, zorder=3)
            present = np.isfinite(values)
            ax.scatter(x[present], values[present], marker="D" if index else "o", s=13,
                       color=MODEL_COLORS[model], edgecolors="white", linewidths=.4, zorder=5)
            _net_labels(ax, x, values, index, stagger=False)
    low = min(0., *all_values) if all_values else 0.
    high = max(0., *all_values) if all_values else 0.
    span = max(high - low, 5.)
    limits = (5 * np.floor((low - .20 * span) / 5),
              5 * np.ceil((high + .15 * span) / 5))
    for ax in axes:
        ax.set_ylim(*limits)
    return data


def build_combined_emissions(europe_frame, openprom_frame, year=2050):
    """Build from the in-memory results of the two raw-source data adapters.

    Each model keeps its own reference quantities and emissions definition.
    Shared scales apply within a row; the two rows retain independent scales.
    """
    from .openprom_comparison import draw_openprom_emissions_panels

    title = "Circular economy effects on reported emissions"
    fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 200 / 25.4), squeeze=False)
    for row, bottom in enumerate((.593, .178)):
        for column, left in enumerate((.105, .610)):
            axes[row, column].set_position([left, bottom, .365, .229])
    fig.text(.035, .980, title, fontsize=8, weight="bold", va="top")
    fig.text(.035, .951, f"{year} | changes from each model's own same-year no-CE/reference case",
             fontsize=6.7, color="#52616A", va="top")
    fig.text(.035, .917, "Europe · reported all-industry GHG · EU-only CE in both models",
             fontsize=6.9, weight="bold", va="top")
    _row_legend(fig, "GCAM", .894)
    europe = _draw_europe_panels(axes[0], europe_frame, year)

    fig.text(.035, .502, "Global steel · OMNIA CE in EU+ · OPEN-PROM global",
             fontsize=6.9, weight="bold", va="top")
    _row_legend(fig, "OPEN-PROM", .479)
    selected = draw_openprom_emissions_panels(axes[1], openprom_frame, year=year,
                                             letters=("c", "d"))
    selected = selected.copy()
    selected["SourcePanel"] = selected.Panel
    selected["ComparisonRow"] = "Global steel"
    selected["Plotted"] = True
    selected["ContextOnly"] = False
    selected["PlotRole"] = "plotted"
    selected["DisplayRole"] = "plotted"
    selected["PlotUnit"] = "%"
    # Keep the separately reported reference observations once, alongside each
    # plotted scenario's own ReferenceValue and full source metadata.
    references = openprom_frame.loc[openprom_frame.Year.eq(year) &
                                    openprom_frame.Metric.eq("Emissions") &
                                    openprom_frame.Ambition.eq("Reference")].copy()
    references["SourcePanel"] = references.get("Panel", "")
    references["Panel"] = ""
    references["ComparisonRow"] = "Global steel"
    references["Role"] = "context"
    references["Plotted"] = False
    references["ContextOnly"] = True
    references["PlotRole"] = "context"
    references["DisplayRole"] = "context"
    references["PlotValue"] = references.PercentChange
    references["PlotUnit"] = "%"
    data = pd.concat([europe, selected, references], ignore_index=True, sort=False)

    notes = [
        "Top: OMNIA reported all-industry GHG; GCAM industry energy-demand + industrial-process Kyoto gases.",
        "Bottom: OMNIA reported steel GHG; OPEN-PROM steel energy-related CO₂. Gas and process boundaries differ.",
        "Global steel: OMNIA CE in EU+; OPEN-PROM intervention area unspecified. Policy equivalence is unconfirmed.",
        f"Reference: each model's own {year} case. Percentage scales match within rows; inventories differ between rows.",
        "Capture is not subtracted again. Missing inputs retain the source treatment; unavailable responses remain gaps.",
    ]
    for y, note in zip((.096, .075, .054, .033, .012), notes):
        fig.text(.035, y, note, fontsize=5.05, color="#52616A", va="bottom")
    caption = (
        f"Emissions responses in {year}, with NDC on the left and NDC+LTT on the right. "
        "Panels a,b reproduce the European all-industry GHG comparison for medium/high EU-only CE, pairing "
        "solid OMNIA-ENGAGE bars on the left with hatched GCAM bars on the right. OMNIA selects its reported "
        "all-industry parent Emissions|GHG|Industry, including sectors beyond cement, iron and steel, and "
        "aluminium. GCAM sums Emissions|Kyoto Gases|Energy|Demand|Industry and "
        "Emissions|Kyoto Gases|Industrial Processes. OMNIA sums EUE, EUM and EUW; GCAM selects EU-27 once. "
        "These regional boundaries differ slightly. Panels c,d compare global steel responses, pairing "
        "OMNIA-ENGAGE CE in the EU plus partners (EU+) with OPEN-PROM medium/high scenarios. The global "
        "output scope is distinct from the policy adoption area. OPEN-PROM's intervention definitions and "
        "adoption area are undocumented, so medium/high labels do not establish policy equivalence. "
        "OMNIA sums all 28 model regions and selects reported steel GHG; OPEN-PROM selects World and "
        "steel energy-related CO2. OPEN-PROM has no steel-specific process-emissions or GHG total, and "
        "its energy-related CO2 is not harmonised to OMNIA's reported steel GHG. All bars show "
        "100 × (scenario emissions − own reference emissions) / own reference emissions, using the "
        "same model, climate pathway, geography, emissions definition and year for numerator and denominator. "
        "References are each model's own no-CE/reference case; no absolute-level alignment is assumed. "
        "The rows have different geography, sector scope, policy coverage and emissions inventories. "
        "Percentage scales match within each row and are independent between rows. Gas coverage, GWP "
        "conventions and process boundaries remain source-defined. Reported negative emissions and "
        "signed increases are retained; carbon capture is not subtracted again. Endpoint markers identify "
        "observed values, including zero, and signed labels are rounded to 0.1%. Missing responses remain "
        "gaps. Original source quantities, references, variables, regions, units and missing treatments "
        "are retained in the source data. European decompositions and focused three-sector calculations "
        "remain unplotted context, identified by PlotRole=context; only industry net rows are plotted in "
        "the upper row. The lower row excludes OMNIA's EU-only CE cases."
    )
    return ReportFigure("fig05_emissions_comparison", "5", title,
                        "How do reported emissions respond within each model and comparison scope?",
                        caption, fig, data, 4)
