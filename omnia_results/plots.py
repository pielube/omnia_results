"""Reusable axes-level plot functions and publication export settings."""

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
import numpy as np
import pandas as pd

from .metrics import Results, SECTORS


STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans", "sans-serif"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#42484C",
    "text.color": "#202629",
    "axes.labelcolor": "#202629",
    "xtick.color": "#42484C",
    "ytick.color": "#42484C",
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.fontsize": 6,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": True,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
}

# Okabe–Ito colours with redundant marker and dash encodings.
REGION_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#000000", "#56B4E9", "#9A6700"]
REGION_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
REGION_STYLES = ["-", "--", "-.", ":", "-", "--", "-."]


@dataclass
class Curve:
    label: str
    values: pd.Series
    color: str
    marker: str = "o"
    linestyle: str = "-"


@dataclass
class Panel:
    slug: str
    title: str
    scope: str
    family: str
    ylabel: str
    unit: str
    curves: list[Curve] = field(default_factory=list)
    caption: str = ""
    wide: bool = False


METRICS = {
    "energy": ("Industrial final energy", "Final energy (EJ yr⁻¹)", "EJ/yr",
               "Sector totals sum the specified carrier-level final-energy variables; nested fuel subcategories are excluded."),
    "energy_intensity": ("Energy intensity", "Energy intensity (GJ t⁻¹)", "GJ/t",
                         "Global final energy divided by global production, multiplied by 1,000 to convert EJ/Mt to GJ/t."),
    "emissions": ("Industrial greenhouse-gas emissions", "GHG emissions (Gt CO₂e yr⁻¹)", "GtCO2e/yr",
                  "Sector greenhouse-gas emissions, selecting MtCO2e/yr source rows and converting Mt to Gt."),
    "emissions_share": ("Share of industry emissions", "Share of industry GHG emissions (%)", "%",
                        "Each sector's global emissions divided by global Emissions|GHG|Industry, multiplied by 100. The three sectors are not assumed to exhaust industry."),
    "emissions_intensity": ("Emissions intensity", "GHG intensity (t CO₂e t⁻¹)", "tCO2e/t",
                            "Global sector greenhouse-gas emissions divided by global sector production. MtCO2e/Mt is numerically equal to tCO2e/t."),
    "capture": ("Industrial carbon capture", "CO₂ captured (Mt CO₂ yr⁻¹)", "MtCO2/yr",
                "Sum of reported sector carbon capture across model regions; capture is kept separate from greenhouse-gas emissions."),
    "sector_costs": ("Annualised industrial costs", "Annualised cost (billion USD₂₀₁₀ yr⁻¹)", "billion USD_2010/yr",
                     "Sector annualised costs converted from millions to billions of 2010 USD. All source years and signed values are retained."),
}


def build_panels(results: Results, config: dict) -> list[Panel]:
    panels = []
    for region in ["Global", *config["europe_regions"]]:
        for sector in SECTORS:
            curves = []
            for i, (label, values) in enumerate(results.production(sector, region).items()):
                curves.append(Curve(label, values, sector.color if i == 0 else "#535C64",
                                    ["o", "s", "^"][i], ["-", "--", ":"][i]))
            detail = ("Cement and clinker are distinct products; they are not added together."
                      if sector.key == "cement" else "Total production is primary plus secondary production.")
            panels.append(Panel(f"production_{sector.key}_{region.lower()}",
                                f"{sector.label} production", region, "Production",
                                "Production (Mt yr⁻¹)", "Mt/yr", curves, detail))
    for key, (title, ylabel, unit, caption) in METRICS.items():
        panels.append(Panel(key, title, "Global", "Global metrics", ylabel, unit,
                            [Curve(s.label, results.global_metric(key, s), s.color, s.marker, s.linestyle)
                             for s in SECTORS], caption))
    panels.append(Panel("system_cost", "Total annualised system cost", "Global", "Global metrics",
                        "Annualised cost (trillion USD₂₀₁₀ yr⁻¹)", "trillion USD_2010/yr",
                        [Curve("System total", results.series("Total Annualised Cost", "Millions USD_2010/yr",
                                                             activity=False) / 1e6, "#313A43")],
                        "Sum of Total Annualised Cost across model regions, converted to trillions of 2010 USD. Negative regional source values are retained. System and sector costs have separate plots and scales."))
    for group_index, (group, regions) in enumerate(config["regional_groups"].items(), 1):
        for sector in [*SECTORS, None]:
            if sector is None:
                values = results.values("Price|Carbon", "USD_2010/t CO2e", activity=False)
                slug, title = "carbon_price", "Carbon price"
                ylabel, unit = "Carbon price (USD₂₀₁₀ t CO₂e⁻¹)", "USD_2010/tCO2e"
                caption = "Regional carbon prices in 2010 USD per tonne CO₂-equivalent. Prices are not aggregated."
            else:
                values = results.values(f"Energy Intensity|Industry|{sector.path}", "EJ/Mt", activity=False) * 1000
                slug, title = f"regional_energy_intensity_{sector.key}", f"{sector.label}: energy intensity"
                ylabel, unit = "Energy intensity (GJ t⁻¹)", "GJ/t"
                caption = "Regional energy intensity as supplied in the CSV, converted from EJ/Mt to GJ/t. These source intensities need not match the derived energy/production definition; see the audit."
            curves = [Curve(region, values.loc[region], REGION_COLORS[i], REGION_MARKERS[i], REGION_STYLES[i])
                      for i, region in enumerate(regions)]
            panels.append(Panel(f"{slug}_{group_index}", title, group, "Regional metrics", ylabel, unit,
                                curves, caption + " Region codes retain the source naming.", wide=True))
    return panels


def draw_panel(ax, panel: Panel):
    """Draw on an existing Axes so the same chart can be composed into future figures."""
    for curve in panel.curves:
        ax.plot(curve.values.index, curve.values.to_numpy(dtype=float), label=curve.label,
                color=curve.color, linestyle=curve.linestyle, marker=curve.marker,
                linewidth=1.0, markersize=2.6, markeredgewidth=0.5,
                markerfacecolor="white", zorder=3, clip_on=False)
    years = panel.curves[0].values.index.tolist()
    ax.set_xlim(min(years) - 0.65, max(years) + 0.65)
    ax.set_xticks(years)
    ax.set_xlabel("Year", labelpad=5)
    ax.set_ylabel(panel.ylabel, labelpad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DEE3E6", linewidth=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, min_n_ticks=3))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.6g}"))
    finite = np.concatenate([c.values.to_numpy(dtype=float) for c in panel.curves])
    finite = finite[np.isfinite(finite)]
    if finite.size:
        low, high = min(0, finite.min()), max(0, finite.max())
        span = high - low or 1
        ax.set_ylim(low - (0.07 * span if low < 0 else 0), high + 0.1 * span)
        if low < 0:
            ax.axhline(0, color="#626A70", linewidth=0.6)
    else:
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.5, "Insufficient complete data", transform=ax.transAxes,
                ha="center", va="center", fontsize=7, color="#535C64")
    legend = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.045, 1, 0),
                       mode="expand", borderaxespad=0, ncol=min(4 if panel.wide else 3, len(panel.curves)),
                       handlelength=2.2, handletextpad=0.5, columnspacing=1, labelspacing=0.6)
    return legend


def export_panel(panel: Panel, directory: Path, scenario_label: str, formats: list[str], dpi: int):
    """Fixed physical dimensions, editable vector text, and a 600 dpi bitmap."""
    with plt.rc_context(STYLE):
        width_mm, height_mm = (120, 85) if panel.wide else (89, 78)
        fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
        ax = fig.add_axes([0.19 if not panel.wide else 0.15, 0.16,
                           0.76 if not panel.wide else 0.80, 0.55])
        fig.text(0.06, 0.965, panel.title, ha="left", va="top", fontsize=7, weight="bold")
        fig.text(0.06, 0.905, f"{scenario_label}  |  {panel.scope}",
                 ha="left", va="top", fontsize=6, color="#535C64")
        if any(curve.values.isna().any() for curve in panel.curves):
            fig.text(0.95, 0.025, "Gaps indicate unavailable inputs", ha="right", va="bottom",
                     fontsize=5, color="#535C64")
        draw_panel(ax, panel)
        fig.canvas.draw()
        # Check rendered text against the canvas before export (no cropping changes the size).
        renderer = fig.canvas.get_renderer()
        for artist in [*fig.texts, ax.xaxis.label, ax.yaxis.label,
                       *ax.get_xticklabels(), *ax.get_yticklabels(), ax.get_legend()]:
            bounds = artist.get_window_extent(renderer)
            if bounds.width and (bounds.x0 < -1 or bounds.y0 < -1 or
                                 bounds.x1 > fig.bbox.width + 1 or bounds.y1 > fig.bbox.height + 1):
                plt.close(fig)
                raise ValueError(f"Text or legend exceeds figure canvas: {panel.slug}")
        for fmt in formats:
            metadata = {"Title": panel.title, "Description": panel.caption} if fmt == "svg" else None
            fig.savefig(directory / f"{panel.slug}.{fmt}", dpi=dpi, metadata=metadata)
        plt.close(fig)
        return {"width_mm": width_mm, "height_mm": height_mm,
                "font": Path(font_manager.findfont("DejaVu Sans")).name}


def panel_data(panel: Panel, scenario: str) -> pd.DataFrame:
    return pd.DataFrame([{"Scenario": scenario, "Scope": panel.scope, "Series": c.label,
                          "Year": int(year), "Value": value, "Unit": panel.unit}
                         for c in panel.curves for year, value in c.values.items()])
