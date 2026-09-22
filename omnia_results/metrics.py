"""Unit-aware calculations; no plotting or implicit scenario aggregation."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Sector:
    key: str
    label: str
    path: str
    color: str
    marker: str
    linestyle: str
    fuels: tuple[str, ...]


SECTORS = (
    Sector("cement", "Cement", "Non-Metallic Minerals|Cement", "#0072B2", "o", "-",
           ("Electricity", "Gases", "Liquids", "Solids")),
    Sector("steel", "Iron and steel", "Iron and Steel", "#D55E00", "s", "--",
           ("Electricity", "Gases", "Hydrogen", "Liquids", "Solids")),
    Sector("aluminium", "Aluminium", "Non-Ferrous Metals|Aluminum", "#009E73", "^", "-.",
           ("Electricity", "Gases", "Hydrogen", "Liquids", "Solids")),
)


def ratio(numerator: pd.Series, denominator: pd.Series, factor: float = 1) -> pd.Series:
    """A non-positive or unavailable denominator is undefined, never zero."""
    return numerator.div(denominator.where(denominator > 0)).mul(factor)


class Results:
    def __init__(self, path: Path, scenario: str, regions: list[str], years: list[int],
                 missing_activity: str = "preserve"):
        if missing_activity not in {"zero", "preserve"}:
            raise ValueError("missing_activity must be 'zero' or 'preserve'")
        self.path, self.scenario = path, scenario
        self.regions, self.years = regions, years
        self.missing_activity = missing_activity
        if len(set(regions)) != len(regions) or len(set(years)) != len(years):
            raise ValueError("Configured regions and years must be unique")
        if years != sorted(years) or not years:
            raise ValueError("Years must be nonempty and increasing")
        raw = pd.read_csv(path)
        keys = ["Scenario", "Region", "Variable", "Unit"]
        required = keys + [str(y) for y in years]
        if missing := set(required) - set(raw.columns):
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        if raw[keys].isna().any().any():
            raise ValueError("Blank scenario, region, variable or unit")
        self.available_scenarios = sorted(raw.Scenario.unique())
        raw = raw.loc[raw.Scenario == scenario, required].copy()
        if raw.empty:
            raise ValueError(f"Unknown scenario {scenario!r}; available: {self.available_scenarios}")
        if raw.duplicated(keys).any():
            raise ValueError("Duplicate scenario/region/variable/unit rows")
        if set(raw.Region) != set(regions):
            raise ValueError("Configured regions must match the scenario's source regions exactly; "
                             "check for overlapping aggregate regions before changing this list")
        for year in years:
            raw[str(year)] = pd.to_numeric(raw[str(year)], errors="raise")
        numeric = raw[[str(y) for y in years]].to_numpy(dtype=float)
        if np.isinf(numeric).any():
            raise ValueError("Infinite values in input")
        self.raw = raw
        self._cache: dict[tuple[str, str, bool], pd.DataFrame] = {}
        self.coverage: list[dict] = []

    def values(self, variable: str, unit: str, activity: bool = True) -> pd.DataFrame:
        """Return region × year values, matching the full variable AND unit.

        Only additive activity quantities may be filled under the explicit zero
        policy. Prices, costs and reported intensities always retain missingness.
        An entirely absent variable/unit is an error under either policy.
        """
        key = (variable, unit, activity)
        if key in self._cache:
            return self._cache[key].copy()
        selected = self.raw[(self.raw.Variable == variable) & (self.raw.Unit == unit)]
        if selected.empty:
            raise ValueError(f"Required series is absent: {variable} [{unit}]")
        data = selected.set_index("Region")[[str(y) for y in self.years]].reindex(self.regions)
        data.columns = self.years
        data.columns.name = "Year"
        policy = "zero" if activity and self.missing_activity == "zero" else "preserve"
        absent = set(self.regions) - set(selected.Region)
        for region in self.regions:
            for year in self.years:
                if pd.isna(data.loc[region, year]):
                    self.coverage.append({"Scenario": self.scenario, "Region": region,
                                          "Variable": variable, "Unit": unit, "Year": year,
                                          "Issue": "absent row" if region in absent else "blank cell",
                                          "Treatment": policy})
        if policy == "zero":
            data = data.fillna(0.0)
        self._cache[key] = data
        return data.copy()

    def aggregate(self, data: pd.DataFrame, region: str = "Global") -> pd.Series:
        if region == "Global":
            # min_count prevents a sum over a changing subset of model regions.
            return data.sum(axis=0, min_count=len(self.regions))
        return data.loc[region].copy()

    def series(self, variable: str, unit: str, region: str = "Global",
               activity: bool = True) -> pd.Series:
        return self.aggregate(self.values(variable, unit, activity), region)

    def production(self, sector: Sector, region: str = "Global") -> dict[str, pd.Series]:
        if sector.key == "cement":
            return {
                "Cement": self.series("Production|" + sector.path, "Mt/yr", region),
                "Clinker": self.series("Production|Non-Metallic Minerals|Cement Clinker", "Mt/yr", region),
            }
        primary = self.series("Production|" + sector.path + "|Primary", "Mt/yr", region)
        secondary = self.series("Production|" + sector.path + "|Secondary", "Mt/yr", region)
        return {"Total": primary + secondary, "Primary": primary, "Secondary": secondary}

    def production_total(self, sector: Sector, region: str = "Global") -> pd.Series:
        return self.production(sector, region)["Cement" if sector.key == "cement" else "Total"]

    def energy_by_region(self, sector: Sector) -> pd.DataFrame:
        # Exact carrier parents only: including |Gases|Gas etc. would double count.
        return sum(self.values(f"Final Energy|Industry|{sector.path}|{fuel}", "EJ/yr")
                   for fuel in sector.fuels)

    def energy(self, sector: Sector) -> pd.Series:
        return self.aggregate(self.energy_by_region(sector))

    def emissions(self, sector: Sector) -> pd.Series:
        return self.series(f"Emissions|GHG|Industry|{sector.path}", "MtCO2e/yr")

    def global_metric(self, kind: str, sector: Sector) -> pd.Series:
        match kind:
            case "energy":
                return self.energy(sector)
            case "energy_intensity":
                # EJ/Mt × 1,000 = GJ/t. Ratio of sums, not mean of intensities.
                return ratio(self.energy(sector), self.production_total(sector), 1000)
            case "emissions":
                return self.emissions(sector) / 1000  # Mt to Gt
            case "emissions_share":
                return ratio(self.emissions(sector),
                             self.series("Emissions|GHG|Industry", "MtCO2e/yr"), 100)
            case "emissions_intensity":
                return ratio(self.emissions(sector), self.production_total(sector))
            case "capture":
                return self.series(f"Carbon Capture|Industry|{sector.path}", "MtCO2/yr")
            case "sector_costs":
                return self.series(f"Total Annualised Cost|Industry|{sector.path}",
                                   "Millions USD_2010/yr", activity=False) / 1000
        raise ValueError(f"Unknown metric: {kind}")

    def intensity_comparison(self) -> pd.DataFrame:
        """Audit source intensities against the document's energy/production formula."""
        rows = []
        for sector in SECTORS:
            reported = self.values(f"Energy Intensity|Industry|{sector.path}", "EJ/Mt", activity=False)
            energy = self.energy_by_region(sector)
            for region in self.regions:
                derived = ratio(energy.loc[region], self.production_total(sector, region), 1000)
                for year in self.years:
                    source_value, derived_value = reported.loc[region, year] * 1000, derived.loc[year]
                    rows.append({"Scenario": self.scenario, "Region": region,
                                 "Sector": sector.label, "Year": year,
                                 "Reported_GJ_per_t": source_value, "Derived_GJ_per_t": derived_value,
                                 "Difference_GJ_per_t": source_value - derived_value})
        return pd.DataFrame(rows)
