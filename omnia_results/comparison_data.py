"""Matched climate-policy comparisons with an explicit, disjoint geography.

All additive quantities follow the configured missing-activity policy. Costs,
prices and the common carbon-price weighting data always retain missingness.
The industry residual is accounting arithmetic, not an identified mechanism.
"""

from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import Results, SECTORS, Sector


def percent_change(value, reference):
    """Percent change only where both values are finite and reference is positive."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = 100 * np.divide(value - reference, reference)
    valid = np.isfinite(value) & np.isfinite(reference) & (reference > 0)
    if isinstance(result, pd.Series):
        return result.where(valid)
    return float(result) if valid else np.nan


class ComparisonData:
    climates = ("baseline", "ndc")
    climate_labels = {"baseline": "Baseline", "ndc": "NDC"}
    policies = ("noce", "medce_eu", "highce_eu", "medce_gbl", "highce_gbl")
    policy_labels = {"noce": "No CE", "medce_eu": "Medium CE, EU",
                     "highce_eu": "High CE, EU", "medce_gbl": "Medium CE, EU + partners",
                     "highce_gbl": "High CE, EU + partners"}
    policy_short = {"noce": "No CE", "medce_eu": "Medium\nEU", "highce_eu": "High\nEU",
                    "medce_gbl": "Medium\nEU + partners", "highce_gbl": "High\nEU + partners"}

    def __init__(self, config: dict, base: Path):
        self.config = config
        self.years = list(config["years"])
        self.regions = list(config["regions"])
        self.comparison_year = config.get("comparison_year", 2050)
        self.price_weight_year = config.get("price_weight_year", 2019)
        for field in ("comparison_year", "price_weight_year"):
            year = getattr(self, field)
            if isinstance(year, bool) or not isinstance(year, Integral) or year not in self.years:
                raise ValueError(f"{field} must be an integer in configured years")
        expected = [self.scenario(climate, policy)
                    for climate in self.climates for policy in self.policies]
        selected = config.get("scenarios", expected)
        if not isinstance(selected, (list, tuple)) or len(selected) != len(expected) or set(selected) != set(expected):
            raise ValueError("Comparison scenarios must select all ten climate/CE combinations exactly once")
        source = Path(base) / config["input"]
        self.results = {scenario: Results(source, scenario, self.regions, self.years,
                                          config.get("missing_activity", "preserve"))
                        for scenario in expected}
        if set(self.results[expected[0]].available_scenarios) != set(expected):
            raise ValueError("Comparison source must contain exactly the ten expected climate/CE scenarios")
        self._membership = self._read_membership(Path(base) / config["policy_regions_csv"])
        self.groups = {group: self._membership.loc[self._membership.Group == group, "Region"].tolist()
                       for group in ("EU", "Partners", "Rest")}
        if any(not members for members in self.groups.values()):
            raise ValueError("Policy geography must contain nonempty EU, Partners and Rest groups")
        self.price_weight_scenario = config.get("price_weight_scenario", "baseline_noce")
        if self.price_weight_scenario not in self.results:
            raise ValueError("price_weight_scenario must be a selected comparison scenario")
        self.price_weights = self._price_weights()

    @staticmethod
    def _flag(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0"}:
            return value.strip().lower() in {"true", "1"}
        raise ValueError(f"Policy geography flag must be boolean, received {value!r}")

    def _read_membership(self, path: Path) -> pd.DataFrame:
        table = pd.read_csv(path)
        if missing := {"Region", "EU", "EU_plus_partners"} - set(table.columns):
            raise ValueError(f"Missing policy geography columns: {sorted(missing)}")
        if table.Region.isna().any() or table.Region.duplicated().any() or set(table.Region) != set(self.regions):
            raise ValueError("Policy geography must partition configured source regions exactly once")
        table = table.set_index("Region").loc[self.regions].reset_index()
        eu = table.EU.map(self._flag)
        participating = table.EU_plus_partners.map(self._flag)
        if (eu & ~participating).any():
            raise ValueError("EU must be a subset of EU_plus_partners in policy geography")
        table["Group"] = np.select([eu, participating], ["EU", "Partners"], default="Rest")
        columns = ["Group", "Region"] + (["Description"] if "Description" in table else [])
        return table[columns].copy()

    def scenario(self, climate: str, policy: str) -> str:
        if climate not in self.climates or policy not in self.policies:
            raise ValueError(f"Unknown comparison climate/policy: {climate!r}/{policy!r}")
        return f"{climate}_{policy}"

    def reference(self, scenario: str) -> str:
        if scenario not in self.results:
            raise ValueError(f"Unknown comparison scenario: {scenario!r}")
        return self.scenario(scenario.split("_", 1)[0], "noce")

    def global_production(self, scenario: str, sector: Sector) -> dict[str, pd.Series]:
        return self.results[scenario].production(sector)

    def energy_by_carrier(self, scenario: str, sector: Sector) -> pd.DataFrame:
        results = self.results[scenario]
        output = pd.DataFrame({fuel: results.series(f"Final Energy|Industry|{sector.path}|{fuel}", "EJ/yr")
                               for fuel in sector.fuels}).T
        output.index.name, output.columns.name = "Carrier", "Year"
        return output

    def industry_emissions(self, scenario: str) -> pd.Series:
        return self.results[scenario].series("Emissions|GHG|Industry", "MtCO2e/yr")

    def emissions_components(self, scenario: str) -> pd.DataFrame:
        components = pd.DataFrame({sector.label: self.results[scenario].emissions(sector)
                                   for sector in SECTORS}).T
        components.loc["Other industry"] = (self.industry_emissions(scenario)
                                             - components.sum(axis=0, min_count=len(SECTORS)))
        components.index.name, components.columns.name = "Component", "Year"
        return components

    def target_emissions(self, scenario: str) -> pd.Series:
        return self.emissions_components(scenario).loc[[sector.label for sector in SECTORS]].sum(
            axis=0, min_count=len(SECTORS))

    def emissions_delta(self, scenario: str) -> pd.DataFrame:
        return self.emissions_components(scenario) - self.emissions_components(self.reference(scenario))

    def regional_emissions_delta(self, scenario: str) -> pd.DataFrame:
        def grouped(scenario_id):
            results = self.results[scenario_id]
            targeted = sum(results.values(f"Emissions|GHG|Industry|{sector.path}", "MtCO2e/yr")
                           for sector in SECTORS)
            return pd.DataFrame({group: targeted.loc[members].sum(axis=0, min_count=len(members))
                                 for group, members in self.groups.items()}).T
        output = grouped(scenario) - grouped(self.reference(scenario))
        output.index.name, output.columns.name = "Group", "Year"
        return output

    def system_cost(self, scenario: str) -> pd.Series:
        return self.results[scenario].series("Total Annualised Cost", "Millions USD_2010/yr",
                                             activity=False) / 1000

    def _price_weights(self) -> pd.DataFrame:
        base = self.results[self.price_weight_scenario].values("Emissions|GHG|Industry", "MtCO2e/yr",
                                                               activity=False)[self.price_weight_year]
        if not np.isfinite(base.to_numpy(dtype=float)).all() or (base < 0).any():
            raise ValueError("Common carbon-price base emissions must be finite and nonnegative; "
                             "missing emissions are not zero-filled")
        rows = []
        for group, members in self.groups.items():
            total = base.loc[members].sum()
            if not np.isfinite(total) or total <= 0:
                raise ValueError(f"Common carbon-price base emissions must have a positive finite sum in {group!r}")
            rows.extend({"Group": group, "Region": region, "WeightScenario": self.price_weight_scenario,
                         "WeightYear": self.price_weight_year, "Weight": base.loc[region] / total,
                         "BaseEmissions_MtCO2e": base.loc[region]} for region in members)
        return pd.DataFrame(rows)

    def carbon_price(self, scenario: str, group: str = "EU") -> pd.Series:
        if group not in self.groups:
            raise ValueError(f"Unknown policy geography group: {group!r}")
        weights = self.price_weights.loc[self.price_weights.Group == group].set_index("Region").Weight
        positive = weights.loc[weights > 0]
        prices = self.results[scenario].values("Price|Carbon", "USD_2010/t CO2e", activity=False)
        return prices.loc[positive.index].mul(positive, axis=0).sum(axis=0, min_count=len(positive))

    def membership(self) -> pd.DataFrame:
        return self._membership.copy()
