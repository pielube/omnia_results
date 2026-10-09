"""Climate-matched CE responses inside and outside the adoption area."""

import json
from numbers import Integral
from pathlib import Path

import pandas as pd

from .comparison_data import ComparisonData, percent_change
from .metrics import SECTORS


SCOPES = ("Inside policy area", "Outside policy area")


class PolicyCoverageData:
    """Sum raw regional activity using each CE scenario's policy geography.

    Material activity measures production, not consumption. Each complement
    is calculated independently, so missing data in one area do not invalidate
    the other. Existing Results objects retain exact selections and their ledger.
    """

    def __init__(self, results, config, base):
        from .consolidation_data import CE_POLICIES, MATERIAL_CLIMATES

        self.results, self.config = results, config
        self.climates, self.policies = MATERIAL_CLIMATES, CE_POLICIES[1:]
        self.regions = list(config["regions"])
        self.year = config.get("comparison_year", 2050)
        if (isinstance(self.year, bool) or not isinstance(self.year, Integral)
                or self.year not in config["years"]):
            raise ValueError("Policy coverage comparison year must be an integer in configured years")
        required = {f"{climate}_{policy}" for climate in self.climates for policy in CE_POLICIES}
        if not required <= results.keys():
            raise ValueError("Policy coverage requires all ten CE scenarios")
        if not config.get("policy_regions_csv"):
            raise ValueError("Policy coverage requires a policy_regions_csv membership table")
        self.policy_source = (Path(base) / config["policy_regions_csv"]).resolve()
        self.policy_membership = self._read_membership()
        self.groups = {}
        for area, flag in (("EU", "EU"), ("EU + partners", "EU_plus_partners")):
            inside = self.policy_membership[flag]
            self.groups[area] = {
                SCOPES[0]: self.policy_membership.loc[inside, "Region"].tolist(),
                SCOPES[1]: self.policy_membership.loc[~inside, "Region"].tolist(),
            }
            if any(not members for members in self.groups[area].values()):
                raise ValueError("Policy coverage requires nonempty inside and outside groups for each area")
        self._regional = {}

    def _read_membership(self):
        table = pd.read_csv(self.policy_source)
        if missing := {"Region", "EU", "EU_plus_partners"} - set(table.columns):
            raise ValueError(f"Missing policy geography columns: {sorted(missing)}")
        if (table.Region.isna().any() or table.Region.duplicated().any()
                or set(table.Region) != set(self.regions)):
            raise ValueError("Policy geography must partition configured source regions exactly once")
        table = table.set_index("Region").loc[self.regions].reset_index()
        for column in ("EU", "EU_plus_partners"):
            table[column] = table[column].map(ComparisonData._flag)
        if (table.EU & ~table.EU_plus_partners).any():
            raise ValueError("EU must be a subset of EU_plus_partners in policy geography")
        if "Description" not in table:
            table["Description"] = table.Region
        return table[["Region", "Description", "EU", "EU_plus_partners"]].copy()

    def membership(self):
        descriptions = self.policy_membership.set_index("Region").Description.to_dict()
        return pd.DataFrame([
            {"PolicyArea": area, "Scope": scope, "Region": region,
             "Description": descriptions[region], "InsidePolicyArea": scope == SCOPES[0]}
            for area, groups in self.groups.items()
            for scope, members in groups.items() for region in members
        ])

    @staticmethod
    def _variables(kind, sector):
        if kind == "material":
            parent = f"Production|{sector.path}"
            return ([parent] if sector.key == "cement"
                    else [parent + "|Primary", parent + "|Secondary"])
        if kind == "energy":
            return [f"Final Energy|Industry|{sector.path}|{fuel}" for fuel in sector.fuels]
        raise ValueError("Policy coverage kind must be material or energy")

    def _quantity(self, kind, scenario, sector):
        key = (kind, scenario, sector.key)
        if key not in self._regional:
            unit = "Mt/yr" if kind == "material" else "EJ/yr"
            selections = [self.results[scenario].values(variable, unit)
                          for variable in self._variables(kind, sector)]
            # Ordinary addition requires every route/carrier within each region.
            self._regional[key] = sum(selections)
        return self._regional[key]

    def figure_data(self, kind):
        from .consolidation_data import climate_label, policy_label

        if kind not in {"material", "energy"}:
            raise ValueError("Policy coverage kind must be material or energy")
        label = "Material production" if kind == "material" else "Final energy"
        unit = "Mt/yr" if kind == "material" else "EJ/yr"
        rows = []
        for row, climate in enumerate(self.climates):
            reference_scenario = f"{climate}_noce"
            for column, sector in enumerate(SECTORS):
                variables = self._variables(kind, sector)
                reference = self._quantity(kind, reference_scenario, sector)[self.year]
                for policy in self.policies:
                    scenario = f"{climate}_{policy}"
                    area = "EU" if policy.endswith("_eu") else "EU + partners"
                    quantity = self._quantity(kind, scenario, sector)[self.year]
                    for scope, members in self.groups[area].items():
                        current = float(quantity.loc[members].sum(min_count=len(members)))
                        noce = float(reference.loc[members].sum(min_count=len(members)))
                        change = current - noce
                        common = {
                            "Scenario": scenario, "ClimatePathway": climate_label(scenario),
                            "CE": policy_label(scenario), "Policy": policy,
                            "Panel": chr(97 + 3 * row + column), "Scope": scope,
                            "PolicyArea": area, "Sector": sector.label, "Series": scope,
                            "Year": self.year, "ScenarioValue": current, "ReferenceValue": noce,
                            "ReferenceScenario": reference_scenario, "ReferenceYear": self.year,
                            "SourceRegions": json.dumps(members), "RegionCount": len(members),
                            "SourceVariables": json.dumps(variables),
                            "SourceCoefficients": json.dumps([1] * len(variables)),
                            "Numerator": change, "NumeratorUnit": unit, "Offset": 0,
                        }
                        rows.append({**common, "Metric": label + " percentage change",
                                     "Value": percent_change(current, noce), "Unit": "%",
                                     "Role": "plotted", "Denominator": noce,
                                     "DenominatorUnit": unit, "ConversionFactor": 100})
                        rows.append({**common, "Metric": label + " absolute change",
                                     "Value": change, "Unit": unit, "Role": "context",
                                     "Denominator": float("nan"), "DenominatorUnit": "",
                                     "ConversionFactor": 1})
        return pd.DataFrame(rows)
