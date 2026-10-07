"""Raw emissions calculations and a fixed partner cohort for Figure 7."""

import json
from pathlib import Path

import pandas as pd

from .comparison_data import ComparisonData
from .metrics import SECTORS


PARTNER_LABELS = {"IND": "India", "CHN": "China", "IDN": "Indonesia group",
                  "USA": "United States", "JPN": "Japan"}
UNIT = "MtCO2e/yr"


class EmissionsResponseData:
    """Reuse the existing raw Results objects without loading unrelated metrics."""

    def __init__(self, results, config, base):
        from .consolidation_data import CE_POLICIES, MATERIAL_CLIMATES

        self.results, self.config = results, config
        self.climates, self.policies = MATERIAL_CLIMATES, CE_POLICIES
        self.year = int(config.get("comparison_year", 2050))
        self.regions = list(config["regions"])
        required = {f"{climate}_{policy}" for climate in self.climates for policy in self.policies}
        if not required <= results.keys():
            raise ValueError("Emissions response requires all ten CE scenarios")
        if self.year not in config["years"]:
            raise ValueError("Emissions response comparison year must be configured")
        if not config.get("policy_regions_csv"):
            raise ValueError("Emissions response requires a policy_regions_csv membership table")
        self.policy_source = (Path(base) / config["policy_regions_csv"]).resolve()
        self.policy_membership = self._read_membership()
        self.groups = {group: self.policy_membership.loc[self.policy_membership.PolicyGroup == group,
                                                        "Region"].tolist()
                       for group in ("EU", "Partners", "Rest")}
        if any(not regions for regions in self.groups.values()):
            raise ValueError("Policy geography requires nonempty EU, Partners and Rest groups")
        if config.get("partner_top_n", 3) != 3 or len(self.groups["Partners"]) <= 3:
            raise ValueError("Figure 7 requires three main partners and a nonempty partner remainder")
        self.sector_variables = [f"Emissions|GHG|Industry|{sector.path}" for sector in SECTORS]
        self.industry_variable = "Emissions|GHG|Industry"
        self.components, self.industry, self.targets, self.regional_targets = {}, {}, {}, {}
        for scenario in results:
            if scenario not in required:
                continue
            source = results[scenario]
            regional = {sector.label: source.values(variable, UNIT)
                        for sector, variable in zip(SECTORS, self.sector_variables)}
            components = pd.DataFrame({label: source.aggregate(values)
                                       for label, values in regional.items()}).T
            target = components.sum(axis=0, min_count=len(SECTORS))
            industry = source.series(self.industry_variable, UNIT)
            components.loc["Other industry"] = industry - target
            self.components[scenario], self.industry[scenario] = components, industry
            self.targets[scenario], self.regional_targets[scenario] = target, sum(regional.values())
        self.ranking_scenarios = [f"{climate}_{policy}"
                                 for climate in self.climates for policy in self.policies[1:]]
        changes = pd.DataFrame({scenario: (self.regional_targets[scenario][self.year] -
                                          self.regional_targets[self.reference(scenario)][self.year])
                                for scenario in self.ranking_scenarios}).loc[self.groups["Partners"]]
        scores = changes.abs().sum(axis=1, min_count=len(self.ranking_scenarios)) / len(self.ranking_scenarios)
        ranking = pd.DataFrame({"Region": scores.index, "MeanAbsoluteChange_MtCO2e_yr": scores.values})
        ranking = ranking.sort_values(["MeanAbsoluteChange_MtCO2e_yr", "Region"],
                                      ascending=[False, True], na_position="last").reset_index(drop=True)
        if ranking.MeanAbsoluteChange_MtCO2e_yr.notna().sum() < 3:
            raise ValueError("Partner ranking needs three regions with complete changes across all eight CE cases")
        descriptions = self.policy_membership.set_index("Region").Description.to_dict()
        ranking["Label"] = ranking.Region.map(lambda region: PARTNER_LABELS.get(region, descriptions[region]))
        ranking["Rank"] = range(1, len(ranking) + 1)
        ranking["Selected"] = ranking.Rank <= 3
        ranking["RankingYear"] = self.year
        ranking["RankingScenarios"] = json.dumps(self.ranking_scenarios)
        ranking["SourceVariables"] = json.dumps(self.sector_variables)
        ranking["SourceCoefficients"] = json.dumps([1, 1, 1])
        ranking["Unit"] = UNIT
        ranking["ChangesByScenario"] = ranking.Region.map(lambda region: json.dumps({
            scenario: None if pd.isna(changes.loc[region, scenario]) else float(changes.loc[region, scenario])
            for scenario in self.ranking_scenarios}))
        self.rankings = ranking
        chosen = ranking.loc[ranking.Selected]
        if (chosen.Label.duplicated().any() or
                set(chosen.Label) & {"EU", "Other partners", "Rest of world", "Global net"}):
            raise ValueError("Selected partner labels must be distinct from geographic group labels")
        self.geographic_groups = {"EU": self.groups["EU"],
                                  **{item.Label: [item.Region] for item in chosen.itertuples()},
                                  "Other partners": [region for region in self.groups["Partners"]
                                                     if region not in set(chosen.Region)],
                                  "Rest of world": self.groups["Rest"]}

    def _read_membership(self):
        table = pd.read_csv(self.policy_source)
        if missing := {"Region", "EU", "EU_plus_partners"} - set(table.columns):
            raise ValueError(f"Missing policy geography columns: {sorted(missing)}")
        if table.Region.isna().any() or table.Region.duplicated().any() or set(table.Region) != set(self.regions):
            raise ValueError("Policy geography must partition configured source regions exactly once")
        table = table.set_index("Region").loc[self.regions].reset_index()
        eu, participating = table.EU.map(ComparisonData._flag), table.EU_plus_partners.map(ComparisonData._flag)
        if (eu & ~participating).any():
            raise ValueError("EU must be a subset of EU_plus_partners in policy geography")
        table["PolicyGroup"] = ["EU" if is_eu else "Partners" if is_partner else "Rest"
                                for is_eu, is_partner in zip(eu, participating)]
        if "Description" not in table:
            table["Description"] = table.Region
        return table[["Region", "Description", "PolicyGroup"]].copy()

    @staticmethod
    def reference(scenario):
        return scenario.split("_", 1)[0] + "_noce"

    def membership(self):
        membership = self.policy_membership.copy()
        mapping = {region: group for group, members in self.geographic_groups.items() for region in members}
        membership["Group"] = membership.Region.map(mapping)
        return membership

    def figure_data(self):
        from .consolidation_data import climate_label, policy_label

        rows = []

        def record(scenario, panel, kind, series, value, variables, coefficients, regions, *,
                   year, role="plotted", reference_value=None, scope="Global", sector="Three sectors"):
            reference = self.reference(scenario) if reference_value is not None else ""
            rows.append({"Scenario": scenario, "ClimatePathway": climate_label(scenario),
                         "CE": policy_label(scenario), "Policy": scenario.split("_", 1)[1],
                         "Panel": panel, "PanelKind": kind, "Scope": scope, "Sector": sector,
                         "Series": series, "Year": year,
                         "Metric": "GHG emissions change" if reference else "GHG emissions",
                         "Value": value if reference_value is None else value - reference_value,
                         "Unit": UNIT, "Role": role, "SourceVariables": json.dumps(variables),
                         "SourceCoefficients": json.dumps(coefficients), "SourceRegions": json.dumps(regions),
                         "ScenarioValue": value,
                         "ReferenceValue": float("nan") if reference_value is None else reference_value,
                         "ReferenceScenario": reference,
                         "ReferenceYear": float("nan") if reference_value is None else year})

        for column, climate in enumerate(self.climates):
            for policy in self.policies:
                scenario = f"{climate}_{policy}"
                for year, value in self.targets[scenario].items():
                    record(scenario, chr(97 + column), "trajectory", "Three sectors", float(value),
                           self.sector_variables, [1, 1, 1], self.regions, year=int(year))
            for policy in self.policies[1:]:
                scenario, reference = f"{climate}_{policy}", f"{climate}_noce"
                for label in self.components[scenario].index:
                    is_residual = label == "Other industry"
                    variables = ([self.industry_variable, *self.sector_variables] if is_residual else
                                 [self.sector_variables[[sector.label for sector in SECTORS].index(label)]])
                    record(scenario, chr(99 + column), "industry", label,
                           float(self.components[scenario].loc[label, self.year]), variables,
                           [1, -1, -1, -1] if is_residual else [1], self.regions, year=self.year,
                           reference_value=float(self.components[reference].loc[label, self.year]), sector=label)
                record(scenario, chr(99 + column), "industry", "Total industry",
                       float(self.industry[scenario].loc[self.year]), [self.industry_variable], [1],
                       self.regions, year=self.year, role="net", sector="Total industry",
                       reference_value=float(self.industry[reference].loc[self.year]))
                for group, members in self.geographic_groups.items():
                    current = self.regional_targets[scenario].loc[members, self.year].sum(min_count=len(members))
                    base = self.regional_targets[reference].loc[members, self.year].sum(min_count=len(members))
                    record(scenario, chr(101 + column), "geography", group, float(current),
                           self.sector_variables, [1, 1, 1], members, year=self.year, scope=group,
                           reference_value=float(base))
                record(scenario, chr(101 + column), "geography", "Global net",
                       float(self.targets[scenario].loc[self.year]), self.sector_variables, [1, 1, 1],
                       self.regions, year=self.year, role="net",
                       reference_value=float(self.targets[reference].loc[self.year]))
        return pd.DataFrame(rows)
