"""Direct source extraction and current scenario labels for report refinement."""

import json
from pathlib import Path

import pandas as pd

from .metrics import Results, SECTORS, ratio
from .comparison_data import percent_change
from .report_data import ReportData


CLIMATE_LABELS = {"baseline": "NDC+LTT", "ndc": "NDC"}
POLICY_LABELS = {"noce": "No CE", "medce_eu": "Medium CE, EU",
                 "highce_eu": "High CE, EU", "medce_gbl": "Medium CE, EU + partners",
                 "highce_gbl": "High CE, EU + partners"}
PRODUCTION_SCENARIOS = ("ndc_noce", "baseline_noce")
MATERIAL_CLIMATES = ("ndc", "baseline")
CE_POLICIES = tuple(POLICY_LABELS)
ENERGY_CARRIERS = ("Electricity", "Gases", "Hydrogen", "Liquids", "Solids")
POLICY_SHORT = {"noce": "No CE", "medce_eu": "M–EU", "highce_eu": "H–EU",
                "medce_gbl": "M–EU+", "highce_gbl": "H–EU+"}


def climate_label(scenario: str) -> str:
    """Interpret the source prefix without renaming or changing source rows."""
    climate = scenario.split("_", 1)[0]
    if climate not in CLIMATE_LABELS:
        raise ValueError(f"Unknown climate pathway in {scenario!r}")
    return CLIMATE_LABELS[climate]


def policy_label(scenario: str) -> str:
    policy = scenario.split("_", 1)[1] if "_" in scenario else ""
    if policy not in POLICY_LABELS:
        raise ValueError(f"Unknown CE setting in {scenario!r}")
    return POLICY_LABELS[policy]


class ConsolidationData:
    """Reuse unit-aware raw-results calculations for all consolidated figures.

    This class never reads previously exported figure data or artwork. Results
    validates source keys and coverage and caches exact variable/unit selections.
    """

    def __init__(self, config: dict, base: Path):
        self.config = config
        self.base = Path(base)
        self.source = (base / config["input"]).resolve()
        scenarios = config["scenarios"]
        if not scenarios or len(set(scenarios)) != len(scenarios):
            raise ValueError("Consolidation scenarios must be nonempty and unique")
        for scenario in scenarios:
            climate_label(scenario)
            policy_label(scenario)
        self.results = {
            scenario: Results(self.source, scenario, config["regions"], config["years"],
                              config["missing_activity"])
            for scenario in scenarios
        }
        self._producer_reports = None
        self._emissions_response = None
        self._policy_coverage = None

    def _policy_coverage_data(self):
        if self._policy_coverage is None:
            from .consolidation_coverage_data import PolicyCoverageData
            self._policy_coverage = PolicyCoverageData(self.results, self.config, self.base)
        return self._policy_coverage

    def material_policy_coverage(self) -> pd.DataFrame:
        return self._policy_coverage_data().figure_data("material")

    def energy_policy_coverage(self) -> pd.DataFrame:
        return self._policy_coverage_data().figure_data("energy")

    def policy_coverage_membership(self) -> pd.DataFrame:
        return self._policy_coverage_data().membership()

    def _emissions_response_data(self):
        if self._emissions_response is None:
            from .consolidation_impact_data import EmissionsResponseData
            self._emissions_response = EmissionsResponseData(self.results, self.config, self.base)
        return self._emissions_response

    def emissions_response(self) -> pd.DataFrame:
        return self._emissions_response_data().figure_data()

    def partner_rankings(self) -> pd.DataFrame:
        return self._emissions_response_data().rankings.copy()

    def geographic_membership(self) -> pd.DataFrame:
        return self._emissions_response_data().membership()

    def production_scenarios(self):
        """Select one raw no-CE pathway for simplified Figures 1–2, or both."""
        scenario = self.config.get("simplified_production_scenario")
        if scenario is None:
            return PRODUCTION_SCENARIOS
        if scenario not in PRODUCTION_SCENARIOS:
            raise ValueError("simplified_production_scenario must be ndc_noce or baseline_noce")
        return (scenario,)

    def producer_ranking_scenario(self):
        return (self.production_scenarios()[0]
                if self.config.get("simplified_production_scenario") is not None else "baseline_noce")

    def _producer_data(self):
        scenarios = self.production_scenarios()
        if not set(scenarios) <= self.results.keys():
            raise ValueError(f"Producer figure requires configured scenarios: {', '.join(scenarios)}")
        if self._producer_reports is None:
            if not self.config.get("producer_groups"):
                raise ValueError("Producer comparison requires a producer_groups partition")
            # Only producer calculations are used here. Supply the same validated
            # partition for ReportData's unused price-group configuration.
            settings = {**self.config, "macroregions": self.config["producer_groups"]}
            self._producer_reports = {
                scenario: ReportData(self.results[scenario], settings)
                for scenario in scenarios
            }
        return self._producer_reports

    @staticmethod
    def _production_variables(sector):
        parent = f"Production|{sector.path}"
        return [parent] if sector.key == "cement" else [parent + "|Primary", parent + "|Secondary"]

    def leading_producers(self) -> pd.DataFrame:
        """One or two no-CE pathways for a shared, explicitly ranked cohort."""
        reports = self._producer_data()
        ranking_scenario = self.producer_ranking_scenario()
        ranking = reports[ranking_scenario]
        first, last = min(self.config["years"]), ranking.ranking_year
        if first == last:
            raise ValueError("Producer endpoints require a ranking year after the first year")
        rows = []
        for column, sector in enumerate(SECTORS):
            cohort = ranking.ranked_producers(sector)
            if not cohort:
                raise ValueError(f"No positive ranking-year production available for {sector.label}")
            members = ranking.producer_groups
            other_labels = [label for label in members if label not in cohort]
            other_regions = [region for label in other_labels for region in members[label]]
            cohort_regions = [region for label in cohort for region in members[label]]
            for scenario, report in reports.items():
                groups = report.production_groups(sector)
                common = {"Scenario": scenario, "ClimatePathway": climate_label(scenario),
                          "CE": policy_label(scenario), "Panel": chr(97 + column),
                          "Sector": sector.label, "RankingScenario": ranking_scenario,
                          "RankingYear": last,
                          "SourceVariables": json.dumps(self._production_variables(sector))}
                plotted = groups.loc[cohort, [first, last]].copy()
                if other_labels:
                    plotted.loc["Other model regions"] = groups.loc[other_labels, [first, last]].sum(
                        axis=0, min_count=len(other_labels))
                for order, (label, values) in enumerate(plotted.iterrows(), start=1):
                    source_regions = other_regions if label == "Other model regions" else members[label]
                    rows.extend({**common, "Scope": "Producer regions",
                                 "Series": "Cement" if sector.key == "cement" else "Total",
                                 "Group": label, "Rank": order if label in cohort else float("nan"),
                                 "DisplayOrder": order, "Year": int(year), "Value": value,
                                 "Unit": "Mt/yr", "Role": "plotted",
                                 "SourceRegions": json.dumps(source_regions)}
                                for year, value in values.items())
                selected = groups.loc[cohort].sum(axis=0, min_count=len(cohort))
                global_output = self.results[scenario].aggregate(report.production_by_region(sector))
                coverage = ratio(selected, global_output, 100)
                rows.extend({**common, "Scope": "Selected producers",
                             "Series": "Selected producer coverage", "Group": "Selected producers",
                             "Rank": float("nan"), "DisplayOrder": float("nan"),
                             "Year": int(year), "Value": value, "Unit": "%", "Role": "context",
                             "SourceRegions": json.dumps(cohort_regions)}
                            for year, value in coverage.items())
        return pd.DataFrame(rows)

    def producer_membership(self) -> pd.DataFrame:
        groups = self._producer_data()[self.producer_ranking_scenario()].producer_groups
        return pd.DataFrame([{"Group": label, "Region": region}
                             for label, members in groups.items() for region in members])

    def producer_rankings(self) -> pd.DataFrame:
        reports = self._producer_data()
        ranking_scenario = self.producer_ranking_scenario()
        ranking = reports[ranking_scenario]
        rows = []
        for sector in SECTORS:
            cohort = ranking.ranked_producers(sector)
            for scenario, report in reports.items():
                groups = report.production_groups(sector)[ranking.ranking_year]
                global_output = self.results[scenario].aggregate(report.production_by_region(sector))
                shares = ratio(groups, pd.Series(global_output.loc[ranking.ranking_year], index=groups.index), 100)
                rows.extend({"Scenario": scenario, "ClimatePathway": climate_label(scenario),
                             "Sector": sector.label, "Rank": rank, "Group": label,
                             "RankingScenario": ranking_scenario, "RankingYear": ranking.ranking_year,
                             "Production_Mt_yr": groups.loc[label], "GlobalShare_percent": shares.loc[label]}
                            for rank, label in enumerate(cohort, start=1))
        return pd.DataFrame(rows)

    def energy(self) -> pd.DataFrame:
        """Global final energy and energy per tonne for the two no-CE pathways."""
        if not set(PRODUCTION_SCENARIOS) <= self.results.keys():
            raise ValueError("Energy comparison requires baseline_noce and ndc_noce")
        rows = []
        for column, sector in enumerate(SECTORS):
            energy_variables = [f"Final Energy|Industry|{sector.path}|{fuel}" for fuel in sector.fuels]
            production_variables = self._production_variables(sector)
            for scenario in PRODUCTION_SCENARIOS:
                results = self.results[scenario]
                energy = results.energy(sector)
                # Clinker is not an input to the cement intensity calculation.
                production = (results.series(production_variables[0], "Mt/yr")
                              if sector.key == "cement" else results.production_total(sector))
                intensity = ratio(energy, production, 1000)
                common = {"Scenario": scenario, "ClimatePathway": climate_label(scenario),
                          "CE": policy_label(scenario), "Scope": "Global", "Sector": sector.label,
                          "Series": sector.label, "Role": "plotted",
                          "SourceRegions": json.dumps(results.regions)}
                for row, (metric, values, unit) in enumerate([
                        ("Final energy", energy, "EJ/yr"),
                        ("Derived energy intensity", intensity, "GJ/t")]):
                    variables = energy_variables + (production_variables if row == 1 else [])
                    rows.extend({**common, "Panel": chr(97 + 3 * row + column),
                                 "Metric": metric, "Year": int(year), "Value": value, "Unit": unit,
                                 "SourceVariables": json.dumps(variables),
                                 "Numerator_EJ_yr": energy.loc[year],
                                 "Denominator_Mt_yr": production.loc[year] if row == 1 else float("nan"),
                                 "ConversionFactor": 1000 if row == 1 else float("nan")}
                                for year, value in values.items())
        return pd.DataFrame(rows)

    def emissions(self) -> pd.DataFrame:
        """Reported sector GHG, industry shares, derived intensity and CO2 capture."""
        if not set(PRODUCTION_SCENARIOS) <= self.results.keys():
            raise ValueError("Emissions comparison requires baseline_noce and ndc_noce")
        rows = []
        industry_variable = "Emissions|GHG|Industry"
        for scenario in PRODUCTION_SCENARIOS:
            results = self.results[scenario]
            industry = results.series(industry_variable, "MtCO2e/yr")
            for sector in SECTORS:
                emissions_variable = f"Emissions|GHG|Industry|{sector.path}"
                capture_variable = f"Carbon Capture|Industry|{sector.path}"
                production_variables = self._production_variables(sector)
                ghg = results.emissions(sector)
                production = (results.series(production_variables[0], "Mt/yr")
                              if sector.key == "cement" else results.production_total(sector))
                capture = results.series(capture_variable, "MtCO2/yr")
                specifications = [
                    ("a", "Sector GHG emissions", ghg / 1000, "GtCO2e/yr",
                     [emissions_variable], ghg, "MtCO2e/yr", None, "", .001),
                    ("b", "Share of industrial GHG emissions", ratio(ghg, industry, 100), "%",
                     [emissions_variable, industry_variable], ghg, "MtCO2e/yr", industry, "MtCO2e/yr", 100),
                    ("c", "Derived emissions intensity", ratio(ghg, production), "tCO2e/t",
                     [emissions_variable, *production_variables], ghg, "MtCO2e/yr", production, "Mt/yr", 1),
                    ("d", "Carbon capture", capture, "MtCO2/yr",
                     [capture_variable], capture, "MtCO2/yr", None, "", 1),
                ]
                common = {"Scenario": scenario, "ClimatePathway": climate_label(scenario),
                          "CE": policy_label(scenario), "Scope": "Global", "Sector": sector.label,
                          "Series": sector.label, "Role": "plotted",
                          "SourceRegions": json.dumps(results.regions)}
                for panel, metric, values, unit, variables, numerator, numerator_unit, denominator, denominator_unit, factor in specifications:
                    rows.extend({**common, "Panel": panel, "Metric": metric, "Year": int(year),
                                 "Value": value, "Unit": unit, "SourceVariables": json.dumps(variables),
                                 "Numerator": numerator.loc[year], "NumeratorUnit": numerator_unit,
                                 "Denominator": denominator.loc[year] if denominator is not None else float("nan"),
                                 "DenominatorUnit": denominator_unit, "ConversionFactor": factor}
                                for year, value in values.items())
        return pd.DataFrame(rows)

    def material_production(self) -> pd.DataFrame:
        """Historical and future material bars for both pathways and every CE case."""
        required = {f"{climate}_{policy}" for climate in MATERIAL_CLIMATES for policy in CE_POLICIES}
        if not required <= self.results.keys():
            raise ValueError("Material production comparison requires all ten CE scenarios")
        historical_year = int(self.config.get("historical_year", 2019))
        comparison_year = int(self.config.get("comparison_year", 2050))
        if (historical_year not in self.config["years"] or
                comparison_year not in self.config["years"] or historical_year >= comparison_year):
            raise ValueError("Material comparison requires configured historical and later comparison years")
        rows = []
        for row, climate in enumerate(MATERIAL_CLIMATES):
            reference_scenario = f"{climate}_noce"
            bars = [(reference_scenario, historical_year, "historical", str(historical_year)),
                    *[(f"{climate}_{policy}", comparison_year, "future", POLICY_SHORT[policy])
                      for policy in CE_POLICIES]]
            for column, sector in enumerate(SECTORS):
                production_variables = self._production_variables(sector)
                variables = ({"Cement": production_variables,
                              "Clinker": ["Production|Non-Metallic Minerals|Cement Clinker"]}
                             if sector.key == "cement" else
                             {"Primary": [production_variables[0]],
                              "Secondary": [production_variables[1]], "Total": production_variables})
                total_key = "Cement" if sector.key == "cement" else "Total"
                reference = float(self.results[reference_scenario].production(sector)[total_key]
                                  .loc[comparison_year])
                for order, (scenario, year, bar_type, label) in enumerate(bars):
                    results = self.results[scenario]
                    quantities = results.production(sector)
                    total = float(quantities[total_key].loc[year])
                    common = {"Scenario": scenario, "ClimatePathway": climate_label(scenario),
                              "CE": policy_label(scenario), "Panel": chr(97 + 3 * row + column),
                              "Scope": "Global", "Sector": sector.label, "Year": year,
                              "SourceRegions": json.dumps(results.regions),
                              "Policy": scenario.split("_", 1)[1], "BarOrder": order,
                              "BarType": bar_type, "BarLabel": label}

                    def record(metric, series, numerator, source_variables, *, denominator=None,
                               factor=1., offset=0., role="plotted", reference_scenario="",
                               reference_year=None):
                        value = (numerator if denominator is None else
                                 (factor * numerator / denominator + offset
                                  if pd.notna(denominator) and denominator > 0 else float("nan")))
                        rows.append({**common, "Metric": metric, "Series": series, "Value": value,
                                     "Unit": "Mt/yr" if denominator is None else "%", "Role": role,
                                     "SourceVariables": json.dumps(source_variables),
                                     "Numerator": numerator, "NumeratorUnit": "Mt/yr",
                                     "Denominator": float("nan") if denominator is None else denominator,
                                     "DenominatorUnit": "" if denominator is None else "Mt/yr",
                                     "ConversionFactor": factor, "Offset": offset,
                                     "ReferenceScenario": reference_scenario,
                                     "ReferenceYear": (float("nan") if reference_year is None
                                                       else reference_year)})

                    for component, series in quantities.items():
                        record("Material production", component, float(series.loc[year]),
                               variables[component], role="context" if component == "Total" else "plotted")
                    if sector.key != "cement":
                        record("Secondary production share", "Secondary",
                               float(quantities["Secondary"].loc[year]), production_variables,
                               denominator=total, factor=100., role="context",
                               reference_scenario=scenario, reference_year=year)
                    if bar_type == "future":
                        record("Production change from no CE", total_key, total, production_variables,
                               denominator=reference, factor=100., offset=-100., role="context",
                               reference_scenario=reference_scenario, reference_year=comparison_year)
        return pd.DataFrame(rows)

    def energy_mix(self) -> pd.DataFrame:
        """Historical and future carrier stacks using exact raw energy parents."""
        required = {f"{climate}_{policy}" for climate in MATERIAL_CLIMATES for policy in CE_POLICIES}
        if not required <= self.results.keys():
            raise ValueError("Energy mix comparison requires all ten CE scenarios")
        historical_year = int(self.config.get("historical_year", 2019))
        comparison_year = int(self.config.get("comparison_year", 2050))
        if (historical_year not in self.config["years"] or
                comparison_year not in self.config["years"] or historical_year >= comparison_year):
            raise ValueError("Energy mix requires configured historical and later comparison years")
        rows = []
        for row, climate in enumerate(MATERIAL_CLIMATES):
            reference_scenario = f"{climate}_noce"
            bars = [(reference_scenario, historical_year, "historical", str(historical_year)),
                    *[(f"{climate}_{policy}", comparison_year, "future", POLICY_SHORT[policy])
                      for policy in CE_POLICIES]]
            for column, sector in enumerate(SECTORS):
                variables = {fuel: f"Final Energy|Industry|{sector.path}|{fuel}"
                             for fuel in ENERGY_CARRIERS if fuel in sector.fuels}
                quantities = {
                    scenario: {fuel: self.results[scenario].series(variable, "EJ/yr")
                               for fuel, variable in variables.items()}
                    for scenario in dict.fromkeys(scenario for scenario, *_ in bars)
                }
                totals = {scenario: pd.DataFrame(carriers).sum(axis=1, min_count=len(carriers))
                          for scenario, carriers in quantities.items()}
                reference = float(totals[reference_scenario].loc[comparison_year])
                for order, (scenario, year, bar_type, label) in enumerate(bars):
                    results = self.results[scenario]
                    total = float(totals[scenario].loc[year])
                    common = {"Scenario": scenario, "ClimatePathway": climate_label(scenario),
                              "CE": policy_label(scenario), "Panel": chr(97 + 3 * row + column),
                              "Scope": "Global", "Sector": sector.label, "Year": year,
                              "SourceRegions": json.dumps(results.regions),
                              "Policy": scenario.split("_", 1)[1], "BarOrder": order,
                              "BarType": bar_type, "BarLabel": label}

                    def record(series, numerator, source_variables, *, role="plotted", denominator=None):
                        is_change = denominator is not None
                        value = (numerator if not is_change else
                                 (100 * numerator / denominator - 100
                                  if pd.notna(denominator) and denominator > 0 else float("nan")))
                        rows.append({**common, "Metric": ("Final energy change from no CE" if is_change
                                                          else "Final energy"),
                                     "Series": series, "Value": value,
                                     "Unit": "%" if is_change else "EJ/yr", "Role": role,
                                     "SourceVariables": json.dumps(source_variables),
                                     "Numerator": numerator, "NumeratorUnit": "EJ/yr",
                                     "Denominator": denominator if is_change else float("nan"),
                                     "DenominatorUnit": "EJ/yr" if is_change else "",
                                     "ConversionFactor": 100. if is_change else 1.,
                                     "Offset": -100. if is_change else 0.,
                                     "ReferenceScenario": reference_scenario if is_change else "",
                                     "ReferenceYear": comparison_year if is_change else float("nan")})

                    for fuel, variable in variables.items():
                        record(fuel, float(quantities[scenario][fuel].loc[year]), [variable])
                    record("Total", total, list(variables.values()), role="context")
                    if bar_type == "future":
                        record("Total", total, list(variables.values()), role="context", denominator=reference)
        return pd.DataFrame(rows)

    def sector_costs(self) -> pd.DataFrame:
        """Global sector-cost percentages against climate/sector/year-matched no CE."""
        required = {f"{climate}_{policy}" for climate in MATERIAL_CLIMATES for policy in CE_POLICIES}
        if not required <= self.results.keys():
            raise ValueError("Sector-cost comparison requires all ten CE scenarios")
        start = int(self.config.get("main_cost_start_year", 2024))
        if not any(year >= start for year in self.config["years"]):
            raise ValueError("Sector-cost comparison requires at least one year in the main cost period")
        rows = []
        for row, climate in enumerate(MATERIAL_CLIMATES):
            reference_scenario = f"{climate}_noce"
            for column, sector in enumerate(SECTORS):
                variable = f"Total Annualised Cost|Industry|{sector.path}"
                unit = "Millions USD_2010/yr"
                reference = self.results[reference_scenario].series(variable, unit, activity=False)
                for policy in CE_POLICIES:
                    scenario = f"{climate}_{policy}"
                    results = self.results[scenario]
                    costs = results.series(variable, unit, activity=False)
                    changes = percent_change(costs, reference)
                    rows.extend({
                        "Scenario": scenario, "ClimatePathway": climate_label(scenario),
                        "CE": policy_label(scenario), "Panel": chr(97 + 3 * row + column),
                        "Scope": "Global", "Sector": sector.label, "Series": sector.label,
                        "Metric": "Annualised sector-cost change", "Policy": policy,
                        "Year": int(year), "Value": value, "Unit": "%",
                        "Role": "plotted" if year >= start else "context",
                        "SourceRegions": json.dumps(results.regions), "SourceVariables": json.dumps([variable]),
                        "Numerator": costs.loc[year], "NumeratorUnit": unit,
                        "Denominator": reference.loc[year], "DenominatorUnit": unit,
                        "ConversionFactor": 100., "Offset": -100.,
                        "ScenarioCost": costs.loc[year], "ReferenceCost": reference.loc[year],
                        "ReferenceScenario": reference_scenario, "ReferenceYear": int(year),
                    } for year, value in changes.items())
        return pd.DataFrame(rows)

    def production(self) -> pd.DataFrame:
        """Global production for the selected no-CE pathways, in panel order."""
        scenarios = self.production_scenarios()
        if not set(scenarios) <= self.results.keys():
            raise ValueError(f"Production figure requires configured scenarios: {', '.join(scenarios)}")
        rows = []
        for row, scenario in enumerate(scenarios):
            for column, sector in enumerate(SECTORS):
                for series, values in self.results[scenario].production(sector).items():
                    if sector.key == "cement":
                        variable = ("Production|Non-Metallic Minerals|Cement Clinker"
                                    if series == "Clinker" else "Production|" + sector.path)
                        variables = [variable]
                    else:
                        routes = ("Primary", "Secondary") if series == "Total" else (series,)
                        variables = [f"Production|{sector.path}|{route}" for route in routes]
                    rows.extend({
                        "Scenario": scenario, "ClimatePathway": climate_label(scenario),
                        "CE": policy_label(scenario), "Panel": chr(97 + 3 * row + column),
                        "Scope": "Global", "Sector": sector.label, "Series": series,
                        "Year": int(year), "Value": value, "Unit": "Mt/yr", "Role": "plotted",
                        "SourceVariables": json.dumps(variables),
                    } for year, value in values.items())
        return pd.DataFrame(rows)
