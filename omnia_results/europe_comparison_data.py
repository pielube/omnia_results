"""Exact, auditable EU-scope extraction from the two supplied model exports.

The sources have different reporting boundaries. This adapter preserves their
variable definitions and does not invent production routes or fuel categories.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import SECTORS


MODELS = ("OMNIA-ENGAGE", "GCAM")
CLIMATES = ("ndc", "baseline")
POLICIES = ("noce", "medce_eu", "highce_eu")
CLIMATE_LABELS = {"ndc": "NDC", "baseline": "NDC+LTT"}
POLICY_LABELS = {"noce": "No CE", "medce_eu": "Medium CE, EU", "highce_eu": "High CE, EU"}
OMNIA_REGIONS = ("EUE", "EUM", "EUW")
GCAM_PRODUCTION = {
    "cement": "Production|Non-Metallic Minerals|Cement",
    "steel": "Production|Iron and Steel|Steel",
    "aluminium": "Production|Non-Ferrous Metals|Aluminum",
}
GCAM_FUELS = {"cement": ("Electricity", "Gases", "Hydrogen", "Liquids", "Solids"),
              "steel": ("Electricity", "Gases", "Hydrogen", "Liquids", "Solids"),
              "aluminium": ("Electricity", "Liquids", "Solids")}


def source_scenario(model, climate, policy):
    if model == "OMNIA-ENGAGE":
        return f"{climate}_{policy}"
    return (f"{'NDC' if climate == 'ndc' else 'LTT'}_"
            f"{dict(noce='NoCE', medce_eu='MedCE_EU', highce_eu='HighCE_EU')[policy]}")


def _percent(value, reference):
    return 100 * (value / reference - 1) if np.isfinite(value) and np.isfinite(reference) and reference > 0 else np.nan


class EuropeComparisonData:
    models, climates, policies = MODELS, CLIMATES, POLICIES

    def __init__(self, config: dict, base: Path):
        self.config, self.base = config, Path(base)
        self.sources = {MODELS[0]: (self.base / config["input"]).resolve(),
                        MODELS[1]: (self.base / config["gcam_input"]).resolve()}
        self.source = self.sources[MODELS[0]]
        self.gcam_source = self.sources[MODELS[1]]
        self.missing_activity = config.get("missing_activity", "zero")
        if self.missing_activity not in {"zero", "preserve"}:
            raise ValueError("missing_activity must be 'zero' or 'preserve'")
        self.historical_years = {MODELS[0]: int(config.get("historical_year", 2019)),
                                 MODELS[1]: int(config.get("gcam_historical_year", 2021))}
        self.comparison_year = int(config.get("comparison_year", 2050))
        self.raw = {MODELS[0]: pd.read_csv(self.source),
                    MODELS[1]: pd.read_excel(self.gcam_source, sheet_name=config.get("gcam_sheet", "Sheet1"))}
        self.years, self.available_years, self._cache, self._missing, self._checks = {}, {}, {}, [], []
        self._frames = {}
        self._validate_sources()
        self.notes = [
            "OMNIA-ENGAGE aggregates EUE, EUM and EUW; GCAM selects its EU-27 aggregate once, without adding the country rows.",
            "The OMNIA EU regions include Greenland, Faroe Islands, Andorra, Monaco and San Marino in the supplied country mapping; these cannot be removed from regional data.",
            "GCAM source LTT scenarios are paired with OMNIA baseline (NDC+LTT); the scenario names alone do not establish identical policy assumptions.",
            "Historical bars use OMNIA 2019 and GCAM 2021 observations separately; no interpolation or historical alignment is imposed.",
            "GCAM reports total cement, steel and aluminium production, without clinker or primary/secondary route splits.",
            "GCAM aluminium energy includes an Unallocated residual between the reported sector total and its reported carrier parents; the residual is not assigned to an unreported carrier.",
            "Industrial GHG boundaries differ: OMNIA reports Emissions|GHG|Industry; GCAM combines industry energy-demand Kyoto gases with industrial-process Kyoto gases. GCAM has no comparable cement/steel/aluminium GHG decomposition.",
        ]

    def _validate_sources(self):
        keys = ["Scenario", "Region", "Variable", "Unit"]
        if "regions" in self.config and set(self.config["regions"]) != set(OMNIA_REGIONS):
            raise ValueError("Europe comparison uses exactly the OMNIA EU regions EUE, EUM and EUW")
        if "scenarios" in self.config:
            expected = {source_scenario(MODELS[0], c, p) for c in CLIMATES for p in POLICIES}
            if set(self.config["scenarios"]) != expected:
                raise ValueError("Europe comparison config must select the six OMNIA EU-only scenarios")
        for model, raw in self.raw.items():
            if missing := set(keys) - set(raw):
                raise ValueError(f"{model} missing source columns: {sorted(missing)}")
            if raw[keys].isna().any().any():
                raise ValueError(f"{model} blank scenario, region, variable or unit")
            if raw.duplicated(keys).any():
                raise ValueError(f"{model} duplicate scenario/region/variable/unit rows")
            years = sorted(int(c) for c in raw.columns if str(c).isdigit())
            if not years:
                raise ValueError(f"{model} has no model-year columns")
            raw.columns = [str(c) for c in raw.columns]
            for year in years:
                raw[str(year)] = pd.to_numeric(raw[str(year)], errors="raise")
            if np.isinf(raw[[str(y) for y in years]].to_numpy(dtype=float)).any():
                raise ValueError(f"{model} contains infinite source values")
            required = {source_scenario(model, c, p) for c in CLIMATES for p in POLICIES}
            if missing := required - set(raw.Scenario):
                raise ValueError(f"{model} missing required EU scenarios: {sorted(missing)}")
            regions = set(raw.Region)
            if missing := set(self._regions(model)) - regions:
                raise ValueError(f"{model} missing comparison regions: {sorted(missing)}")
            selection_key = "years" if model == MODELS[0] else "gcam_years"
            configured_years = self.config.get(selection_key, years)
            if not configured_years or any(not isinstance(year, int) for year in configured_years):
                raise ValueError(f"{selection_key} must contain integer source years")
            if configured_years != sorted(set(configured_years)):
                raise ValueError(f"{selection_key} must be unique and increasing")
            if missing := set(configured_years) - set(years):
                raise ValueError(f"{model} configured years absent from source: {sorted(missing)}")
            for year in (self.historical_years[model], self.comparison_year):
                if year not in years:
                    raise ValueError(f"{model} has no source observation for required year {year}")
                if year not in configured_years:
                    raise ValueError(f"{model} required year {year} is omitted from {selection_key}")
            if self.historical_years[model] >= self.comparison_year:
                raise ValueError("Historical year must precede comparison year")
            if model == "GCAM" and "Model" in raw and raw.Model.nunique() != 1:
                raise ValueError("GCAM workbook must contain exactly one source model")
            self.available_years[model], self.years[model] = years, list(configured_years)

    @staticmethod
    def _regions(model):
        return list(OMNIA_REGIONS) if model == MODELS[0] else ["EU-27"]

    @property
    def missing_inputs(self):
        columns = ["Model", "Scenario", "Region", "Variable", "Unit", "Year", "Issue", "Treatment"]
        return pd.DataFrame(self._missing, columns=columns).drop_duplicates().reset_index(drop=True)

    @property
    def aggregation_checks(self):
        return pd.DataFrame(self._checks)

    def _series(self, model, scenario, variable, unit):
        key = model, scenario, variable, unit
        if key in self._cache:
            return self._cache[key].copy()
        raw = self.raw[model]
        selected = raw.loc[(raw.Scenario == scenario) & (raw.Variable == variable) & (raw.Unit == unit)]
        if selected.empty:
            raise ValueError(f"{model} required source series absent: {scenario}: {variable} [{unit}]")
        regions, years = self._regions(model), self.years[model]
        values = selected.set_index("Region")[[str(y) for y in years]].reindex(regions)
        values.columns = years
        absent = set(regions) - set(selected.Region)
        for region in regions:
            for year in years:
                if pd.isna(values.loc[region, year]):
                    self._missing.append(dict(Model=model, Scenario=scenario, Region=region, Variable=variable,
                                              Unit=unit, Year=year, Issue="absent row" if region in absent else "blank cell",
                                              Treatment=self.missing_activity if model == MODELS[0] else "preserve"))
        # An absent EU-27 aggregate is an error, rather than a fabricated zero.
        if model == "GCAM" and absent:
            raise ValueError(f"GCAM EU-27 aggregate absent for {scenario}: {variable} [{unit}]")
        if model == MODELS[0] and self.missing_activity == "zero":
            values = values.fillna(0)
        result = values.sum(axis=0, min_count=len(regions))
        if model == "GCAM":
            countries = selected.loc[selected.Region.ne("EU-27")]
            if not countries.empty:
                sums = countries[[str(y) for y in years]].sum(axis=0, min_count=len(countries))
                for year in years:
                    total = result.loc[year]
                    other = sums.loc[str(year)]
                    self._checks.append(dict(Model=model, Scenario=scenario, Variable=variable, Unit=unit,
                                             Year=year, Aggregate=total, SumReportedCountries=other,
                                             CountryRowCount=len(countries), Difference=total-other))
                    if np.isfinite(total) and np.isfinite(other) and not np.isclose(total, other, rtol=1e-8, atol=1e-8):
                        raise ValueError(f"GCAM EU-27 aggregate differs from reported country sum: {scenario}: {variable} ({year})")
        self._cache[key] = result
        return result.copy()

    def _common(self, model, climate, policy, sector, year, order=np.nan):
        return dict(Model=model, Scenario=source_scenario(model, climate, policy), Climate=climate,
                    ClimatePathway=CLIMATE_LABELS[climate], Policy=policy, CE=POLICY_LABELS[policy],
                    Sector=sector.label if hasattr(sector, "label") else sector, Year=year,
                    Scope="EU model regions" if model == MODELS[0] else "EU-27",
                    BarOrder=order, BarType="historical" if order == 0 else "future" if np.isfinite(order) else "trajectory",
                    SourceFile=self.sources[model].name,
                    SourceModel=(self.raw[model].Model.iloc[0] if model == "GCAM" and "Model" in self.raw[model] else model),
                    SourceSheet=self.config.get("gcam_sheet", "Sheet1") if model == "GCAM" else "",
                    SourceRegions=json.dumps(self._regions(model)),
                    MissingActivityPolicy=self.missing_activity if model == MODELS[0] else "preserve")

    @staticmethod
    def _provenance(variables, units, coefficients=None):
        return dict(SourceVariables=json.dumps(variables), SourceUnits=json.dumps(units),
                    SourceCoefficients=json.dumps(coefficients or [1] * len(variables)))

    def _quantity(self, common, metric, series, value, unit, variables, source_units=None,
                  coefficients=None, role="plotted", **extra):
        return {**common, "Metric": metric, "Series": series, "Value": value, "Unit": unit,
                "Role": role, "Numerator": value, "NumeratorUnit": unit, "ConversionFactor": 1,
                "Offset": 0, **self._provenance(variables, source_units or [unit] * len(variables), coefficients), **extra}

    def _change(self, common, metric, value, reference, unit, variables, source_units=None):
        model, climate = common["Model"], common["Climate"]
        return {**common, "Metric": metric, "Series": "Total", "Value": _percent(value, reference), "Unit": "%",
                "Role": "annotation", "Numerator": value, "Denominator": reference,
                "NumeratorUnit": unit, "DenominatorUnit": unit, "ConversionFactor": 100, "Offset": -100,
                "ScenarioValue": value, "ReferenceValue": reference,
                "ReferenceScenario": source_scenario(model, climate, "noce"), "ReferenceYear": self.comparison_year,
                "ReferenceSourceVariables": json.dumps(variables), "ReferenceSourceRegions": json.dumps(self._regions(model)),
                "ReferenceSourceUnits": json.dumps(source_units or [unit] * len(variables)),
                **self._provenance(variables, source_units or [unit] * len(variables))}

    def _bars(self, model, climate):
        yield "noce", self.historical_years[model], 0
        for order, policy in enumerate(POLICIES, start=1):
            yield policy, self.comparison_year, order

    def material_production(self):
        if "production" in self._frames:
            return self._frames["production"].copy()
        rows = []
        for model in MODELS:
            for climate in CLIMATES:
                for sector in SECTORS:
                    parent = f"Production|{sector.path}"
                    if model == "GCAM":
                        definitions = {"Cement" if sector.key == "cement" else "Total": [GCAM_PRODUCTION[sector.key]]}
                    elif sector.key == "cement":
                        definitions = {"Cement": [parent], "Clinker": ["Production|Non-Metallic Minerals|Cement Clinker"]}
                    else:
                        definitions = {"Primary": [parent + "|Primary"], "Secondary": [parent + "|Secondary"],
                                       "Total": [parent + "|Primary", parent + "|Secondary"]}
                    totals = definitions["Cement" if sector.key == "cement" else "Total"]
                    reference_scenario = source_scenario(model, climate, "noce")
                    reference = sum(self._series(model, reference_scenario, v, "Mt/yr") for v in totals).loc[self.comparison_year]
                    for policy, year, order in self._bars(model, climate):
                        common = self._common(model, climate, policy, sector, year, order)
                        values = {}
                        for series, variables in definitions.items():
                            value = sum(self._series(model, common["Scenario"], v, "Mt/yr") for v in variables).loc[year]
                            values[series] = value
                            rows.append(self._quantity(common, "Material production", series, value, "Mt/yr", variables,
                                                       role="context" if series == "Clinker" or series == "Total" and model == MODELS[0] else "plotted"))
                        total = values["Cement" if sector.key == "cement" else "Total"]
                        if order:
                            rows.append(self._change(common, "Production change from no CE", total, reference, "Mt/yr", totals))
                        if "Secondary" in values:
                            share = 100 * values["Secondary"] / total if np.isfinite(total) and total > 0 else np.nan
                            rows.append({**common, "Metric": "Secondary production share", "Series": "Secondary",
                                         "Value": share, "Unit": "%", "Role": "annotation", "Numerator": values["Secondary"],
                                         "Denominator": total, "NumeratorUnit": "Mt/yr", "DenominatorUnit": "Mt/yr",
                                         "NumeratorVariables": json.dumps(definitions["Secondary"]),
                                         "DenominatorVariables": json.dumps(definitions["Total"]),
                                         "NumeratorSourceVariables": json.dumps(definitions["Secondary"]),
                                         "DenominatorSourceVariables": json.dumps(definitions["Total"]),
                                         "ConversionFactor": 100, "Offset": 0,
                                         **self._provenance(definitions["Total"], ["Mt/yr"] * 2)})
        self._frames["production"] = pd.DataFrame(rows)
        self._add_panels(self._frames["production"])
        return self._frames["production"].copy()

    def energy_mix(self):
        if "energy" in self._frames:
            return self._frames["energy"].copy()
        rows = []
        for model in MODELS:
            for climate in CLIMATES:
                for sector in SECTORS:
                    parent = f"Final Energy|Industry|{sector.path}"
                    fuels = sector.fuels if model == MODELS[0] else GCAM_FUELS[sector.key]
                    variables = [parent + "|" + fuel for fuel in fuels]
                    total_variables = variables if model == MODELS[0] else [parent]
                    reference_scenario = source_scenario(model, climate, "noce")
                    reference = sum(self._series(model, reference_scenario, v, "EJ/yr") for v in total_variables).loc[self.comparison_year]
                    for policy, year, order in self._bars(model, climate):
                        common = self._common(model, climate, policy, sector, year, order)
                        values = []
                        for fuel, variable in zip(fuels, variables):
                            value = self._series(model, common["Scenario"], variable, "EJ/yr").loc[year]
                            values.append(value)
                            rows.append(self._quantity(common, "Final energy", fuel, value, "EJ/yr", [variable]))
                        total = sum(values) if model == MODELS[0] else self._series(model, common["Scenario"], parent, "EJ/yr").loc[year]
                        if model == "GCAM":
                            residual = total - sum(values)
                            # Round only negligible floating arithmetic residue.
                            if np.isfinite(residual) and abs(residual) < 1e-12:
                                residual = 0.0
                            rows.append(self._quantity(common, "Final energy", "Unallocated", residual, "EJ/yr",
                                                       [parent] + variables, coefficients=[1] + [-1] * len(variables),
                                                       Definition="Reported sector total minus reported carrier parents"))
                        rows.append(self._quantity(common, "Final energy", "Total", total, "EJ/yr", total_variables, role="context"))
                        if order:
                            rows.append(self._change(common, "Final energy change from no CE", total, reference, "EJ/yr", total_variables))
        self._frames["energy"] = pd.DataFrame(rows)
        self._add_panels(self._frames["energy"])
        return self._frames["energy"].copy()

    @staticmethod
    def _add_panels(frame):
        columns = {sector.label: index for index, sector in enumerate(SECTORS)}
        frame["Panel"] = [chr(97 + CLIMATES.index(climate) * 3 + columns[sector])
                          for climate, sector in zip(frame.Climate, frame.Sector)]

    def _emissions_definitions(self, model):
        if model == MODELS[0]:
            variables = [f"Emissions|GHG|Industry|{sector.path}" for sector in SECTORS]
            return {"Total industry": (["Emissions|GHG|Industry"], [1]),
                    **{sector.label: ([variable], [1]) for sector, variable in zip(SECTORS, variables)},
                    "Other industry": (["Emissions|GHG|Industry"] + variables, [1, -1, -1, -1])}, "MtCO2e/yr"
        energy = "Emissions|Kyoto Gases|Energy|Demand|Industry"
        process = "Emissions|Kyoto Gases|Industrial Processes"
        return {"Total industry": ([energy, process], [1, 1]),
                "Industrial energy demand": ([energy], [1]),
                "Industrial processes": ([process], [1])}, "Mt CO2-equiv/yr"

    def emissions_response(self):
        """Source-defined industry totals, model-specific components and CE effects.

        All trajectories retain the original observations. Absolute changes use
        the same model's no-CE scenario in the same climate pathway and year.
        """
        if "emissions" in self._frames:
            return self._frames["emissions"].copy()
        rows = []
        for model in MODELS:
            definitions, source_unit = self._emissions_definitions(model)
            scope = ("Reported all-industry GHG" if model == MODELS[0] else
                     "Industry energy-demand Kyoto gases plus industrial-process Kyoto gases")
            for climate in CLIMATES:
                reference_scenario = source_scenario(model, climate, "noce")
                references = {series: sum(coef * self._series(model, reference_scenario, variable, source_unit)
                                           for variable, coef in zip(variables, coefficients))
                              for series, (variables, coefficients) in definitions.items()}
                for policy in POLICIES:
                    scenario = source_scenario(model, climate, policy)
                    values = {series: sum(coef * self._series(model, scenario, variable, source_unit)
                                          for variable, coef in zip(variables, coefficients))
                              for series, (variables, coefficients) in definitions.items()}
                    variables, coefficients = definitions["Total industry"]
                    for year in self.years[model]:
                        # Historical trajectory observations before the respective
                        # historical bar anchor are context; plots begin at 2019/2021.
                        common = self._common(model, climate, policy, "Industry", year)
                        value = values["Total industry"].loc[year]
                        rows.append(self._quantity(common, "Industrial GHG emissions", "Total industry", value,
                                                   "MtCO2e/yr", variables, [source_unit] * len(variables), coefficients,
                                                   role="plotted" if year >= self.historical_years[model] else "context",
                                                   EmissionsScope=scope))
                    year = self.comparison_year
                    common = self._common(model, climate, policy, "Industry", year, POLICIES.index(policy) + 1)
                    for series, (variables, coefficients) in definitions.items():
                        value, reference = values[series].loc[year], references[series].loc[year]
                        rows.append({**self._quantity(common, "Industrial GHG emissions change", series, value-reference,
                                                      "MtCO2e/yr", variables, [source_unit] * len(variables), coefficients,
                                                      role="net" if series == "Total industry" else "plotted"),
                                     "ScenarioValue": value, "ReferenceValue": reference,
                                     "ReferenceScenario": reference_scenario, "ReferenceYear": year,
                                     "ReferenceSourceVariables": json.dumps(variables),
                                     "ReferenceSourceUnits": json.dumps([source_unit] * len(variables)),
                                     "ReferenceSourceCoefficients": json.dumps(coefficients),
                                     "ReferenceSourceRegions": json.dumps(self._regions(model)),
                                     "EmissionsScope": scope})
                    variables, _ = definitions["Total industry"]
                    change = self._change(common, "Industrial GHG emissions change from no CE",
                                          values["Total industry"].loc[year], references["Total industry"].loc[year],
                                          "MtCO2e/yr", variables, [source_unit] * len(variables))
                    change["Series"], change["EmissionsScope"] = "Total industry", scope
                    change["Role"] = "context" if policy == "noce" else "plotted"
                    rows.append(change)
        self._frames["emissions"] = pd.DataFrame(rows)
        offsets = {"Industrial GHG emissions": 0, "Industrial GHG emissions change": 2,
                   "Industrial GHG emissions change from no CE": 4}
        self._frames["emissions"]["Panel"] = [chr(97 + offsets[metric] + CLIMATES.index(climate))
                                                for metric, climate in zip(self._frames["emissions"].Metric,
                                                                           self._frames["emissions"].Climate)]
        return self._frames["emissions"].copy()

    def scenario_mapping(self):
        return pd.DataFrame([dict(Model=model, Scenario=source_scenario(model, climate, policy), Climate=climate,
                                  ClimatePathway=CLIMATE_LABELS[climate], Policy=policy, CE=POLICY_LABELS[policy],
                                  SourceClimateLabel=climate if model == MODELS[0] else "NDC" if climate == "ndc" else "LTT")
                             for model in MODELS for climate in CLIMATES for policy in POLICIES])

    def geography_membership(self):
        path = self.config.get("region_mapping", "data/OMNIA_region_mapping_241120.csv")
        mapping = pd.read_csv(self.base / path)
        mapping = mapping.loc[mapping.region.isin(OMNIA_REGIONS)].drop_duplicates(["region", "ISO3"])
        countries = sorted(r for r in self.raw["GCAM"].Region.unique() if r != "EU-27")
        # Match Czech Republic in GCAM to Czechia in the source country table.
        gcam_names = {"Czechia": "Czech Republic"}
        rows = [dict(Model=MODELS[0], SourceRegion=r.region, Country=r.country_OMNIA, ISO3=r.ISO3,
                     Aggregation="Sum EUE, EUM and EUW once", MatchedGCAMCountry=gcam_names.get(r.country_OMNIA, r.country_OMNIA),
                     IncludedInGCAM=gcam_names.get(r.country_OMNIA, r.country_OMNIA) in countries)
                for r in mapping.itertuples()]
        rows.extend(dict(Model="GCAM", SourceRegion="EU-27", Country=country,
                         Aggregation="Use EU-27 aggregate once; country rows document membership only",
                         IncludedInGCAM=True) for country in countries)
        return pd.DataFrame(rows)

    def variable_mapping(self):
        """Variable definitions, including source series that cannot be aligned."""
        rows = []
        for model in MODELS:
            for sector in SECTORS:
                production = ([GCAM_PRODUCTION[sector.key]] if model == "GCAM" else
                              [f"Production|{sector.path}"] if sector.key == "cement" else
                              [f"Production|{sector.path}|Primary", f"Production|{sector.path}|Secondary"])
                rows.append(dict(Model=model, Metric="Material production", Sector=sector.label,
                                 SourceVariables=json.dumps(production), SourceUnit="Mt/yr", FigureUnit="Mt/yr",
                                 Definition="Total reported production" if model == "GCAM" or sector.key == "cement" else "Primary plus secondary production"))
                fuels = sector.fuels if model == MODELS[0] else GCAM_FUELS[sector.key]
                parent = f"Final Energy|Industry|{sector.path}"
                rows.append(dict(Model=model, Metric="Final energy", Sector=sector.label,
                                 SourceVariables=json.dumps([parent + "|" + f for f in fuels]), SourceUnit="EJ/yr", FigureUnit="EJ/yr",
                                 TotalSourceVariables=json.dumps([parent] if model == "GCAM" else [parent + "|" + f for f in fuels]),
                                 Definition="Reported sector energy total, with reported carriers and unallocated residual" if model == "GCAM" else "Sum exact carrier parents"))
            definitions, unit = self._emissions_definitions(model)
            for series, (variables, coefficients) in definitions.items():
                rows.append(dict(Model=model, Metric="Industrial GHG emissions", Sector="Industry", Series=series,
                                 SourceVariables=json.dumps(variables), SourceCoefficients=json.dumps(coefficients),
                                 SourceUnit=unit, FigureUnit="MtCO2e/yr",
                                 Definition="Reported all-industry GHG" if model == MODELS[0] else
                                 "Reported industrial energy-demand and industrial-process Kyoto gases; not a sector-aligned decomposition"))
        return pd.DataFrame(rows)
