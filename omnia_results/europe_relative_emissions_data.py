"""Relative industrial and material-sector GHG responses with explicit scopes."""

import json

import numpy as np
import pandas as pd

from .europe_comparison_data import CLIMATES, MODELS, OMNIA_REGIONS, POLICIES, source_scenario
from .metrics import SECTORS


INDUSTRY_METRIC = "Industrial GHG response"
THREE_SECTOR_METRIC = "Three-sector GHG response"
ALUMINIUM_GHG = "Emissions|GHG|Industry|Non-Ferrous Metals|Aluminum"


class RelativeEmissionsDataMixin:
    """Extend EuropeRelativeData using its raw-source extraction and row schema."""

    def _emissions_missingness(self, model, scenario, reference_scenario, definition, total_definition, unit):
        """Record every missing input relevant to the component and its denominator."""
        frame = self.source_data.missing_inputs
        variables, _ = definition
        total_variables, _ = total_definition
        selected = frame.loc[(frame.Model == model) & (frame.Unit == unit) & (frame.Year == self.comparison_year)
                             & (((frame.Scenario == scenario) & frame.Variable.isin(variables))
                                | ((frame.Scenario == reference_scenario)
                                   & frame.Variable.isin(set(variables) | set(total_variables))))].copy()
        zero = selected.loc[selected.Treatment.eq("zero")]
        aluminium_scenarios = []
        for name, rows in zero.loc[zero.Variable.eq(ALUMINIUM_GHG)].groupby("Scenario"):
            if set(rows.Region) == set(OMNIA_REGIONS):
                aluminium_scenarios.append(name)
        return dict(MissingInputCount=len(selected), ProvisionalZeroCount=len(zero),
                    UsesProvisionalZeros=bool(len(zero)),
                    MissingSourceInputs=selected.to_json(orient="records"),
                    ProvisionalZeroInputs=zero.to_json(orient="records"),
                    AllEUAluminiumMissingAsZero=bool(aluminium_scenarios),
                    MissingAluminiumScenarios=json.dumps(sorted(aluminium_scenarios)),
                    ProvisionalZeroNote=("All three OMNIA EU regions have missing aluminium GHG inputs, provisionally treated as zero, in the listed scenarios."
                                         if aluminium_scenarios else
                                         "Missing additive OMNIA GHG inputs were provisionally treated as zero."
                                         if len(zero) else ""))

    def _emissions_case(self, model, climate, policy, panel_kind):
        if panel_kind == "industry":
            definitions, source_unit = self.source_data._emissions_definitions(model)
            total_definition = definitions["Total industry"]
            components = {name: definition for name, definition in definitions.items() if name != "Total industry"}
            metric, sector, offset = INDUSTRY_METRIC, "Industry", 0
            scope = ("Reported Emissions|GHG|Industry" if model == MODELS[0] else
                     "Industry energy-demand Kyoto gases plus industrial-process Kyoto gases")
            coverage = "Reported all industry; model emissions boundaries differ"
        else:
            components = {sector.label: ([f"Emissions|GHG|Industry|{sector.path}"], [1]) for sector in SECTORS}
            total_definition = ([f"Emissions|GHG|Industry|{sector.path}" for sector in SECTORS], [1, 1, 1])
            source_unit, metric, sector, offset = "MtCO2e/yr", THREE_SECTOR_METRIC, "Three material sectors", 2
            scope = "Sum of reported cement, iron and steel, and aluminium GHG; excludes other industry"
            coverage = "OMNIA-ENGAGE material-sector GHG only; no matched GCAM sector GHG decomposition"

        scenario = source_scenario(model, climate, policy)
        reference_scenario = source_scenario(model, climate, "noce")
        total_reference = self._value(model, reference_scenario, total_definition, source_unit)
        total_value = self._value(model, scenario, total_definition, source_unit)
        # Load all component values before documenting missingness, so the audit
        # includes numerator, component-reference and shared-denominator inputs.
        values = {name: self._value(model, scenario, definition, source_unit) for name, definition in components.items()}
        references = {name: self._value(model, reference_scenario, definition, source_unit) for name, definition in components.items()}
        common = self.source_data._common(model, climate, policy, sector, self.comparison_year)
        common.update(ReferenceScenario=reference_scenario, ReferenceYear=self.comparison_year,
                      Panel=chr(97 + offset + CLIMATES.index(climate)), PanelKind=panel_kind,
                      BarType="future", BarOrder=POLICIES[1:].index(policy),
                      EmissionsScope=scope, EmissionsCoverage=coverage)
        rows = []
        for name, definition in components.items():
            row = self._row(common, metric, name, "component", values[name], references[name], total_reference,
                            definition, total_definition, source_unit)
            row.update(self._emissions_missingness(model, scenario, reference_scenario, definition, total_definition, source_unit),
                       ContextOnly=panel_kind == "industry", DisplayRole="context" if panel_kind == "industry" else "plotted")
            rows.append(row)
        net = self._row(common, metric, "Total", "net", total_value, total_reference, total_reference,
                        total_definition, total_definition, source_unit)
        net.update(self._emissions_missingness(model, scenario, reference_scenario, total_definition, total_definition, source_unit),
                   ContextOnly=False, DisplayRole="plotted")
        contributions = np.array([row["Value"] for row in rows], dtype=float)
        complete = bool(np.isfinite(contributions).all() and np.isfinite(net["Value"]))
        summed = float(contributions.sum()) if complete else np.nan
        difference = summed - net["Value"] if complete else np.nan
        self._closure_checks.append({**common, "Metric": metric, "Complete": complete,
                                     "ComponentCount": len(rows), "SumContributions": summed,
                                     "NetResponse": net["Value"], "Difference": difference,
                                     "Unit": "percentage points"})
        if complete and not np.isclose(summed, net["Value"], rtol=1e-10, atol=1e-10):
            raise ValueError(f"GHG contributions do not reconstruct {model} {climate} {policy} {panel_kind} response")
        return rows + [net]

    def emissions_response(self):
        if "emissions" in self._frames:
            return self._frames["emissions"].copy()
        rows = []
        for model in MODELS:
            for climate in CLIMATES:
                for policy in POLICIES[1:]:
                    rows.extend(self._emissions_case(model, climate, policy, "industry"))
                    if model == MODELS[0]:
                        rows.extend(self._emissions_case(model, climate, policy, "three_sector"))
        self._frames["emissions"] = pd.DataFrame(rows)
        return self._frames["emissions"].copy()

    def emissions_availability(self):
        """Inventory GCAM emissions without substituting narrower CO2 variables."""
        raw = self.source_data.raw["GCAM"]
        selected = raw.loc[raw.Variable.str.startswith("Emissions|")]
        industry = {"Emissions|Kyoto Gases|Energy|Demand|Industry", "Emissions|Kyoto Gases|Industrial Processes"}
        explanations = {
            "Emissions|CO2|Industrial Processes|Cement": "Cement process CO2 only; excludes cement energy emissions and non-CO2 GHG. Not a matched cement-sector GHG total.",
            "Emissions|CO2|Energy|Demand|Industry|Iron and Steel": "Iron and steel energy-demand CO2 only; not a reported all-GHG sector total.",
            "Emissions|CO2|Energy|Demand|Industry|Non-Ferrous Metals": "Broader non-ferrous-metals energy-demand CO2; neither aluminium-specific nor an all-GHG sector total.",
        }
        rows = []
        for (variable, unit), values in selected.groupby(["Variable", "Unit"], sort=True):
            explanation = ("Used as a component of the reported industry GHG definition; no matched material-sector GHG decomposition."
                           if variable in industry else explanations.get(variable,
                           "Not mapped to a comparable cement, iron-and-steel or aluminium GHG sector total."))
            rows.append(dict(Model="GCAM", Variable=variable, Unit=unit,
                             SourceFile=self.source_data.sources["GCAM"].name,
                             SourceSheet=self.config.get("gcam_sheet", "Sheet1"),
                             SourceRegions=json.dumps(sorted(values.Region.unique())),
                             SourceScenarios=json.dumps(sorted(values.Scenario.unique())),
                             EU27Available=bool(values.Region.eq("EU-27").any()),
                             SelectedForIndustry=variable in industry, UsedInThreeSector=False,
                             ComparableSectorGHG=False, Explanation=explanation))
        return pd.DataFrame(rows)
