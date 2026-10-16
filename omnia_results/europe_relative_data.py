"""EU-only CE responses relative to each model's own no-CE sector total.

Component changes use the total reference denominator, so their signed
percentage-point contributions reconstruct the net percentage response. They
are not percentage changes within the component itself.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .europe_comparison_data import (CLIMATES, GCAM_FUELS, GCAM_PRODUCTION,
                                     MODELS, POLICIES, EuropeComparisonData,
                                     source_scenario)
from .metrics import SECTORS
from .europe_relative_emissions_data import RelativeEmissionsDataMixin


MATERIAL_METRIC = "Material production response"
ENERGY_METRIC = "Final energy response"
RESPONSE_POLICIES = POLICIES[1:]


def _contribution(change, total_reference):
    """Return a gap for an unavailable or nonpositive reference total."""
    if not np.isfinite(change) or not np.isfinite(total_reference) or total_reference <= 0:
        return np.nan
    return 100 * change / total_reference


class EuropeRelativeData(RelativeEmissionsDataMixin):
    """Read the supplied raw sources through the existing validated adapter."""

    models, climates, policies = MODELS, CLIMATES, RESPONSE_POLICIES

    def __init__(self, config: dict, base: Path):
        self.config, self.base = config, Path(base)
        self.source_data = EuropeComparisonData(config, self.base)
        self.comparison_year = self.source_data.comparison_year
        self._frames, self._closure_checks = {}, []
        self.notes = [
            "Every response uses the same model, climate pathway, sector and comparison-year no-CE total as its denominator.",
            "Component values are signed percentage-point contributions: 100 times the component CE-minus-no-CE change divided by the no-CE sector total. Net values use the same calculation for the sector total and are percentages.",
            "GCAM metal output has no exported primary/secondary route split. Its single reported-metal contribution does not establish equivalent route coverage across models.",
            "GCAM aluminium energy retains an Unallocated contribution from reported sector total minus reported carrier parents; no process or carrier attribution is assumed.",
            "Incomplete inputs retain the source adapter's explicit missing-activity treatment. Nonfinite or nonpositive no-CE sector totals produce gaps for every contribution and net response.",
        ]

    @property
    def closure_checks(self):
        return pd.DataFrame(self._closure_checks)

    @staticmethod
    def _definitions(model, sector, kind):
        """Return exact component and total variable/coefficient definitions."""
        if kind == "material":
            if model == "GCAM":
                total = ([GCAM_PRODUCTION[sector.key]], [1])
                label = "Cement" if sector.key == "cement" else "Reported metal"
                return {label: total}, total, "Mt/yr"
            parent = f"Production|{sector.path}"
            if sector.key == "cement":
                total = ([parent], [1])
                return {"Cement": total}, total, "Mt/yr"
            components = {"Primary": ([parent + "|Primary"], [1]),
                          "Secondary": ([parent + "|Secondary"], [1])}
            return components, ([parent + "|Primary", parent + "|Secondary"], [1, 1]), "Mt/yr"

        parent = f"Final Energy|Industry|{sector.path}"
        fuels = sector.fuels if model == MODELS[0] else GCAM_FUELS[sector.key]
        carriers = [parent + "|" + fuel for fuel in fuels]
        components = {fuel: ([variable], [1]) for fuel, variable in zip(fuels, carriers)}
        if model == "GCAM":
            components["Unallocated"] = ([parent] + carriers, [1] + [-1] * len(carriers))
            total = ([parent], [1])
        else:
            total = (carriers, [1] * len(carriers))
        return components, total, "EJ/yr"

    def _value(self, model, scenario, definition, unit):
        variables, coefficients = definition
        return sum(coefficient * self.source_data._series(model, scenario, variable, unit).loc[self.comparison_year]
                   for variable, coefficient in zip(variables, coefficients))

    def _row(self, common, metric, series, role, value, reference, denominator, definition,
             total_definition, source_unit):
        variables, coefficients = definition
        total_variables, total_coefficients = total_definition
        change = value - reference
        scenario, reference_scenario = common["Scenario"], common["ReferenceScenario"]
        return {
            **common, "Metric": metric, "Series": series, "Role": role,
            "Value": _contribution(change, denominator),
            "Unit": "percentage points" if role == "component" else "%",
            "Numerator": change, "Denominator": denominator,
            "NumeratorUnit": source_unit, "DenominatorUnit": source_unit,
            "ConversionFactor": 100, "Offset": 0,
            "ScenarioValue": value, "ReferenceValue": reference,
            "ComponentChange": change, "TotalReferenceValue": denominator,
            "SourceVariables": json.dumps(variables), "SourceUnits": json.dumps([source_unit] * len(variables)),
            "SourceCoefficients": json.dumps(coefficients),
            "ReferenceSourceVariables": json.dumps(variables),
            "ReferenceSourceUnits": json.dumps([source_unit] * len(variables)),
            "ReferenceSourceCoefficients": json.dumps(coefficients),
            "ReferenceSourceRegions": common["SourceRegions"],
            "NumeratorSourceVariables": json.dumps(variables + variables),
            "NumeratorSourceUnits": json.dumps([source_unit] * (2 * len(variables))),
            "NumeratorSourceCoefficients": json.dumps(coefficients + [-coefficient for coefficient in coefficients]),
            "NumeratorSourceScenarios": json.dumps([scenario] * len(variables) + [reference_scenario] * len(variables)),
            "DenominatorSourceVariables": json.dumps(total_variables),
            "DenominatorSourceUnits": json.dumps([source_unit] * len(total_variables)),
            "DenominatorSourceCoefficients": json.dumps(total_coefficients),
            "DenominatorSourceRegions": common["SourceRegions"],
            "DenominatorSourceScenario": reference_scenario,
            "DenominatorSourceYear": common["ReferenceYear"],
            "DenominatorSourceValue": denominator,
            "DenominatorValue": denominator,
            "Calculation": "100 * (ScenarioValue - ReferenceValue) / TotalReferenceValue",
        }

    def _response(self, kind):
        if kind in self._frames:
            return self._frames[kind].copy()
        metric = MATERIAL_METRIC if kind == "material" else ENERGY_METRIC
        rows = []
        for model in MODELS:
            for climate in CLIMATES:
                reference_scenario = source_scenario(model, climate, "noce")
                for column, sector in enumerate(SECTORS):
                    components, total_definition, unit = self._definitions(model, sector, kind)
                    total_reference = self._value(model, reference_scenario, total_definition, unit)
                    reference_components = {series: self._value(model, reference_scenario, definition, unit)
                                            for series, definition in components.items()}
                    for policy in RESPONSE_POLICIES:
                        common = self.source_data._common(model, climate, policy, sector, self.comparison_year)
                        common.update(ReferenceScenario=reference_scenario, ReferenceYear=self.comparison_year,
                                      Panel=chr(97 + CLIMATES.index(climate) * len(SECTORS) + column),
                                      BarType="future", BarOrder=RESPONSE_POLICIES.index(policy))
                        component_rows = []
                        for series, definition in components.items():
                            value = self._value(model, common["Scenario"], definition, unit)
                            component_rows.append(self._row(common, metric, series, "component", value,
                                                            reference_components[series], total_reference,
                                                            definition, total_definition, unit))
                        total_value = self._value(model, common["Scenario"], total_definition, unit)
                        net = self._row(common, metric, "Total", "net", total_value, total_reference,
                                        total_reference, total_definition, total_definition, unit)
                        contribution_values = np.array([row["Value"] for row in component_rows], dtype=float)
                        complete = bool(np.isfinite(contribution_values).all() and np.isfinite(net["Value"]))
                        summed = float(contribution_values.sum()) if complete else np.nan
                        difference = summed - net["Value"] if complete else np.nan
                        self._closure_checks.append({**common, "Metric": metric,
                                                     "Complete": complete, "ComponentCount": len(component_rows),
                                                     "SumContributions": summed, "NetResponse": net["Value"],
                                                     "Difference": difference, "Unit": "percentage points"})
                        if complete and not np.isclose(summed, net["Value"], rtol=1e-10, atol=1e-10):
                            raise ValueError(f"Component contributions do not reconstruct {model} {climate} {policy} {sector.label} {kind} response")
                        rows.extend(component_rows)
                        rows.append(net)
        self._frames[kind] = pd.DataFrame(rows)
        return self._frames[kind].copy()

    def material_response(self):
        return self._response("material")

    def energy_response(self):
        return self._response("energy")
