"""Own-route production responses using the reported GCAM steel route split.

The existing material response figure deliberately retains its original total
steel denominator. This additional dataset instead compares each steel route
with the same route in its own model/pathway no-CE scenario.
"""

import json

import numpy as np
import pandas as pd

from .europe_comparison_data import CLIMATES, GCAM_PRODUCTION, MODELS, POLICIES, source_scenario
from .metrics import SECTORS


ROUTE_MATERIAL_METRIC = "Material production response by steel route"
PRODUCTION_GROUPS = ("Cement", "Primary steel", "Secondary steel", "Aluminium")


class RelativeRouteProductionDataMixin:
    """Additional production comparison; leaves the original dataset intact."""

    @property
    def steel_route_checks(self):
        """GCAM reported primary + secondary versus reported total, in Mt/yr."""
        return pd.DataFrame(getattr(self, "_steel_route_checks", [])).copy()

    def _validate_gcam_steel_routes(self):
        if getattr(self, "_steel_routes_validated", False):
            return
        checks = []
        parent = GCAM_PRODUCTION["steel"]
        for climate in CLIMATES:
            for policy in POLICIES:
                scenario = source_scenario("GCAM", climate, policy)
                total, primary, secondary = (
                    self.source_data._series("GCAM", scenario, variable, "Mt/yr")
                    for variable in (parent, parent + "|Primary", parent + "|Secondary")
                )
                for year in self.source_data.years["GCAM"]:
                    values = [float(series.loc[year]) for series in (total, primary, secondary)]
                    complete = bool(np.isfinite(values).all())
                    route_sum = values[1] + values[2] if complete else np.nan
                    difference = route_sum - values[0] if complete else np.nan
                    checks.append(dict(
                        Model="GCAM", Scenario=scenario, Climate=climate, Policy=policy,
                        Region="EU-27", Year=year, Unit="Mt/yr", Complete=complete,
                        ReportedTotal=values[0], Primary=values[1], Secondary=values[2],
                        SumRoutes=route_sum, Difference=difference,
                        SourceVariables=json.dumps([parent, parent + "|Primary", parent + "|Secondary"]),
                        SourceFile=self.source_data.gcam_source.name,
                        SourceSheet=self.config.get("gcam_sheet", "Sheet1"),
                        Check="Reported EU-27 primary plus secondary equals reported total steel",
                    ))
                    if complete and not np.isclose(route_sum, values[0], rtol=1e-8, atol=1e-8):
                        raise ValueError(
                            "GCAM primary + secondary steel differs from reported EU-27 total: "
                            f"{scenario} ({year})"
                        )
        self._steel_route_checks = checks
        self._steel_routes_validated = True

    def _route_production_missingness(self, model, scenario, reference_scenario, definition, unit,
                                     reference):
        """Distinguish an absent route reference from physical zero output."""
        variables, _ = definition
        missing = self.source_data.missing_inputs
        selected = missing.loc[missing.Model.eq(model) & missing.Unit.eq(unit)
                               & missing.Year.eq(self.comparison_year)
                               & missing.Variable.isin(variables)
                               & missing.Scenario.isin([scenario, reference_scenario])].copy()
        reference_missing = selected.loc[selected.Scenario.eq(reference_scenario)]
        scenario_missing = selected.loc[selected.Scenario.eq(scenario)]
        provisional = selected.loc[selected.Treatment.eq("zero")]
        reference_zeros = reference_missing.loc[reference_missing.Treatment.eq("zero")]
        if len(reference_missing):
            reference_issue = (
                "Missing no-CE production-group inputs; "
                + ("provisionally treated as zero under configured missing-activity policy."
                   if len(reference_zeros) else "missing values preserved.")
            )
        elif not np.isfinite(reference):
            reference_issue = "Nonfinite no-CE production-group output; percentage response is undefined."
        elif reference <= 0:
            reference_issue = "Nonpositive reported no-CE production-group output; percentage response is undefined."
        else:
            reference_issue = ""
        return dict(
            MissingInputCount=len(selected), UsesProvisionalZeros=bool(len(provisional)),
            ProvisionalZeroCount=len(provisional), MissingSourceInputs=selected.to_json(orient="records"),
            ProvisionalZeroInputs=provisional.to_json(orient="records"),
            ReferenceMissingInputCount=len(reference_missing),
            ReferenceUsesProvisionalZeros=bool(len(reference_zeros)),
            ReferenceComplete=not bool(len(reference_missing)), ReferenceIssue=reference_issue,
            ReferenceMissingInputs=reference_missing.to_json(orient="records"),
            ScenarioMissingInputCount=len(scenario_missing),
        )

    def material_response_by_steel_route(self):
        """Return own-group percentage changes for four production groups.

        Each steel route uses its own same-year no-CE output as denominator;
        these percentages therefore cannot be added to obtain total steel's
        percentage change. Cement and aluminium retain their existing totals.
        """
        cache_key = "material_by_steel_route"
        if cache_key in self._frames:
            return self._frames[cache_key].copy()
        self._validate_gcam_steel_routes()
        by_key = {sector.key: sector for sector in SECTORS}
        groups = (("Cement", by_key["cement"], "Total"),
                  ("Primary steel", by_key["steel"], "Primary"),
                  ("Secondary steel", by_key["steel"], "Secondary"),
                  ("Aluminium", by_key["aluminium"], "Total"))
        rows = []
        for model in MODELS:
            for climate in CLIMATES:
                reference_scenario = source_scenario(model, climate, "noce")
                for column, (group, sector, route) in enumerate(groups):
                    if sector.key == "steel":
                        parent = (GCAM_PRODUCTION["steel"] if model == "GCAM"
                                  else "Production|" + sector.path)
                        definition, unit = ([parent + "|" + route], [1]), "Mt/yr"
                    else:
                        _, definition, unit = self._definitions(model, sector, "material")
                    reference = self._value(model, reference_scenario, definition, unit)
                    for order, policy in enumerate(POLICIES[1:]):
                        common = self.source_data._common(model, climate, policy, sector, self.comparison_year)
                        common.update(
                            ReferenceScenario=reference_scenario, ReferenceYear=self.comparison_year,
                            Panel=chr(97 + CLIMATES.index(climate) * len(groups) + column),
                            BarType="future", BarOrder=order, ProductionGroup=group, Route=route,
                        )
                        value = self._value(model, common["Scenario"], definition, unit)
                        row = self._row(common, ROUTE_MATERIAL_METRIC, route, "net", value,
                                        reference, reference, definition, definition, unit)
                        row.update(
                            DenominatorScope="Same model, pathway, year and production group no-CE output",
                            Calculation="100 * (ScenarioValue - ReferenceValue) / ReferenceValue",
                            Interpretation="Own-route percentage change" if sector.key == "steel"
                            else "Total-sector percentage change",
                            Plotted=True,
                            ResponseDefined=bool(np.isfinite(row["Value"])),
                        )
                        row.update(self._route_production_missingness(
                            model, common["Scenario"], reference_scenario, definition, unit, reference))
                        rows.append(row)
        self._frames[cache_key] = pd.DataFrame(rows)
        return self._frames[cache_key].copy()
