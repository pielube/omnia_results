"""Independent examples for the added own-route production comparison."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from omnia_results.europe_comparison_data import (CLIMATES, GCAM_PRODUCTION, MODELS,
                                                  OMNIA_REGIONS, POLICIES, source_scenario)
from omnia_results.europe_relative_data import EuropeRelativeData
from omnia_results.europe_relative_routes_data import (PRODUCTION_GROUPS, ROUTE_MATERIAL_METRIC,
                                                      RelativeRouteProductionDataMixin)


class RouteData(EuropeRelativeData):
    # Exercise the isolated implementation both before and after its integration
    # into EuropeRelativeData; no alternate arithmetic is supplied by the test.
    material_response_by_steel_route = RelativeRouteProductionDataMixin.material_response_by_steel_route
    _validate_gcam_steel_routes = RelativeRouteProductionDataMixin._validate_gcam_steel_routes
    _route_production_missingness = RelativeRouteProductionDataMixin._route_production_missingness
    steel_route_checks = RelativeRouteProductionDataMixin.steel_route_checks


class EuropeRelativeRouteDataTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(input="omnia.csv", gcam_input="gcam.xlsx", missing_activity="preserve",
                           historical_year=2019, gcam_historical_year=2021, comparison_year=2050)
        omnia, gcam = [], []

        def add(model, scenario, variable, values):
            if model == MODELS[0]:
                for region, weight in zip((*OMNIA_REGIONS, "ENW"), (.2, .3, .5, 10000)):
                    omnia.append([scenario, region, variable, "Mt/yr", values * weight / 2, values * weight])
            else:
                for region, weight in (("EU-27", 1), ("Austria", .4), ("Belgium", .6)):
                    gcam.append(["GCAM Europe8.7", scenario, region, variable, "Mt/yr",
                                 values * weight / 2, values * weight])

        for model in MODELS:
            for climate in CLIMATES:
                scale = (1 if climate == "ndc" else 2) * (1 if model == MODELS[0] else 5)
                for p, policy in enumerate(POLICIES):
                    scenario = source_scenario(model, climate, policy)
                    steel_parent = (GCAM_PRODUCTION["steel"] if model == "GCAM"
                                    else "Production|Iron and Steel")
                    primary, secondary = (80, 60, 40)[p], (20, 35, 50)[p]
                    for route, amount in (("Primary", primary), ("Secondary", secondary)):
                        add(model, scenario, steel_parent + "|" + route, amount * scale)
                    if model == "GCAM":
                        add(model, scenario, steel_parent, (primary + secondary) * scale)
                    add(model, scenario, GCAM_PRODUCTION["cement"], (100, 90, 80)[p] * scale)
                    aluminium = GCAM_PRODUCTION["aluminium"]
                    if model == "GCAM":
                        add(model, scenario, aluminium, (30, 28, 25)[p] * scale)
                    else:
                        add(model, scenario, aluminium + "|Primary", (20, 15, 10)[p] * scale)
                        add(model, scenario, aluminium + "|Secondary", (10, 13, 15)[p] * scale)
        self.omnia = pd.DataFrame(omnia, columns=["Scenario", "Region", "Variable", "Unit", "2019", "2050"])
        self.gcam = pd.DataFrame(gcam, columns=["Model", "Scenario", "Region", "Variable", "Unit", "2021", "2050"])

    def data(self, omnia=None, gcam=None, **settings):
        with patch("omnia_results.europe_comparison_data.pd.read_csv", return_value=(self.omnia if omnia is None else omnia).copy()), \
             patch("omnia_results.europe_comparison_data.pd.read_excel", return_value=(self.gcam if gcam is None else gcam).copy()):
            return RouteData({**self.config, **settings}, Path("."))

    @staticmethod
    def select(frame, group, model="GCAM", climate="ndc", policy="medce_eu"):
        return frame.loc[frame.ProductionGroup.eq(group) & frame.Model.eq(model)
                         & frame.Climate.eq(climate) & frame.Policy.eq(policy)].iloc[0]

    def test_each_steel_route_uses_its_own_reference_not_total_or_route_share(self):
        frame = self.data().material_response_by_steel_route()
        for model in MODELS:
            primary = self.select(frame, "Primary steel", model)
            secondary = self.select(frame, "Secondary steel", model)
            self.assertEqual(primary.Value, -25)
            self.assertEqual(secondary.Value, 75)
            self.assertEqual(primary.Denominator, 80 * (5 if model == "GCAM" else 1))
            self.assertEqual(secondary.Denominator, 20 * (5 if model == "GCAM" else 1))
            self.assertNotAlmostEqual(secondary.Value, 100 * 35 / 95)  # The CE route share.
            self.assertNotAlmostEqual(secondary.Value, 15)  # Total-steel referenced pp contribution.
            self.assertEqual((primary.Unit, secondary.Unit), ("%", "%"))
        self.assertEqual(len(frame), 32)
        self.assertEqual(set(frame.Metric), {ROUTE_MATERIAL_METRIC})
        self.assertEqual(set(frame.Role), {"net"})
        self.assertEqual(set(frame.ProductionGroup), set(PRODUCTION_GROUPS))
        self.assertEqual(set(frame.Panel), set("abcdefgh"))
        self.assertTrue(frame.Plotted.all())

    def test_sources_and_denominators_keep_own_model_climate_route_and_year(self):
        frame = self.data().material_response_by_steel_route()
        for model in MODELS:
            for climate in CLIMATES:
                for route in ("Primary", "Secondary"):
                    row = self.select(frame, route + " steel", model, climate)
                    parent = "Production|Iron and Steel" + ("|Steel" if model == "GCAM" else "")
                    variables = [parent + "|" + route]
                    self.assertEqual(row.Route, route)
                    self.assertEqual(row.Sector, "Iron and steel")
                    self.assertEqual(json.loads(row.SourceVariables), variables)
                    self.assertEqual(json.loads(row.DenominatorSourceVariables), variables)
                    self.assertEqual(json.loads(row.NumeratorSourceCoefficients), [1, -1])
                    self.assertEqual(json.loads(row.SourceUnits), ["Mt/yr"])
                    self.assertEqual(row.ReferenceScenario, source_scenario(model, climate, "noce"))
                    self.assertEqual(row.DenominatorSourceScenario, row.ReferenceScenario)
                    self.assertEqual(row.DenominatorSourceYear, 2050)
                    self.assertEqual(row.ReferenceYear, row.Year)
                    regions = ["EU-27"] if model == "GCAM" else list(OMNIA_REGIONS)
                    self.assertEqual(json.loads(row.SourceRegions), regions)
                    self.assertEqual(json.loads(row.DenominatorSourceRegions), regions)

    def test_cement_and_aluminium_equal_original_total_responses(self):
        data = self.data()
        original = data.material_response()
        additional = data.material_response_by_steel_route()
        for _, row in additional.loc[additional.ProductionGroup.isin(["Cement", "Aluminium"])].iterrows():
            prior = original.loc[original.Model.eq(row.Model) & original.Climate.eq(row.Climate)
                                 & original.Policy.eq(row.Policy) & original.Sector.eq(row.Sector)
                                 & original.Role.eq("net")].iloc[0]
            for field in ("Value", "ScenarioValue", "ReferenceValue", "Denominator", "SourceVariables",
                          "DenominatorSourceVariables", "Numerator", "ReferenceScenario"):
                self.assertEqual(row[field], prior[field])
            self.assertEqual(row.Route, "Total")

    def test_reported_gcam_routes_close_for_all_selected_scenarios_and_years(self):
        data = self.data()
        data.material_response_by_steel_route()
        checks = data.steel_route_checks
        self.assertEqual(len(checks), 12)
        self.assertTrue(checks.Complete.all())
        self.assertEqual(set(checks.Year), {2021, 2050})
        self.assertEqual(set(checks.Scenario), set(self.gcam.Scenario))
        self.assertEqual(set(checks.Region), {"EU-27"})
        self.assertLess(checks.Difference.abs().max(), 1e-10)
        self.assertTrue(np.allclose(checks.Primary + checks.Secondary, checks.ReportedTotal))

    def test_inconsistent_reported_total_is_rejected_even_outside_comparison_year(self):
        gcam = self.gcam.copy()
        mask = gcam.Scenario.eq("LTT_HighCE_EU") & gcam.Variable.eq(GCAM_PRODUCTION["steel"])
        gcam.loc[mask, "2021"] *= 2  # Keep country aggregation valid; break route closure only.
        with self.assertRaisesRegex(ValueError, r"primary \+ secondary steel differs.*2021"):
            self.data(gcam=gcam).material_response_by_steel_route()

    def test_blank_route_reference_remains_gap_under_zero_policy(self):
        gcam = self.gcam.copy()
        mask = gcam.Scenario.eq("NDC_NoCE") & gcam.Variable.eq(GCAM_PRODUCTION["steel"] + "|Secondary")
        gcam.loc[mask & gcam.Region.eq("EU-27"), "2050"] = np.nan
        data = self.data(gcam=gcam, missing_activity="zero")
        frame = data.material_response_by_steel_route()
        self.assertTrue(np.isnan(self.select(frame, "Secondary steel").Value))
        self.assertEqual(self.select(frame, "Primary steel").Value, -25)
        check = data.steel_route_checks.loc[data.steel_route_checks.Scenario.eq("NDC_NoCE")
                                           & data.steel_route_checks.Year.eq(2050)].iloc[0]
        self.assertFalse(check.Complete)
        self.assertTrue(np.isnan(check.SumRoutes))
        self.assertEqual(set(data.source_data.missing_inputs.Treatment), {"preserve"})

    def test_nonpositive_route_reference_gives_gap_without_affecting_other_route(self):
        for reference in (0, -5):
            gcam = self.gcam.copy()
            mask = gcam.Scenario.eq("NDC_NoCE")
            weights = gcam.Region.map({"EU-27": 1, "Austria": .4, "Belgium": .6})
            primary = mask & gcam.Variable.eq(GCAM_PRODUCTION["steel"] + "|Primary")
            total = mask & gcam.Variable.eq(GCAM_PRODUCTION["steel"])
            gcam.loc[primary, "2050"] = reference * weights[primary]
            gcam.loc[total, "2050"] = (reference + 100) * weights[total]
            frame = self.data(gcam=gcam).material_response_by_steel_route()
            self.assertTrue(np.isnan(self.select(frame, "Primary steel").Value))
            self.assertFalse(self.select(frame, "Primary steel").ResponseDefined)
            self.assertIn("Nonpositive reported", self.select(frame, "Primary steel").ReferenceIssue)
            self.assertEqual(self.select(frame, "Primary steel").ReferenceMissingInputCount, 0)
            self.assertEqual(self.select(frame, "Secondary steel").Value, 75)

    def test_missing_omnia_route_reference_is_identified_under_zero_and_preserve(self):
        omnia = self.omnia.copy()
        mask = (omnia.Scenario.eq("baseline_noce") & omnia.Region.isin(OMNIA_REGIONS)
                & omnia.Variable.eq("Production|Iron and Steel|Secondary"))
        omnia.loc[mask, "2050"] = np.nan
        for treatment in ("zero", "preserve"):
            frame = self.data(omnia=omnia, missing_activity=treatment).material_response_by_steel_route()
            row = self.select(frame, "Secondary steel", MODELS[0], "baseline")
            self.assertTrue(np.isnan(row.Value))
            self.assertFalse(row.ResponseDefined)
            self.assertFalse(row.ReferenceComplete)
            self.assertEqual(row.ReferenceMissingInputCount, 3)
            self.assertEqual(row.MissingInputCount, 3)
            self.assertEqual(row.ScenarioMissingInputCount, 0)
            self.assertEqual(row.ReferenceUsesProvisionalZeros, treatment == "zero")
            self.assertEqual(row.ProvisionalZeroCount, 3 if treatment == "zero" else 0)
            self.assertIn("Missing no-CE production-group inputs", row.ReferenceIssue)
            self.assertEqual({item["Region"] for item in json.loads(row.ReferenceMissingInputs)},
                             set(OMNIA_REGIONS))
            if treatment == "zero":
                self.assertEqual(row.ReferenceValue, 0)
                self.assertIn("provisionally treated as zero", row.ReferenceIssue)
            else:
                self.assertTrue(np.isnan(row.ReferenceValue))
                self.assertIn("missing values preserved", row.ReferenceIssue)
            self.assertTrue(self.select(frame, "Primary steel", MODELS[0], "baseline").ResponseDefined)

    def test_absent_country_route_does_not_fabricate_zero_eu_aggregate(self):
        gcam = self.gcam.copy()
        # Belgium has no secondary production row. The reported EU total agrees
        # with Austria's reported route, rather than an inferred Belgium value.
        variable = GCAM_PRODUCTION["steel"] + "|Secondary"
        secondary = gcam.Variable.eq(variable)
        keep = ~(secondary & gcam.Region.eq("Belgium"))
        for year in ("2021", "2050"):
            gcam.loc[secondary & gcam.Region.eq("Austria"), year] *= 2.5
        data = self.data(gcam=gcam.loc[keep])
        self.assertEqual(self.select(data.material_response_by_steel_route(), "Secondary steel").Value, 75)
        absent_eu = gcam.loc[~(secondary & gcam.Region.eq("EU-27"))]
        with self.assertRaisesRegex(ValueError, "EU-27 aggregate absent"):
            self.data(gcam=absent_eu).material_response_by_steel_route()

    def test_cached_results_and_checks_are_copies(self):
        data = self.data()
        frame = data.material_response_by_steel_route()
        frame.loc[:, "Value"] = 9999
        checks = data.steel_route_checks
        checks.loc[:, "Difference"] = 9999
        self.assertFalse(data.material_response_by_steel_route().Value.eq(9999).any())
        self.assertFalse(data.steel_route_checks.Difference.eq(9999).any())
        self.assertEqual(len(data.steel_route_checks), 12)


if __name__ == "__main__":
    unittest.main()
