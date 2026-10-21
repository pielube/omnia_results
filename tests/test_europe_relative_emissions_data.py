"""Independent examples for three-sector GHG/CO2 responses and industry context."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from omnia_results.europe_comparison_data import (CLIMATES, MODELS, OMNIA_REGIONS,
                                                  POLICIES, source_scenario)
from omnia_results.europe_relative_data import EuropeRelativeData
from omnia_results.europe_relative_emissions_data import (ALUMINIUM_GHG, INDUSTRY_METRIC,
                                                         THREE_SECTOR_METRIC,
                                                         RelativeEmissionsDataMixin)
from omnia_results.metrics import SECTORS


class EmissionsHarness(EuropeRelativeData, RelativeEmissionsDataMixin):
    pass


class EuropeRelativeEmissionsDataTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(input="omnia.csv", gcam_input="gcam.xlsx", missing_activity="preserve",
                           historical_year=2019, gcam_historical_year=2021, comparison_year=2050)
        omnia, gcam = [], []

        def add(model, scenario, variable, unit, value):
            if model == MODELS[0]:
                for region, weight in zip((*OMNIA_REGIONS, "ENW"), (1, 2, 3, 10000)):
                    omnia.append([scenario, region, variable, unit, 99999 * weight, value * weight])
            else:
                for region, weight in (("EU-27", 1), ("Austria", .4), ("Belgium", .6)):
                    gcam.append(["GCAM Europe8.7", scenario, region, variable, unit, 99999 * weight, value * weight])

        for model in MODELS:
            for climate in CLIMATES:
                scale = (1 if climate == "ndc" else 2) * (1 if model == MODELS[0] else 5)
                for p, policy in enumerate(POLICIES):
                    scenario = source_scenario(model, climate, policy)
                    if model == MODELS[0]:
                        add(model, scenario, "Emissions|GHG|Industry", "MtCO2e/yr", (100, 90, 80)[p] * scale)
                        for sector, amounts in zip(SECTORS, ((30, 27, 24), (-10, -5, -20), (20, 16, 10))):
                            add(model, scenario, "Emissions|GHG|Industry|" + sector.path, "MtCO2e/yr", amounts[p] * scale)
                            add(model, scenario, "Emissions|GHG|Industry|" + sector.path, "MtCO2e/Mt", 99999)
                            add(model, scenario, "Carbon Capture|Industry|" + sector.path, "MtCO2/yr", 99999)
                    else:
                        add(model, scenario, "Emissions|Kyoto Gases|Energy|Demand|Industry", "Mt CO2-equiv/yr", (70, 63, 56)[p] * scale)
                        add(model, scenario, "Emissions|Kyoto Gases|Industrial Processes", "Mt CO2-equiv/yr", (30, 24, 21)[p] * scale)
                        for variable, amounts in (
                            ("Emissions|CO2|Energy|Demand|Industry|Non-Metallic Minerals", (10, 8, 6)),
                            ("Emissions|CO2|Industrial Processes|Non-Metallic Minerals|Cement", (20, 16, 12)),
                            ("Emissions|CO2|Energy|Demand|Industry|Iron and Steel", (15, 14, 10)),
                            ("Emissions|CO2|Energy|Demand|Industry|Non-Ferrous Metals|Aluminum", (5, 4, 3)),
                        ):
                            add(model, scenario, variable, "Mt CO2/yr", amounts[p] * scale)
                        for variable in ("Final Energy|Industry|Non-Metallic Minerals",
                                         "Final Energy|Industry|Non-Metallic Minerals|Cement"):
                            add(model, scenario, variable, "EJ/yr", 2 * scale)
                        for variable in ("Emissions|CO2|Industrial Processes|Cement",
                                         "Emissions|CO2|Industrial Processes|Non-Metallic Minerals",
                                         "Emissions|CO2|Energy|Demand|Industry",
                                         "Emissions|CO2|Energy|Demand|Industry|Non-Ferrous Metals"):
                            add(model, scenario, variable, "Mt CO2/yr", (99999, 0, 99999)[p])
                        add(model, scenario, "Carbon Capture|Energy|Demand|Industry", "Mt CO2/yr", 99999)
        self.omnia = pd.DataFrame(omnia, columns=["Scenario", "Region", "Variable", "Unit", "2019", "2050"])
        self.gcam = pd.DataFrame(gcam, columns=["Model", "Scenario", "Region", "Variable", "Unit", "2021", "2050"])

    def data(self, omnia=None, gcam=None, **settings):
        with patch("omnia_results.europe_comparison_data.pd.read_csv", return_value=(self.omnia if omnia is None else omnia).copy()), \
             patch("omnia_results.europe_comparison_data.pd.read_excel", return_value=(self.gcam if gcam is None else gcam).copy()):
            return EmissionsHarness({**self.config, **settings}, Path("."))

    @staticmethod
    def select(frame, panel_kind, series="Total", model=MODELS[0], climate="ndc", policy="medce_eu"):
        return frame.loc[(frame.Model == model) & (frame.Climate == climate) & (frame.Policy == policy)
                         & (frame.PanelKind == panel_kind) & (frame.Series == series)].iloc[0]

    def test_industry_nets_match_the_existing_adapter_and_keep_own_future_references(self):
        data = self.data()
        frame = data.emissions_response()
        previous = data.source_data.emissions_response()
        for model in MODELS:
            for climate in CLIMATES:
                for policy in POLICIES[1:]:
                    row = self.select(frame, "industry", model=model, climate=climate, policy=policy)
                    old = previous.loc[(previous.Model == model) & (previous.Climate == climate)
                                       & (previous.Policy == policy)
                                       & previous.Metric.eq("Industrial GHG emissions change from no CE")].iloc[0]
                    self.assertAlmostEqual(row.Value, old.Value)
                    self.assertEqual(row.ReferenceScenario, source_scenario(model, climate, "noce"))
                    self.assertEqual(row.ReferenceYear, 2050)
                    self.assertEqual(row.Unit, "%")
        self.assertEqual(self.select(frame, "industry").Denominator, 600)
        self.assertEqual(self.select(frame, "industry", climate="baseline").Denominator, 1200)
        self.assertEqual(self.select(frame, "industry", model="GCAM").Denominator, 500)
        self.assertEqual(set(frame.Metric), {INDUSTRY_METRIC, THREE_SECTOR_METRIC})
        self.assertEqual(set(frame.Policy), {"medce_eu", "highce_eu"})
        self.assertEqual(set(frame.Year), {2050})

    def test_focused_sector_contributions_use_three_sector_total_instead_of_whole_industry(self):
        frame = self.data().emissions_response()
        cement = self.select(frame, "three_sector", "Cement")
        steel = self.select(frame, "three_sector", "Iron and steel")
        aluminium = self.select(frame, "three_sector", "Aluminium")
        net = self.select(frame, "three_sector")
        self.assertEqual(cement.Denominator, 240)  # EU sum: (30 - 10 + 20) × 6.
        self.assertAlmostEqual(cement.Value, -7.5)
        self.assertAlmostEqual(steel.Value, 12.5)
        self.assertAlmostEqual(aluminium.Value, -10)
        self.assertAlmostEqual(net.Value, -5)
        self.assertAlmostEqual(self.select(frame, "industry").Value, -10)
        self.assertEqual(steel.ReferenceValue, -60)
        self.assertEqual(steel.ScenarioValue, -30)
        self.assertTrue(np.isfinite(steel.Value))
        self.assertEqual(cement.Unit, "percentage points")
        self.assertEqual(len(json.loads(cement.DenominatorSourceVariables)), 3)
        self.assertNotIn("Emissions|GHG|Industry", json.loads(cement.DenominatorSourceVariables))
        self.assertEqual(set(frame.loc[frame.PanelKind.eq("three_sector"), "Model"]), set(MODELS))
        self.assertEqual(set(frame.loc[frame.PanelKind.eq("three_sector"), "Panel"]), {"a", "b"})

    def test_gcam_aggregate_combines_selected_sector_co2_with_its_own_total_reference(self):
        frame = self.data().emissions_response()
        medium = self.select(frame, "three_sector", model="GCAM")
        high = self.select(frame, "three_sector", model="GCAM", policy="highce_eu")
        self.assertEqual(medium.Denominator, 250)
        self.assertEqual(medium.ReferenceValue, 250)
        self.assertEqual(medium.ScenarioValue, 210)
        self.assertAlmostEqual(medium.Value, -16)
        self.assertAlmostEqual(high.Value, -38)
        cement = self.select(frame, "three_sector", "Cement", model="GCAM")
        self.assertEqual(cement.ReferenceValue, 150)
        self.assertEqual(cement.ScenarioValue, 120)
        self.assertAlmostEqual(cement.Value, -12)
        self.assertEqual(json.loads(cement.SourceVariables), [
            "Emissions|CO2|Energy|Demand|Industry|Non-Metallic Minerals",
            "Emissions|CO2|Industrial Processes|Non-Metallic Minerals|Cement",
        ])
        self.assertEqual(json.loads(cement.SourceUnits), ["Mt CO2/yr", "Mt CO2/yr"])
        self.assertEqual(len(json.loads(medium.SourceVariables)), 4)
        self.assertIn("CO2", medium.EmissionsCoverage)
        self.assertIn("GHG", self.select(frame, "three_sector").EmissionsCoverage)
        self.assertNotEqual(medium.Denominator, self.select(frame, "industry", model="GCAM").Denominator)

    def test_gcam_cement_energy_proxy_requires_parent_child_scope_agreement(self):
        gcam = self.gcam.copy()
        for region, change in (("EU-27", .5), ("Austria", .2), ("Belgium", .3)):
            mask = (gcam.Scenario.eq("NDC_MedCE_EU") & gcam.Region.eq(region)
                    & gcam.Variable.eq("Final Energy|Industry|Non-Metallic Minerals"))
            gcam.loc[mask, "2050"] += change
        with self.assertRaisesRegex(ValueError, "cement energy"):
            self.data(gcam=gcam).emissions_response()

    def test_signed_components_close_to_net_and_industry_components_are_context(self):
        data = self.data()
        frame = data.emissions_response()
        for _, case in frame.groupby(["Model", "Climate", "Policy", "PanelKind"]):
            self.assertAlmostEqual(case.loc[case.Role.eq("component"), "Value"].sum(),
                                   case.loc[case.Role.eq("net"), "Value"].iloc[0])
        focused_high = self.select(frame, "three_sector", policy="highce_eu")
        self.assertAlmostEqual(focused_high.Value, -65)
        industry_components = frame.loc[frame.PanelKind.eq("industry") & frame.Role.eq("component")]
        self.assertTrue(industry_components.ContextOnly.all())
        self.assertEqual(set(industry_components.DisplayRole), {"context"})
        focused_components = frame.loc[frame.PanelKind.eq("three_sector") & frame.Role.eq("component")]
        self.assertFalse(focused_components.ContextOnly.any())
        self.assertTrue(data.closure_checks.Complete.all())
        self.assertLess(data.closure_checks.Difference.abs().max(), 1e-10)

    def test_broader_parents_and_capture_do_not_enter_three_sector_totals(self):
        frame = self.data().emissions_response()
        source_variables = {variable for values in frame.SourceVariables for variable in json.loads(values)}
        self.assertFalse(any(variable.startswith("Carbon Capture|") for variable in source_variables))
        selected = json.loads(self.select(frame, "three_sector", model="GCAM").SourceVariables)
        self.assertEqual(set(selected), {
            "Emissions|CO2|Energy|Demand|Industry|Non-Metallic Minerals",
            "Emissions|CO2|Industrial Processes|Non-Metallic Minerals|Cement",
            "Emissions|CO2|Energy|Demand|Industry|Iron and Steel",
            "Emissions|CO2|Energy|Demand|Industry|Non-Ferrous Metals|Aluminum",
        })
        self.assertNotIn("Emissions|CO2|Industrial Processes|Non-Metallic Minerals", selected)
        self.assertNotIn("Emissions|CO2|Energy|Demand|Industry|Non-Ferrous Metals", selected)
        self.assertNotIn("Emissions|CO2|Industrial Processes|Cement", selected)
        self.assertNotIn("Emissions|CO2|Energy|Demand|Industry", selected)
        self.assertEqual(json.loads(self.select(frame, "three_sector", model="GCAM").SourceCoefficients), [1, 1, 1, 1])
        omnia = self.select(frame, "three_sector")
        self.assertTrue(all(variable.startswith("Emissions|GHG|") for variable in json.loads(omnia.SourceVariables)))
        gc = self.select(frame, "industry", model="GCAM")
        self.assertAlmostEqual(gc.Value, -13)
        self.assertEqual(json.loads(gc.SourceVariables), ["Emissions|Kyoto Gases|Energy|Demand|Industry",
                                                         "Emissions|Kyoto Gases|Industrial Processes"])
        self.assertEqual(json.loads(gc.SourceUnits), ["Mt CO2-equiv/yr", "Mt CO2-equiv/yr"])
        self.assertIn("boundaries differ", gc.EmissionsCoverage)

    def test_missing_aluminium_zero_policy_is_explicit_and_changes_focused_result(self):
        omnia = self.omnia.copy()
        mask = omnia.Scenario.str.startswith("baseline_") & omnia.Region.isin(OMNIA_REGIONS) & omnia.Variable.eq(ALUMINIUM_GHG) & omnia.Unit.eq("MtCO2e/yr")
        omnia.loc[mask, "2050"] = np.nan
        zero_data = self.data(omnia=omnia, missing_activity="zero")
        zero = zero_data.emissions_response()
        row = self.select(zero, "three_sector", climate="baseline")
        self.assertAlmostEqual(row.Value, 10)
        self.assertTrue(row.UsesProvisionalZeros)
        self.assertTrue(row.AllEUAluminiumMissingAsZero)
        self.assertEqual(row.ProvisionalZeroCount, 6)
        self.assertEqual(json.loads(row.MissingAluminiumScenarios), ["baseline_medce_eu", "baseline_noce"])
        self.assertIn("All three OMNIA EU regions", row.ProvisionalZeroNote)
        aluminium = self.select(zero, "three_sector", "Aluminium", climate="baseline")
        self.assertEqual((aluminium.ScenarioValue, aluminium.ReferenceValue, aluminium.Value), (0, 0, 0))
        preserve = self.data(omnia=omnia, missing_activity="preserve").emissions_response()
        focused = preserve.loc[preserve.PanelKind.eq("three_sector") & preserve.Model.eq(MODELS[0])
                               & preserve.Climate.eq("baseline")]
        self.assertTrue(focused.Value.isna().all())
        self.assertFalse(focused.UsesProvisionalZeros.any())
        self.assertTrue(np.isfinite(self.select(preserve, "industry", climate="baseline").Value))

    def test_missing_ce_component_does_not_create_partial_total(self):
        omnia = self.omnia.copy()
        mask = omnia.Scenario.eq("ndc_medce_eu") & omnia.Region.eq("EUM") & omnia.Variable.eq(ALUMINIUM_GHG) & omnia.Unit.eq("MtCO2e/yr")
        omnia.loc[mask, "2050"] = np.nan
        data = self.data(omnia=omnia)
        frame = data.emissions_response()
        self.assertTrue(np.isnan(self.select(frame, "three_sector").Value))
        self.assertTrue(np.isnan(self.select(frame, "three_sector", "Aluminium").Value))
        self.assertTrue(np.isfinite(self.select(frame, "three_sector", "Cement").Value))
        check = data.closure_checks.loc[data.closure_checks.PanelKind.eq("three_sector")
                                        & data.closure_checks.Model.eq(MODELS[0])
                                        & data.closure_checks.Climate.eq("ndc")
                                        & data.closure_checks.Policy.eq("medce_eu")].iloc[0]
        self.assertFalse(check.Complete)

    def test_missing_or_nonpositive_focused_total_reference_gives_all_gaps(self):
        for amounts in ((10, -20, 10), (5, -20, 5), (np.nan, -10, 20)):
            omnia = self.omnia.copy()
            for sector, amount in zip(SECTORS, amounts):
                variable = "Emissions|GHG|Industry|" + sector.path
                for region, weight in zip(OMNIA_REGIONS, (1, 2, 3)):
                    mask = omnia.Scenario.eq("ndc_noce") & omnia.Region.eq(region) & omnia.Variable.eq(variable) & omnia.Unit.eq("MtCO2e/yr")
                    omnia.loc[mask, "2050"] = amount * weight
            frame = self.data(omnia=omnia).emissions_response()
            focused = frame.loc[frame.PanelKind.eq("three_sector") & frame.Model.eq(MODELS[0])
                                & frame.Climate.eq("ndc")]
            self.assertTrue(focused.Value.isna().all())
            self.assertTrue(np.isfinite(self.select(frame, "industry").Value))

    def test_gcam_aggregate_missingness_remains_preserved_under_omnia_zero_policy(self):
        gcam = self.gcam.copy()
        mask = (gcam.Scenario.eq("NDC_NoCE") & gcam.Region.eq("EU-27")
                & gcam.Variable.eq("Emissions|CO2|Industrial Processes|Non-Metallic Minerals|Cement"))
        gcam.loc[mask, "2050"] = np.nan
        frame = self.data(gcam=gcam, missing_activity="zero").emissions_response()
        gc = frame.loc[frame.Model.eq("GCAM") & frame.Climate.eq("ndc") & frame.PanelKind.eq("three_sector")]
        self.assertTrue(gc.Value.isna().all())
        self.assertFalse(gc.UsesProvisionalZeros.any())
        self.assertTrue(np.isfinite(self.select(frame, "industry", model="GCAM").Value))

    def test_missing_gcam_ce_sector_co2_keeps_gap_without_fabricating_partial_total(self):
        gcam = self.gcam.copy()
        mask = (gcam.Scenario.eq("NDC_MedCE_EU") & gcam.Region.eq("EU-27")
                & gcam.Variable.eq("Emissions|CO2|Energy|Demand|Industry|Non-Ferrous Metals|Aluminum"))
        gcam.loc[mask, "2050"] = np.nan
        data = self.data(gcam=gcam, missing_activity="zero")
        frame = data.emissions_response()
        net = self.select(frame, "three_sector", model="GCAM")
        self.assertTrue(np.isnan(net.Value))
        self.assertFalse(net.UsesProvisionalZeros)
        self.assertEqual(net.MissingInputCount, 1)
        self.assertTrue(np.isfinite(self.select(frame, "three_sector", "Cement", model="GCAM").Value))
        self.assertTrue(np.isfinite(self.select(frame, "three_sector", model="GCAM", policy="highce_eu").Value))

    def test_availability_inventory_identifies_the_unmatched_co2_boundaries(self):
        inventory = self.data().emissions_availability()
        self.assertEqual(len(inventory), 10)
        self.assertEqual(int(inventory.SelectedForIndustry.sum()), 2)
        self.assertFalse(inventory.ComparableSectorGHG.any())
        self.assertEqual(int(inventory.UsedInThreeSector.sum()), 4)
        selected = inventory.loc[inventory.UsedInThreeSector]
        self.assertEqual(set(selected.Unit), {"Mt CO2/yr"})
        self.assertFalse(selected.SelectedForIndustry.any())
        cement = inventory.loc[inventory.Variable.eq("Emissions|CO2|Industrial Processes|Cement")].iloc[0]
        nonferrous = inventory.loc[inventory.Variable.eq("Emissions|CO2|Energy|Demand|Industry|Non-Ferrous Metals")].iloc[0]
        self.assertIn("process CO2 only", cement.Explanation)
        self.assertIn("Broader non-ferrous", nonferrous.Explanation)
        self.assertFalse(nonferrous.UsedInThreeSector)
        selected_cement = selected.loc[selected.Variable.eq("Emissions|CO2|Industrial Processes|Non-Metallic Minerals|Cement")].iloc[0]
        self.assertIn("without adding its overlapping process parent", selected_cement.Explanation)
        self.assertEqual(cement.Unit, "Mt CO2/yr")

    def test_cached_emissions_are_copies_and_do_not_duplicate_audit(self):
        data = self.data()
        first = data.emissions_response()
        count = len(data.closure_checks)
        first.loc[:, "Value"] = 12345
        self.assertFalse(data.emissions_response().Value.eq(12345).any())
        self.assertEqual(len(data.closure_checks), count)


if __name__ == "__main__":
    unittest.main()
