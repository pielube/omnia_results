"""Independent examples for total-referenced, signed CE contributions."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from omnia_results.europe_comparison_data import (CLIMATES, GCAM_FUELS, GCAM_PRODUCTION,
                                                  MODELS, OMNIA_REGIONS, POLICIES,
                                                  source_scenario)
from omnia_results.europe_relative_data import (ENERGY_METRIC, MATERIAL_METRIC,
                                                EuropeRelativeData, _contribution)
from omnia_results.metrics import SECTORS


class EuropeRelativeDataTests(unittest.TestCase):
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
                    for sector in SECTORS:
                        if model == "GCAM":
                            amount = (100, 80, 70)[p] if sector.key == "cement" else (100, 90, 75)[p]
                            add(model, scenario, GCAM_PRODUCTION[sector.key], "Mt/yr", amount * scale)
                        elif sector.key == "cement":
                            add(model, scenario, "Production|" + sector.path, "Mt/yr", (100, 80, 70)[p] * scale)
                        else:
                            add(model, scenario, "Production|" + sector.path + "|Primary", "Mt/yr", (10, 8, 6)[p] * scale)
                            add(model, scenario, "Production|" + sector.path + "|Secondary", "Mt/yr", (2, 5, 7)[p] * scale)
                        parent = "Final Energy|Industry|" + sector.path
                        fuels = sector.fuels if model == MODELS[0] else GCAM_FUELS[sector.key]
                        if len(fuels) == 4:
                            values = ((10, 20, 30, 40), (11, 10, 33, 20), (8, 24, 15, 60))[p]
                        elif len(fuels) == 5:
                            values = ((10, 20, 30, 40, 50), (15, 15, 35, 20, 80), (20, 0, 30, 60, 40))[p]
                        else:
                            values = ((10, 20, 30), (12, 10, 25), (5, 30, 15))[p]
                        for fuel, value in zip(fuels, values):
                            add(model, scenario, parent + "|" + fuel, "EJ/yr", value * scale)
                            add(model, scenario, parent + "|" + fuel + "|Detail", "EJ/yr", 99999)
                            add(model, scenario, parent + "|" + fuel, "PJ/yr", 99999)
                        if model == "GCAM":
                            residual = (10, 20, -5)[p] if sector.key == "aluminium" else 0
                            add(model, scenario, parent, "EJ/yr", (sum(values) + residual) * scale)
                        else:
                            # The OMNIA reported aggregate is not the denominator.
                            add(model, scenario, parent, "EJ/yr", 99999)
        self.omnia = pd.DataFrame(omnia, columns=["Scenario", "Region", "Variable", "Unit", "2019", "2050"])
        self.gcam = pd.DataFrame(gcam, columns=["Model", "Scenario", "Region", "Variable", "Unit", "2021", "2050"])

    def data(self, omnia=None, gcam=None, **settings):
        with patch("omnia_results.europe_comparison_data.pd.read_csv", return_value=(self.omnia if omnia is None else omnia).copy()), \
             patch("omnia_results.europe_comparison_data.pd.read_excel", return_value=(self.gcam if gcam is None else gcam).copy()):
            return EuropeRelativeData({**self.config, **settings}, Path("."))

    @staticmethod
    def select(frame, model, policy, series, sector="Iron and steel", climate="ndc"):
        return frame.loc[(frame.Model == model) & (frame.Climate == climate) & (frame.Policy == policy)
                         & (frame.Sector == sector) & (frame.Series == series)].iloc[0]

    def test_route_contributions_use_total_reference_and_retain_opposite_signs(self):
        data = self.data()
        frame = data.material_response()
        primary = self.select(frame, MODELS[0], "medce_eu", "Primary")
        secondary = self.select(frame, MODELS[0], "medce_eu", "Secondary")
        net = self.select(frame, MODELS[0], "medce_eu", "Total")
        self.assertAlmostEqual(primary.Value, -100 * 2 / 12)
        self.assertAlmostEqual(secondary.Value, 100 * 3 / 12)
        self.assertAlmostEqual(net.Value, 100 / 12)
        self.assertEqual(primary.Unit, "percentage points")
        self.assertEqual(net.Unit, "%")
        self.assertEqual(primary.Denominator, 72)  # 12 × sum of EU-region weights.
        self.assertEqual(secondary.Denominator, primary.Denominator)
        self.assertNotAlmostEqual(secondary.Value, 150)  # Within-secondary growth.
        self.assertEqual((primary.ScenarioValue, primary.ReferenceValue, primary.Numerator), (48, 60, -12))
        self.assertEqual((net.ConversionFactor, net.Offset), (100, 0))
        self.assertEqual(set(frame.Metric), {MATERIAL_METRIC})
        self.assertEqual(set(frame.Year), {2050})
        self.assertEqual(set(frame.Policy), {"medce_eu", "highce_eu"})
        self.assertTrue(data.closure_checks.Complete.all())
        self.assertLess(data.closure_checks.Difference.abs().max(), 1e-10)

    def test_models_and_climates_keep_their_own_future_reference_and_scope(self):
        frame = self.data().material_response()
        for model in MODELS:
            for climate in CLIMATES:
                row = self.select(frame, model, "medce_eu", "Total", sector="Cement", climate=climate)
                self.assertAlmostEqual(row.Value, -20)
                self.assertEqual(row.ReferenceScenario, source_scenario(model, climate, "noce"))
                self.assertEqual(row.ReferenceYear, 2050)
                self.assertEqual(row.DenominatorSourceScenario, row.ReferenceScenario)
                self.assertEqual(json.loads(row.SourceRegions), list(OMNIA_REGIONS) if model == MODELS[0] else ["EU-27"])
        a = self.select(frame, MODELS[0], "medce_eu", "Total", sector="Cement")
        b = self.select(frame, "GCAM", "medce_eu", "Total", sector="Cement")
        self.assertEqual((a.Denominator, b.Denominator), (600, 500))
        self.assertEqual(set(frame.Panel), set("abcdef"))

    def test_gcam_metals_remain_one_reported_component_without_assumed_route_split(self):
        frame = self.data().material_response()
        gcam_metals = frame.loc[(frame.Model == "GCAM") & frame.Sector.ne("Cement")]
        self.assertEqual(set(gcam_metals.Series), {"Reported metal", "Total"})
        row = self.select(frame, "GCAM", "medce_eu", "Reported metal")
        net = self.select(frame, "GCAM", "medce_eu", "Total")
        self.assertEqual(row.Value, net.Value)
        self.assertAlmostEqual(row.Value, -10)
        self.assertEqual(json.loads(row.SourceVariables), ["Production|Iron and Steel|Steel"])

    def test_energy_exact_carriers_and_unallocated_signed_contributions_close_to_parent_net(self):
        data = self.data()
        frame = data.energy_response()
        gc_electricity = self.select(frame, "GCAM", "medce_eu", "Electricity", sector="Aluminium")
        gc_residual = self.select(frame, "GCAM", "medce_eu", "Unallocated", sector="Aluminium")
        gc_net = self.select(frame, "GCAM", "medce_eu", "Total", sector="Aluminium")
        self.assertAlmostEqual(gc_electricity.Value, 100 * 2 / 70)
        self.assertAlmostEqual(gc_residual.Value, 100 * 10 / 70)
        self.assertAlmostEqual(gc_net.Value, -100 * 3 / 70)
        self.assertEqual(gc_net.Denominator, 350)
        self.assertEqual(json.loads(gc_net.DenominatorSourceVariables), ["Final Energy|Industry|Non-Ferrous Metals|Aluminum"])
        self.assertEqual(json.loads(gc_residual.SourceCoefficients), [1, -1, -1, -1])
        negative = self.select(frame, "GCAM", "highce_eu", "Unallocated", sector="Aluminium")
        self.assertLess(negative.ScenarioValue, 0)
        self.assertAlmostEqual(negative.Value, -100 * 15 / 70)
        om_net = self.select(frame, MODELS[0], "medce_eu", "Total", sector="Cement")
        self.assertEqual(om_net.Denominator, 600)
        self.assertAlmostEqual(om_net.Value, -26)
        self.assertEqual(len(json.loads(om_net.DenominatorSourceVariables)), 4)
        self.assertEqual(set(frame.Metric), {ENERGY_METRIC})
        self.assertTrue(data.closure_checks.Complete.all())
        self.assertLess(data.closure_checks.Difference.abs().max(), 1e-10)

    def test_provenance_distinguishes_component_reference_from_total_denominator(self):
        row = self.select(self.data().material_response(), MODELS[0], "medce_eu", "Secondary")
        secondary = "Production|Iron and Steel|Secondary"
        primary = "Production|Iron and Steel|Primary"
        self.assertEqual(json.loads(row.SourceVariables), [secondary])
        self.assertEqual(json.loads(row.ReferenceSourceVariables), [secondary])
        self.assertEqual(json.loads(row.DenominatorSourceVariables), [primary, secondary])
        self.assertEqual(json.loads(row.DenominatorSourceUnits), ["Mt/yr", "Mt/yr"])
        self.assertEqual(json.loads(row.DenominatorSourceCoefficients), [1, 1])
        self.assertEqual(json.loads(row.NumeratorSourceCoefficients), [1, -1])
        self.assertEqual(json.loads(row.NumeratorSourceScenarios), ["ndc_medce_eu", "ndc_noce"])
        self.assertEqual((row.DenominatorSourceValue, row.TotalReferenceValue), (72, 72))

    def test_zero_component_reference_is_valid_when_sector_total_remains_positive(self):
        omnia = self.omnia.copy()
        mask = omnia.Scenario.eq("ndc_noce") & omnia.Variable.eq("Production|Iron and Steel|Secondary")
        omnia.loc[mask, "2050"] = 0
        frame = self.data(omnia=omnia).material_response()
        row = self.select(frame, MODELS[0], "medce_eu", "Secondary")
        self.assertEqual(row.ReferenceValue, 0)
        self.assertEqual(row.Denominator, 60)
        self.assertEqual(row.Value, 50)

    def test_missing_component_keeps_net_undefined_and_does_not_sum_available_subset(self):
        omnia = self.omnia.copy()
        mask = (omnia.Scenario.eq("ndc_medce_eu") & omnia.Region.eq("EUM")
                & omnia.Variable.eq("Production|Iron and Steel|Primary"))
        omnia.loc[mask, "2050"] = np.nan
        data = self.data(omnia=omnia)
        frame = data.material_response()
        self.assertTrue(np.isnan(self.select(frame, MODELS[0], "medce_eu", "Primary").Value))
        self.assertTrue(np.isnan(self.select(frame, MODELS[0], "medce_eu", "Total").Value))
        self.assertTrue(np.isfinite(self.select(frame, MODELS[0], "medce_eu", "Secondary").Value))
        check = data.closure_checks.loc[(data.closure_checks.Model == MODELS[0])
                                       & (data.closure_checks.Climate == "ndc")
                                       & (data.closure_checks.Policy == "medce_eu")
                                       & (data.closure_checks.Sector == "Iron and steel")].iloc[0]
        self.assertFalse(check.Complete)
        self.assertTrue(np.isnan(check.SumContributions))

    def test_missing_or_nonpositive_total_reference_gives_gaps_for_every_contribution(self):
        for value in (np.nan, 0, -5):
            gcam = self.gcam.copy()
            mask = gcam.Scenario.eq("NDC_NoCE") & gcam.Variable.eq(GCAM_PRODUCTION["steel"])
            gcam.loc[mask, "2050"] = value
            # Keep country totals consistent for finite nonpositive aggregate tests.
            if np.isfinite(value):
                gcam.loc[mask & gcam.Region.eq("Austria"), "2050"] = value * .4
                gcam.loc[mask & gcam.Region.eq("Belgium"), "2050"] = value * .6
            data = self.data(gcam=gcam, missing_activity="zero")
            frame = data.material_response()
            selected = frame.loc[(frame.Model == "GCAM") & (frame.Climate == "ndc") & (frame.Sector == "Iron and steel")]
            self.assertTrue(selected.Value.isna().all())
            if np.isnan(value):
                self.assertEqual(set(data.source_data.missing_inputs.loc[data.source_data.missing_inputs.Model.eq("GCAM"), "Treatment"]), {"preserve"})
        for denominator in (np.nan, np.inf, -np.inf, 0, -1):
            self.assertTrue(np.isnan(_contribution(1, denominator)))
        self.assertTrue(np.isnan(_contribution(np.inf, 100)))

    def test_cached_results_are_copies_and_do_not_duplicate_closure_audit(self):
        data = self.data()
        first = data.material_response()
        checks = len(data.closure_checks)
        first.loc[:, "Value"] = 9999
        second = data.material_response()
        self.assertFalse(second.Value.eq(9999).any())
        self.assertEqual(len(data.closure_checks), checks)


if __name__ == "__main__":
    unittest.main()
