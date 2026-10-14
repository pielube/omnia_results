"""Asymmetric examples verify cross-model boundaries and calculation references."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from omnia_results.europe_comparison_data import (CLIMATES, GCAM_FUELS, GCAM_PRODUCTION,
                                                  MODELS, OMNIA_REGIONS, POLICIES,
                                                  EuropeComparisonData, source_scenario)
from omnia_results.metrics import SECTORS


class EuropeComparisonDataTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(input="omnia.csv", gcam_input="gcam.xlsx", missing_activity="preserve",
                           historical_year=2019, gcam_historical_year=2021, comparison_year=2050)
        omnia, gcam = [], []

        def add(model, scenario, variable, unit, start, final):
            if model == MODELS[0]:
                for region, coefficient in zip((*OMNIA_REGIONS, "ENW"), (1, 2, 3, 10000)):
                    omnia.append([scenario, region, variable, unit, start * coefficient, final * coefficient])
            else:
                # Two reported country rows sum to the authoritative aggregate.
                # There is no assumption that 27 production rows must be present.
                for region, coefficient in (("EU-27", 1), ("Austria", .25), ("Belgium", .75)):
                    gcam.append(["GCAM Europe8.7", scenario, region, variable, unit,
                                 start * coefficient, final * coefficient])

        for model in MODELS:
            for climate in CLIMATES:
                scale = (1 if climate == "ndc" else 2) * (1 if model == MODELS[0] else 5)
                for p, policy in enumerate(POLICIES):
                    scenario = source_scenario(model, climate, policy)
                    factor = (1, .8, 1.2)[p]
                    # CE histories deliberately differ from no-CE observations.
                    historical = scale * (1 + p * 100)
                    for sector in SECTORS:
                        parent = f"Production|{sector.path}"
                        if model == "GCAM":
                            add(model, scenario, GCAM_PRODUCTION[sector.key], "Mt/yr", 10 * historical, 100 * scale * factor)
                        elif sector.key == "cement":
                            add(model, scenario, parent, "Mt/yr", 10 * historical, 100 * scale * factor)
                            add(model, scenario, "Production|Non-Metallic Minerals|Cement Clinker", "Mt/yr", 7 * historical, 70 * scale * factor)
                        else:
                            add(model, scenario, parent + "|Primary", "Mt/yr", 8 * historical, 80 * scale * factor)
                            add(model, scenario, parent + "|Secondary", "Mt/yr", 2 * historical, 20 * scale * factor)
                        energy = f"Final Energy|Industry|{sector.path}"
                        fuels = sector.fuels if model == MODELS[0] else GCAM_FUELS[sector.key]
                        for i, fuel in enumerate(fuels, 1):
                            add(model, scenario, energy + "|" + fuel, "EJ/yr", i * historical, i * scale * factor)
                            add(model, scenario, energy + "|" + fuel + "|Detail", "EJ/yr", 99999, 99999)
                            add(model, scenario, energy + "|" + fuel, "PJ/yr", 99999, 99999)
                        if model == "GCAM":
                            residual = 2 if sector.key == "aluminium" else 0
                            add(model, scenario, energy, "EJ/yr", sum(range(1, len(fuels) + 1)) * historical + residual,
                                sum(range(1, len(fuels) + 1)) * scale * factor + residual)
                        else:
                            add(model, scenario, energy, "EJ/yr", 99999, 99999)
                    if model == MODELS[0]:
                        add(model, scenario, "Emissions|GHG|Industry", "MtCO2e/yr", 20 * historical,
                            20 * scale * (1, .7, 1.3)[p])
                        for sector, amount in zip(SECTORS, (3, 4, 2)):
                            add(model, scenario, f"Emissions|GHG|Industry|{sector.path}", "MtCO2e/yr",
                                amount * historical, amount * scale * (1, .8, .9)[p])
                    else:
                        add(model, scenario, "Emissions|Kyoto Gases|Energy|Demand|Industry", "Mt CO2-equiv/yr",
                            100 * historical, 100 * scale * factor)
                        add(model, scenario, "Emissions|Kyoto Gases|Industrial Processes", "Mt CO2-equiv/yr",
                            30 * historical, 30 * scale * (1, 1.1, .7)[p])
        self.omnia = pd.DataFrame(omnia, columns=["Scenario", "Region", "Variable", "Unit", "2019", "2050"])
        self.gcam = pd.DataFrame(gcam, columns=["Model", "Scenario", "Region", "Variable", "Unit", "2021", "2050"])

    def data(self, omnia=None, gcam=None, **settings):
        with patch("omnia_results.europe_comparison_data.pd.read_csv", return_value=(self.omnia if omnia is None else omnia).copy()), \
             patch("omnia_results.europe_comparison_data.pd.read_excel", return_value=(self.gcam if gcam is None else gcam).copy()):
            return EuropeComparisonData({**self.config, **settings}, Path("."))

    @staticmethod
    def select(frame, model, climate, policy, metric, series, sector="Cement", year=2050):
        return frame.loc[(frame.Model == model) & (frame.Climate == climate) & (frame.Policy == policy)
                         & (frame.Metric == metric) & (frame.Series == series) & (frame.Sector == sector)
                         & (frame.Year == year)].iloc[0]

    def test_production_uses_eu_scope_once_and_preserves_unavailable_gcam_routes(self):
        frame = self.data().material_production()
        a = self.select(frame, MODELS[0], "ndc", "noce", "Material production", "Cement")
        b = self.select(frame, "GCAM", "ndc", "noce", "Material production", "Cement")
        self.assertEqual(a.Value, 600)
        self.assertEqual(b.Value, 500)
        self.assertEqual(json.loads(a.SourceRegions), ["EUE", "EUM", "EUW"])
        self.assertEqual(json.loads(b.SourceRegions), ["EU-27"])
        self.assertEqual(set(frame.loc[(frame.Model == "GCAM") & (frame.Metric == "Material production"), "Series"]), {"Cement", "Total"})
        self.assertEqual(set(frame.loc[frame.Series.eq("Clinker"), "Role"]), {"context"})
        self.assertFalse(frame.Policy.str.endswith("gbl").any())

    def test_historical_bars_and_percentage_references_keep_each_models_observations(self):
        frame = self.data().material_production()
        a = self.select(frame, MODELS[0], "ndc", "noce", "Material production", "Cement", year=2019)
        b = self.select(frame, "GCAM", "ndc", "noce", "Material production", "Cement", year=2021)
        self.assertEqual(a.Value, 60)
        self.assertEqual(b.Value, 50)
        self.assertEqual((a.BarOrder, b.BarOrder), (0, 0))
        for model in MODELS:
            for climate in CLIMATES:
                row = self.select(frame, model, climate, "medce_eu", "Production change from no CE", "Total")
                self.assertAlmostEqual(row.Value, -20)
                self.assertEqual(row.ReferenceScenario, source_scenario(model, climate, "noce"))
                self.assertEqual(row.ReferenceYear, 2050)
                self.assertEqual((row.ConversionFactor, row.Offset), (100, -100))

    def test_energy_uses_exact_carrier_parents_and_preserves_gcam_residual(self):
        d = self.data()
        frame = d.energy_mix()
        a = self.select(frame, MODELS[0], "ndc", "noce", "Final energy", "Total", sector="Aluminium")
        b = self.select(frame, "GCAM", "ndc", "noce", "Final energy", "Total", sector="Aluminium")
        residual = self.select(frame, "GCAM", "ndc", "noce", "Final energy", "Unallocated", sector="Aluminium")
        self.assertEqual(a.Value, 90)
        self.assertEqual(b.Value, 32)
        self.assertEqual(residual.Value, 2)
        self.assertEqual(json.loads(residual.SourceCoefficients), [1, -1, -1, -1])
        self.assertEqual(set(frame.loc[(frame.Model == "GCAM") & (frame.Sector == "Aluminium") & (frame.Metric == "Final energy"), "Series"]),
                         {"Electricity", "Liquids", "Solids", "Unallocated", "Total"})
        plotted = frame.loc[(frame.Model == "GCAM") & (frame.Sector == "Aluminium") & (frame.Role == "plotted")
                            & (frame.BarOrder == 1) & (frame.Climate == "ndc")]
        self.assertEqual(plotted.Value.sum(), b.Value)
        self.assertEqual(d.aggregation_checks.Difference.abs().max(), 0)

    def test_emissions_components_reconstruct_signed_total_changes_without_matching_sectors(self):
        frame = self.data().emissions_response()
        for model in MODELS:
            for climate in CLIMATES:
                for policy in POLICIES:
                    change = frame.loc[(frame.Model == model) & (frame.Climate == climate) & (frame.Policy == policy)
                                       & (frame.Metric == "Industrial GHG emissions change")]
                    self.assertAlmostEqual(change.loc[change.Role.eq("plotted"), "Value"].sum(),
                                           change.loc[change.Role.eq("net"), "Value"].iloc[0])
        gc = frame.loc[(frame.Model == "GCAM") & (frame.Metric == "Industrial GHG emissions change")]
        self.assertEqual(set(gc.Series), {"Total industry", "Industrial energy demand", "Industrial processes"})
        medium = self.select(frame, "GCAM", "ndc", "medce_eu", "Industrial GHG emissions change", "Industrial processes", sector="Industry")
        self.assertGreater(medium.Value, 0)
        total = self.select(frame, "GCAM", "ndc", "medce_eu", "Industrial GHG emissions change", "Total industry", sector="Industry")
        self.assertLess(total.Value, 0)
        percent = self.select(frame, "GCAM", "ndc", "medce_eu", "Industrial GHG emissions change from no CE", "Total industry", sector="Industry")
        self.assertAlmostEqual(percent.Value, 100 * ((100 * .8 + 30 * 1.1) / 130 - 1))
        self.assertEqual(json.loads(percent.SourceUnits), ["Mt CO2-equiv/yr", "Mt CO2-equiv/yr"])
        self.assertEqual(percent.Role, "plotted")
        noce_percent = frame.loc[frame.Policy.eq("noce") & frame.Metric.eq("Industrial GHG emissions change from no CE")]
        self.assertEqual(set(noce_percent.Role), {"context"})

    def test_missingness_is_model_specific_and_gcam_aggregate_blanks_remain_missing(self):
        omnia, gcam = self.omnia.copy(), self.gcam.copy()
        omask = (omnia.Scenario == "ndc_noce") & (omnia.Region == "EUM") & (omnia.Variable == "Production|Non-Metallic Minerals|Cement") & (omnia.Unit == "Mt/yr")
        gmask = (gcam.Scenario == "NDC_NoCE") & (gcam.Region == "EU-27") & (gcam.Variable == "Production|Non-Metallic Minerals|Cement") & (gcam.Unit == "Mt/yr")
        omnia.loc[omask, "2050"], gcam.loc[gmask, "2050"] = np.nan, np.nan
        d = self.data(omnia, gcam, missing_activity="zero")
        frame = d.material_production()
        self.assertEqual(self.select(frame, MODELS[0], "ndc", "noce", "Material production", "Cement").Value, 400)
        self.assertTrue(np.isnan(self.select(frame, "GCAM", "ndc", "noce", "Material production", "Cement").Value))
        self.assertEqual(set(d.missing_inputs.loc[d.missing_inputs.Model.eq("GCAM"), "Treatment"]), {"preserve"})
        self.assertTrue(np.isnan(self.select(self.data(omnia, gcam).material_production(), MODELS[0], "ndc", "noce", "Material production", "Cement").Value))

    def test_required_variable_aggregate_or_units_are_never_fabricated(self):
        mask = (self.gcam.Scenario == "NDC_NoCE") & (self.gcam.Variable == GCAM_PRODUCTION["cement"])
        wrong_unit = self.gcam.copy()
        wrong_unit.loc[mask, "Unit"] = "kt/yr"
        with self.assertRaisesRegex(ValueError, "required source series absent"):
            self.data(gcam=wrong_unit).material_production()
        absent_aggregate = self.gcam.loc[~(mask & self.gcam.Region.eq("EU-27"))]
        with self.assertRaisesRegex(ValueError, "EU-27 aggregate absent"):
            self.data(gcam=absent_aggregate).material_production()
        wrong_total = self.gcam.copy()
        wrong_total.loc[mask & wrong_total.Region.eq("EU-27"), "2050"] += 1
        with self.assertRaisesRegex(ValueError, "aggregate differs"):
            self.data(gcam=wrong_total).material_production()

    def test_config_cannot_silently_change_scope_or_drop_required_observations(self):
        with self.assertRaisesRegex(ValueError, "exactly the OMNIA EU regions"):
            self.data(regions=["EUE", "EUM", "EUW", "ENW"])
        with self.assertRaisesRegex(ValueError, "six OMNIA EU-only scenarios"):
            self.data(scenarios=["ndc_noce"])
        with self.assertRaisesRegex(ValueError, "required year 2019 is omitted"):
            self.data(years=[2050])
        with self.assertRaisesRegex(ValueError, "configured years absent"):
            self.data(gcam_years=[2019, 2021, 2050])
        with self.assertRaisesRegex(ValueError, "unique and increasing"):
            self.data(years=[2050, 2019])

        source = self.gcam.copy()
        source["2030"] = source["2021"]
        d = self.data(gcam=source, years=[2019, 2050], gcam_years=[2021, 2050])
        frame = d.emissions_response()
        self.assertEqual(d.available_years["GCAM"], [2021, 2030, 2050])
        self.assertEqual(d.years["GCAM"], [2021, 2050])
        self.assertFalse(frame.Year.eq(2030).any())


if __name__ == "__main__":
    unittest.main()
