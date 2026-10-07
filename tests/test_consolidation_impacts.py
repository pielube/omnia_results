"""Asymmetric source examples for industrial and regional emissions changes."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np
import pandas as pd

from omnia_results.consolidation_data import ConsolidationData
from omnia_results.metrics import SECTORS


POLICIES = ("noce", "medce_eu", "highce_eu", "medce_gbl", "highce_gbl")


class ConsolidationImpactTests(unittest.TestCase):
    def setUp(self):
        self.regions = ["EUE", "EUW", "CHN", "IND", "USA", "JPN", "AFE", "MEA"]
        self.config = {
            "input": "raw_impacts.csv", "policy_regions_csv": "policy_regions.csv",
            "scenarios": [f"{climate}_{policy}" for climate in ("baseline", "ndc") for policy in POLICIES],
            "regions": self.regions, "years": [2019, 2030, 2050], "missing_activity": "preserve",
            "comparison_year": 2050,
            "partner_top_n": 3,
        }
        self.membership = pd.DataFrame({
            "Region": self.regions,
            "Description": ["EU East", "EU West", "China", "India", "United States", "Japan",
                            "Eastern Africa", "Middle East"],
            "EU": [True, True, False, False, False, False, False, False],
            "EU_plus_partners": [True, True, True, True, True, True, False, False],
        })
        # NDC's largest partner response is Japan, while NDC+LTT ranks China,
        # India and the USA. Both plots must retain one shared selected cohort.
        self.target_changes = {"baseline": [-.5, -.25, -10, -7.5, -5, -2.5, .2, .3],
                               "ndc": [-1, -.5, -1.25, -1.75, -2, -25, 1, 2]}
        self.ghg = {}
        self.industry = {}
        rows = []

        def add(scenario, region, variable, values):
            rows.append([scenario, region, variable, "MtCO2e/yr", *values])
            rows.append([scenario, region, variable, "MtCO2e/Mt", 99999, 99999, 99999])
            rows.append([scenario, region, variable + "|Detail", "MtCO2e/yr", 99999, 99999, 99999])

        for climate in ("baseline", "ndc"):
            for p, policy in enumerate(POLICIES):
                scenario = f"{climate}_{policy}"
                self.ghg[scenario], self.industry[scenario] = {}, []
                for region_index, region in enumerate(self.regions):
                    sector_values = []
                    for sector_index, (sector, share) in enumerate(zip(SECTORS, (.5, .3, .2))):
                        base = ([40 + 2 * region_index, 20 + region_index, 3 + .5 * region_index]
                                if climate == "baseline" else
                                [80 + 3 * region_index, 40 + 2 * region_index, 6 + region_index])[sector_index]
                        values = np.asarray([base * .7, base * .85, base]) + np.asarray([0, .5, 1]) * p * self.target_changes[climate][region_index] * share
                        sector_values.append(values)
                        self.ghg[scenario].setdefault(sector.label, []).append(values)
                        add(scenario, region, f"Emissions|GHG|Industry|{sector.path}", values)
                        rows.append([scenario, region, f"Carbon Capture|Industry|{sector.path}",
                                     "MtCO2/yr", 99999, 99999, 99999])
                    other_base = (30 + region_index if climate == "baseline" else 70 + 2 * region_index)
                    other_change = (1 + .3 * region_index if climate == "baseline" else 2 + .4 * region_index)
                    other = np.asarray([other_base * .7, other_base * .85, other_base]) + np.asarray([0, .5, 1]) * p * other_change
                    industry = np.sum(sector_values, axis=0) + other
                    self.industry[scenario].append(industry)
                    add(scenario, region, "Emissions|GHG|Industry", industry)
                    rows.append([scenario, region, "Emissions|GHG|Industry|Other", "MtCO2e/yr", 99999, 99999, 99999])
        self.source = pd.DataFrame(rows, columns=["Scenario", "Region", "Variable", "Unit", "2019", "2030", "2050"])

    def data(self, *, source=None, membership=None, config=None):
        source = self.source if source is None else source
        membership = self.membership if membership is None else membership
        config = self.config if config is None else config

        def read(path, *args, **kwargs):
            return (membership if Path(path).name == "policy_regions.csv" else source).copy()

        with patch("omnia_results.metrics.pd.read_csv", side_effect=read):
            data = ConsolidationData(config, Path("."))
            # Mapping is loaded lazily by the comparison calculations.
            data.emissions_response()
        return data

    @staticmethod
    def select(frame, scenario, kind, series):
        return frame[(frame.Scenario == scenario) & (frame.PanelKind == kind) & (frame.Series == series)]

    def target(self, scenario):
        return sum(np.asarray(regional).sum(axis=0) for regional in self.ghg[scenario].values())

    def test_all_trajectory_scenarios_are_direct_global_three_sector_sums_in_source_units(self):
        frame = self.data().emissions_response()
        self.assertEqual(len(frame), 126)
        for scenario in self.config["scenarios"]:
            selected = self.select(frame, scenario, "trajectory", "Three sectors")
            with self.subTest(scenario=scenario):
                self.assertEqual(selected.Year.tolist(), self.config["years"])
                np.testing.assert_allclose(selected.Value, self.target(scenario))
                np.testing.assert_allclose(selected.ScenarioValue, self.target(scenario))
                self.assertEqual(selected.ReferenceScenario.unique().tolist(), [""])
                self.assertEqual(selected.Unit.unique().tolist(), ["MtCO2e/yr"])
        for panels, climate, label in (("ace", "ndc", "NDC"), ("bdf", "baseline", "NDC+LTT")):
            selected = frame[frame.Panel.isin(list(panels))]
            self.assertTrue(selected.Scenario.str.startswith(climate + "_").all())
            self.assertEqual(selected.ClimatePathway.unique().tolist(), [label])

    def test_sector_residual_and_industry_net_changes_close_with_own_climate_references(self):
        frame = self.data().emissions_response()
        for climate in ("baseline", "ndc"):
            reference = f"{climate}_noce"
            reference_industry = np.asarray(self.industry[reference]).sum(axis=0)[-1]
            for p, policy in enumerate(POLICIES[1:], start=1):
                scenario = f"{climate}_{policy}"
                selected = frame[(frame.Scenario == scenario) & (frame.PanelKind == "industry")]
                current_industry = np.asarray(self.industry[scenario]).sum(axis=0)[-1]
                for sector in SECTORS:
                    observation = selected[selected.Series == sector.label].iloc[0]
                    current = np.asarray(self.ghg[scenario][sector.label]).sum(axis=0)[-1]
                    base = np.asarray(self.ghg[reference][sector.label]).sum(axis=0)[-1]
                    self.assertAlmostEqual(observation.Value, current - base)
                    self.assertAlmostEqual(observation.ScenarioValue, current)
                    self.assertAlmostEqual(observation.ReferenceValue, base)
                residual = selected[selected.Series == "Other industry"].iloc[0]
                self.assertAlmostEqual(residual.ScenarioValue, current_industry - self.target(scenario)[-1])
                self.assertAlmostEqual(residual.ReferenceValue, reference_industry - self.target(reference)[-1])
                net = selected[selected.Role == "net"].iloc[0]
                self.assertEqual(net.Series, "Total industry")
                self.assertAlmostEqual(net.Value, current_industry - reference_industry)
                self.assertAlmostEqual(selected.loc[selected.Role == "plotted", "Value"].sum(), net.Value)
                self.assertAlmostEqual(net.Value, (-8.85 if climate == "baseline" else -1.3) * p)
                self.assertEqual(selected.ReferenceScenario.unique().tolist(), [reference])
                self.assertEqual(selected.ReferenceYear.unique().tolist(), [2050])

    def test_fixed_geographic_partition_preserves_each_region_once_and_closes_global_change(self):
        data = self.data()
        frame = data.emissions_response()
        membership = data.geographic_membership()
        expected_groups = {"EU": ["EUE", "EUW"], "Japan": ["JPN"], "China": ["CHN"],
                           "India": ["IND"], "Other partners": ["USA"], "Rest of world": ["AFE", "MEA"]}
        self.assertEqual(len(membership), len(self.regions))
        self.assertEqual(set(membership.Region), set(self.regions))
        self.assertFalse(membership.Region.duplicated().any())
        for group, regions in expected_groups.items():
            self.assertEqual(membership.loc[membership.Group == group, "Region"].tolist(), regions)
        self.assertEqual(membership.loc[membership.PolicyGroup == "EU", "Region"].tolist(), ["EUE", "EUW"])
        for climate in ("ndc", "baseline"):
            for p, policy in enumerate(POLICIES[1:], start=1):
                scenario = f"{climate}_{policy}"
                selected = frame[(frame.Scenario == scenario) & (frame.PanelKind == "geography")]
                for group, regions in expected_groups.items():
                    observation = selected[selected.Series == group].iloc[0]
                    expected = sum(self.target_changes[climate][self.regions.index(region)] for region in regions) * p
                    self.assertAlmostEqual(observation.Value, expected)
                    self.assertEqual(json.loads(observation.SourceRegions), regions)
                net = selected[selected.Role == "net"].iloc[0]
                self.assertEqual(net.Series, "Global net")
                self.assertAlmostEqual(net.Value, sum(self.target_changes[climate]) * p)
                self.assertAlmostEqual(selected.loc[selected.Role == "plotted", "Value"].sum(), net.Value)

    def test_partner_cohort_uses_pooled_absolute_changes_in_all_eight_cases_and_stays_fixed(self):
        data = self.data()
        ranks = data.partner_rankings()
        self.assertEqual(ranks.Region.tolist(), ["JPN", "CHN", "IND", "USA"])
        np.testing.assert_allclose(ranks.MeanAbsoluteChange_MtCO2e_yr, [34.375, 14.0625, 11.5625, 8.75])
        self.assertEqual(ranks.Selected.tolist(), [True, True, True, False])
        self.assertEqual(ranks.Rank.tolist(), [1, 2, 3, 4])
        cases = {f"{climate}_{policy}" for climate in ("ndc", "baseline") for policy in POLICIES[1:]}
        for observation in ranks.itertuples():
            self.assertEqual(set(json.loads(observation.RankingScenarios)), cases)
            changes = json.loads(observation.ChangesByScenario)
            self.assertEqual(set(changes), cases)
            region_index = self.regions.index(observation.Region)
            for scenario, value in changes.items():
                climate, policy = scenario.split("_", 1)
                self.assertAlmostEqual(value, self.target_changes[climate][region_index] * POLICIES.index(policy))
        frame = data.emissions_response()
        for scenario in cases:
            groups = frame[(frame.Scenario == scenario) & (frame.PanelKind == "geography") & (frame.Role == "plotted")]
            self.assertEqual(groups.Series.tolist(), ["EU", "Japan", "China", "India", "Other partners", "Rest of world"])

    def test_partner_ties_use_region_codes_and_missing_cases_are_not_averaged_away(self):
        source = self.source.copy()
        for climate in ("ndc", "baseline"):
            for p, policy in enumerate(POLICIES[1:], start=1):
                for region in ("CHN", "IND"):
                    for sector in SECTORS:
                        variable = f"Emissions|GHG|Industry|{sector.path}"
                        base = source[(source.Scenario == f"{climate}_noce") & (source.Region == region)
                                      & (source.Variable == variable) & (source.Unit == "MtCO2e/yr")]
                        selected = ((source.Scenario == f"{climate}_{policy}") & (source.Region == region)
                                    & (source.Variable == variable) & (source.Unit == "MtCO2e/yr"))
                        source.loc[selected, ["2019", "2030", "2050"]] = base[["2019", "2030", "2050"]].to_numpy() + [0, -2.5 * p, -5 * p]
        ranks = self.data(source=source).partner_rankings()
        self.assertEqual(ranks.Region.iloc[:2].tolist(), ["CHN", "IND"])
        self.assertAlmostEqual(ranks.MeanAbsoluteChange_MtCO2e_yr.iloc[0], 37.5)
        source = self.source.copy()
        selected = ((source.Scenario == "ndc_medce_eu") & (source.Region == "CHN")
                    & (source.Variable == "Emissions|GHG|Industry|Non-Metallic Minerals|Cement")
                    & (source.Unit == "MtCO2e/yr"))
        source.loc[selected, "2050"] = np.nan
        preserve = self.data(source=source).partner_rankings().set_index("Region")
        self.assertFalse(preserve.loc["CHN", "Selected"])
        self.assertTrue(np.isnan(preserve.loc["CHN", "MeanAbsoluteChange_MtCO2e_yr"]))
        self.assertIsNone(json.loads(preserve.loc["CHN", "ChangesByScenario"])["ndc_medce_eu"])
        zero = self.data(source=source, config={**self.config, "missing_activity": "zero"}).partner_rankings().set_index("Region")
        self.assertTrue(zero.loc["CHN", "Selected"])
        self.assertTrue(np.isfinite(zero.loc["CHN", "MeanAbsoluteChange_MtCO2e_yr"]))

    def test_exact_ghg_parents_units_and_residual_coefficients_preserve_source_provenance(self):
        data = self.data()
        frame = data.emissions_response()
        sector_variables = [f"Emissions|GHG|Industry|{sector.path}" for sector in SECTORS]
        variables = ["Emissions|GHG|Industry", *sector_variables]
        exact = self.source[(self.source.Unit == "MtCO2e/yr") & self.source.Variable.isin(variables)]
        pd.testing.assert_frame_equal(frame, self.data(source=exact).emissions_response())
        for observation in frame.itertuples():
            source_variables = json.loads(observation.SourceVariables)
            coefficients = json.loads(observation.SourceCoefficients)
            if observation.PanelKind != "industry":
                self.assertEqual(source_variables, sector_variables)
                self.assertEqual(coefficients, [1, 1, 1])
            elif observation.Series == "Other industry":
                self.assertEqual(source_variables, variables)
                self.assertEqual(coefficients, [1, -1, -1, -1])
            elif observation.Series == "Total industry":
                self.assertEqual(source_variables, ["Emissions|GHG|Industry"])
                self.assertEqual(coefficients, [1])
            else:
                self.assertEqual(source_variables, [sector_variables[[sector.label for sector in SECTORS].index(observation.Series)]])
                self.assertEqual(coefficients, [1])
            if observation.PanelKind != "trajectory":
                self.assertAlmostEqual(observation.Value, observation.ScenarioValue - observation.ReferenceValue)

    def test_missing_target_inputs_preserve_unknown_totals_without_erasing_known_industry_net(self):
        source = self.source.copy()
        selected = ((source.Scenario == "ndc_highce_gbl") & (source.Region == "EUE")
                    & (source.Variable == "Emissions|GHG|Industry|Non-Metallic Minerals|Cement")
                    & (source.Unit == "MtCO2e/yr"))
        source.loc[selected, "2050"] = np.nan
        for policy in ("preserve", "zero"):
            with self.subTest(policy=policy):
                data = self.data(source=source, config={**self.config, "missing_activity": policy})
                frame = data.emissions_response()
                target = self.select(frame, "ndc_highce_gbl", "trajectory", "Three sectors").Value.iloc[-1]
                cement = self.select(frame, "ndc_highce_gbl", "industry", "Cement").Value.iloc[0]
                residual = self.select(frame, "ndc_highce_gbl", "industry", "Other industry").Value.iloc[0]
                eu = self.select(frame, "ndc_highce_gbl", "geography", "EU").Value.iloc[0]
                global_net = self.select(frame, "ndc_highce_gbl", "geography", "Global net").Value.iloc[0]
                if policy == "preserve":
                    self.assertTrue(all(np.isnan(value) for value in (target, cement, residual, eu, global_net)))
                else:
                    self.assertAlmostEqual(target, 984)
                    self.assertAlmostEqual(cement, -135)
                    self.assertAlmostEqual(residual, 186.8)
                    self.assertAlmostEqual(eu, -84)
                    self.assertAlmostEqual(global_net, -192)
                net = self.select(frame, "ndc_highce_gbl", "industry", "Total industry").Value.iloc[0]
                self.assertAlmostEqual(net, -5.2)
                self.assertEqual(data.results["ndc_highce_gbl"].coverage[0]["Treatment"], policy)

    def test_missing_industry_only_changes_industry_net_and_residual_and_signed_values_remain(self):
        source = self.source.copy()
        selected = ((source.Scenario == "ndc_highce_gbl") & (source.Region == "EUE")
                    & (source.Variable == "Emissions|GHG|Industry") & (source.Unit == "MtCO2e/yr"))
        source.loc[selected, "2050"] = np.nan
        for policy in ("preserve", "zero"):
            frame = self.data(source=source, config={**self.config, "missing_activity": policy}).emissions_response()
            net = self.select(frame, "ndc_highce_gbl", "industry", "Total industry").Value.iloc[0]
            residual = self.select(frame, "ndc_highce_gbl", "industry", "Other industry").Value.iloc[0]
            if policy == "preserve":
                self.assertTrue(np.isnan(net))
                self.assertTrue(np.isnan(residual))
            else:
                self.assertAlmostEqual(net, -205.2)
                self.assertAlmostEqual(residual, -91.2)
            self.assertAlmostEqual(self.select(frame, "ndc_highce_gbl", "geography", "Global net").Value.iloc[0], -114)
        # Japan's NDC aluminium output has negative reported GHG. Its negative
        # value must enter the global target rather than being clipped to zero.
        frame = self.data().emissions_response()
        self.assertLess(self.ghg["ndc_highce_gbl"]["Aluminium"][5][-1], 0)
        self.assertAlmostEqual(self.select(frame, "ndc_highce_gbl", "trajectory", "Three sectors").Value.iloc[-1], 1062)
        source = self.source.copy()
        selected = ((source.Scenario == "baseline_noce") & (source.Variable == "Emissions|GHG|Industry")
                    & (source.Unit == "MtCO2e/yr"))
        source.loc[selected, "2050"] = -2
        frame = self.data(source=source).emissions_response()
        net = self.select(frame, "baseline_highce_gbl", "industry", "Total industry").iloc[0]
        residual = self.select(frame, "baseline_highce_gbl", "industry", "Other industry").iloc[0]
        self.assertEqual(net.ReferenceValue, -16)
        self.assertEqual(residual.ReferenceValue, -618)
        self.assertAlmostEqual(net.Value, 850.6)

    def test_geography_rejects_duplicate_missing_and_inconsistent_membership(self):
        invalid = [pd.concat([self.membership, self.membership.iloc[[0]]], ignore_index=True),
                   self.membership.iloc[:-1],
                   self.membership.assign(EU_plus_partners=[False, True, True, True, True, True, False, False])]
        for membership in invalid:
            with self.subTest(membership=membership.to_dict()), self.assertRaises(ValueError):
                self.data(membership=membership)

    def test_known_partial_stack_extents_remain_visible_when_late_residual_and_net_are_missing(self):
        from omnia_results.consolidation_impacts import EmissionsResponseFigures

        class Builder(EmissionsResponseFigures):
            def __init__(self, data, config):
                self.data, self.config = data, config

        source = self.source.copy()
        row = ((source.Scenario == "ndc_highce_gbl") & (source.Region == "EUE")
               & (source.Unit == "MtCO2e/yr"))
        # Make known early segments much larger than every complete stack.
        # Missing industry makes the final residual and its net unknown; it
        # must not cause known contributions to disappear from axis bounds.
        source.loc[row & (source.Variable == "Emissions|GHG|Industry|Non-Metallic Minerals|Cement"), "2050"] = -5000
        source.loc[row & (source.Variable == "Emissions|GHG|Industry"), "2050"] = np.nan
        spec = Builder(self.data(source=source), self.config).emissions_response()
        try:
            ax = spec.figure.axes[2]  # NDC industrial-change panel.
            containers = {container.get_label(): container for container in ax.containers}
            known = [containers[label][-1] for label in ("Cement", "Iron and steel", "Aluminium")]
            self.assertAlmostEqual(sum(bar.get_height() for bar in known), -5192)
            lower, upper = ax.get_ylim()
            for bar in known:
                edges = (bar.get_y(), bar.get_y() + bar.get_height())
                self.assertTrue(np.isfinite(edges).all())
                self.assertGreaterEqual(min(edges), lower)
                self.assertLessEqual(max(edges), upper)
            self.assertTrue(np.isnan(containers["Other industry"][-1].get_height()))
            selected = self.select(spec.data, "ndc_highce_gbl", "industry", "Total industry")
            self.assertTrue(selected.Value.isna().all())
            markers = [collection for collection in ax.collections if isinstance(collection, PathCollection)]
            self.assertEqual(len(markers[0].get_offsets()), 3)
        finally:
            plt.close(spec.figure)

    def test_rendered_trajectories_signed_stacks_and_net_markers_match_source_values(self):
        from omnia_results.consolidation_impacts import EmissionsResponseFigures

        class Builder(EmissionsResponseFigures):
            def __init__(self, data, config):
                self.data, self.config = data, config

        spec = Builder(self.data(), self.config).emissions_response()
        try:
            self.assertEqual(len(spec.figure.axes), 6)
            for index, (ax, panel) in enumerate(zip(spec.figure.axes, "abcdef")):
                selected = spec.data[spec.data.Panel == panel]
                if index < 2:
                    actual = {line.get_label(): line for line in ax.lines if not line.get_label().startswith("_")}
                    self.assertEqual(set(actual), set(selected.Scenario))
                    for scenario, rows in selected.groupby("Scenario", sort=False):
                        np.testing.assert_array_equal(actual[scenario].get_xdata(), self.config["years"])
                        np.testing.assert_allclose(actual[scenario].get_ydata(), rows.Value)
                else:
                    components = selected[selected.Role == "plotted"]
                    actual = {container.get_label(): container for container in ax.containers}
                    self.assertEqual(set(actual), set(components.Series))
                    positive, negative = np.zeros(4), np.zeros(4)
                    for series, bars in actual.items():
                        values = components[components.Series == series].set_index("Policy").reindex(POLICIES[1:]).Value.to_numpy()
                        np.testing.assert_allclose([bar.get_height() for bar in bars], values)
                        np.testing.assert_allclose([bar.get_y() for bar in bars], np.where(values >= 0, positive, negative))
                        positive += np.maximum(values, 0)
                        negative += np.minimum(values, 0)
                    markers = [collection for collection in ax.collections if isinstance(collection, PathCollection)]
                    self.assertEqual(len(markers), 1)
                    offsets = markers[0].get_offsets()
                    expected = selected[selected.Role == "net"].set_index("Policy").reindex(POLICIES[1:]).Value.to_numpy()
                    np.testing.assert_allclose(offsets[np.argsort(offsets[:, 0]), 1], expected)
                if index % 2 == 0:
                    self.assertEqual(ax.get_ylim(), spec.figure.axes[index + 1].get_ylim())
        finally:
            plt.close(spec.figure)


if __name__ == "__main__":
    unittest.main()
