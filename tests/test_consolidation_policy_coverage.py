"""Raw group-total changes within and outside the CE policy geography."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from omnia_results.consolidation_data import ConsolidationData
from omnia_results.metrics import SECTORS


POLICIES = ("noce", "medce_eu", "highce_eu", "medce_gbl", "highce_gbl")
INSIDE, OUTSIDE = "Inside policy area", "Outside policy area"


class ConsolidationPolicyCoverageTests(unittest.TestCase):
    def setUp(self):
        self.regions = ["EUE", "EUW", "CHN", "IND", "USA"]
        self.config = {
            "input": "raw_coverage.csv", "policy_regions_csv": "policy_regions.csv",
            "scenarios": [f"{climate}_{policy}" for climate in ("baseline", "ndc")
                          for policy in POLICIES],
            "regions": self.regions, "years": [2019, 2030, 2050],
            "comparison_year": 2050, "missing_activity": "preserve",
        }
        self.membership = pd.DataFrame({
            "Region": self.regions,
            "Description": ["EU East", "EU West", "China", "India", "United States"],
            "EU": [True, True, False, False, False],
            "EU_plus_partners": [True, True, True, True, False],
        })
        self.production, self.energy = {}, {}
        rows = []

        def add(scenario, region, variable, unit, final, policy_index):
            rows.append([scenario, region, variable, unit,
                         final * .3 + policy_index * 10000,
                         final * .8 + policy_index * 333, final])
            wrong_unit = "Gt/yr" if unit == "Mt/yr" else "PJ/yr"
            rows.append([scenario, region, variable, wrong_unit, 99999, 99999, 99999])
            rows.append([scenario, region, variable + "|Detail", unit, 99999, 99999, 99999])

        for climate in ("baseline", "ndc"):
            raw_base = np.asarray([40, 20, 80, 10, 50] if climate == "baseline"
                                  else [8, 22, 100, 5, 65], dtype=float)
            production_factor = 1.5 if climate == "baseline" else 1
            energy_factor = 2.5 if climate == "baseline" else 1
            for p, policy in enumerate(POLICIES):
                scenario = f"{climate}_{policy}"
                self.production[scenario], self.energy[scenario] = {}, {}
                for sector, material_scale, energy_scale in zip(SECTORS, (1, 2, .1), (1, .7, .3)):
                    output = material_scale * (raw_base + p * production_factor * np.asarray([-3, 4, -8, 5, 7]))
                    self.production[scenario][sector.label] = output
                    carrier_values = []
                    for fuel_index, fuel in enumerate(sector.fuels, start=1):
                        quantity = energy_factor * energy_scale * fuel_index * (
                            np.asarray([2, 5, 4, 8, 12]) + p * np.asarray([-.1, .2, -.3, .4, -.5]))
                        carrier_values.append(quantity)
                        for region, value in zip(self.regions, quantity):
                            add(scenario, region, f"Final Energy|Industry|{sector.path}|{fuel}",
                                "EJ/yr", value, p)
                    self.energy[scenario][sector.label] = np.sum(carrier_values, axis=0)
                    for region, value in zip(self.regions, output):
                        parent = f"Production|{sector.path}"
                        if sector.key == "cement":
                            add(scenario, region, parent, "Mt/yr", value, p)
                        else:
                            add(scenario, region, parent + "|Primary", "Mt/yr", .7 * value, p)
                            add(scenario, region, parent + "|Secondary", "Mt/yr", .3 * value, p)
                            rows.append([scenario, region, parent, "Mt/yr", 99999, 99999, 99999])
                        rows.append([scenario, region, f"Final Energy|Industry|{sector.path}",
                                     "EJ/yr", 99999, 99999, 99999])
                        if sector.key == "cement":
                            rows.append([scenario, region, "Production|Non-Metallic Minerals|Cement Clinker",
                                         "Mt/yr", 99999, 99999, 99999])
                            rows.append([scenario, region, f"Final Energy|Industry|{sector.path}|Hydrogen",
                                         "EJ/yr", 99999, 99999, 99999])
        self.source = pd.DataFrame(rows, columns=["Scenario", "Region", "Variable", "Unit",
                                                  "2019", "2030", "2050"])

    def data(self, *, source=None, membership=None, config=None):
        source = self.source if source is None else source
        membership = self.membership if membership is None else membership
        config = self.config if config is None else config

        def read(path, *args, **kwargs):
            return (membership if Path(path).name == "policy_regions.csv" else source).copy()

        with patch("omnia_results.metrics.pd.read_csv", side_effect=read):
            data = ConsolidationData(config, Path("."))
            # Load the geography lazily while its in-memory source is available.
            data.policy_coverage_membership()
        return data

    def group_regions(self, scenario, series):
        inside = self.regions[:2] if scenario.endswith("_eu") else self.regions[:4]
        return inside if series == INSIDE else [region for region in self.regions if region not in inside]

    @staticmethod
    def select(frame, scenario, sector, series, unit="%"):
        return frame[(frame.Scenario == scenario) & (frame.Sector == sector)
                     & (frame.Series == series) & (frame.Unit == unit)].iloc[0]

    def assert_group_totals(self, frame, quantities, unit):
        self.assertEqual(len(frame), 96)
        self.assertEqual(frame.Year.unique().tolist(), [2050])
        self.assertEqual(len(frame[frame.Role == "plotted"]), 48)
        self.assertEqual(len(frame[frame.Role == "context"]), 48)
        self.assertFalse(frame.Scenario.str.endswith("_noce").any())
        for climate in ("ndc", "baseline"):
            reference = f"{climate}_noce"
            for policy in POLICIES[1:]:
                scenario = f"{climate}_{policy}"
                for sector in SECTORS:
                    absolute_sum = 0
                    for series in (INSIDE, OUTSIDE):
                        regions = self.group_regions(scenario, series)
                        indices = [self.regions.index(region) for region in regions]
                        current = quantities[scenario][sector.label][indices].sum()
                        base = quantities[reference][sector.label][indices].sum()
                        delta = current - base
                        percentage = self.select(frame, scenario, sector.label, series)
                        absolute = self.select(frame, scenario, sector.label, series, unit)
                        with self.subTest(scenario=scenario, sector=sector.key, series=series):
                            self.assertAlmostEqual(percentage.Value, 100 * delta / base)
                            self.assertAlmostEqual(absolute.Value, delta)
                            for row in (percentage, absolute):
                                self.assertAlmostEqual(row.ScenarioValue, current)
                                self.assertAlmostEqual(row.ReferenceValue, base)
                                self.assertAlmostEqual(row.Numerator, delta)
                                self.assertEqual(row.NumeratorUnit, unit)
                                self.assertEqual(row.ReferenceScenario, reference)
                                self.assertEqual(row.ReferenceYear, 2050)
                                self.assertEqual(row.Offset, 0)
                                self.assertEqual(json.loads(row.SourceRegions), regions)
                                self.assertEqual(row.RegionCount, len(regions))
                                self.assertEqual(row.Scope, series)
                                self.assertEqual(row.PolicyArea, "EU" if policy.endswith("_eu") else "EU + partners")
                            self.assertAlmostEqual(percentage.Denominator, base)
                            self.assertEqual(percentage.DenominatorUnit, unit)
                            self.assertTrue(np.isnan(absolute.Denominator))
                            self.assertEqual(absolute.DenominatorUnit, "")
                            self.assertEqual(percentage.ConversionFactor, 100)
                            self.assertEqual(absolute.ConversionFactor, 1)
                        absolute_sum += absolute.Value
                    self.assertAlmostEqual(absolute_sum,
                                           (quantities[scenario][sector.label] - quantities[reference][sector.label]).sum())

    def test_material_changes_use_own_climate_and_policy_group_totals_and_close_global_delta(self):
        frame = self.data().material_policy_coverage()
        self.assert_group_totals(frame, self.production, "Mt/yr")
        inside = self.select(frame, "ndc_medce_eu", "Cement", INSIDE)
        self.assertAlmostEqual(inside.Value, 100 / 30)
        regional_mean = np.mean(np.asarray([-3, 4]) / np.asarray([8, 22])) * 100
        self.assertNotAlmostEqual(inside.Value, regional_mean)
        self.assertNotAlmostEqual(inside.Value, 100 / 200)
        expanded = self.select(frame, "ndc_medce_gbl", "Cement", INSIDE)
        self.assertAlmostEqual(expanded.Value, -600 / 135)

    def test_energy_changes_sum_exact_carriers_then_take_the_ratio_of_policy_group_totals(self):
        frame = self.data().energy_policy_coverage()
        self.assert_group_totals(frame, self.energy, "EJ/yr")
        inside = self.select(frame, "ndc_medce_eu", "Cement", INSIDE)
        self.assertAlmostEqual(inside.Value, 100 * 1 / 70)
        self.assertNotAlmostEqual(inside.Value, np.mean(np.asarray([-.1, .2]) / np.asarray([2, 5])) * 100)
        outside = self.select(frame, "baseline_highce_gbl", "Iron and steel", OUTSIDE)
        self.assertAlmostEqual(outside.Value, -100 * 2 / 12)

    def test_exact_parent_and_unit_selection_excludes_clinker_nested_children_and_cement_hydrogen(self):
        data = self.data()
        material_variables = {sector.label: ([f"Production|{sector.path}"] if sector.key == "cement" else
                                             [f"Production|{sector.path}|Primary", f"Production|{sector.path}|Secondary"])
                              for sector in SECTORS}
        energy_variables = {sector.label: [f"Final Energy|Industry|{sector.path}|{fuel}" for fuel in sector.fuels]
                            for sector in SECTORS}
        exact = self.source[((self.source.Unit == "Mt/yr") & self.source.Variable.isin(sum(material_variables.values(), [])))
                            | ((self.source.Unit == "EJ/yr") & self.source.Variable.isin(sum(energy_variables.values(), [])))]
        clean = self.data(source=exact)
        for getter, variables in (("material_policy_coverage", material_variables),
                                  ("energy_policy_coverage", energy_variables)):
            original = getattr(data, getter)()
            pd.testing.assert_frame_equal(original, getattr(clean, getter)())
            for row in original.itertuples():
                self.assertEqual(json.loads(row.SourceVariables), variables[row.Sector])
                self.assertEqual(json.loads(row.SourceCoefficients), [1] * len(variables[row.Sector]))
            prefix = "Material production" if getter == "material_policy_coverage" else "Final energy"
            self.assertEqual(original.loc[original.Role == "plotted", "Metric"].unique().tolist(),
                             [prefix + " percentage change"])
            self.assertEqual(original.loc[original.Role == "context", "Metric"].unique().tolist(),
                             [prefix + " absolute change"])

    def test_each_policy_area_membership_partitions_all_regions_once_and_matches_plot_sources(self):
        data = self.data()
        membership = data.policy_coverage_membership()
        self.assertEqual(len(membership), 2 * len(self.regions))
        self.assertEqual(set(membership.PolicyArea), {"EU", "EU + partners"})
        for area, inside in (("EU", self.regions[:2]), ("EU + partners", self.regions[:4])):
            selected = membership[membership.PolicyArea == area]
            self.assertEqual(selected.Region.tolist(), self.regions)
            self.assertFalse(selected.Region.duplicated().any())
            self.assertEqual(selected.loc[selected.InsidePolicyArea, "Region"].tolist(), inside)
            self.assertEqual(selected.loc[selected.Scope == INSIDE, "Region"].tolist(), inside)
            outside = [region for region in self.regions if region not in inside]
            self.assertEqual(selected.loc[selected.Scope == OUTSIDE, "Region"].tolist(), outside)
            for frame in (data.material_policy_coverage(), data.energy_policy_coverage()):
                for scope in (INSIDE, OUTSIDE):
                    expected = selected.loc[selected.Scope == scope, "Region"].tolist()
                    sources = frame.loc[(frame.PolicyArea == area) & (frame.Scope == scope), "SourceRegions"]
                    self.assertTrue(all(json.loads(source) == expected for source in sources))

    def test_scenario_order_is_fixed_and_each_sector_panel_uses_its_own_climate(self):
        data = self.data()
        reversed_data = self.data(config={**self.config, "scenarios": self.config["scenarios"][::-1]})
        for getter in ("material_policy_coverage", "energy_policy_coverage"):
            frame = getattr(data, getter)()
            pd.testing.assert_frame_equal(frame, getattr(reversed_data, getter)())
            for panels, climate, label in (("abc", "ndc", "NDC"), ("def", "baseline", "NDC+LTT")):
                selected = frame[frame.Panel.isin(list(panels))]
                self.assertTrue(selected.Scenario.str.startswith(climate + "_").all())
                self.assertEqual(selected.ClimatePathway.unique().tolist(), [label])
                for panel in panels:
                    self.assertEqual(selected[selected.Panel == panel].Policy.drop_duplicates().tolist(), list(POLICIES[1:]))

    def test_blank_material_values_affect_only_their_policy_group_and_follow_the_activity_policy(self):
        source = self.source.copy()
        selected = ((source.Scenario == "ndc_highce_eu") & (source.Region == "EUE")
                    & (source.Variable == "Production|Non-Metallic Minerals|Cement") & (source.Unit == "Mt/yr"))
        source.loc[selected, "2050"] = np.nan
        for policy in ("preserve", "zero"):
            data = self.data(source=source, config={**self.config, "missing_activity": policy})
            frame = data.material_policy_coverage()
            inside = self.select(frame, "ndc_highce_eu", "Cement", INSIDE)
            absolute = self.select(frame, "ndc_highce_eu", "Cement", INSIDE, "Mt/yr")
            if policy == "preserve":
                self.assertTrue(np.isnan(inside.Value))
                self.assertTrue(np.isnan(absolute.Value))
            else:
                self.assertAlmostEqual(inside.Value, 0)
                self.assertAlmostEqual(absolute.Value, 0)
            self.assertAlmostEqual(self.select(frame, "ndc_highce_eu", "Cement", OUTSIDE, "Mt/yr").Value, 8)
            self.assertEqual(data.results["ndc_highce_eu"].coverage[0]["Treatment"], policy)

    def test_absent_metal_reference_route_preserves_unknown_group_without_dropping_region(self):
        selected = ((self.source.Scenario == "baseline_noce") & (self.source.Region == "CHN")
                    & (self.source.Variable == "Production|Iron and Steel|Secondary") & (self.source.Unit == "Mt/yr"))
        for policy in ("preserve", "zero"):
            data = self.data(source=self.source.loc[~selected], config={**self.config, "missing_activity": policy})
            frame = data.material_policy_coverage()
            eu_inside = self.select(frame, "baseline_medce_eu", "Iron and steel", INSIDE)
            self.assertAlmostEqual(eu_inside.Value, 2.5)
            expanded = self.select(frame, "baseline_medce_gbl", "Iron and steel", INSIDE)
            absolute = self.select(frame, "baseline_medce_gbl", "Iron and steel", INSIDE, "Mt/yr")
            if policy == "preserve":
                self.assertTrue(np.isnan(expanded.Value))
                self.assertTrue(np.isnan(absolute.Value))
                self.assertTrue(np.isnan(expanded.ReferenceValue))
            else:
                self.assertAlmostEqual(expanded.ReferenceValue, 252)
                self.assertAlmostEqual(absolute.Value, 30)
                self.assertAlmostEqual(expanded.Value, 100 * 30 / 252)
            issues = [entry for entry in data.results["baseline_noce"].coverage if entry["Variable"].endswith("|Secondary")]
            self.assertEqual(len(issues), 3)
            self.assertTrue(all(entry["Issue"] == "absent row" and entry["Treatment"] == policy for entry in issues))

    def test_missing_energy_carrier_or_reference_cell_preserves_unknown_sum_and_complement(self):
        for scenario, region, fuel in (("ndc_highce_gbl", "IND", "Gases"),
                                       ("ndc_noce", "USA", "Solids")):
            source = self.source.copy()
            variable = f"Final Energy|Industry|Non-Metallic Minerals|Cement|{fuel}"
            selected = ((source.Scenario == scenario) & (source.Region == region)
                        & (source.Variable == variable) & (source.Unit == "EJ/yr"))
            source.loc[selected, "2050"] = np.nan
            for policy in ("preserve", "zero"):
                with self.subTest(scenario=scenario, policy=policy):
                    frame = self.data(source=source, config={**self.config, "missing_activity": policy}).energy_policy_coverage()
                    series = INSIDE if region == "IND" else OUTSIDE
                    row = self.select(frame, "ndc_highce_gbl", "Cement", series)
                    absolute = self.select(frame, "ndc_highce_gbl", "Cement", series, "EJ/yr")
                    if policy == "preserve":
                        self.assertTrue(np.isnan(row.Value))
                        self.assertTrue(np.isnan(absolute.Value))
                    else:
                        expected_delta, denominator = (-11.2, 190) if region == "IND" else (28, 72)
                        self.assertAlmostEqual(absolute.Value, expected_delta)
                        self.assertAlmostEqual(row.Value, 100 * expected_delta / denominator)
                    other = self.select(frame, "ndc_highce_gbl", "Cement", OUTSIDE if series == INSIDE else INSIDE)
                    self.assertTrue(np.isfinite(other.Value))

    def test_nonpositive_references_leave_signed_absolute_changes_but_percentages_undefined(self):
        for reference_value in (0, -1):
            source = self.source.copy()
            production_variable = "Production|Non-Metallic Minerals|Cement"
            energy_variables = [f"Final Energy|Industry|Non-Metallic Minerals|Cement|{fuel}" for fuel in SECTORS[0].fuels]
            selected = ((source.Scenario == "baseline_noce") & source.Region.isin(self.regions[:2])
                        & (((source.Variable == production_variable) & (source.Unit == "Mt/yr"))
                           | (source.Variable.isin(energy_variables) & (source.Unit == "EJ/yr"))))
            source.loc[selected, "2050"] = reference_value
            data = self.data(source=source)
            for getter, unit, count in (("material_policy_coverage", "Mt/yr", 2),
                                        ("energy_policy_coverage", "EJ/yr", 8)):
                frame = getattr(data, getter)()
                for policy in POLICIES[1:3]:
                    scenario = f"baseline_{policy}"
                    percent = self.select(frame, scenario, "Cement", INSIDE)
                    absolute = self.select(frame, scenario, "Cement", INSIDE, unit)
                    self.assertTrue(np.isnan(percent.Value))
                    self.assertAlmostEqual(percent.ReferenceValue, reference_value * count)
                    self.assertAlmostEqual(absolute.Value, absolute.ScenarioValue - reference_value * count)
                    self.assertTrue(np.isfinite(absolute.Value))
        source = self.source.copy()
        selected = ((source.Scenario == "ndc_medce_eu") & source.Region.isin(self.regions[:2])
                    & (source.Variable == "Production|Non-Metallic Minerals|Cement") & (source.Unit == "Mt/yr"))
        source.loc[selected, "2050"] = -10
        frame = self.data(source=source).material_policy_coverage()
        self.assertAlmostEqual(self.select(frame, "ndc_medce_eu", "Cement", INSIDE, "Mt/yr").Value, -50)
        self.assertAlmostEqual(self.select(frame, "ndc_medce_eu", "Cement", INSIDE).Value, -100 * 50 / 30)

    def test_policy_geography_requires_unique_complete_regions_valid_flags_and_nested_areas(self):
        invalid = [self.membership.iloc[:-1],
                   pd.concat([self.membership, self.membership.iloc[[0]]], ignore_index=True),
                   self.membership.assign(Region=["EUE", "EUW", "CHN", "IND", "MISSING"]),
                   self.membership.assign(EU=[True, True, False, False, True]),
                   self.membership.assign(EU=[True, "unknown", False, False, False]),
                   self.membership.drop(columns="EU_plus_partners")]
        for membership in invalid:
            with self.subTest(membership=membership.to_dict()), self.assertRaises(ValueError):
                self.data(membership=membership)
        for config in ({**self.config, "scenarios": self.config["scenarios"][:-1]},
                       {**self.config, "comparison_year": 2040}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.data(config=config)

    def test_paired_bars_use_exact_signed_percentages_in_policy_order_and_shared_sector_scales(self):
        from omnia_results.consolidation_policy_coverage import PolicyCoverageFigures

        class Builder(PolicyCoverageFigures):
            def __init__(self, data, config):
                self.data, self.config = data, config

        builder = Builder(self.data(), self.config)
        for getter in ("material_policy_coverage", "energy_policy_coverage"):
            spec = getattr(builder, getter)()
            try:
                self.assertEqual(len(spec.figure.axes), 6)
                self.assertEqual(len(spec.data), 96)
                for index, (ax, panel) in enumerate(zip(spec.figure.axes, "abcdef")):
                    selected = spec.data[(spec.data.Panel == panel) & (spec.data.Role == "plotted")]
                    containers = {container.get_label(): container for container in ax.containers}
                    self.assertEqual(set(containers), {INSIDE, OUTSIDE})
                    centers = {}
                    for scope, bars in containers.items():
                        expected = selected[selected.Scope == scope].set_index("Policy").loc[list(POLICIES[1:]), "Value"]
                        self.assertEqual(len(bars), 4)
                        np.testing.assert_allclose([bar.get_height() for bar in bars], expected)
                        np.testing.assert_allclose([bar.get_y() for bar in bars], 0)
                        centers[scope] = np.asarray([bar.get_x() + bar.get_width() / 2 for bar in bars])
                        for value in expected:
                            self.assertLessEqual(ax.get_ylim()[0], value)
                            self.assertGreaterEqual(ax.get_ylim()[1], value)
                    self.assertTrue((centers[INSIDE] < centers[OUTSIDE]).all())
                    np.testing.assert_allclose((centers[INSIDE] + centers[OUTSIDE]) / 2, ax.get_xticks(), atol=1e-12)
                    title = ax.get_title(loc="left")
                    self.assertIn(SECTORS[index % 3].label, title)
                    self.assertTrue(title.endswith("NDC" if index < 3 else "NDC+LTT"))
                    if index < 3:
                        self.assertEqual(ax.get_ylim(), spec.figure.axes[index + 3].get_ylim())
            finally:
                plt.close(spec.figure)


if __name__ == "__main__":
    unittest.main()
