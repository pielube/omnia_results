"""Asymmetric fixtures exercise boundaries, references and source extraction."""

import json
from io import BytesIO, StringIO
from pathlib import Path
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd

from omnia_results.openprom_comparison_data import (
    CAPTURE, CLIMATES, EMISSIONS, ENERGY, GROSS_EMISSIONS, MODELS, MTOE_TO_EJ,
    OMNIA_CAPTURE, OMNIA_EMISSIONS, POLICIES, STEEL, OpenPromComparisonData,
    _ROW, _xml_rows, percent_change, read_openprom_csv, read_openprom_workbook,
)


class OpenPromComparisonDataTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(input="omnia.csv", openprom_workbook="openprom.xlsx",
                           years=[2019, 2050], openprom_years=[2019, 2050],
                           historical_year=2019, comparison_year=2050,
                           regions=["EU", "Rest"], missing_activity="preserve",
                           openprom_references={}, workbook_scenarios={})
        self.omnia, self.references, self.workbook = [], {}, {}
        for climate, climate_scale in zip(CLIMATES, (1, 3)):
            reference = f"OPEN_{climate}_Reference"
            self.config["openprom_references"][climate] = dict(path=f"{climate}.csv", scenario=reference)
            self.config["workbook_scenarios"][climate] = {}
            for index, (policy, ambition, _) in enumerate(POLICIES):
                scenario, factor = f"{climate}_{policy}", (1, .8, .6, .7, .4)[index]
                # CE history is deliberately different from the shared reference.
                for region, region_scale in (("EU", 1), ("Rest", 4)):
                    def add(variable, unit, start, final):
                        self.omnia.append([scenario, region, variable, unit,
                                           start * region_scale, final * region_scale])
                    for carrier, coefficient in zip(STEEL.fuels, (1, 2, 3, 4, 5)):
                        add(ENERGY + "|" + carrier, "EJ/yr", coefficient * climate_scale * (1 + index * 10),
                            coefficient * climate_scale * factor)
                        add(ENERGY + "|" + carrier + "|Detail", "EJ/yr", 99999, 99999)
                        add(ENERGY + "|" + carrier, "PJ/yr", 99999, 99999)
                    add(ENERGY, "EJ/yr", 99999, 99999)
                    add(OMNIA_EMISSIONS, "MtCO2e/yr", 20 * climate_scale, 100 * climate_scale * factor)
                    add(OMNIA_CAPTURE, "MtCO2/yr", 0, 50 * climate_scale * factor)
            for ambition, factor in (("Reference", 1), ("Medium", .5), ("High", .3)):
                scenario = f"OPEN_{climate}_{ambition}"
                rows = []
                # Country rows overlap World and must never be included in totals.
                for region, scale in (("World", climate_scale), ("AUT", climate_scale * 1000)):
                    for variable, unit, start, final in (
                            (ENERGY, "Mtoe", 100, 200 * factor),
                            (ENERGY + "|Coal", "Mtoe", 60, 120 * factor),
                            (ENERGY + "|Electricity", "Mtoe", 40, 80 * factor),
                            (ENERGY + "|Coal|Detail", "Mtoe", 99999, 99999),
                            (EMISSIONS, "Mt CO2/yr", 200, 400 * factor),
                            (GROSS_EMISSIONS, "Mt CO2/yr", 300, 500 * factor),
                            (CAPTURE, "Mt CO2/yr", 100, 100 * factor)):
                        rows.append(["OPEN-PROM 2.2", scenario, region, variable, unit,
                                     str(start * scale), str(final * scale)])
                table = pd.DataFrame(rows, columns=["Model", "Scenario", "Region", "Variable", "Unit", "2019", "2050"])
                if ambition == "Reference":
                    self.references[climate] = table
                else:
                    sheet = f"sheet_{climate}_{ambition}"
                    self.workbook[sheet] = table
                    self.config["workbook_scenarios"][climate][ambition.lower()] = dict(sheet=sheet, scenario=scenario)
        self.omnia = pd.DataFrame(self.omnia, columns=["Scenario", "Region", "Variable", "Unit", "2019", "2050"])

    def data(self, omnia=None, references=None, workbook=None, **settings):
        refs = self.references if references is None else references
        with patch("omnia_results.openprom_comparison_data.read_openprom_workbook",
                   return_value=self.workbook if workbook is None else workbook), \
             patch("omnia_results.openprom_comparison_data.read_openprom_csv",
                   side_effect=lambda path: refs[path.stem].copy()), \
             patch("omnia_results.metrics.pd.read_csv",
                   return_value=(self.omnia if omnia is None else omnia).copy()):
            return OpenPromComparisonData({**self.config, **settings}, Path("."))

    @staticmethod
    def select(frame, model=MODELS[0], climate="ndc", ambition="Medium", coverage="EU", metric="Energy", year=2050):
        return frame.loc[frame.Model.eq(model) & frame.Climate.eq(climate)
                         & frame.Ambition.eq(ambition) & frame.Coverage.eq(coverage)
                         & frame.Metric.eq(metric) & frame.Year.eq(year)].iloc[0]

    def test_global_energy_uses_exact_parents_world_once_and_unit_conversion(self):
        data = self.data()
        frame = data.frame()
        omnia = self.select(frame)
        openprom = self.select(frame, model=MODELS[1], coverage="Unspecified")
        self.assertEqual(omnia.Value, 60)
        self.assertAlmostEqual(openprom.Value, 100 * MTOE_TO_EJ)
        self.assertEqual(json.loads(omnia.SourceRegions), ["EU", "Rest"])
        self.assertEqual(json.loads(openprom.SourceRegions), ["World"])
        self.assertEqual(json.loads(omnia.SourceVariables), [ENERGY + "|" + fuel for fuel in STEEL.fuels])
        self.assertEqual((openprom.Unit, openprom.SourceUnit, openprom.ConversionFactor), ("EJ/yr", "Mtoe", MTOE_TO_EJ))
        self.assertTrue(data.aggregation_checks.Passed.all())

    def test_percentage_references_match_model_climate_year_and_history(self):
        frame = self.data().frame()
        for climate, scale in zip(CLIMATES, (1, 3)):
            omnia = self.select(frame, climate=climate)
            openprom = self.select(frame, model=MODELS[1], climate=climate, coverage="Unspecified")
            self.assertAlmostEqual(omnia.PercentChange, -20)
            self.assertAlmostEqual(openprom.PercentChange, -50)
            self.assertEqual(omnia.ReferenceScenario, f"{climate}_noce")
            self.assertEqual(openprom.ReferenceScenario, f"OPEN_{climate}_Reference")
            self.assertAlmostEqual(omnia.ReferenceValue, 75 * scale)
            self.assertAlmostEqual(openprom.ReferenceValue, 200 * scale * MTOE_TO_EJ)
            self.assertAlmostEqual(omnia.HistoricalValue, 75 * scale)
            self.assertAlmostEqual(omnia.HistoricalPercentChange, -20)
            self.assertEqual(omnia.AbsoluteChange, -15 * scale)
        historical_ce = self.select(frame, year=2019)
        self.assertEqual(historical_ce.Value, 825)
        self.assertEqual(historical_ce.HistoricalValue, 75)

    def test_policy_geography_never_changes_global_scope_or_invents_openprom_match(self):
        data = self.data()
        frame = data.frame()
        self.assertEqual(set(frame.loc[frame.Model.eq(MODELS[0]), "Coverage"]), {"Reference", "EU", "EU+"})
        self.assertEqual(set(frame.loc[frame.Model.eq(MODELS[1]), "Coverage"]), {"Unspecified"})
        self.assertEqual(self.select(frame, coverage="EU+").Value, 52.5)
        self.assertEqual(len(data.scenario_mapping()), 16)
        self.assertEqual(len(frame), 16 * 2 * 2)
        self.assertEqual(len(data.sources), 4)
        self.assertTrue(any("unconfirmed" in note and "geography" in note for note in data.notes))

    def test_emissions_keep_ghg_vs_energy_co2_and_capture_is_not_subtracted_twice(self):
        data = self.data()
        frame = data.frame()
        omnia = self.select(frame, metric="Emissions")
        openprom = self.select(frame, model=MODELS[1], coverage="Unspecified", metric="Emissions")
        self.assertEqual((omnia.Value, omnia.Unit), (400, "MtCO2e/yr"))
        self.assertEqual((openprom.Value, openprom.Unit), (200, "MtCO2/yr"))
        diagnostics = data.diagnostics()
        self.assertEqual(set(diagnostics.loc[diagnostics.Model.eq(MODELS[0]), "Metric"]), {"Capture"})
        net = self.select(diagnostics, model=MODELS[1], coverage="Unspecified", metric="Net emissions")
        gross = self.select(diagnostics, model=MODELS[1], coverage="Unspecified", metric="Gross emissions")
        capture = self.select(diagnostics, model=MODELS[1], coverage="Unspecified", metric="Capture")
        self.assertEqual(net.Value, gross.Value - capture.Value)
        self.assertEqual(openprom.Value, net.Value)

    def test_omnia_missing_activity_policy_is_explicit_and_logged(self):
        raw = self.omnia.copy()
        mask = raw.Scenario.eq("ndc_medce_eu") & raw.Region.eq("Rest") & raw.Variable.eq(ENERGY + "|Electricity") & raw.Unit.eq("EJ/yr")
        raw.loc[mask, "2050"] = np.nan
        preserved = self.data(omnia=raw)
        self.assertTrue(np.isnan(self.select(preserved.frame()).Value))
        zero = self.data(omnia=raw, missing_activity="zero")
        self.assertAlmostEqual(self.select(zero.frame()).Value, 56.8)
        self.assertEqual(set(zero.missing_inputs.Treatment), {"zero"})
        self.assertEqual(set(zero.missing_inputs.Model), {MODELS[0]})

    def test_openprom_world_blanks_absent_units_and_duplicate_rows_are_rejected(self):
        for issue, expected in (("blank", "must be complete"), ("absent", "required World"),
                                ("unit", "required World"), ("duplicate", "duplicate")):
            with self.subTest(issue=issue):
                references = {climate: frame.copy() for climate, frame in self.references.items()}
                raw = references["ndc"]
                mask = raw.Region.eq("World") & raw.Variable.eq(ENERGY)
                if issue == "blank":
                    raw.loc[mask, "2050"] = np.nan
                elif issue == "absent":
                    references["ndc"] = raw.loc[~mask]
                elif issue == "unit":
                    raw.loc[mask, "Unit"] = "EJ/yr"
                else:
                    references["ndc"] = pd.concat([raw, raw.loc[mask]], ignore_index=True)
                with self.assertRaisesRegex(ValueError, expected):
                    self.data(references=references, missing_activity="zero")

    def test_source_accounting_mismatches_are_rejected(self):
        for variable, check in ((EMISSIONS, "gross minus capture"), (ENERGY, "carrier sum")):
            with self.subTest(variable=variable):
                references = {climate: frame.copy() for climate, frame in self.references.items()}
                raw = references["ndc"]
                raw.loc[raw.Region.eq("World") & raw.Variable.eq(variable), "2050"] = "999"
                with self.assertRaisesRegex(ValueError, check):
                    self.data(references=references)

    def test_configuration_does_not_silently_drop_scenarios_years_or_regions(self):
        with self.assertRaisesRegex(ValueError, "all ten"):
            self.data(scenarios=["ndc_noce"])
        with self.assertRaisesRegex(ValueError, "required year 2019"):
            self.data(openprom_years=[2050])
        with self.assertRaisesRegex(ValueError, "source regions exactly"):
            self.data(regions=["EU"])
        with self.assertRaisesRegex(ValueError, "unique increasing"):
            self.data(years=[2050, 2019])

    def test_zero_negative_and_nonfinite_reference_percentages_are_undefined(self):
        for reference in (0, -10, np.nan, np.inf):
            self.assertTrue(np.isnan(percent_change(5, reference)))
        self.assertTrue(np.isnan(percent_change(np.inf, 10)))
        self.assertEqual(percent_change(5, 10), -50)


class SourceReaderTests(unittest.TestCase):
    def test_self_closing_formatted_rows_stream_with_a_bounded_buffer(self):
        class SmallChunks(BytesIO):
            def read(self, size=-1):
                return super().read(min(size, 97))

        formatted = [f'<row r="{index}" customFormat="1"/>'.encode()
                     for index in range(1, 1001)]
        actual = b'<row r="1001"><c r="A1001"><v>42</v></c></row>'
        trailing = [f'<row r="{index}" customFormat="1"/>'.encode()
                    for index in range(1002, 2002)]
        source = b"<worksheet><sheetData>" + b"".join(formatted) + actual + b"".join(trailing) + b"</sheetData></worksheet>"
        buffer_sizes = []

        def record_buffer(data):
            buffer_sizes.append(len(data))
            return _ROW.finditer(data)

        with patch("omnia_results.openprom_comparison_data._ROW") as pattern:
            pattern.finditer.side_effect = record_buffer
            result = list(_xml_rows(SmallChunks(source)))
        self.assertEqual(result, formatted + [actual] + trailing)
        # Whole formatted runs must be consumed on each chunk. Checking work
        # and maximum buffer size avoids a machine-dependent timing assertion.
        self.assertLessEqual(max(buffer_sizes), 97 + max(map(len, result)))
        self.assertLess(sum(buffer_sizes), 2 * len(source))

    def test_semicolon_csv_selects_steel_without_including_subsector_prices(self):
        source = StringIO("Model;Scenario;Region;Variable;Unit;2019;2050;\n"
                          f"OPEN-PROM 2.2;ref;World;{ENERGY};Mtoe;10;20;\n"
                          f"OPEN-PROM 2.2;ref;World;Price|{ENERGY};USD;3;4;\n")
        raw = read_openprom_csv(source)
        self.assertEqual(raw.Variable.tolist(), [ENERGY])
        self.assertEqual(raw.iloc[0]["2050"], 20)

    def test_xlsx_relationships_shared_strings_numeric_strings_and_trailing_formatting(self):
        with BytesIO() as path:
            shared = ["Model", "Scenario", "Region", "Variable", "Unit", "OPEN-PROM 2.2", "ref", "World", ENERGY, "Mtoe", "EU"]
            xml_strings = "<sst>" + "".join(f"<si><t>{value}</t></si>" for value in shared) + "</sst>"
            header = "".join(f'<c r="{column}1" t="s"><v>{index}</v></c>' for index, column in enumerate("ABCDE"))
            header += '<c r="F1"><v>2019</v></c><c r="G1"><v>2050</v></c>'
            metadata = "".join(f'<c r="{column}2" t="s"><v>{index}</v></c>' for column, index in zip("ABCDE", range(5, 10)))
            values = '<c r="F2" t="inlineStr"><is><t>10.5</t></is></c><c r="G2" t="str"><v>20.75</v></c>'
            before = "".join(f'<row r="{index}" customFormat="1"/>' for index in range(2, 1026))
            between = "".join(f'<row r="{index}" customFormat="1"/>' for index in range(1027, 2051))
            trailing = "".join(f'<row r="{index}" customFormat="1"/>' for index in range(2052, 42052))
            # More than one parser chunk of trailing formatting, and another
            # selected steel row after an earlier formatted run.
            self.assertGreater(len(trailing), 1024 * 1024)
            second_metadata = metadata.replace('<c r="C2" t="s"><v>7</v></c>', '<c r="C2" t="s"><v>10</v></c>')
            first_row = (metadata + values).replace('2"', '1026"')
            second_row = (second_metadata + values).replace('2"', '2051"')
            sheet = (f'<worksheet><sheetData><row r="1">{header}</row>{before}'
                     f'<row r="1026">{first_row}</row>{between}'
                     f'<row r="2051">{second_row}</row>{trailing}'
                     '<row r="900000"><c r="A900000" s="1"/></row></sheetData></worksheet>')
            with ZipFile(path, "w") as archive:
                archive.writestr("xl/workbook.xml", '<workbook xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Chosen" sheetId="42" r:id="rId7"/></sheets></workbook>')
                archive.writestr("xl/_rels/workbook.xml.rels", '<Relationships><Relationship Id="rId7" Type="worksheet" Target="worksheets/sheet42.xml"/><Relationship Id="s" Type="x/sharedStrings" Target="sharedStrings.xml"/></Relationships>')
                archive.writestr("xl/sharedStrings.xml", xml_strings)
                archive.writestr("xl/worksheets/sheet42.xml", sheet)
            raw = read_openprom_workbook(path, {"Chosen": "ref"})["Chosen"]
            self.assertEqual(len(raw), 2)
            self.assertEqual(raw.Region.tolist(), ["World", "EU"])
            self.assertEqual(raw.iloc[0]["2019"], "10.5")
            self.assertEqual(raw.iloc[0]["2050"], "20.75")
            with self.assertRaisesRegex(ValueError, "expected scenario"):
                read_openprom_workbook(path, {"Chosen": "wrong"})
            with self.assertRaisesRegex(ValueError, "missing workbook sheets"):
                read_openprom_workbook(path, {"Missing": "ref"})


if __name__ == "__main__":
    unittest.main()
