"""Auditable global steel comparison; preserves the two models' boundaries.

OPEN-PROM reports final energy and energy-demand CO2, while OMNIA reports
final energy and sector GHG. Percentage responses use the corresponding
model and climate reference; they do not harmonise the emissions boundaries.
The workbook is streamed from its source ZIP/XML, keeping selected steel
rows only. No cached extraction is accepted as an analysis input.
"""

import json
from numbers import Integral
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .metrics import Results, SECTORS


MODELS = ("OMNIA-ENGAGE", "OPEN-PROM")
CLIMATES = ("ndc", "baseline")
CLIMATE_LABELS = {"ndc": "NDC", "baseline": "NDC+LTT"}
POLICIES = (("noce", "Reference", "Reference"),
            ("medce_eu", "Medium", "EU"), ("highce_eu", "High", "EU"),
            ("medce_gbl", "Medium", "EU+"), ("highce_gbl", "High", "EU+"))
MTOE_TO_EJ = 0.041868
STEEL = next(sector for sector in SECTORS if sector.key == "steel")
ENERGY = "Final Energy|Industry|Iron and Steel"
EMISSIONS = "Emissions|CO2|Energy|Demand|Industry|Iron and Steel"
GROSS_EMISSIONS = "Gross Emissions|CO2|Energy|Demand|Industry|Iron and Steel"
CAPTURE = "Carbon Capture|Energy|Demand|Iron and Steel"
OMNIA_EMISSIONS = "Emissions|GHG|Industry|Iron and Steel"
OMNIA_CAPTURE = "Carbon Capture|Industry|Iron and Steel"
# EU is selected once; its 27 country entries are overlapping diagnostics.
OPENPROM_WORLD_REGIONS = ("CAZ", "CHA", "EU", "GBR", "IND", "JPN", "LAM",
                          "MEA", "NEU", "OAS", "REF", "SSA", "USA")
KEYS = ["Model", "Scenario", "Region", "Variable", "Unit"]
_ROW = re.compile(rb"<(?:\w+:)?row\b[^>]*?(?:/>|>.*?</(?:\w+:)?row>)", re.DOTALL)
_CELL = re.compile(rb"<(?:\w+:)?c\b[^>]*?(?:/>|>.*?</(?:\w+:)?c>)", re.DOTALL)
_CELL_ADDRESS = re.compile(rb'\br="([A-Z]+)\d+"')
_VALUE = re.compile(rb"<(?:\w+:)?v(?:\s[^>]*)?>(.*?)</(?:\w+:)?v>", re.DOTALL)
_TYPE = re.compile(rb'\bt="([^\"]+)"')


def percent_change(value, reference):
    """Undefined for a missing/nonfinite quantity or a nonpositive reference."""
    if not (np.isfinite(value) and np.isfinite(reference) and reference > 0):
        return np.nan
    return 100 * (value / reference - 1)


def _wanted_variable(value):
    return (value in {ENERGY, EMISSIONS, GROSS_EMISSIONS, CAPTURE}
            or isinstance(value, str) and value.startswith(ENERGY + "|"))


def _xml_cell_value(cell, strings):
    """Read a cached XLSX value, including numeric values stored as strings."""
    kind = _TYPE.search(cell)
    kind = kind.group(1) if kind else b"n"
    value = _VALUE.search(cell)
    if kind == b"s":
        return strings[int(value.group(1))] if value else None
    if kind == b"inlineStr":
        element = ET.fromstring(cell)
        return "".join(child.text or "" for child in element.iter()
                       if child.tag.rsplit("}", 1)[-1] == "t") or None
    if kind == b"e":
        raise ValueError("Excel error cell in selected OPEN-PROM source data")
    return value.group(1).decode("utf-8") if value else None


def _xml_rows(stream):
    """Yield complete XML rows without parsing unselected numeric cells.

    A bounded chunk plus its final incomplete row is retained. Empty trailing
    worksheet formatting is ignored by the caller rather than interpreted as
    extra observations. XML entities remain escaped until cell decoding.
    """
    buffer = b""
    while chunk := stream.read(1024 * 1024):
        buffer += chunk
        last = 0
        for match in _ROW.finditer(buffer):
            yield match.group()
            last = match.end()
        if last:
            buffer = buffer[last:]
    if re.search(rb"<(?:\w+:)?row\b", buffer):
        raise ValueError("Incomplete XML row in OPEN-PROM workbook")


def read_openprom_workbook(path, scenarios):
    """Select steel rows in each configured sheet, reading the original XLSX.

    ``scenarios`` maps sheet names to their exact expected scenario names.
    Sheet names are resolved through workbook relationships, not sheet order.
    """
    output = {}
    with ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {item.attrib["Id"]: item.attrib["Target"] for item in relations}
        strings = []
        shared = next((item.attrib["Target"] for item in relations
                       if item.attrib.get("Type", "").endswith("/sharedStrings")), None)

        def member(target):
            return str(PurePosixPath(target.lstrip("/")) if target.startswith("/")
                       else PurePosixPath("xl") / target)

        if shared:
            root = ET.fromstring(archive.read(member(shared)))
            strings = ["".join(node.text or "" for node in item.iter()
                               if node.tag.rsplit("}", 1)[-1] == "t") for item in root]
        sheets = {}
        for sheet in workbook.iter():
            if sheet.tag.rsplit("}", 1)[-1] == "sheet":
                relationship = next(value for key, value in sheet.attrib.items()
                                    if key.rsplit("}", 1)[-1] == "id")
                sheets[sheet.attrib["name"]] = member(targets[relationship])
        if absent := set(scenarios) - set(sheets):
            raise ValueError(f"OPEN-PROM missing workbook sheets: {sorted(absent)}")
        for sheet, expected in scenarios.items():
            header, rows = None, []
            with archive.open(sheets[sheet]) as stream:
                for row_xml in _xml_rows(stream):
                    cells = _CELL.finditer(row_xml)
                    if header is None:
                        header = {}
                        for cell in cells:
                            address = _CELL_ADDRESS.search(cell.group())
                            if address:
                                value = _xml_cell_value(cell.group(), strings)
                                if value is not None:
                                    header[address.group(1).decode()] = str(value)
                        if set(KEYS) - set(header.values()):
                            raise ValueError(f"OPEN-PROM sheet {sheet} missing metadata columns")
                        if len(header.values()) != len(set(header.values())):
                            raise ValueError(f"OPEN-PROM sheet {sheet} duplicate source columns")
                        continue
                    # Metadata columns precede years in the IAMC source. Use
                    # addresses rather than assuming every blank cell exists.
                    metadata, remaining = {}, []
                    for cell in cells:
                        address = _CELL_ADDRESS.search(cell.group())
                        if not address:
                            continue
                        column = address.group(1).decode()
                        name = header.get(column)
                        if name in KEYS:
                            metadata[name] = _xml_cell_value(cell.group(), strings)
                        elif name is not None:
                            remaining.append((name, cell.group()))
                        if len(metadata) == len(KEYS):
                            break
                    if not any(value is not None for value in metadata.values()):
                        continue
                    if any(metadata.get(key) is None for key in KEYS):
                        raise ValueError(f"OPEN-PROM sheet {sheet} blank source metadata")
                    if metadata["Scenario"] != expected:
                        raise ValueError(f"OPEN-PROM sheet {sheet} expected scenario {expected!r}, "
                                         f"found {metadata['Scenario']!r}")
                    if not _wanted_variable(metadata["Variable"]):
                        continue
                    values = dict(metadata)
                    for name, cell in remaining:
                        if name.isdigit():
                            values[name] = _xml_cell_value(cell, strings)
                    for cell in cells:
                        address = _CELL_ADDRESS.search(cell.group())
                        name = header.get(address.group(1).decode()) if address else None
                        if name and name.isdigit():
                            values[name] = _xml_cell_value(cell.group(), strings)
                    rows.append(values)
            if header is None:
                raise ValueError(f"OPEN-PROM sheet {sheet} is empty")
            output[sheet] = pd.DataFrame(rows, columns=list(header.values()))
            print(f"Read OPEN-PROM sheet {sheet}: {len(rows)} selected steel rows", flush=True)
    return output


def read_openprom_csv(path):
    raw = pd.read_csv(path, sep=";")
    if absent := set(KEYS) - set(raw):
        raise ValueError(f"OPEN-PROM CSV missing source columns: {sorted(absent)}")
    return raw.loc[raw.Variable.map(_wanted_variable)].copy()


class OpenPromComparisonData:
    models, climates = MODELS, CLIMATES

    def __init__(self, config, base):
        self.config, self.base = config, Path(base)
        self.regions = list(config["regions"])
        self.years = {MODELS[0]: list(config["years"]),
                      MODELS[1]: list(config["openprom_years"])}
        self.historical_year = config.get("historical_year", 2019)
        self.comparison_year = config.get("comparison_year", 2050)
        self.missing_activity = config.get("missing_activity", "preserve")
        self._checks, self._missing, self._frame, self._diagnostics = [], [], None, None
        if self.missing_activity not in {"zero", "preserve"}:
            raise ValueError("missing_activity must be 'zero' or 'preserve'")
        if not self.regions or len(self.regions) != len(set(self.regions)):
            raise ValueError("Configured OMNIA regions must be nonempty and unique")
        for model, years in self.years.items():
            if (not years or any(isinstance(y, bool) or not isinstance(y, Integral) for y in years)
                    or years != sorted(set(years))):
                raise ValueError(f"{model} years must contain unique increasing integers")
            for year in (self.historical_year, self.comparison_year):
                if isinstance(year, bool) or not isinstance(year, Integral) or year not in years:
                    raise ValueError(f"{model} required year {year} is omitted from configured years")
        if self.historical_year >= self.comparison_year:
            raise ValueError("Historical year must precede comparison year")
        expected = [f"{climate}_{policy}" for climate in CLIMATES for policy, _, _ in POLICIES]
        selected = config.get("scenarios", expected)
        if len(selected) != len(expected) or set(selected) != set(expected):
            raise ValueError("Select all ten OMNIA climate/CE scenarios exactly once")
        self.source = (self.base / config["input"]).resolve()
        self.sources = {MODELS[0]: self.source,
                        "OPEN-PROM workbook": (self.base / config["openprom_workbook"]).resolve()}
        self.results = {scenario: Results(self.source, scenario, self.regions, self.years[MODELS[0]],
                                          self.missing_activity) for scenario in expected}
        self._mapping = []
        for climate in CLIMATES:
            reference = f"{climate}_noce"
            self._mapping.extend(dict(Model=MODELS[0], Climate=climate,
                                      ClimateLabel=CLIMATE_LABELS[climate], Ambition=ambition,
                                      Coverage=coverage, Scenario=f"{climate}_{policy}",
                                      ReferenceScenario=reference, Source=str(self.source), Sheet="",
                                      Scope="Global; all configured OMNIA regions")
                                 for policy, ambition, coverage in POLICIES)
        workbook_specs = config["workbook_scenarios"]
        sheets = {workbook_specs[c][a]["sheet"]: workbook_specs[c][a]["scenario"]
                  for c in CLIMATES for a in ("medium", "high")}
        if len(sheets) != 4 or len(set(sheets.values())) != 4:
            raise ValueError("OPEN-PROM requires four distinct ambition sheets and scenarios")
        workbook = read_openprom_workbook(self.sources["OPEN-PROM workbook"], sheets)
        self.openprom = {}
        for climate in CLIMATES:
            spec = config["openprom_references"][climate]
            path, reference = (self.base / spec["path"]).resolve(), spec["scenario"]
            self.sources[f"OPEN-PROM {CLIMATE_LABELS[climate]} reference"] = path
            tables = [("Reference", reference, read_openprom_csv(path), path, "")]
            for ambition in ("medium", "high"):
                item = workbook_specs[climate][ambition]
                tables.append((ambition.title(), item["scenario"], workbook[item["sheet"]],
                               self.sources["OPEN-PROM workbook"], item["sheet"]))
            for ambition, scenario, raw, source, sheet in tables:
                if scenario in self.openprom:
                    raise ValueError(f"Repeated OPEN-PROM scenario {scenario!r}")
                self.openprom[scenario] = self._validate_openprom(raw, scenario)
                self._mapping.append(dict(Model=MODELS[1], Climate=climate,
                                          ClimateLabel=CLIMATE_LABELS[climate], Ambition=ambition,
                                          Coverage="Unspecified", Scenario=scenario,
                                          ReferenceScenario=reference, Source=str(source), Sheet=sheet,
                                          Scope="Global; reported World aggregate"))
        self.notes = [
            "Global OMNIA totals sum all configured model regions once. EU and EU+ describe CE policy coverage, not the plotted geography; EU+ means EU plus partners.",
            "OPEN-PROM uses its reported World aggregate once; country rows and the overlapping EU aggregate are never added to World.",
            "OPEN-PROM Medium/High are paired by climate and ambition label only. Their intervention geography and assumptions are unconfirmed; no match to OMNIA EU or EU+ coverage is asserted.",
            "OMNIA final energy sums the five exact steel carrier parents in EJ/yr. OPEN-PROM uses its steel final-energy parent in Mtoe, multiplied by 0.041868 EJ/Mtoe.",
            "Emissions retain different accounting scopes: OMNIA steel sector GHG (MtCO2e/yr) versus OPEN-PROM steel energy-demand CO2 after capture (MtCO2/yr). OPEN-PROM steel process emissions and non-CO2 gases are unavailable. Percentage changes retain this difference and do not establish equal absolute emissions.",
            "CE responses use each model's corresponding NDC or NDC+LTT reference at the same year. Historical changes use that model/climate reference's historical observation; neither model is scaled to the other.",
            "OPEN-PROM capture and gross energy CO2 are supplementary diagnostics; capture is already reflected in net energy CO2 and is not subtracted again. No OMNIA gross emissions total is inferred from GHG plus CO2 capture.",
            f"OMNIA missing additive activity uses the explicit {self.missing_activity!r} policy. Missing OPEN-PROM World data are rejected; no zero filling is applied to OPEN-PROM.",
        ]

    def _validate_openprom(self, raw, scenario):
        raw = raw.copy()
        raw.columns = [str(column) for column in raw.columns]
        if absent := set(KEYS) - set(raw):
            raise ValueError(f"OPEN-PROM missing source columns: {sorted(absent)}")
        if raw[KEYS].isna().any().any():
            raise ValueError("OPEN-PROM blank source metadata")
        if set(raw.Scenario) != {scenario}:
            raise ValueError(f"OPEN-PROM expected exact scenario {scenario!r}")
        if set(raw.Model) != {"OPEN-PROM 2.2"}:
            raise ValueError("OPEN-PROM source must identify exactly OPEN-PROM 2.2")
        if raw.duplicated(KEYS).any():
            raise ValueError("OPEN-PROM duplicate scenario/region/variable/unit rows")
        years = sorted(int(column) for column in raw if column.isdigit())
        if absent := set(self.years[MODELS[1]]) - set(years):
            raise ValueError(f"OPEN-PROM configured years absent from source: {sorted(absent)}")
        for year in years:
            raw[str(year)] = pd.to_numeric(raw[str(year)], errors="raise")
        if np.isinf(raw[[str(y) for y in years]].to_numpy(dtype=float)).any():
            raise ValueError("OPEN-PROM infinite source values")
        required = {ENERGY: "Mtoe", EMISSIONS: "Mt CO2/yr",
                    GROSS_EMISSIONS: "Mt CO2/yr", CAPTURE: "Mt CO2/yr"}
        for variable, unit in required.items():
            selected = raw.loc[raw.Variable.eq(variable) & raw.Unit.eq(unit)]
            if selected.loc[selected.Region.eq("World")].empty:
                raise ValueError(f"OPEN-PROM required World source series absent: {scenario}: {variable} [{unit}]")
            if not np.isfinite(selected.loc[selected.Region.eq("World"), [str(y) for y in years]].to_numpy(dtype=float)).all():
                raise ValueError(f"OPEN-PROM World source observations must be complete: {scenario}: {variable}")
        self._openprom_checks(raw, scenario, years)
        return raw

    def _check(self, scenario, kind, year, value, reconstructed, terms, unit):
        # Exported OPEN-PROM observations have four decimal places. A sum of
        # n rounded terms can differ by at most (n+1)*0.00005.
        tolerance = (terms + 1) * 0.00005001
        difference = value - reconstructed
        passed = bool(np.isfinite(difference) and abs(difference) <= tolerance)
        self._checks.append(dict(Model=MODELS[1], Scenario=scenario, Check=kind, Year=year,
                                 Reported=value, Reconstructed=reconstructed, Difference=difference,
                                 Tolerance=tolerance, Unit=unit, Passed=passed))
        if not passed:
            raise ValueError(f"OPEN-PROM {kind} does not reconcile for {scenario} in {year}: {difference:g} [{unit}]")

    def _openprom_checks(self, raw, scenario, years):
        def rows(variable, unit):
            return raw.loc[raw.Variable.eq(variable) & raw.Unit.eq(unit)].set_index("Region")[[str(y) for y in years]]

        net, gross, capture = (rows(v, "Mt CO2/yr").loc["World"] for v in (EMISSIONS, GROSS_EMISSIONS, CAPTURE))
        for year in years:
            self._check(scenario, "Net CO2 equals gross minus capture", year,
                        net.loc[str(year)], gross.loc[str(year)] - capture.loc[str(year)], 2, "Mt CO2/yr")
        parent = rows(ENERGY, "Mtoe").loc["World"]
        carriers = raw.loc[raw.Region.eq("World") & raw.Unit.eq("Mtoe")
                           & raw.Variable.str.startswith(ENERGY + "|")
                           & raw.Variable.str.count(r"\|").eq(ENERGY.count("|") + 1)]
        if not carriers.empty:
            total = carriers[[str(y) for y in years]].sum(min_count=len(carriers))
            for year in years:
                self._check(scenario, "Final energy equals carrier sum", year, parent.loc[str(year)],
                            total.loc[str(year)], len(carriers), "Mtoe")
        for variable, unit in ((ENERGY, "Mtoe"), (EMISSIONS, "Mt CO2/yr"),
                               (GROSS_EMISSIONS, "Mt CO2/yr"), (CAPTURE, "Mt CO2/yr")):
            regional = rows(variable, unit)
            if set(OPENPROM_WORLD_REGIONS).issubset(regional.index):
                total = regional.loc[list(OPENPROM_WORLD_REGIONS)].sum(min_count=len(OPENPROM_WORLD_REGIONS))
                for year in years:
                    self._check(scenario, f"World equals disjoint region sum: {variable}", year,
                                regional.loc["World", str(year)], total.loc[str(year)],
                                len(OPENPROM_WORLD_REGIONS), unit)

    def scenario_mapping(self):
        return pd.DataFrame(self._mapping).copy()

    @property
    def missing_inputs(self):
        rows = [{"Model": MODELS[0], **row} for result in self.results.values() for row in result.coverage]
        columns = ["Model", "Scenario", "Region", "Variable", "Unit", "Year", "Issue", "Treatment"]
        return pd.DataFrame(rows, columns=columns).drop_duplicates().reset_index(drop=True)

    @property
    def aggregation_checks(self):
        return pd.DataFrame(self._checks).copy()

    @property
    def checks(self):
        return self.aggregation_checks

    def _metric_series(self, model, scenario, metric):
        if model == MODELS[0]:
            result = self.results[scenario]
            if metric == "Energy":
                return result.energy(STEEL)
            if metric in {"Emissions", "Net emissions"}:
                return result.emissions(STEEL)
            if metric == "Capture":
                return result.series(OMNIA_CAPTURE, "MtCO2/yr")
            raise ValueError(f"OMNIA has no source metric {metric!r}")
        variable = {"Energy": ENERGY, "Emissions": EMISSIONS, "Net emissions": EMISSIONS,
                    "Gross emissions": GROSS_EMISSIONS, "Capture": CAPTURE}[metric]
        unit = "Mtoe" if metric == "Energy" else "Mt CO2/yr"
        row = self.openprom[scenario].loc[lambda x: x.Variable.eq(variable) & x.Unit.eq(unit) & x.Region.eq("World")]
        series = row.iloc[0][[str(y) for y in self.years[model]]].astype(float)
        series.index = self.years[model]
        return series * (MTOE_TO_EJ if metric == "Energy" else 1)

    def _build(self, diagnostic=False):
        rows = []
        for mapping in self._mapping:
            model, scenario, reference = (mapping[key] for key in ("Model", "Scenario", "ReferenceScenario"))
            metrics = (("Capture",) if model == MODELS[0] else ("Net emissions", "Gross emissions", "Capture")) if diagnostic else ("Energy", "Emissions")
            for metric in metrics:
                values, references = (self._metric_series(model, name, metric) for name in (scenario, reference))
                historical = references.loc[self.historical_year]
                if metric == "Energy":
                    unit, sourceunit = "EJ/yr", "EJ/yr" if model == MODELS[0] else "Mtoe"
                    variables = [f"{ENERGY}|{fuel}" for fuel in STEEL.fuels] if model == MODELS[0] else [ENERGY]
                    scope = "Steel final energy; exact carrier parents" if model == MODELS[0] else "Steel final energy; reported sector total"
                else:
                    unit = "MtCO2e/yr" if model == MODELS[0] and metric != "Capture" else "MtCO2/yr"
                    sourceunit = unit if model == MODELS[0] else "Mt CO2/yr"
                    variable = (OMNIA_CAPTURE if metric == "Capture" else OMNIA_EMISSIONS) if model == MODELS[0] else {"Emissions": EMISSIONS, "Net emissions": EMISSIONS, "Gross emissions": GROSS_EMISSIONS, "Capture": CAPTURE}[metric]
                    variables = [variable]
                    scope = ("Steel CO2 capture; separately reported" if metric == "Capture" else
                             "Steel sector GHG; source reporting boundary" if model == MODELS[0] else
                             "Steel energy-demand CO2 before capture" if metric == "Gross emissions" else
                             "Steel energy-demand CO2 after capture; process emissions and non-CO2 excluded")
                for year in self.years[model]:
                    value, reference_value = values.loc[year], references.loc[year]
                    rows.append({**mapping, "Year": year, "Metric": metric, "Value": value, "Unit": unit,
                                 "ReferenceValue": reference_value, "AbsoluteChange": value - reference_value,
                                 "PercentChange": percent_change(value, reference_value),
                                 "HistoricalValue": historical, "HistoricalPercentChange": percent_change(value, historical),
                                 "HistoricalScenario": reference, "HistoricalYear": self.historical_year,
                                 "SourceVariables": json.dumps(variables), "SourceUnit": sourceunit,
                                 "ConversionFactor": MTOE_TO_EJ if sourceunit == "Mtoe" else 1,
                                 "SourceRegions": json.dumps(self.regions if model == MODELS[0] else ["World"]),
                                 "Scope": scope})
        return pd.DataFrame(rows)

    def frame(self):
        if self._frame is None:
            self._frame = self._build()
        return self._frame.copy()

    def diagnostics(self):
        if self._diagnostics is None:
            self._diagnostics = self._build(diagnostic=True)
        return self._diagnostics.copy()
