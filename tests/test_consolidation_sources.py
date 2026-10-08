"""Source-independent provenance for regenerated consolidated collections."""

from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

from omnia_results.consolidation import (ConsolidationBuilder, _write_gallery,
                                         _write_methods, generate_consolidation)
from omnia_results.consolidation_data import ConsolidationData


class ConsolidationSourceTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "input": "results_allCE_261008.csv", "output": "unwritten_updated_reports",
            "config_file": "figures.consolidation.261008.json", "figures": ["production"],
            "scenarios": ["baseline_noce", "ndc_noce"], "regions": ["EUE", "CHN"],
            "years": [2019, 2050], "missing_activity": "preserve",
            "preserve_individual_figures": [], "formats": ["pdf", "svg", "png"], "dpi": 72,
        }
        variables = ["Production|Non-Metallic Minerals|Cement",
                     "Production|Non-Metallic Minerals|Cement Clinker",
                     "Production|Iron and Steel|Primary", "Production|Iron and Steel|Secondary",
                     "Production|Non-Ferrous Metals|Aluminum|Primary",
                     "Production|Non-Ferrous Metals|Aluminum|Secondary"]
        rows = []
        for scenario, factor in (("baseline_noce", 1), ("ndc_noce", 7)):
            for region_index, region in enumerate(self.config["regions"]):
                for index, variable in enumerate(variables, start=1):
                    rows.append([scenario, region, variable, "Mt/yr",
                                 factor * (index + region_index), factor * (2 * index + region_index)])
        self.source = pd.DataFrame(rows, columns=["Scenario", "Region", "Variable", "Unit", "2019", "2050"])

    def capture_text(self, function, *args):
        written = {}

        def write(path, value, *unused, **kwargs):
            written[path.name] = value
            return len(value)

        with patch.object(Path, "write_text", write):
            function(*args)
        return written

    def test_production_caption_attributes_the_configured_source_without_old_data_claims(self):
        with patch("omnia_results.metrics.pd.read_csv", return_value=self.source):
            data = ConsolidationData(self.config, Path("."))
        spec = ConsolidationBuilder(data, self.config).production()
        try:
            self.assertIn(self.config["input"], spec.caption)
            self.assertNotIn("results_allCE_261002.csv", spec.caption)
            self.assertNotIn("nearly identical", spec.caption)
            self.assertEqual(spec.data.ClimatePathway.unique().tolist(), ["NDC", "NDC+LTT"])
        finally:
            plt.close(spec.figure)

    def test_gallery_attributes_and_escapes_a_configurable_source_name(self):
        config = {**self.config, "input": "updated & checked <results>.csv"}
        page = self.capture_text(_write_gallery, Path("unwritten"), [], config)["index.html"]
        self.assertIn("updated &amp; checked &lt;results&gt;.csv", page)
        self.assertNotIn(config["input"], page)
        self.assertNotIn("results_allCE_261002.csv", page)

    def test_methods_command_and_preservation_state_follow_the_run_configuration(self):
        text = self.capture_text(_write_methods, Path("unwritten"), self.config)["methods.md"]
        self.assertIn(self.config["input"], text)
        self.assertIn(f"--config {self.config['config_file']}", text)
        self.assertNotIn("--config figures.consolidation.json", text)
        self.assertNotIn("archive/before_ndc_first_", text)
        reviewed = {**self.config, "preserve_individual_figures": ["reviewed_example"]}
        reviewed_text = self.capture_text(_write_methods, Path("unwritten"), reviewed)["methods.md"]
        self.assertNotIn("reviewed_example", text)
        self.assertIn("reviewed_example", reviewed_text)

    def test_export_attributes_current_source_bytes_and_config_without_writing_old_collections(self):
        source_bytes = self.source.to_csv(index=False).encode("utf-8")
        written = {}

        class InMemoryPdf:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def savefig(self, figure):
                pass

            def __exit__(self, *args):
                pass

        def write(path, value, *unused, **kwargs):
            written[path.name] = value
            return len(value)

        with patch("omnia_results.metrics.pd.read_csv", return_value=self.source), \
                patch.object(Path, "mkdir"), \
                patch.object(Path, "read_bytes", return_value=source_bytes), \
                patch.object(Path, "write_text", write), \
                patch.object(pd.DataFrame, "to_csv"), \
                patch.object(Figure, "savefig") as save, \
                patch("omnia_results.consolidation.PdfPages", InMemoryPdf), \
                patch("omnia_results.consolidation.check_figure_bounds"), \
                redirect_stdout(StringIO()):
            generate_consolidation(self.config, Path("."))
        audit = json.loads(written["audit.json"])
        self.assertEqual(audit["source"], self.config["input"])
        self.assertEqual(audit["source_sha256"], hashlib.sha256(source_bytes).hexdigest())
        self.assertEqual(audit["config"]["config_file"], self.config["config_file"])
        self.assertEqual(audit["config"]["preserve_individual_figures"], [])
        self.assertEqual(audit["scenario_labels"], {"baseline_noce": "NDC+LTT", "ndc_noce": "NDC"})
        manifest = json.loads(written["manifest.json"])
        self.assertFalse(manifest[0]["individual_artifacts_preserved"])
        self.assertIn(self.config["input"], manifest[0]["caption"])
        self.assertIn(self.config["input"], written["captions.md"])
        self.assertIn(self.config["input"], written["index.html"])
        self.assertIn(f"--config {self.config['config_file']}", written["methods.md"])
        # Individual artwork goes to the configured output, and SVG metadata
        # carries the same source attribution as its caption and manifest.
        for call in save.call_args_list:
            self.assertEqual(Path(call.args[0]).parent.name, self.config["output"])
        svg = next(call for call in save.call_args_list if Path(call.args[0]).suffix == ".svg")
        self.assertIn(self.config["input"], svg.kwargs["metadata"]["Description"])


if __name__ == "__main__":
    unittest.main()
