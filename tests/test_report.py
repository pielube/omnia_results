"""Regression checks for report PDF generation."""

from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd

from omnia_results.plots import STYLE
from omnia_results.report import generate_report


class ReportPdfTests(unittest.TestCase):
    def test_combined_pdf_keeps_font_type_through_finalization(self):
        font_types = {}

        class RecordingPdf:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def savefig(self, figure):
                font_types["save"] = plt.rcParams["pdf.fonttype"]

            def __exit__(self, *args):
                # Matplotlib chooses embedded font types when the PDF closes.
                font_types["close"] = plt.rcParams["pdf.fonttype"]

        def render_scenarios(config, base, source, mapping_path, mapping, book):
            # Scenario rendering has its own style context; the combined PDF
            # must retain that font setting after this context exits.
            with plt.rc_context(STYLE):
                book.savefig(object())

        config = {"input": "fixture.csv", "region_mapping": "mapping.csv",
                  "regions": ["A"], "output": "unused_test_output",
                  "combined_pdf": "all_scenarios.pdf"}
        mapping = pd.DataFrame({"region": ["A"], "country_OMNIA": ["A"],
                                "ISO3": ["AAA"]})
        original_font_type = plt.rcParams["pdf.fonttype"]
        with plt.rc_context({"pdf.fonttype": 3}):
            with patch("omnia_results.report.pd.read_csv", return_value=mapping), \
                    patch("omnia_results.report.PdfPages", RecordingPdf), \
                    patch("omnia_results.report._generate_scenarios", side_effect=render_scenarios), \
                    patch("omnia_results.report.write_collection_gallery"), \
                    patch.object(Path, "mkdir"):
                generate_report(config, Path(__file__).resolve().parents[1])
            self.assertEqual(font_types, {"save": 42, "close": 42})
            self.assertEqual(plt.rcParams["pdf.fonttype"], 3)
        self.assertEqual(plt.rcParams["pdf.fonttype"], original_font_type)


if __name__ == "__main__":
    unittest.main()
