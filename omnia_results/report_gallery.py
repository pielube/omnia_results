"""A dependency-free, sequential review page for the report figure suite."""

from html import escape
from pathlib import Path
from urllib.parse import quote


def _text(value: object) -> str:
    return escape(str(value), quote=True)


def _figure_card(record: dict) -> str:
    number = _text(record["number"])
    title = _text(record["title"])
    slug = quote(str(record["slug"]), safe="")
    formats = {str(fmt).lower() for fmt in record["formats"]}
    links = [
        f'<a href="{slug}.{fmt}">{fmt.upper()}</a>'
        for fmt in ("pdf", "svg", "png") if fmt in formats
    ]
    links.append(f'<a href="{slug}.csv">Source CSV</a>')
    preview = next((fmt for fmt in ("svg", "png") if fmt in formats), None)
    if preview:
        artwork = (
            f'<img src="{slug}.{preview}" alt="Figure {number}: {title}" '
            'decoding="async">'
        )
    elif "pdf" in formats:
        artwork = f'<p class="no-preview"><a href="{slug}.pdf">View figure PDF</a></p>'
    else:
        artwork = '<p class="no-preview">No image preview was exported.</p>'
    return f'''<article class="figure-card" id="figure{number}" aria-labelledby="title{number}">
  <header class="figure-heading">
    <p class="figure-number">Figure {number}</p>
    <h3 id="title{number}">{title}</h3>
    <p class="purpose">{_text(record.get("question", ""))}</p>
  </header>
  <figure>
    <div class="artwork">{artwork}</div>
    <figcaption>{_text(record["caption"])}</figcaption>
  </figure>
  <nav class="downloads" aria-label="Download figure {number}">{"".join(links)}</nav>
</article>'''


def write_report_gallery(directory: Path, records: list[dict], audit: dict) -> None:
    """Write the report's index and a named copy for direct linking.

    The supplied records determine figure order and available artwork formats.
    Every figure has a source CSV exported by the report pipeline.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    main = [record for record in records if record["section"] == "Main report"]
    supporting = [record for record in records if record["section"] != "Main report"]
    ordered = main + supporting
    scenario = _text(audit["scenario"])
    scenario_label = _text(audit.get("scenario_label", audit["scenario"]))
    navigation = "".join(
        f'<a href="#figure{_text(record["number"])}" '
        f'title="{_text(record["title"])}">{_text(record["number"])}</a>'
        for record in ordered
    )
    notes = "".join(f"<li>{_text(note)}</li>" for note in audit.get("notes", []))
    assumption = (
        '<aside class="assumption"><strong>Draft assumption:</strong> '
        'blank activity values are treated as zero; prices and reported '
        'intensities are not filled.</aside>'
        if audit.get("missing_activity") == "zero" else ""
    )
    support_section = ""
    if supporting:
        support_section = f'''<section class="support-section" aria-labelledby="support-title">
  <details class="supporting" open>
    <summary id="support-title">Supporting figures <span>Appendix · {len(supporting)} figures</span></summary>
    <div class="supporting-content">{"".join(_figure_card(record) for record in supporting)}</div>
  </details>
</section>'''
    page = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light">
<title>Industrial transitions | {scenario}</title>
<style>
:root{{font-family:Arial,Helvetica,sans-serif;color:#202b32;background:#f5f6f4;font-size:16px;line-height:1.55}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth;scroll-padding-top:24px}}body{{margin:0}}
a{{color:#006887;text-decoration-thickness:1px;text-underline-offset:3px}}
a:hover{{color:#003f55}}a:focus-visible,summary:focus-visible{{outline:3px solid #1385a5;outline-offset:4px}}
main{{max-width:1136px;margin:0 auto;padding:52px 32px 40px}}
.eyebrow{{margin:0 0 12px;color:#647078;font-size:12px;letter-spacing:.14em;text-transform:uppercase}}
h1{{font-size:42px;font-weight:600;line-height:1.14;letter-spacing:-.035em;margin:0 0 16px}}
.intro{{max-width:780px;margin:0;color:#56646c;font-size:17px}}
.suite-links{{display:flex;flex-wrap:wrap;align-items:center;gap:10px 20px;margin:24px 0}}
.suite-links a{{font-size:14px;font-weight:600}}.suite-links .primary-link{{display:inline-block;background:#173f4b;color:white;padding:10px 16px;border-radius:4px;text-decoration:none}}
.assumption{{border-left:3px solid #ad7e2e;background:#fff7e8;padding:12px 16px;margin:22px 0;color:#694e24;font-size:14px}}
.figure-nav{{display:flex;flex-wrap:wrap;align-items:center;gap:7px;margin:24px 0 40px}}
.figure-nav span{{font-size:13px;color:#5b6a72;margin-right:8px}}
.figure-nav a{{display:inline-flex;min-width:34px;height:32px;justify-content:center;align-items:center;border:1px solid #cad3d5;border-radius:4px;background:white;font-size:13px;text-decoration:none}}
.figure-nav a:hover{{background:#e6eff0;border-color:#7f9ca5}}
.section-heading{{display:flex;gap:14px;align-items:baseline;margin:0 0 18px;padding-bottom:12px;border-bottom:1px solid #ced7d9}}
h2{{font-size:19px;font-weight:600;margin:0}}.section-heading p{{font-size:13px;color:#69777e;margin:0}}
.figure-card{{background:white;border:1px solid #dce2e2;border-radius:5px;margin-bottom:32px;overflow:hidden;scroll-margin-top:24px}}
.figure-heading{{padding:24px 28px 8px}}.figure-number{{font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:#65737b;margin:0 0 6px}}
h3{{font-size:23px;line-height:1.3;letter-spacing:-.012em;font-weight:600;margin:0 0 8px}}
.purpose{{color:#617078;font-size:14px;margin:0;max-width:900px}}figure{{margin:0}}
.artwork{{padding:12px 22px 16px}}.artwork img{{display:block;width:100%;height:auto;margin:auto}}
.no-preview{{padding:32px 10px;text-align:center}}
figcaption{{margin:0 28px;padding:18px 0 0;border-top:1px solid #e5e9e9;font-size:13px;line-height:1.65;color:#52616a}}
.downloads{{display:flex;flex-wrap:wrap;gap:18px;padding:15px 28px 22px}}
.downloads a{{font-size:12px;font-weight:600;letter-spacing:.035em}}
.support-section{{margin-top:46px}}summary{{cursor:pointer}}
.supporting>summary{{font-size:19px;font-weight:600;padding:0 0 16px;border-bottom:1px solid #ced7d9}}
.supporting>summary span{{font-size:13px;font-weight:400;color:#69777e;margin-left:14px}}
.supporting-content{{padding-top:20px}}.methods{{margin-top:38px;background:#eaf0ee;border:1px solid #d8e1dd;border-radius:5px;padding:18px 22px}}
.methods>summary{{font-size:15px;font-weight:600}}.methods ul{{padding-left:20px;margin:16px 0}}
.methods li{{font-size:13px;color:#52616a;margin:8px 0;line-height:1.6}}
.method-links{{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:18px}}.method-links a{{font-size:13px}}
footer{{margin-top:28px;font-size:12px;line-height:1.6;color:#69777e}}
@media(max-width:700px){{main{{padding:30px 16px}}h1{{font-size:34px}}.intro{{font-size:15px}}.figure-heading{{padding:20px 18px 6px}}h3{{font-size:20px}}.artwork{{padding:8px 6px 12px}}figcaption{{margin:0 18px}}.downloads{{padding:14px 18px 20px}}.figure-nav{{margin-bottom:30px}}.supporting>summary span{{display:block;margin-left:18px;margin-top:5px}}}}
@media(prefers-reduced-motion:reduce){{html{{scroll-behavior:auto}}}}
@media print{{
  @page{{size:A4 portrait;margin:13mm}}
  :root{{background:white;color:black;font-size:10pt}}main{{max-width:none;padding:0}}
  h1{{font-size:25pt}}.intro{{font-size:10pt}}.eyebrow{{font-size:8pt}}
  .suite-links,.figure-nav,.downloads,.methods,footer{{display:none}}
  .assumption{{background:white;color:black;border:1px solid #888;font-size:9pt}}
  .section-heading{{margin-top:20px}}.section-heading p{{font-size:9pt}}
  .figure-card{{break-before:page;break-inside:avoid;border:none;border-radius:0;margin:0}}
  .figure-heading{{padding:0 0 6px}}h3{{font-size:15pt}}.purpose{{font-size:9pt}}
  .artwork{{padding:6px 0}}.artwork img{{max-height:200mm;object-fit:contain}}
  figcaption{{margin:0;padding-top:8px;font-size:8pt;line-height:1.45}}
  .support-section{{margin-top:0}}.supporting>summary{{font-size:13pt;list-style:none}}
  .supporting>summary::marker{{content:""}}.supporting-content{{display:block!important;padding:0}}
  details::details-content{{display:block!important;content-visibility:visible!important}}
}}
</style>
</head>
<body>
<main>
<header>
  <p class="eyebrow">OMNIA / Report figures</p>
  <h1>Industrial transitions</h1>
  <p class="intro">{scenario_label} · 2019–2050.<br>
  {len(main)} main figures present production, regional change, carbon prices, energy, emissions and costs.</p>
  <div class="suite-links">
    <a class="primary-link" href="report_figures.pdf">Download complete figure set · PDF</a>
    <a href="captions.md">Figure captions</a>
    <a href="methods.md">Methods &amp; definitions</a>
  </div>
  {assumption}
  <nav class="figure-nav" aria-label="Jump to figure"><span>Jump to figure</span>{navigation}</nav>
</header>
<section aria-labelledby="main-title">
  <div class="section-heading"><h2 id="main-title">Main report</h2><p>{len(main)} figures</p></div>
  {"".join(_figure_card(record) for record in main)}
</section>
{support_section}
<details class="methods">
  <summary>Methods, source data and calculation checks</summary>
  <ul>{notes}</ul>
  <div class="method-links">
    <a href="methods.md">Methods</a>
    <a href="audit.json">Calculation audit</a>
    <a href="source_data.csv">All plotted data</a>
    <a href="region_membership.csv">Region definitions</a>
    <a href="producer_rankings.csv">Producer rankings</a>
    <a href="carbon_price_weights.csv">Carbon-price weights</a>
    <a href="missing_inputs.csv">Missing-input ledger</a>
    <a href="intensity_comparison.csv">Intensity comparison</a>
  </div>
</details>
<footer>PDF and SVG figures preserve vector artwork. Individual source-data tables accompany each figure.</footer>
</main>
</body>
</html>
'''
    (directory / "index.html").write_text(page, encoding="utf-8")
    (directory / "report_figures.html").write_text(page, encoding="utf-8")
