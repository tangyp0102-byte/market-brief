"""Static HTML rendering of the daily brief.

The layout encodes the one thing that matters about this product: how far each
layer can be trusted.

Above the divider is arithmetic. Levels, changes, z-scores, the classification
and its confirmations - all of it reproducible from the stored history and all of
it checked by the test suite. Below the divider is prose written by a language
model whose numerals are verified but whose direction, attribution and choice of
concepts are not. The divider is labelled, and the layers are set differently, so
the distinction is visible rather than a matter of the reader's memory.

That ordering is deliberate and the reverse of the usual dashboard, which leads
with a narrative and buries the table. Here the checkable thing comes first.

No build step, no framework, no JavaScript. One self-contained file per session.
"""

from __future__ import annotations

import datetime as dt
import html
from pathlib import Path

import pandas as pd

from . import classify as classify_mod
from . import brief as brief_mod
from . import tape as tape_mod
from .registry import Registry, load_registry

ASSET_CLASS_LABELS = {
    "rates": "Rates",
    "credit": "Credit",
    "equity": "Equity",
    "vol": "Volatility",
    "commodity": "Commodities",
    "fx": "Foreign exchange",
}

CSS = """
:root {
  --ground: #e9eaec;
  --panel: #fcfcfd;
  --ink: #14161a;
  --muted: #767c86;
  --hair: #d2d5da;
  --rise: #1b6b54;
  --fall: #9e3524;
  --provisional: #6e6480;
  --provisional-bg: #edeaf1;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 14px;
  line-height: 1.55;
  font-variant-numeric: tabular-nums;
}
.sheet {
  max-width: 940px;
  margin: 0 auto;
  padding: 40px 24px 80px;
}
.eyebrow {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 18px;
  align-items: baseline;
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  border-bottom: 1px solid var(--hair);
  padding-bottom: 12px;
}
.eyebrow .date { color: var(--ink); font-weight: 600; }
.verdict {
  margin: 34px 0 4px;
  font-family: Newsreader, Georgia, "Times New Roman", serif;
  font-size: clamp(28px, 5vw, 44px);
  line-height: 1.12;
  font-weight: 400;
  letter-spacing: -0.015em;
}
.confidence {
  font-size: 12px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
}
.meter { margin: 26px 0 8px; }
.meter-track {
  position: relative;
  height: 3px;
  background: var(--hair);
}
.meter-fill { position: absolute; inset: 0 auto 0 0; background: var(--ink); }
.meter-label {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  margin-top: 8px;
}
h2 {
  margin: 46px 0 14px;
  font-size: 11px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  font-weight: 600;
  color: var(--muted);
}
table { width: 100%; border-collapse: collapse; }
th {
  text-align: right;
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  font-weight: 500;
  color: var(--muted);
  padding: 0 0 6px;
  border-bottom: 1px solid var(--hair);
}
th:first-child, td:first-child { text-align: left; }
td {
  padding: 5px 0;
  border-bottom: 1px solid rgba(210, 213, 218, 0.5);
  white-space: nowrap;
}
td.name { white-space: normal; padding-right: 12px; }
td.num { text-align: right; padding-left: 14px; }
.rise { color: var(--rise); }
.fall { color: var(--fall); }
.flat { color: var(--muted); }
.undef { color: var(--muted); font-style: italic; }
a { color: inherit; text-decoration: none; border-bottom: 1px solid var(--hair); }
a:hover { border-bottom-color: var(--ink); }
a:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }
.tag {
  display: inline-block;
  font-size: 10px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 1px 5px;
  margin-left: 6px;
  border: 1px solid var(--hair);
  color: var(--muted);
}
.moved { border-color: var(--ink); color: var(--ink); }
.holds { color: var(--rise); }
.breaks { color: var(--fall); }
.note {
  margin: 12px 0 0;
  padding-left: 14px;
  border-left: 2px solid var(--hair);
  color: var(--muted);
  font-size: 13px;
}
.note.conflict { border-left-color: var(--fall); color: var(--fall); }

/* Signature: the boundary between record and commentary. */
.boundary { margin: 64px 0 0; }
.boundary-rule {
  border: 0;
  border-top: 1px solid var(--ink);
  margin: 0;
}
.boundary-label {
  margin-top: -0.72em;
  text-align: center;
}
.boundary-label span {
  background: var(--ground);
  padding: 0 14px;
  font-size: 10px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--ink);
}
.commentary {
  margin-top: 34px;
  padding: 26px 28px;
  background: var(--provisional-bg);
  color: var(--provisional);
}
.commentary p { margin: 0 0 15px; }
.commentary p:last-child { margin-bottom: 0; }
.commentary .section {
  font-size: 10px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--provisional);
  opacity: 0.75;
  margin-bottom: 4px;
}
.commentary .caveat {
  margin-top: 20px;
  padding-top: 16px;
  border-top: 1px solid rgba(110, 100, 128, 0.28);
  font-size: 12px;
  opacity: 0.85;
}
.withheld {
  margin-top: 34px;
  padding: 20px 24px;
  border: 1px solid var(--fall);
  color: var(--fall);
  font-size: 13px;
}
footer {
  margin-top: 56px;
  padding-top: 16px;
  border-top: 1px solid var(--hair);
  font-size: 11px;
  color: var(--muted);
  display: flex;
  flex-wrap: wrap;
  gap: 8px 20px;
  justify-content: space-between;
}
@media (max-width: 620px) {
  .sheet { padding: 26px 16px 56px; }
  body { font-size: 13px; }
  td.num { padding-left: 8px; }
  .hide-narrow { display: none; }
}
@media (prefers-reduced-motion: no-preference) {
  .sheet > * { animation: rise 0.4s ease-out both; }
  @keyframes rise { from { opacity: 0; transform: translateY(4px); } }
}
"""


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def _change_cell(change, unit: str) -> str:
    if change is None or pd.isna(change):
        return '<td class="num undef">undefined</td>'
    cls = "rise" if change > 0 else "fall" if change < 0 else "flat"
    places = 1 if unit == "bp" else 2
    return f'<td class="num {cls}">{change:+,.{places}f}{_esc(unit)}</td>'


def _z_cell(z) -> str:
    if z is None or pd.isna(z):
        return '<td class="num flat">&mdash;</td>'
    return f'<td class="num">{z:+.2f}</td>'


def _signals_table(result: classify_mod.Classification) -> str:
    rows = []
    for bloc in classify_mod.BLOCS:
        signal = result.signals.get(bloc.id)
        if signal is None:
            continue
        tags = ""
        if signal.moved:
            tags += '<span class="tag moved">moved</span>'
        if signal.quiet_vol:
            tags += '<span class="tag">quiet vol</span>'
        pct = "&mdash;" if signal.percentile is None else f"{signal.percentile:.0%}"
        rows.append(
            f"<tr><td class='name'>{_esc(bloc.name)}{tags}</td>"
            f"<td class='num flat hide-narrow'>{_esc(signal.instrument_id)}</td>"
            f"{_change_cell(signal.change, signal.unit)}"
            f"{_z_cell(signal.z)}"
            f"<td class='num flat'>{pct}</td></tr>"
        )
    if not rows:
        return "<p class='note'>No signals available for this session.</p>"
    return (
        "<table><thead><tr><th>Bloc</th>"
        "<th class='hide-narrow'>Via</th><th>Change</th><th>Z</th>"
        "<th>Pctile</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )


def _confirmations_block(result: classify_mod.Classification) -> str:
    if not result.confirmations:
        return ""
    rows = []
    for c in result.confirmations:
        if c.holds is True:
            mark, cls = "holds", "holds"
        elif c.holds is False:
            mark, cls = "does not hold", "breaks"
        else:
            mark, cls = "not evaluable", "flat"
        rows.append(
            f"<tr><td class='name'>{_esc(c.id.replace('_', ' '))}</td>"
            f"<td class='{cls}'>{mark}</td>"
            f"<td class='num flat'>{_esc(c.detail)}</td></tr>"
        )
    return (
        "<h2>Confirmation</h2><table><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _tape_tables(built: tape_mod.Tape, registry: Registry) -> str:
    out = []
    for asset_class in tape_mod.ASSET_CLASS_ORDER:
        group = built.by_class(asset_class)
        if not group:
            continue
        rows = []
        for row in sorted(group, key=lambda r: (r.is_derived, r.instrument_id)):
            inst = registry.get(row.instrument_id)
            url = inst.tradingview_url() if inst else None
            name = _esc(row.name)
            # Deep link to the reader's own charts: the pipeline says what to
            # look at, their subscription is where they look.
            label = (
                f'<a href="{_esc(url)}" target="_blank" rel="noopener">{name}</a>'
                if url else name
            )
            if row.is_derived:
                label += '<span class="tag">derived</span>'
            if row.quiet_vol:
                label += '<span class="tag">quiet vol</span>'
            level_places = row.level_places if row.level_places is not None else 2
            if abs(row.level) < 1 and row.level_places is None:
                level_places = 6 if abs(row.level) < 0.01 else 4
            rows.append(
                f"<tr><td class='name'>{label}</td>"
                f"<td class='num flat'>{row.level:,.{level_places}f}</td>"
                f"{_change_cell(row.change, row.unit_label)}"
                f"{_z_cell(row.z)}</tr>"
            )
        out.append(
            f"<h2>{_esc(ASSET_CLASS_LABELS.get(asset_class, asset_class))}</h2>"
            "<table><thead><tr><th>Instrument</th><th>Level</th>"
            "<th>Change</th><th>Z</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>"
        )
    return "".join(out)


def _commentary(result_brief: brief_mod.Brief) -> str:
    if result_brief.unverified_numbers:
        figures = ", ".join(str(n) for n in result_brief.unverified_numbers)
        return (
            "<div class='withheld'>The written summary was withheld. It quoted "
            f"figures that do not appear in the data: {_esc(figures)}. "
            "The tape above is unaffected.</div>"
        )

    body = result_brief.body
    paragraphs = []
    for chunk in [c.strip() for c in body.split("\n") if c.strip()]:
        if chunk.isupper() and len(chunk) < 40:
            paragraphs.append(f"<p class='section'>{_esc(chunk.title())}</p>")
        else:
            paragraphs.append(f"<p>{_esc(chunk)}</p>")

    caveat = (
        "Written from the figures above and nothing else: no news, no events, no "
        "policy. Quoted numbers are checked against the data; direction, "
        "attribution and choice of concepts are not. Read it as commentary."
        if result_brief.generated else
        "Generated from the figures above without a language model."
    )
    return (
        "<div class='commentary'>" + "".join(paragraphs)
        + f"<p class='caveat'>{caveat}</p></div>"
    )


def render_page(
    built: tape_mod.Tape,
    result: classify_mod.Classification,
    result_brief: brief_mod.Brief,
    registry: Registry | None = None,
    archive: list[str] | None = None,
) -> str:
    registry = registry or load_registry()
    session = built.session

    gate = built.gate_result
    gate_bits = [f"gate {built.verdict}"]
    if gate and gate.is_early_close:
        gate_bits.append("half session")
    if gate and gate.undefined:
        gate_bits.append(f"{len(gate.undefined)} undefined")
    if gate and gate.review:
        gate_bits.append(f"{len(gate.review)} for review")

    pct = result.intensity_percentile
    fill = min(100, max(2, round((pct or 0) * 100)))
    meter_right = (
        f"more active than {pct:.0%} of sessions" if pct is not None
        else "no baseline yet"
    )

    confidence = (
        f"{result.confidence} confidence" if result.classified
        else "not classified"
    )

    notes = "".join(
        f"<p class='note{' conflict' if n.startswith(('CONFLICT', 'WITHDRAWN')) else ''}'>"
        f"{_esc(n)}</p>"
        for n in result.notes
    )

    archive_links = ""
    if archive:
        links = " ".join(
            f'<a href="{_esc(d)}.html">{_esc(d)}</a>' for d in archive[:8]
        )
        archive_links = f"<span>Earlier: {links}</span>"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Market brief &middot; {_esc(session)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<main class="sheet">

  <div class="eyebrow">
    <span class="date">{_esc(session)}</span>
    <span>{_esc(' &middot; '.join(gate_bits))}</span>
    <span>z-window {built.window} sessions</span>
  </div>

  <h1 class="verdict">{_esc(result.regime_name)}</h1>
  <div class="confidence">{_esc(confidence)}</div>

  <div class="meter">
    <div class="meter-track"><div class="meter-fill" style="width:{fill}%"></div></div>
    <div class="meter-label">
      <span>Intensity {result.intensity:.2f}</span>
      <span>{_esc(meter_right)}</span>
    </div>
  </div>

  {notes}

  <h2>Signals &mdash; one per bloc</h2>
  {_signals_table(result)}
  {_confirmations_block(result)}

  {_tape_tables(built, registry)}

  <div class="boundary">
    <hr class="boundary-rule">
    <div class="boundary-label"><span>Below: commentary, not record</span></div>
  </div>

  {_commentary(result_brief)}

  <footer>
    <span>Instrument names link to TradingView.</span>
    {archive_links}
  </footer>

</main>
</body>
</html>
"""


def write_page(
    path: str | Path,
    built: tape_mod.Tape,
    result: classify_mod.Classification,
    result_brief: brief_mod.Brief,
    registry: Registry | None = None,
    archive: list[str] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_page(built, result, result_brief, registry, archive),
        encoding="utf-8",
    )
    return path
