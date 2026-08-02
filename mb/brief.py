"""The written brief.

This is the first stage that can produce fluent, confident, wrong prose.
Everything before it was arithmetic you could check. Three defences:

1. QUIET DAYS NEVER REACH THE MODEL. On the ~85% of sessions with no clean
   signal, a fixed template lists what moved and stops. Handing a language model
   a quiet day invites it to manufacture significance, which is exactly what the
   intensity gate exists to prevent.

2. THE MODEL SEES ONLY THE FACT SHEET. No raw prices, no free text, no news. It
   receives the classification, the bloc signals, the confirmations and the gate
   findings as JSON, and is instructed to introduce no figure that is not there.

3. EVERY NUMERAL IS VERIFIED. After generation, each digit sequence in the output
   is matched against the fact sheet within a rounding tolerance. An unmatched
   number fails the brief rather than publishing it. A model that invents "the
   10-year rose 12bp" when it rose 7bp produces text indistinguishable from
   correct text; only mechanical checking catches that.

WHAT THE VALIDATOR DOES NOT CATCH, and which the prompt alone defends against:

  * DIRECTION. Prose carries direction in words, so numerals cannot be
    sign-matched. "The S&P rose 7.60%" passes when it fell.
  * ATTRIBUTION. A figure is checked for membership in the fact sheet, not for
    being attached to the right instrument. Swapping two instruments' moves
    passes, because both numbers are present.
  * INVENTED CONCEPTS. A claim carrying no digits is invisible here. An early
    run asserted the term premium "remains elevated" when no term premium
    measure exists anywhere in the fact sheet.

These are the reasons the narrative is presented as commentary rather than
record, and the reason the tape and the classification are printed alongside it.

The narrative is commentary, not conclusion. Attribution of a day's moves to a
cause is post-hoc storytelling, and the prompt requires it to be phrased as
hypothesis.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import classify, rolls, store, tape
from .registry import Registry, load_registry

# Model IDs change; verify against https://docs.claude.com before relying on
# this default. Override with MB_MODEL rather than editing the constant.
# Current self-serve IDs as of August 2026: claude-haiku-4-5-20251001,
# claude-sonnet-5, claude-opus-5, claude-fable-5.
#
# Haiku is the default deliberately. The model here writes three short
# paragraphs from a JSON fact sheet it is forbidden to reason beyond; the
# judgement was already made by the classifier. Paying five times more for a
# larger model buys better prose, not better analysis.
DEFAULT_MODEL = os.environ.get("MB_MODEL", "claude-haiku-4-5-20251001")
MAX_TOKENS = 700

# Digit sequences, optionally signed, with thousands separators and decimals.
_NUMBER = re.compile(r"[-+]?\d[\d,]*\.?\d*")

# Rounding slack when matching a quoted figure against the fact sheet.
ABS_TOLERANCE = 0.051
REL_TOLERANCE = 0.01


@dataclass
class Brief:
    session: dt.date
    headline: str
    body: str
    generated: bool                       # True if a model wrote it
    factsheet: dict = field(default_factory=dict)
    unverified_numbers: list[float] = field(default_factory=list)
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.error is None and not self.unverified_numbers

    def render(self) -> str:
        lines = ["=" * 78, f"MARKET BRIEF  {self.session}", "=" * 78, "",
                 self.headline, "", self.body]
        if self.unverified_numbers:
            lines += ["", "!" * 78,
                      "WITHHELD: the narrative contains figures absent from the "
                      "fact sheet: " + ", ".join(str(n) for n in self.unverified_numbers),
                      "!" * 78]
        if self.error:
            lines += ["", f"ERROR: {self.error}"]
        lines.append("=" * 78)
        return "\n".join(lines)


# --------------------------------------------------------------- fact sheet

def build_factsheet(
    history: pd.DataFrame,
    registry: Registry | None = None,
    session: dt.date | pd.Timestamp | None = None,
    flags: pd.DataFrame | None = None,
    window: int = 60,
) -> dict:
    """Everything the narrative is permitted to know, as plain JSON."""
    registry = registry or load_registry()
    flags = rolls.load_flags() if flags is None else flags
    session = pd.Timestamp(session or history["date"].max()).normalize()

    built = tape.build_tape(history, registry, session, flags, window)
    result = classify.classify(history, registry, session, flags, window)

    gate = built.gate_result
    sheet = {
        "session": session.date().isoformat(),
        "gate": {
            "verdict": built.verdict,
            "half_session": bool(gate.is_early_close) if gate else False,
            "missing": [str(f) for f in gate.missing] if gate else [],
            "stale": [str(f) for f in gate.stale] if gate else [],
            "undefined": [str(f) for f in gate.undefined] if gate else [],
            "review": [str(f) for f in gate.review] if gate else [],
        },
        "regime": {
            "id": result.regime,
            "name": result.regime_name,
            "confidence": result.confidence,
            "intensity": round(result.intensity, 2),
            "intensity_percentile": (
                round(result.intensity_percentile, 2)
                if result.intensity_percentile is not None else None
            ),
        },
        "signals": [
            {
                "bloc": s.bloc,
                "instrument": s.instrument_id,
                "change": None if s.change is None or pd.isna(s.change)
                else round(float(s.change), 2),
                "unit": s.unit,
                "z": None if s.z is None or pd.isna(s.z) else round(float(s.z), 2),
                "percentile": None if s.percentile is None
                else round(s.percentile, 2),
                "moved": bool(s.moved),
                "quiet_vol": bool(s.quiet_vol),
            }
            for s in result.signals.values()
        ],
        "confirmations": [
            {"id": c.id, "holds": c.holds, "detail": c.detail}
            for c in result.confirmations
        ],
        "largest_moves": [
            {
                "name": r.name,
                "change": None if r.change is None or pd.isna(r.change)
                else round(float(r.change), 2),
                "unit": r.unit_label,
                "z": None if r.z is None or pd.isna(r.z) else round(float(r.z), 2),
                "quiet_vol": bool(r.quiet_vol),
            }
            for r in built.largest(6)
        ],
        "notes": list(result.notes),
    }
    return sheet


# ------------------------------------------------------------- verification

def extract_numbers(text: str) -> list[float]:
    """Every numeric token in the text."""
    out = []
    for match in _NUMBER.findall(text):
        cleaned = match.replace(",", "").rstrip(".")
        if cleaned in ("", "-", "+"):
            continue
        try:
            out.append(float(cleaned))
        except ValueError:
            continue
    return out


# Numbers that are part of a NAME, not a quantity. "S&P 500" and "the 10-year"
# contain digits that no fact sheet will ever list, and demanding they be
# traceable withholds correct prose. These are stripped before extraction rather
# than added to the allowed set, so that a genuine quantity of 500 or 10 is still
# checked.
_IDENTIFIERS = re.compile(
    r"""
      S\s*&\s*P\s*-?\s*500
    | SP\s*-?\s*500
    | Nasdaq[\s-]*100
    | Russell[\s-]*2000
    | Dow[\s-]*30
    | VIX\s*-?\s*3M
    | \b\d+s\d+s\b                                   # 2s10s, 5s30s
    | \b\d+\s*-?\s*(?:year|yr|y)\b                   # 10-year, 30yr, 2y
    | \b\d+\s*-?\s*(?:month|mo)\b                    # 3-month, 6mo
    """,
    re.IGNORECASE | re.VERBOSE,
)


def strip_identifiers(text: str) -> str:
    """Remove index names and tenor references before numeral extraction."""
    return _IDENTIFIERS.sub(" ", text)


def _walk(value) -> list[float]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [n for v in value.values() for n in _walk(v)]
    if isinstance(value, (list, tuple)):
        return [n for v in value for n in _walk(v)]
    if isinstance(value, str):
        # Strip identifiers here as well as in the narrative. Otherwise an
        # instrument called "S&P 500" silently authorises the model to write
        # "yields rose 500 basis points" - the name would make the quantity
        # traceable. Both sides must ignore the same tokens.
        return extract_numbers(strip_identifiers(value))
    return []


def allowed_values(factsheet: dict) -> list[float]:
    """Numbers the narrative may quote.

    Includes percentiles rendered as percentages, since 0.89 in the fact sheet
    is naturally written as 89% in prose.
    """
    values = set(_walk(factsheet))

    for key in ("intensity_percentile",):
        v = factsheet.get("regime", {}).get(key)
        if v is not None:
            values.add(round(v * 100))
    for signal in factsheet.get("signals", []):
        if signal.get("percentile") is not None:
            values.add(round(signal["percentile"] * 100))

    session = factsheet.get("session", "")
    values.update(extract_numbers(session))
    return sorted(values)


def unverified_numbers(text: str, factsheet: dict) -> list[float]:
    """Figures in the text that do not appear in the fact sheet.

    Matching is on MAGNITUDE, not signed value: prose carries direction in words
    ("the S&P fell 7.60%") while the fact sheet stores -7.60, so a signed match
    would reject almost every correct sentence.

    The consequence is a real limitation worth stating: this catches invented and
    misquoted figures, but NOT a reversed direction. A model writing "the S&P
    rose 7.60%" would pass. Direction is defended by the prompt and by the fact
    sheet sitting in front of the model, not by this check.
    """
    allowed = [abs(a) for a in allowed_values(factsheet)]
    bad = []
    for found in extract_numbers(strip_identifiers(text)):
        magnitude = abs(found)
        if any(
            abs(magnitude - a) <= max(ABS_TOLERANCE, abs(a) * REL_TOLERANCE)
            for a in allowed
        ):
            continue
        bad.append(found)
    return bad


# ------------------------------------------------------------ quiet template

def factual_summary(factsheet: dict) -> str:
    """The moves, stated and nothing more.

    Used both for quiet sessions and as the fallback when a narrative is
    withheld. It deliberately makes no claim about whether a regime exists,
    because those two cases differ: a withheld narrative still had a
    classification, and printing "no signature was identified" there would be a
    second falsehood layered over the first.
    """
    moves = factsheet.get("largest_moves", [])
    lines = []
    if moves:
        lines.append("Largest moves by z-score:")
        for m in moves:
            if m["change"] is None:
                continue
            z = "" if m["z"] is None else f"  (z {m['z']:+.2f})"
            quiet = "  [quiet vol period]" if m.get("quiet_vol") else ""
            lines.append(f"  {m['name']}: {m['change']:+.2f}{m['unit']}{z}{quiet}")
    else:
        lines.append("No instruments recorded a change this session.")

    moved = [s["bloc"] for s in factsheet.get("signals", []) if s["moved"]]
    if moved:
        lines.append("")
        lines.append("Blocs clearing their move threshold: " + ", ".join(moved) + ".")
    return "\n".join(lines)


def quiet_template(factsheet: dict) -> tuple[str, str]:
    """Deterministic output for a session with no clean signal.

    Lists what moved and stops. No model involved, so no possibility of
    manufactured significance on a day the data says is unremarkable.
    """
    regime = factsheet["regime"]
    pct = regime.get("intensity_percentile")
    intensity_note = (
        f"Composite intensity {regime['intensity']:.2f}"
        + (f", above {pct:.0%} of sessions." if pct is not None else ".")
    )

    headline = f"{regime['name']}. {intensity_note}"

    body = factual_summary(factsheet)
    body += (
        "\n\nNo cross-asset signature was identified. Most sessions are noise."
    )
    return headline, body


# ---------------------------------------------------------------- narrative

PROMPT = """You are writing one section of a daily cross-asset market brief for \
a single reader who follows markets closely.

You will be given a JSON fact sheet produced by a deterministic pipeline. Write \
three short sections:

WHAT MOVED - the notable moves, in plain prose. Two or three sentences.
LIKELY DRIVER - the most plausible reading of the cross-asset pattern. Two or \
three sentences.
WHAT TO WATCH - one or two sentences on what would confirm or refute this.

Absolute rules:

1. Every number you write must appear in the fact sheet. Do not compute, round \
differently, aggregate, or estimate any figure. If you want to state a number \
that is not in the fact sheet, describe it qualitatively instead.
2. Attach every figure to the instrument it belongs to. Do not transfer a number \
from one instrument to another. If you are unsure which instrument a figure \
describes, omit the figure.
3. Do not introduce measures, concepts or indicators that are not in the fact \
sheet. No term premium, positioning, flows, valuations, liquidity, breadth or \
sentiment unless a field of that name is present. You may only discuss what you \
were given.
4. Write counts as words, not digits: "all four confirmations", not "4".
5. You have no information about news, events, earnings, or policy. You cannot \
know WHY anything moved. Phrase the driver as a hypothesis consistent with the \
pattern: "consistent with", "the pattern suggests", "would be typical of". \
Never assert a cause as fact.
6. If confirmations conflict, say so plainly. Do not resolve the conflict or \
argue past it.
7. No investment advice, no forecasts, no price targets.
8. Plain prose. No headers beyond the three section names, no bullet points, no \
markdown.

FACT SHEET:
{factsheet}
"""


def build_prompt(factsheet: dict) -> str:
    return PROMPT.format(factsheet=json.dumps(factsheet, indent=2))


def call_anthropic(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Default model caller. Requires ANTHROPIC_API_KEY."""
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "the anthropic package is required; pip install anthropic"
        ) from exc

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Use --prompt to inspect the prompt "
            "without calling a model."
        )

    client = anthropic.Anthropic(api_key=key)
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    )


def generate(
    factsheet: dict,
    call_model=None,
) -> Brief:
    """Produce the brief for one session.

    A session with no regime never reaches the model.
    """
    session = dt.date.fromisoformat(factsheet["session"])
    regime = factsheet["regime"]

    if factsheet["gate"]["verdict"] == "blocked":
        return Brief(
            session=session,
            headline="No brief for this session.",
            body="The validation gate blocked publication.",
            generated=False, factsheet=factsheet,
        )

    if regime["id"] is None:
        headline, body = quiet_template(factsheet)
        return Brief(session=session, headline=headline, body=body,
                     generated=False, factsheet=factsheet)

    caller = call_model or call_anthropic
    try:
        text = caller(build_prompt(factsheet))
    except Exception as exc:  # noqa: BLE001
        return Brief(
            session=session,
            headline=f"{regime['name']} ({regime['confidence']} confidence).",
            body=factual_summary(factsheet), generated=False,
            factsheet=factsheet,
            error=f"model call failed, fell back to the facts: {exc}",
        )

    bad = unverified_numbers(text, factsheet)
    confidence = regime["confidence"]
    headline = f"{regime['name']} ({confidence} confidence)."

    if bad:
        # Do not publish prose containing figures we cannot trace. The
        # classification still stands, so the headline is kept and only the
        # narrative is replaced by the verifiable facts.
        return Brief(
            session=session, headline=headline,
            body=factual_summary(factsheet),
            generated=False, factsheet=factsheet, unverified_numbers=bad,
        )

    return Brief(session=session, headline=headline, body=text.strip(),
                 generated=True, factsheet=factsheet)


def main(argv: list[str] | None = None) -> int:
    import argparse

    default_store = Path(__file__).resolve().parent.parent / "data" / "history.parquet"

    parser = argparse.ArgumentParser(description="Write the daily brief")
    parser.add_argument("--session", default=None)
    parser.add_argument("--store", default=str(default_store))
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--factsheet", action="store_true",
                        help="print the fact sheet JSON and exit")
    parser.add_argument("--prompt", action="store_true",
                        help="print the prompt without calling a model")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    history = store.read(Path(args.store))
    if history.empty:
        print("store is empty; run the backfill first")
        return 1

    sheet = build_factsheet(
        history, load_registry(), args.session, rolls.load_flags(), args.window
    )

    if args.factsheet:
        print(json.dumps(sheet, indent=2))
        return 0

    if args.prompt:
        if sheet["regime"]["id"] is None:
            print("No regime for this session, so no model call would be made.")
            print("The deterministic template would produce:\n")
            headline, body = quiet_template(sheet)
            print(headline + "\n\n" + body)
            return 0
        print(build_prompt(sheet))
        return 0

    brief = generate(
        sheet, call_model=lambda p: call_anthropic(p, args.model)
    )
    print(brief.render())
    return 0 if brief.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
