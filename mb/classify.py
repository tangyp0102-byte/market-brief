"""Cross-asset regime classification.

Everything below this line is judgement. Everything above it was arithmetic you
could check. The separation is deliberate, and the design here is shaped by four
findings from the earlier stages rather than by theory.

1. CORRELATED SIGNALS ARE NOT INDEPENDENT EVIDENCE. On 2026-07-31 six FX rows
   cleared two sigma; they were one dollar move counted six times. On 2026-07-29
   seven rates rows cleared three sigma; they were one bear-steepening counted
   seven times. Instruments are therefore grouped into blocs, and each bloc
   contributes exactly one signal through a designated representative.

2. SIGMA IS NOT CALIBRATED HERE. The replay showed every instrument's z-score
   standard deviation between 1.05 and 1.19, never below 1, and |z|>2 on about
   6% of sessions rather than the 4.55% a normal distribution implies. With 55
   tape rows that means roughly three two-sigma moves on a purely random day.
   Thresholds are therefore empirical percentiles of each signal's own history,
   which are calibrated by construction.

3. CREDIT IS STRUCTURALLY T+1. The FRED OAS series publish a day late, so the
   same-day credit signal is the duration-hedged HYG proxy. See mb.derived.

4. NATURAL GAS IS EXCLUDED. A third of its notable history is roll-flagged, and
   Henry Hub is driven by US weather and regional storage rather than global risk
   appetite. WTI and Brent already cover energy.

The classifier refuses to label a session unless the day is genuinely active,
and reports conflicts rather than downgrading to the next-best story. Most days
are noise, and saying so is the honest output.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import derived, rolls, store
from . import transforms as tf
from .registry import Registry, load_registry

UP, DOWN, EITHER = 1, -1, 0

# Z-scores are clipped before entering any composite. One instrument in a quiet
# volatility period reached -8.6 sigma on 2026-07-31; unclipped, it would have
# determined the intensity score by itself.
Z_CLIP = 4.0

# A leg must reach this percentile of its own |z| history to count as a move.
LEG_PERCENTILE = 0.70

# Sessions below this composite percentile are not classified at all.
INTENSITY_PERCENTILE = 0.65


@dataclass(frozen=True)
class Bloc:
    id: str
    name: str
    representative: str
    members: tuple[str, ...]
    select: str = "fixed"     # fixed | strongest
    note: str = ""


# One signal per bloc. Members are listed to document what is being deflated.
BLOCS: tuple[Bloc, ...] = (
    Bloc("equity", "Equity index", "spx",
         ("spx", "spy", "ndx", "rut"),
         "index level; sector rotation is handled separately as confirmation"),
    Bloc("rates", "Nominal yields", "ust_10y",
         ("ust_2y", "ust_5y", "ust_10y", "ust_30y"), select="strongest",
         note="whichever tenor moved most: a fixed 10y proxy misses front-end-led "
              "days such as a CPI surprise, where the 2y carries the signal"),
    Bloc("bills", "T-bills", "ust_3m", ("ust_3m",)),
    Bloc("curve", "Curve slope", "ust_2s10s",
         ("ust_2s10s", "ust_3m10y", "ust_5s30s")),
    Bloc("real_rates", "Real yields", "ust_real_10y",
         ("ust_real_5y", "ust_real_10y", "ust_real_30y")),
    Bloc("inflation", "Breakevens", "be_10y", ("be_5y", "be_10y")),
    Bloc("dollar", "US dollar", "dxy",
         ("dxy", "eurusd", "usdjpy", "gbpusd", "usdcad", "usdsek", "usdchf",
          "audusd", "usdmxn", "usdcny"),
         "the six DXY components plus the commodity and EM crosses"),
    Bloc("gold", "Precious metals", "gold", ("gold", "silver")),
    Bloc("energy", "Crude oil", "wti", ("wti", "brent"),
         "natural gas deliberately excluded: roll-contaminated and weather-driven"),
    Bloc("industrial", "Industrial metals", "copper", ("copper",)),
    Bloc("vol", "Equity volatility", "vix", ("vix", "vix3m", "vix_term")),
    Bloc("credit", "Credit risk", "credit_excess", ("credit_excess",),
         "duration-hedged HYG; the FRED OAS series are authoritative but T+1"),
)

BLOC_BY_ID = {b.id: b for b in BLOCS}

# Legs the regime rules are written against.
CORE_BLOCS = ("equity", "rates", "dollar", "gold", "energy")

# Instruments excluded from classification entirely.
EXCLUDED = frozenset({"natgas"})


@dataclass
class Signal:
    bloc: str
    name: str
    instrument_id: str
    change: float | None
    unit: str
    z: float | None
    percentile: float | None      # rank of |z| within this signal's own history
    quiet_vol: bool = False

    @property
    def direction(self) -> int:
        if self.change is None or pd.isna(self.change):
            return 0
        return UP if self.change > 0 else DOWN

    @property
    def moved(self) -> bool:
        """Cleared the empirical bar for counting as a real move."""
        return self.percentile is not None and self.percentile >= LEG_PERCENTILE

    @property
    def clipped_z(self) -> float:
        if self.z is None or pd.isna(self.z):
            return 0.0
        return float(np.clip(self.z, -Z_CLIP, Z_CLIP))


@dataclass(frozen=True)
class Rule:
    id: str
    name: str
    legs: dict[str, int]
    confirmations: tuple[str, ...]
    description: str
    min_intensity_percentile: float | None = None
    """Severity gate, separate from the leg conditions.

    Leg count is specificity, not severity. Liquidation's legs (equity down,
    gold down, dollar up) describe an ordinary risk-off day with a firm dollar,
    yet having three legs made it outrank the mundane reading on any such day.
    It fired on a 1.45-intensity session, which is not a funding crisis. A
    crisis label needs crisis-level evidence, so the rules that name one carry
    an explicit floor on how active the session must be.
    """


# The primary axis is the sign pair (equity, rates): four quadrants that between
# them cover every ordinary session. The first rule table demanded three or four
# legs including a specific dollar direction, and consequently failed to classify
# 2020-03-09 at nearly four times the intensity threshold. Every risk-off rule
# required a STRONGER dollar, but when rate-cut repricing outweighs haven demand
# the dollar falls during a selloff - which is what happened on the COVID crash,
# the SVB weekend, the yen carry unwind and the 2025 tariff selloff alike.
#
# The dollar and gold are now confirmations rather than requirements, except in
# the two configurations where they are the whole point: a term-premium episode
# is defined by yields rising while the dollar falls, and a liquidation is
# defined by gold being sold alongside everything else.
RULES: tuple[Rule, ...] = (
    # Specific patterns first: more legs wins the tie-break in classify().
    Rule(
        "term_premium", "Term premium / fiscal stress",
        {"equity": DOWN, "rates": UP, "dollar": DOWN, "gold": UP},
        ("long_end_leads", "curve_steepens"),
        "yields up while the dollar falls and gold bids: an unusual and "
        "important combination pointing at term premium rather than policy",
        min_intensity_percentile=0.85,
    ),
    Rule(
        "broad_derisking", "Broad de-risking / dollar bid",
        {"equity": DOWN, "gold": DOWN, "dollar": UP},
        ("vol_up", "credit_weak"),
        "equities and gold both sold with the dollar bid: cash preferred to "
        "every alternative. At ordinary intensity this is routine de-risking; "
        "at extreme intensity it is the funding-stress signature, as on "
        "2020-03-16. The legs are the same and severity is what separates "
        "them, so read the intensity percentile alongside the label",
    ),
    # The quadrant grid.
    Rule(
        "growth_scare", "Growth scare / flight to quality",
        {"equity": DOWN, "rates": DOWN},
        ("vol_up", "credit_weak", "defensives_lead", "gold_up", "dollar_up"),
        "equities and yields fall together: demand for duration and safety",
    ),
    Rule(
        "hawkish_repricing", "Hawkish repricing",
        {"equity": DOWN, "rates": UP},
        ("front_end_leads", "curve_flattens", "dollar_up", "growth_underperforms"),
        "rates doing the damage: yields up and equities down together",
    ),
    Rule(
        "goldilocks", "Dovish / goldilocks",
        {"equity": UP, "rates": DOWN},
        ("vol_down", "credit_firm", "dollar_down"),
        "yields fall and equities rally: easing without a growth scare",
    ),
    Rule(
        "reflation", "Reflation / good growth",
        {"equity": UP, "rates": UP},
        ("cyclicals_lead", "curve_steepens", "copper_gold_up", "energy_up"),
        "yields and equities rise together on stronger growth expectations",
    ),
)


@dataclass
class Confirmation:
    id: str
    holds: bool | None            # None = could not be evaluated
    detail: str

    def __post_init__(self):
        # Comparisons on pandas/numpy values return np.bool_, and `np.True_ is
        # True` is False. Without this coercion every confirmation silently
        # counted as unsupported and confidence never reached "strong".
        if self.holds is not None:
            object.__setattr__(self, "holds", bool(self.holds))


@dataclass
class Classification:
    session: dt.date
    regime: str | None
    regime_name: str
    intensity: float
    intensity_percentile: float | None
    signals: dict[str, Signal]
    confirmations: list[Confirmation] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def classified(self) -> bool:
        return self.regime is not None

    @property
    def supported(self) -> list[Confirmation]:
        return [c for c in self.confirmations if c.holds is True]

    @property
    def conflicting(self) -> list[Confirmation]:
        return [c for c in self.confirmations if c.holds is False]

    @property
    def confidence(self) -> str:
        if not self.classified:
            return "none"
        checked = [c for c in self.confirmations if c.holds is not None]
        if not checked:
            return "unconfirmed"
        ratio = len(self.supported) / len(checked)
        if self.conflicting and ratio < 0.5:
            return "contradicted"
        if ratio == 1.0:
            return "strong"
        return "mixed"


def build_signals(
    history: pd.DataFrame,
    registry: Registry | None = None,
    session: dt.date | pd.Timestamp | None = None,
    flags: pd.DataFrame | None = None,
    window: int = 60,
    min_obs: int = 30,
) -> dict[str, Signal]:
    """One signal per bloc, with an empirical percentile for today's move."""
    registry = registry or load_registry()
    flags = rolls.load_flags() if flags is None else flags
    session = pd.Timestamp(session or history["date"].max()).normalize()

    wide = store.wide(history[history["date"] <= session], "close")
    derived_series = derived.build(wide, registry)

    def measure(rep: str):
        """Change, z, percentile and vol ratio for one instrument or derived id."""
        if rep in derived_series:
            levels = derived_series[rep]
            spec = derived.BY_ID[rep]
            change_unit, unit_label = spec.change_unit, spec.unit_label
        elif rep in wide.columns:
            levels = wide[rep].dropna()
            inst = registry[rep]
            change_unit = inst.change_unit
            unit_label = {"bp": "bp", "pct": "%", "pts": "pt"}[change_unit]
        else:
            return None
        if session not in levels.index or len(levels) < window + min_obs:
            return None

        changes = rolls.mask_changes(
            tf.change_series(levels, change_unit), flags, rep
        )
        z = tf.rolling_zscore(changes, window, min_obs)
        today_z = z.get(session)
        history_abs = z.dropna().abs()
        percentile = (
            float((history_abs <= abs(today_z)).mean())
            if today_z is not None and not pd.isna(today_z) and len(history_abs) > 30
            else None
        )
        ratio = tf.vol_ratio(changes, window).get(session)
        return {
            "instrument_id": rep, "change": changes.get(session), "unit": unit_label,
            "z": today_z, "percentile": percentile,
            "quiet_vol": ratio is not None and not pd.isna(ratio) and ratio < 0.6,
        }

    signals: dict[str, Signal] = {}
    for bloc in BLOCS:
        if bloc.select == "strongest":
            best = None
            for member in bloc.members:
                if member in EXCLUDED:
                    continue
                m = measure(member)
                if m is None or m["z"] is None or pd.isna(m["z"]):
                    continue
                if best is None or abs(m["z"]) > abs(best["z"]):
                    best = m
            if best is not None:
                signals[bloc.id] = Signal(bloc=bloc.id, name=bloc.name, **best)
            continue

        rep = bloc.representative
        if rep in EXCLUDED:
            continue

        m = measure(rep)
        if m is not None:
            signals[bloc.id] = Signal(bloc=bloc.id, name=bloc.name, **m)
    return signals


def composite_intensity(signals: dict[str, Signal]) -> float:
    """Mean clipped |z| across the core blocs.

    Clipping matters: without it a single quiet-vol instrument at 8 sigma sets
    the score regardless of what the other four legs did.
    """
    values = [
        abs(signals[b].clipped_z) for b in CORE_BLOCS if b in signals
    ]
    return float(np.mean(values)) if values else 0.0


def intensity_history(
    history: pd.DataFrame,
    registry: Registry | None = None,
    flags: pd.DataFrame | None = None,
    window: int = 60,
    min_obs: int = 30,
) -> pd.Series:
    """Composite intensity across all sessions, for percentile calibration."""
    registry = registry or load_registry()
    flags = rolls.load_flags() if flags is None else flags
    wide = store.wide(history, "close")
    derived_series = derived.build(wide, registry)

    frames = {}
    for bloc_id in CORE_BLOCS:
        rep = BLOC_BY_ID[bloc_id].representative
        if rep in derived_series:
            levels, unit = derived_series[rep], derived.BY_ID[rep].change_unit
        elif rep in wide.columns:
            levels, unit = wide[rep].dropna(), registry[rep].change_unit
        else:
            continue
        changes = rolls.mask_changes(tf.change_series(levels, unit), flags, rep)
        z = tf.rolling_zscore(changes, window, min_obs)
        frames[bloc_id] = z.clip(-Z_CLIP, Z_CLIP).abs()

    if not frames:
        return pd.Series(dtype=float)
    return pd.DataFrame(frames).mean(axis=1, skipna=True).dropna()


def _confirmations(
    rule: Rule,
    signals: dict[str, Signal],
    wide: pd.DataFrame,
    session: pd.Timestamp,
) -> list[Confirmation]:
    """Evaluate each of a rule's corroborating checks."""
    out: list[Confirmation] = []

    def sector_move(ticker: str) -> float | None:
        if ticker not in wide.columns:
            return None
        series = wide[ticker].dropna()
        if session not in series.index or len(series) < 2:
            return None
        return float(tf.change_series(series, "pct").get(session, np.nan))

    def group(tickers) -> float | None:
        values = [sector_move(t) for t in tickers]
        values = [v for v in values if v is not None and not pd.isna(v)]
        return float(np.mean(values)) if values else None

    defensives = group(["xlu", "xlp", "xlv"])
    cyclicals = group(["xli", "xlf", "xlb", "xly"])

    for name in rule.confirmations:
        if name == "defensives_lead":
            ok = None if None in (defensives, cyclicals) else defensives > cyclicals
            out.append(Confirmation(
                name, ok,
                f"defensives {defensives:+.2f}% vs cyclicals {cyclicals:+.2f}%"
                if ok is not None else "sector data unavailable"))

        elif name == "cyclicals_lead":
            ok = None if None in (defensives, cyclicals) else cyclicals > defensives
            out.append(Confirmation(
                name, ok,
                f"cyclicals {cyclicals:+.2f}% vs defensives {defensives:+.2f}%"
                if ok is not None else "sector data unavailable"))

        elif name == "growth_underperforms":
            tech, value = sector_move("xlk"), sector_move("xlf")
            ok = None if None in (tech, value) else tech < value
            out.append(Confirmation(
                name, ok,
                f"tech {tech:+.2f}% vs financials {value:+.2f}%"
                if ok is not None else "sector data unavailable"))

        elif name in ("vol_up", "vol_down"):
            sig = signals.get("vol")
            if sig is None or sig.change is None or pd.isna(sig.change):
                out.append(Confirmation(name, None, "VIX unavailable"))
            else:
                rising = sig.change > 0
                ok = rising if name == "vol_up" else not rising
                out.append(Confirmation(name, ok, f"VIX {sig.change:+.2f}pt"))

        elif name in ("credit_weak", "credit_firm"):
            sig = signals.get("credit")
            if sig is None or sig.change is None or pd.isna(sig.change):
                out.append(Confirmation(name, None, "credit proxy unavailable"))
            else:
                weak = sig.change < 0
                ok = weak if name == "credit_weak" else not weak
                out.append(Confirmation(
                    name, ok, f"duration-hedged HY {sig.change:+.2f}%"))

        elif name in ("curve_flattens", "curve_steepens"):
            sig = signals.get("curve")
            if sig is None or sig.change is None or pd.isna(sig.change):
                out.append(Confirmation(name, None, "curve unavailable"))
            else:
                flattening = sig.change < 0
                ok = flattening if name == "curve_flattens" else not flattening
                out.append(Confirmation(name, ok, f"2s10s {sig.change:+.1f}bp"))

        elif name in ("front_end_leads", "long_end_leads"):
            legs = {}
            for tenor in ("ust_2y", "ust_10y", "ust_30y"):
                if tenor in wide.columns:
                    series = wide[tenor].dropna()
                    if session in series.index and len(series) > 1:
                        legs[tenor] = tf.change_series(series, "bp").get(session)
            legs = {k: v for k, v in legs.items() if v is not None and not pd.isna(v)}
            if len(legs) < 2:
                out.append(Confirmation(name, None, "yield legs unavailable"))
            elif name == "front_end_leads":
                ok = abs(legs.get("ust_2y", 0)) > abs(legs.get("ust_10y", 0))
                out.append(Confirmation(
                    name, ok,
                    f"2y {legs.get('ust_2y', float('nan')):+.0f}bp vs "
                    f"10y {legs.get('ust_10y', float('nan')):+.0f}bp"))
            else:
                ok = abs(legs.get("ust_30y", 0)) > abs(legs.get("ust_2y", 0))
                out.append(Confirmation(
                    name, ok,
                    f"30y {legs.get('ust_30y', float('nan')):+.0f}bp vs "
                    f"2y {legs.get('ust_2y', float('nan')):+.0f}bp"))

        elif name in ("dollar_up", "dollar_down", "gold_up", "gold_down",
                      "energy_up"):
            bloc_id = {"dollar_up": "dollar", "dollar_down": "dollar",
                       "gold_up": "gold", "gold_down": "gold",
                       "energy_up": "energy"}[name]
            sig = signals.get(bloc_id)
            if sig is None or sig.change is None or pd.isna(sig.change):
                out.append(Confirmation(name, None, f"{bloc_id} unavailable"))
            else:
                rising = sig.change > 0
                ok = rising if name.endswith("_up") else not rising
                out.append(Confirmation(
                    name, ok, f"{sig.instrument_id} {sig.change:+.2f}{sig.unit}"))

        elif name == "copper_gold_up":
            copper, gold = signals.get("industrial"), signals.get("gold")
            if copper is None or gold is None or copper.change is None:
                out.append(Confirmation(name, None, "copper/gold unavailable"))
            else:
                ok = copper.change > (gold.change or 0)
                out.append(Confirmation(
                    name, ok,
                    f"copper {copper.change:+.2f}% vs gold {gold.change or 0:+.2f}%"))

    return out


def classify(
    history: pd.DataFrame,
    registry: Registry | None = None,
    session: dt.date | pd.Timestamp | None = None,
    flags: pd.DataFrame | None = None,
    window: int = 60,
    min_obs: int = 30,
    intensity_threshold: float | None = None,
) -> Classification:
    """Classify one session, or decline to."""
    registry = registry or load_registry()
    flags = rolls.load_flags() if flags is None else flags
    session = pd.Timestamp(session or history["date"].max()).normalize()

    signals = build_signals(history, registry, session, flags, window, min_obs)
    intensity = composite_intensity(signals)

    # The percentile is needed for per-rule severity gates, so it is computed
    # even when the caller supplies a threshold (as --scan and --events do).
    series = intensity_history(history, registry, flags, window, min_obs)
    series = series[series.index <= session]
    have_history = len(series) > 100
    pct = float((series <= intensity).mean()) if have_history else None
    if intensity_threshold is None:
        intensity_threshold = (
            float(series.quantile(INTENSITY_PERCENTILE)) if have_history else 0.75
        )

    result = Classification(
        session=session.date(), regime=None, regime_name="No clean signal",
        intensity=intensity, intensity_percentile=pct, signals=signals,
    )

    missing_core = [b for b in CORE_BLOCS if b not in signals]
    if missing_core:
        result.notes.append(
            f"Core blocs unavailable: {', '.join(missing_core)}. Not classified."
        )
        return result

    if intensity < intensity_threshold:
        result.notes.append(
            f"Composite intensity {intensity:.2f} is below the "
            f"{INTENSITY_PERCENTILE:.0%} threshold of {intensity_threshold:.2f}. "
            "Most sessions are noise, and this is one of them."
        )
        return result

    wide = store.wide(history[history["date"] <= session], "close")

    matches = []
    gated = []
    for rule in RULES:
        satisfied = True
        for bloc_id, required in rule.legs.items():
            sig = signals.get(bloc_id)
            if sig is None or not sig.moved or sig.direction != required:
                satisfied = False
                break
        if not satisfied:
            continue
        floor = rule.min_intensity_percentile
        if floor is not None and pct is not None and pct < floor:
            gated.append((rule, floor))
            continue
        matches.append(rule)

    for rule, floor in gated:
        result.notes.append(
            f"{rule.name} fits the legs but the session is only at the "
            f"{pct:.0%} intensity percentile, below the {floor:.0%} this "
            "label requires. A crisis name needs crisis-level evidence."
        )

    if not matches:
        moved = [s.bloc for s in signals.values() if s.moved]
        result.notes.append(
            "Active session but no rule matched. Blocs that moved: "
            f"{', '.join(moved) if moved else 'none'}. "
            "A day can be busy without having a coherent cross-asset signature."
        )
        return result

    # Prefer the most specific rule; a tie means genuinely ambiguous.
    matches.sort(key=lambda r: -len(r.legs))
    chosen = matches[0]
    result.candidates = [r.id for r in matches]

    result.regime = chosen.id
    result.regime_name = chosen.name
    result.notes.append(chosen.description)
    result.confirmations = _confirmations(chosen, signals, wide, session)

    if len(matches) > 1 and len(matches[1].legs) == len(chosen.legs):
        result.notes.append(
            f"Ambiguous: {matches[1].name} fits the same legs equally well."
        )

    if result.conflicting:
        names = ", ".join(c.id for c in result.conflicting)
        if result.confidence == "contradicted":
            # Detecting a mismatch and then asserting the label anyway with a
            # footnote is the worst of both. Withdraw the claim and say what was
            # closest, so the reader gets the shape without a false conclusion.
            result.notes.append(
                f"WITHDRAWN: the legs fit {chosen.name.lower()}, but {names} "
                "contradicted it. Reporting no regime rather than one the "
                "evidence argues against."
            )
            result.regime = None
            result.regime_name = f"No clean signal (closest: {chosen.name})"
        else:
            result.notes.append(
                f"CONFLICT: {names} did not corroborate. The tape looks like "
                f"{chosen.name.lower()} but some confirming signals disagree. "
                "Reported rather than resolved."
            )
    return result


def render(result: Classification) -> str:
    lines = [
        "=" * 78,
        f"REGIME  {result.session}",
        "=" * 78,
        "",
        f"  {result.regime_name}",
    ]
    if result.classified:
        lines.append(f"  confidence: {result.confidence}")
    pct = (
        f" ({result.intensity_percentile:.0%} of sessions)"
        if result.intensity_percentile is not None else ""
    )
    lines.append(f"  intensity: {result.intensity:.2f}{pct}")

    lines.append("")
    lines.append("SIGNALS (one per bloc; correlated members deflated):")
    lines.append(f"  {'bloc':<12}{'via':<16}{'change':>11}{'z':>8}{'pct':>7}")
    for bloc in BLOCS:
        sig = result.signals.get(bloc.id)
        if sig is None:
            lines.append(f"  {bloc.id:<12}{'--':<16}{'unavailable':>11}")
            continue
        change = "--" if sig.change is None or pd.isna(sig.change) else \
            f"{sig.change:+.2f}{sig.unit}"
        z = "--" if sig.z is None or pd.isna(sig.z) else f"{sig.z:+.2f}"
        p = "--" if sig.percentile is None else f"{sig.percentile:.0%}"
        mark = " <" if sig.moved else ""
        q = " q" if sig.quiet_vol else ""
        lines.append(
            f"  {bloc.id:<12}{sig.instrument_id:<16}{change:>11}{z:>8}{p:>7}{mark}{q}"
        )

    if result.confirmations:
        lines.append("")
        lines.append("CONFIRMATION:")
        for c in result.confirmations:
            mark = {True: "  ok ", False: "  XX ", None: "  -- "}[c.holds]
            lines.append(f"  {mark}{c.id:<22}{c.detail}")

    for note in result.notes:
        lines.append("")
        lines.append(f"  {note}")

    lines.append("")
    lines.append("< = cleared the empirical move threshold   q = quiet vol period")
    lines.append("=" * 78)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    default_store = Path(__file__).resolve().parent.parent / "data" / "history.parquet"

    parser = argparse.ArgumentParser(description="Classify the cross-asset regime")
    parser.add_argument("--session", default=None)
    parser.add_argument("--store", default=str(default_store))
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--events", action="store_true",
                        help="classify the known replay events")
    parser.add_argument("--scan", type=int, metavar="N",
                        help="classify the last N sessions and summarise")
    args = parser.parse_args(argv)

    history = store.read(Path(args.store))
    if history.empty:
        print("store is empty; run the backfill first")
        return 1

    registry = load_registry()
    flags = rolls.load_flags()

    if args.events:
        from .replay import EVENTS

        series = intensity_history(history, registry, flags, args.window)
        threshold = float(series.quantile(INTENSITY_PERCENTILE))
        print(f"\nintensity threshold ({INTENSITY_PERCENTILE:.0%}): {threshold:.2f}\n")
        for event in EVENTS:
            day = pd.Timestamp(event.date)
            if day < history["date"].min():
                continue
            result = classify(history, registry, day, flags, args.window,
                              intensity_threshold=threshold)
            conf = f"  [{result.confidence}]" if result.classified else ""
            print(f"{event.date}  {event.name}")
            print(f"            -> {result.regime_name}{conf}   "
                  f"intensity {result.intensity:.2f}")
            if result.conflicting:
                print(f"               conflicts: "
                      f"{', '.join(c.id for c in result.conflicting)}")
        return 0

    if args.scan:
        from . import calendars

        latest = history["date"].max()
        start = (latest - dt.timedelta(days=args.scan * 3)).date().isoformat()
        days = calendars.spine_sessions(start, latest.date().isoformat())[-args.scan:]
        series = intensity_history(history, registry, flags, args.window)
        threshold = float(series.quantile(INTENSITY_PERCENTILE))

        rows = []
        for day in days:
            result = classify(history, registry, day, flags, args.window,
                              intensity_threshold=threshold)
            rows.append({
                "session": result.session,
                "regime": result.regime or "-",
                "confidence": result.confidence,
                "intensity": round(result.intensity, 2),
                "conflicts": len(result.conflicting),
            })
        frame = pd.DataFrame(rows)
        print(f"\nCLASSIFICATION SCAN, threshold {threshold:.2f}\n")
        print(frame.to_string(index=False))
        print("\nRegime counts:")
        print(frame["regime"].value_counts().to_string())
        unclassified = int((frame["regime"] == "-").sum())
        print(f"\n{unclassified} of {len(frame)} sessions left unclassified "
              f"({unclassified / len(frame):.0%}).")
        return 0

    print(render(classify(history, registry, args.session, flags, args.window)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
