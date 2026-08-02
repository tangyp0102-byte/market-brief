"""The daily tape: layers A and B, and nothing else.

Layer A is the level and its change. Layer B is that change scored against the
preceding 60 sessions.

There is deliberately NO interpretation here. No regime label, no narrative, no
"risk-off". The point of this stage is to look at real numbers for a stretch of
sessions and judge whether they are sane before anything exists that could be
wrong about them. A sign convention inverted here is obvious in a bare table and
nearly invisible once a classifier has wrapped a story around it.

Two things the z-score does that matter:

  * Roll-flagged and sign-crossing changes are NaN before scoring, so fabricated
    moves never enter the volatility window.
  * The window is the 60 sessions BEFORE the one being scored, so a large move is
    measured against a distribution that does not contain it.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import calendars, derived, gate, rolls, store
from . import transforms as tf
from .registry import Registry, load_registry

# Display precision by quote type. FX rates carry four decimals by convention
# (USD/JPY is the exception at two, handled by the magnitude fallback); yields
# and index levels carry two; ratios fall through to magnitude-scaled precision.
_LEVEL_PLACES = {"percent": 2, "index": 2, "usd": 2}

ASSET_CLASS_ORDER = ["rates", "credit", "equity", "vol", "commodity", "fx"]


@dataclass
class TapeRow:
    instrument_id: str
    name: str
    asset_class: str
    tier: str
    level: float
    change: float | None
    unit_label: str
    z: float | None
    z_robust: float | None
    vol_ratio: float | None = None
    level_places: int | None = None
    is_derived: bool = False
    note: str = ""

    @property
    def quiet_vol(self) -> bool:
        """Trailing vol well below its longer-run level, which inflates z.

        Deliberately not called a "regime": that word is reserved for the step-5
        classifier, and overloading it here would blur a statistical
        observation into a market judgement.
        """
        return (
            self.vol_ratio is not None
            and not pd.isna(self.vol_ratio)
            and self.vol_ratio < 0.6
        )

    @property
    def significance(self) -> str:
        """Plain magnitude description. Not a signal, just arithmetic."""
        if self.z is None or pd.isna(self.z):
            return ""
        a = abs(self.z)
        if a >= 3:
            return "***"
        if a >= 2:
            return "**"
        if a >= 1:
            return "*"
        return ""


@dataclass
class Tape:
    session: dt.date
    rows: list[TapeRow]
    verdict: str
    gate_result: gate.GateResult | None = None
    window: int = 60
    notes: list[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(r) for r in self.rows])

    def by_class(self, asset_class: str) -> list[TapeRow]:
        return [r for r in self.rows if r.asset_class == asset_class]

    def largest(self, n: int = 8) -> list[TapeRow]:
        scored = [r for r in self.rows if r.z is not None and not pd.isna(r.z)]
        return sorted(scored, key=lambda r: -abs(r.z))[:n]


def _fmt(value: float | None, places: int | None = None) -> str:
    """Format a level with precision scaled to its magnitude.

    A fixed two decimals renders the copper/gold ratio (0.00159) as "0.00",
    hiding the entire signal. Ratios and rates span six orders of magnitude in
    this tape, so the precision has to follow the number.
    """
    if value is None or pd.isna(value):
        return "--"
    if places is not None:
        return f"{value:,.{places}f}"

    magnitude = abs(value)
    if magnitude >= 1000:
        places = 2
    elif magnitude >= 1:
        places = 2
    elif magnitude >= 0.01:
        places = 4
    elif magnitude > 0:
        places = 6
    else:
        places = 2
    return f"{value:,.{places}f}"


def _fmt_z(value: float | None) -> str:
    """Z-scores always render at two decimals.

    Magnitude-scaled precision belongs on levels, where the copper/gold ratio
    needs six decimals and an equity index needs two. Applied to a z-score it
    produces 0.8769 next to 1.50 in the same column, which reads as a data
    difference rather than a formatting one.
    """
    if value is None or pd.isna(value):
        return "--"
    return f"{value:.2f}"


def _fmt_change(value: float | None, unit: str) -> str:
    if value is None or pd.isna(value):
        return "undefined"
    places = 1 if unit == "bp" else 2
    return f"{value:+,.{places}f}{unit}"


def build_tape(
    history: pd.DataFrame,
    registry: Registry | None = None,
    session: dt.date | pd.Timestamp | None = None,
    flags: pd.DataFrame | None = None,
    window: int = 60,
    min_obs: int = 30,
    run_gate: bool = True,
) -> Tape:
    """Assemble one session's tape."""
    registry = registry or load_registry()
    flags = rolls.load_flags() if flags is None else flags

    if session is None:
        session = history["date"].max()
    session = pd.Timestamp(session).normalize()

    gate_result = gate.run_gate(history, registry, session, flags) if run_gate else None
    verdict = gate_result.verdict if gate_result else "unchecked"

    if gate_result is not None and not gate_result.publishable:
        return Tape(
            session=session.date(), rows=[], verdict=verdict, gate_result=gate_result,
            window=window,
            notes=["Gate blocked this session; no tape produced."],
        )

    wide = store.wide(history[history["date"] <= session], "close")
    rows: list[TapeRow] = []

    for inst in registry:
        if inst.id not in wide.columns:
            continue
        levels = wide[inst.id].dropna()
        if session not in levels.index:
            continue

        changes = rolls.mask_changes(
            tf.change_series(levels, inst.change_unit), flags, inst.id
        )
        z = tf.rolling_zscore(changes, window, min_obs)
        z_rob = tf.rolling_zscore(changes, window, min_obs, robust=True)
        vr = tf.vol_ratio(changes, window)

        unit_label = {"bp": "bp", "pct": "%", "pts": "pt"}[inst.change_unit]
        rows.append(
            TapeRow(
                instrument_id=inst.id,
                name=inst.name,
                asset_class=inst.asset_class,
                tier=inst.tier,
                level=float(levels.loc[session]),
                change=changes.get(session),
                unit_label=unit_label,
                z=z.get(session),
                z_robust=z_rob.get(session),
                vol_ratio=vr.get(session),
                level_places=_LEVEL_PLACES.get(inst.quote_unit),
            )
        )

    for did, series in derived.build(wide, registry).items():
        spec = derived.BY_ID[did]
        if session not in series.index:
            continue
        changes = tf.change_series(series, spec.change_unit)
        z = tf.rolling_zscore(changes, window, min_obs)
        z_rob = tf.rolling_zscore(changes, window, min_obs, robust=True)
        vr = tf.vol_ratio(changes, window)
        rows.append(
            TapeRow(
                instrument_id=did,
                name=spec.name,
                asset_class=spec.asset_class,
                tier=spec.tier,
                level=float(series.loc[session]),
                change=changes.get(session),
                unit_label=spec.unit_label,
                z=z.get(session),
                z_robust=z_rob.get(session),
                vol_ratio=vr.get(session),
                level_places=1 if spec.unit_label == "bp" else None,
                is_derived=True,
                note=spec.note,
            )
        )

    notes = []
    missing_derived = derived.unavailable(wide)
    if missing_derived:
        notes.append(f"Derived unavailable (inputs missing): {', '.join(missing_derived)}")

    return Tape(
        session=session.date(), rows=rows, verdict=verdict,
        gate_result=gate_result, window=window, notes=notes,
    )


def render(tape: Tape, show_all: bool = True) -> str:
    """Plain-text tape. Columns are fixed width so misaligned units are visible."""
    lines = [
        "=" * 84,
        f"MARKET TAPE  {tape.session}   gate: {tape.verdict}   z-window: {tape.window} sessions",
        "=" * 84,
    ]

    if not tape.rows:
        lines.extend(tape.notes or ["No data."])
        lines.append("=" * 84)
        return "\n".join(lines)

    if tape.gate_result and tape.gate_result.is_early_close:
        lines.append("HALF SESSION - z-scores understate moves against full-day vol.")
        lines.append("")

    header = f"{'instrument':<32}{'level':>13}{'change':>13}{'z':>8}{'z(rob)':>9}  "
    for asset_class in ASSET_CLASS_ORDER:
        group = tape.by_class(asset_class)
        if not group:
            continue
        lines.append("")
        lines.append(f"--- {asset_class.upper()} " + "-" * (78 - len(asset_class)))
        lines.append(header)
        group = sorted(group, key=lambda r: (r.is_derived, r.instrument_id))
        for row in group:
            marker = "~" if row.is_derived else " "
            core = "!" if row.tier == "core" else " "
            lines.append(
                f"{marker}{core}{row.name[:30]:<30}"
                f"{_fmt(row.level, row.level_places):>13}"
                f"{_fmt_change(row.change, row.unit_label):>13}"
                f"{_fmt_z(row.z):>8}"
                f"{_fmt_z(row.z_robust):>9}  {row.significance}"
                f"{' q' if row.quiet_vol else ''}"
            )

    lines.append("")
    lines.append("-" * 84)
    lines.append("LARGEST MOVES BY Z-SCORE (magnitude only - no interpretation):")
    for row in tape.largest(8):
        lines.append(
            f"  {row.name[:34]:<34}{_fmt_change(row.change, row.unit_label):>12}"
            f"   z={_fmt_z(row.z)}  {row.significance}"
            + (f"   [quiet: 60d vol {row.vol_ratio:.0%} of 250d]"
               if row.quiet_vol else "")
        )

    if tape.gate_result:
        g = tape.gate_result
        for label, findings in (
            ("undefined", g.undefined), ("review", g.review),
            ("missing", g.missing), ("stale", g.stale),
        ):
            if findings:
                lines.append("")
                lines.append(f"{label.upper()}:")
                for f in findings:
                    lines.append(f"  {f}")

    for note in tape.notes:
        lines.append("")
        lines.append(note)

    lines.append("")
    lines.append(
        "~ = derived   ! = core tier   q = quiet vol period (inflates z)   "
        "* >1s  ** >2s  *** >3s"
    )
    lines.append("=" * 84)
    return "\n".join(lines)


def verify(
    history: pd.DataFrame,
    registry: Registry | None = None,
    session: pd.Timestamp | None = None,
    flags: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Internal consistency checks a plain table should let you confirm by eye.

    Each check compares a derived value against an independent route to the same
    number. A mismatch means a unit or sign convention is wrong somewhere.
    """
    registry = registry or load_registry()
    flags = rolls.load_flags() if flags is None else flags
    session = pd.Timestamp(session or history["date"].max()).normalize()

    wide = store.wide(history[history["date"] <= session], "close")
    series = derived.build(wide, registry)
    checks = []

    def add(name, got, expected, tol, explanation):
        ok = (
            got is not None and expected is not None
            and not pd.isna(got) and not pd.isna(expected)
            and abs(got - expected) <= tol
        )
        checks.append(
            {"check": name, "computed": got, "independent": expected,
             "ok": bool(ok), "means": explanation}
        )

    # 2s10s change must equal the 10y change minus the 2y change, in bp.
    if "ust_2s10s" in series:
        spread_chg = tf.change_series(series["ust_2s10s"], "pts").get(session)
        legs = {
            k: tf.change_series(wide[k].dropna(), "bp").get(session)
            for k in ("ust_10y", "ust_2y")
        }
        if all(v is not None for v in legs.values()):
            add("2s10s = d10y - d2y", spread_chg, legs["ust_10y"] - legs["ust_2y"],
                0.6, "curve spread units are consistent with the legs")

    # DXY change must approximate the weighted sum of component log changes.
    if "dxy" in series:
        dxy_chg = tf.change_series(series["dxy"], "pct").get(session)
        weights = {c.id: c.dxy_weight for c in registry.dxy_components()}
        approx = 0.0
        complete = True
        for iid, w in weights.items():
            leg = tf.change_series(wide[iid].dropna(), "pct").get(session)
            if leg is None or pd.isna(leg):
                complete = False
                break
            approx += w * leg
        if complete:
            add("DXY = sum(weight x component)", dxy_chg, approx, 0.05,
                "dollar sign conventions agree across all six crosses")

    # A rise in EUR/USD must move the dollar index down.
    eur = tf.change_series(wide["eurusd"].dropna(), "pct").get(session) if "eurusd" in wide else None
    dxy = tf.change_series(series["dxy"], "pct").get(session) if "dxy" in series else None
    if eur is not None and dxy is not None and abs(eur) > 0.05:
        checks.append({
            "check": "EURUSD up => DXY down",
            "computed": dxy, "independent": eur,
            "ok": bool((eur > 0) != (dxy > 0)),
            "means": "dollar_direction is not inverted",
        })

    # 10y breakeven must equal nominal minus real.
    if "be_10y" in series:
        be = series["be_10y"].get(session)
        nom = wide["ust_10y"].dropna().get(session)
        real = wide["ust_real_10y"].dropna().get(session)
        if None not in (nom, real):
            add("breakeven = nominal - real", be, (nom - real) * 100, 0.6,
                "breakeven is in bp, not percent")

    return pd.DataFrame(checks)


def main(argv: list[str] | None = None) -> int:
    import argparse

    default_store = Path(__file__).resolve().parent.parent / "data" / "history.parquet"

    parser = argparse.ArgumentParser(description="Print the daily market tape")
    parser.add_argument("--session", default=None, help="ISO date; default = latest")
    parser.add_argument("--store", default=str(default_store))
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--last", type=int, metavar="N",
                        help="print the last N sessions in sequence")
    parser.add_argument("--verify", action="store_true",
                        help="run internal consistency checks")
    args = parser.parse_args(argv)

    history = store.read(Path(args.store))
    if history.empty:
        print("store is empty; run the backfill first")
        return 1

    registry = load_registry()
    flags = rolls.load_flags()

    if args.verify:
        session = pd.Timestamp(args.session) if args.session else history["date"].max()
        result = verify(history, registry, session, flags)
        print(f"\nCONSISTENCY CHECKS  {session.date()}\n")
        if result.empty:
            print("  no checks could be run (missing inputs)")
            return 0
        for _, row in result.iterrows():
            status = "OK  " if row["ok"] else "FAIL"
            print(f"[{status}] {row['check']}")
            print(f"         computed={row['computed']:.4f}  "
                  f"independent={row['independent']:.4f}")
            print(f"         {row['means']}")
        failures = int((~result["ok"]).sum())
        print(f"\n{failures} failure(s) of {len(result)} checks.")
        return 1 if failures else 0

    if args.last:
        latest = history["date"].max()
        start = (latest - dt.timedelta(days=args.last * 3)).date().isoformat()
        days = calendars.spine_sessions(start, latest.date().isoformat())[-args.last:]
        for day in days:
            print(render(build_tape(history, registry, day, flags, args.window)))
            print()
        return 0

    tape = build_tape(history, registry, args.session, flags, args.window)
    print(render(tape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
