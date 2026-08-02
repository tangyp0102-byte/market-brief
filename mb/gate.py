"""The validation gate.

Decides whether a given session's data is good enough to publish a brief, and
what to say about it if not. Runs before any interpretation.

Three verdicts:

  blocked   Not an NYSE session, or a core instrument is missing or stale.
            No brief. Say why.
  degraded  Core tape intact, some confirmations absent. Brief is publishable
            but the classifier must lower its confidence and name what is gone.
  pass      Everything expected is present and fresh.

The central design decision is that ABSENCE IS NOT FAILURE. On Columbus Day the
bond market is shut while NYSE trades, so missing Treasury yields are correct.
A gate that cannot tell a holiday from an outage gets ignored within a fortnight,
and an ignored gate is worse than no gate.

max_abs_change breaches are reported for REVIEW, never dropped. Of nine breaches
in seven years of real data, six were genuine market events (the March 2020
energy crash, the January 2022 gas squeeze, the July 2025 copper collapse).
Enforcing the threshold as a filter would delete real history.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import calendars, rolls, store
from . import transforms as tf
from .registry import Instrument, Registry, load_registry

PASS = "pass"
DEGRADED = "degraded"
BLOCKED = "blocked"


@dataclass
class Finding:
    instrument_id: str
    detail: str
    tier: str = "confirmation"

    def __str__(self) -> str:
        return f"{self.instrument_id}: {self.detail}"


@dataclass
class GateResult:
    session: dt.date
    verdict: str
    is_trading_day: bool = True
    is_early_close: bool = False
    missing: list[Finding] = field(default_factory=list)
    stale: list[Finding] = field(default_factory=list)
    review: list[Finding] = field(default_factory=list)
    undefined: list[Finding] = field(default_factory=list)
    not_expected: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def publishable(self) -> bool:
        return self.verdict in (PASS, DEGRADED)

    @property
    def core_problems(self) -> list[Finding]:
        return [f for f in self.missing + self.stale if f.tier == "core"]

    def render(self) -> str:
        icon = {PASS: "PASS", DEGRADED: "DEGRADED", BLOCKED: "BLOCKED"}[self.verdict]
        lines = [
            "=" * 66,
            f"VALIDATION GATE  {self.session}  ->  {icon}",
            "=" * 66,
        ]

        if not self.is_trading_day:
            lines.append("NYSE was closed. No brief for this date.")
            lines.append("=" * 66)
            return "\n".join(lines)

        if self.is_early_close:
            lines.append(
                "Half session (1pm ET close). Volumes and ranges are muted, so "
                "z-scores computed against full-day volatility understate moves."
            )

        for label, findings in (
            ("MISSING", self.missing),
            ("STALE", self.stale),
            ("UNDEFINED CHANGE", self.undefined),
            ("FOR REVIEW (not dropped)", self.review),
        ):
            if findings:
                lines.append("")
                lines.append(f"{label}:")
                for f in findings:
                    marker = "CORE " if f.tier == "core" else "     "
                    lines.append(f"  {marker}{f}")

        if self.not_expected:
            lines.append("")
            lines.append(
                f"Legitimately absent ({len(self.not_expected)}): market closed "
                f"for {', '.join(sorted(self.not_expected)[:6])}"
                + (" ..." if len(self.not_expected) > 6 else "")
            )

        for note in self.notes:
            lines.append("")
            lines.append(note)

        lines.append("=" * 66)
        return "\n".join(lines)


def _levels_for(history: pd.DataFrame, session: pd.Timestamp) -> pd.DataFrame:
    """Wide close frame up to and including `session`."""
    upto = history[history["date"] <= session]
    return store.wide(upto, "close")


def run_gate(
    history: pd.DataFrame,
    registry: Registry | None = None,
    session: dt.date | pd.Timestamp | None = None,
    flags: pd.DataFrame | None = None,
) -> GateResult:
    """Validate one session's data.

    `flags` are contract-roll flags from mb.probe --rollcheck --save. Loaded from
    the default path when not supplied; an absent file simply means no flags.
    """
    registry = registry or load_registry()
    flags = rolls.load_flags() if flags is None else flags

    if session is None:
        session = history["date"].max() if not history.empty else pd.Timestamp.today()
    session = pd.Timestamp(session).normalize()
    iso = session.date().isoformat()

    if not calendars.is_trading_day(session):
        return GateResult(
            session=session.date(), verdict=BLOCKED, is_trading_day=False,
            notes=["NYSE closed."],
        )

    result = GateResult(
        session=session.date(),
        verdict=PASS,
        is_early_close=calendars.is_early_close(session),
    )

    levels = _levels_for(history, session)
    present_today = set(levels.columns[levels.loc[session].notna()]) if session in levels.index else set()

    for inst in registry:
        if not calendars.expected_on(inst, session):
            result.not_expected.append(inst.id)
            continue

        if inst.id not in levels.columns:
            result.missing.append(Finding(inst.id, "no data at all", inst.tier))
            continue

        series = levels[inst.id].dropna()
        if series.empty:
            result.missing.append(Finding(inst.id, "no data at all", inst.tier))
            continue

        last = series.index[-1]

        if inst.id not in present_today:
            behind = calendars.sessions_between(last, session, inst.calendar)
            if behind > inst.stale_tolerance_days:
                result.stale.append(
                    Finding(
                        inst.id,
                        f"last observation {last.date()} ({behind} sessions behind, "
                        f"tolerance {inst.stale_tolerance_days})",
                        inst.tier,
                    )
                )
            else:
                result.not_expected.append(inst.id)
            continue

        _check_change(inst, series, session, result, flags)

    core_broken = bool(result.core_problems)
    if core_broken:
        result.verdict = BLOCKED
        names = ", ".join(sorted({f.instrument_id for f in result.core_problems}))
        result.notes.append(f"Core tape incomplete ({names}). No brief.")
    elif result.missing or result.stale:
        result.verdict = DEGRADED
        n = len({f.instrument_id for f in result.missing + result.stale})
        result.notes.append(
            f"{n} confirmation signal(s) unavailable. Classify with reduced "
            "confidence and name the gap in the brief."
        )

    return result


def _check_change(
    inst: Instrument,
    series: pd.Series,
    session: pd.Timestamp,
    result: GateResult,
    flags: pd.DataFrame | None = None,
) -> None:
    """Flag undefined changes and threshold breaches for this session."""
    if len(series) < 2:
        return

    flags = rolls.empty_flags() if flags is None else flags
    flag = rolls.lookup(flags, inst.id, session)

    if flag and flag.get("label") == rolls.ROLL:
        proxy = flag.get("proxy")
        proxy_pct = flag.get("proxy_pct")
        result.undefined.append(
            Finding(
                inst.id,
                f"contract roll - change undefined "
                f"(series moved {flag.get('futures_pct')}%, {proxy} moved {proxy_pct}%)",
                inst.tier,
            )
        )
        return

    changes = rolls.mask_changes(
        tf.change_series(series, inst.change_unit), flags, inst.id
    )
    value = changes.get(session)

    if flag and flag.get("label") == rolls.DIVERGENT:
        result.review.append(
            Finding(
                inst.id,
                f"diverges from {flag.get('proxy')} (ratio {flag.get('ratio')}) - "
                "usually a settlement-time mismatch rather than a roll, "
                "but worth a glance",
                inst.tier,
            )
        )

    if value is None:
        return

    if pd.isna(value):
        prev = series.iloc[-2]
        curr = series.iloc[-1]
        if inst.change_unit == "pct" and (prev <= 0 or curr <= 0):
            result.undefined.append(
                Finding(
                    inst.id,
                    f"percent change undefined across zero ({prev:.2f} -> {curr:.2f})",
                    inst.tier,
                )
            )
        return

    if abs(value) > inst.max_abs_change:
        result.review.append(
            Finding(
                inst.id,
                f"{value:+.1f}{inst.change_unit} exceeds threshold "
                f"{inst.max_abs_change}{inst.change_unit} - verify against a chart "
                "before publishing",
                inst.tier,
            )
        )


# Sessions that exercise a specific code path. A 60-session scan of a quiet
# stretch tells you the gate does not crash; these tell you it works.
STRESS_SESSIONS: tuple[tuple[str, str], ...] = (
    ("2020-03-09", "COVID + oil price war: XLE -20.1%"),
    ("2020-04-20", "WTI settles negative (-$37.63)"),
    ("2020-04-21", "day after negative settlement"),
    ("2022-01-27", "natgas expiry squeeze +46.5%"),
    ("2023-10-09", "Columbus Day: NYSE open, bonds closed"),
    ("2024-10-14", "Columbus Day: NYSE open, bonds closed"),
    ("2024-11-11", "Veterans Day: NYSE open, bonds closed"),
    ("2024-11-29", "half session (day after Thanksgiving)"),
    ("2024-12-24", "half session (Christmas Eve)"),
    ("2024-12-25", "NYSE closed"),
    ("2025-07-31", "copper tariff collapse -22.3%"),
    ("2026-01-29", "natgas -47.5% (unverified: real or roll?)"),
    ("2026-01-30", "silver -31.3% (unverified: real or roll?)"),
)


def _scan(history, registry, sessions_to_check, labels=None) -> pd.DataFrame:
    rows = []
    for day in sessions_to_check:
        result = run_gate(history, registry, day)
        row = {
            "session": result.session,
            "verdict": result.verdict,
            "missing": len(result.missing),
            "stale": len(result.stale),
            "review": len(result.review),
            "undef": len(result.undefined),
            "absent_ok": len(result.not_expected),
            "half": "Y" if result.is_early_close else "",
        }
        if labels:
            row["exercises"] = labels.get(str(result.session), "")
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    import argparse

    default_store = Path(__file__).resolve().parent.parent / "data" / "history.parquet"

    parser = argparse.ArgumentParser(description="Run the validation gate")
    parser.add_argument("--session", default=None, help="ISO date; default = latest stored")
    parser.add_argument("--store", default=str(default_store))
    parser.add_argument(
        "--scan", type=int, metavar="N",
        help="run the gate over the last N sessions and summarise",
    )
    parser.add_argument("--from", dest="date_from", help="scan from this ISO date")
    parser.add_argument("--to", dest="date_to", help="scan to this ISO date")
    parser.add_argument(
        "--stress", action="store_true",
        help="scan a curated set of sessions that exercise each code path",
    )
    args = parser.parse_args(argv)

    history = store.read(Path(args.store))
    if history.empty:
        print("store is empty; run the backfill first")
        return 1

    registry = load_registry()
    latest = history["date"].max()

    if args.stress:
        labels = {d: why for d, why in STRESS_SESSIONS}
        days = [pd.Timestamp(d) for d, _ in STRESS_SESSIONS]
        frame = _scan(history, registry, days, labels)
        print("\nSTRESS SCAN (sessions chosen to exercise specific paths):\n")
        print(frame.to_string(index=False))
        print(
            "\nExpected: BLOCKED on 2024-12-25 only; absent_ok > 0 on the three "
            "bond holidays; half = Y on the two half sessions; review or undef "
            "> 0 on the extreme-move dates."
        )
        return 0

    if args.date_from or args.date_to:
        start = args.date_from or "2019-01-01"
        end = args.date_to or latest.date().isoformat()
        days = calendars.spine_sessions(start, end)
        days = days[days <= latest]
        frame = _scan(history, registry, days)
        interesting = frame[
            (frame["verdict"] != PASS)
            | (frame[["missing", "stale", "review", "undef", "absent_ok"]].sum(axis=1) > 0)
        ]
        print(f"\nGATE SCAN {start} to {end}: {len(frame)} sessions\n")
        print("Verdict counts:")
        print(frame["verdict"].value_counts().to_string())
        print(f"\nSessions with anything to report ({len(interesting)}):\n")
        print("  none" if interesting.empty else interesting.to_string(index=False))
        return 0

    if args.scan:
        start = (latest - dt.timedelta(days=args.scan * 2)).date().isoformat()
        days = calendars.spine_sessions(start, latest.date().isoformat())[-args.scan:]
        frame = _scan(history, registry, days)
        print(f"\nGATE SCAN over {len(frame)} NYSE sessions:\n")
        print(frame.to_string(index=False))
        print("\nVerdict counts:")
        print(frame["verdict"].value_counts().to_string())
        return 0

    result = run_gate(history, registry, args.session)
    print(result.render())
    return 0 if result.publishable else 1


if __name__ == "__main__":
    raise SystemExit(main())
