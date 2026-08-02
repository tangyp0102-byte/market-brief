"""Replay harness: does the data show what we already know happened?

Two independent checks, neither of which is a classifier.

EVENT SIGNATURES. For sessions with a well-documented outcome, assert the
direction of specific instruments. If the SVB weekend does not show a collapse
in 2y yields, something is wrong with the data, and finding that out here is far
cheaper than finding it out through a classifier that has wrapped a story around
it.

The expected signatures below are a HYPOTHESIS, not ground truth. They encode
what I believe happened on each date, and a mismatch means one of two things:
the data is wrong, or my recollection is. The harness cannot tell you which, so
it flags disagreements for a human rather than declaring a verdict. Directions
only, never magnitudes; being right about the sign is a much weaker claim than
being right about the size, and the weaker claim is the one worth testing.

DISTRIBUTION. Across all history, z-scores should be roughly centred near zero
with a standard deviation near one and fat tails. Systematic drift means the
window or the centring is wrong; a standard deviation far from one means the
scaling is wrong; too few tail events means the scores are over-smoothed.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import rolls, store, tape
from . import transforms as tf
from .registry import Registry, load_registry

UP, DOWN = 1, -1


@dataclass(frozen=True)
class Event:
    date: str
    name: str
    expect: dict[str, int]          # instrument_id -> UP | DOWN
    confidence: str = "high"        # high | medium
    note: str = ""


# Directions only. Yields are quoted as yields, so "10y DOWN" means the yield
# fell (a bond rally). FX is quoted natively, so "usdjpy DOWN" means yen strength.
EVENTS: tuple[Event, ...] = (
    Event(
        "2020-03-09", "COVID crash + Saudi/Russia oil price war",
        {"spx": DOWN, "ust_10y": DOWN, "wti": DOWN, "vix": UP, "xle": DOWN},
        note="flight to quality: equities and oil collapse, yields collapse with them",
    ),
    Event(
        "2020-03-16", "Worst single session of the COVID crash",
        {"spx": DOWN, "vix": UP, "ust_10y": DOWN},
    ),
    Event(
        "2022-02-24", "Russia invades Ukraine",
        {"wti": UP, "gold": UP, "brent": UP},
        note="equities reversed intraday, so no equity direction asserted",
    ),
    Event(
        "2022-09-13", "August CPI upside surprise",
        {"spx": DOWN, "ust_2y": UP, "ndx": DOWN},
        note="hawkish repricing: front-end yields up, equities down",
    ),
    Event(
        "2023-03-13", "SVB failure fallout",
        {"ust_2y": DOWN, "xlf": DOWN, "gold": UP},
        confidence="high",
        note="2y yield collapsed on rate-cut repricing; banks led equities lower",
    ),
    Event(
        "2024-08-05", "Yen carry unwind",
        {"vix": UP, "usdjpy": DOWN, "spx": DOWN},
        note="yen strength, vol spike",
    ),
    Event(
        "2025-04-03", "Tariff announcement selloff",
        {"spx": DOWN, "vix": UP},
        confidence="medium",
    ),
    Event(
        "2025-04-04", "Tariff selloff continues",
        {"spx": DOWN, "vix": UP},
        confidence="medium",
    ),
)


@dataclass
class EventResult:
    event: Event
    session: dt.date
    checks: list[dict] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)
    gate_verdict: str = ""

    @property
    def matched(self) -> int:
        return sum(1 for c in self.checks if c["match"])

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def clean(self) -> bool:
        return self.total > 0 and self.matched == self.total


def check_event(
    history: pd.DataFrame,
    event: Event,
    registry: Registry | None = None,
    flags: pd.DataFrame | None = None,
) -> EventResult:
    """Compare one event's expected directions against the tape."""
    registry = registry or load_registry()
    flags = rolls.load_flags() if flags is None else flags
    session = pd.Timestamp(event.date)

    built = tape.build_tape(history, registry, session, flags, run_gate=True)
    result = EventResult(event=event, session=session.date(),
                         gate_verdict=built.verdict)

    rows = {r.instrument_id: r for r in built.rows}
    for iid, expected in event.expect.items():
        row = rows.get(iid)
        if row is None or row.change is None or pd.isna(row.change):
            result.unavailable.append(iid)
            continue
        observed = UP if row.change > 0 else DOWN
        result.checks.append(
            {
                "instrument_id": iid,
                "expected": "up" if expected == UP else "down",
                "observed": "up" if observed == UP else "down",
                "change": round(float(row.change), 2),
                "unit": row.unit_label,
                "z": None if row.z is None or pd.isna(row.z) else round(float(row.z), 2),
                "match": observed == expected,
            }
        )
    return result


def replay_events(
    history: pd.DataFrame,
    registry: Registry | None = None,
    flags: pd.DataFrame | None = None,
    events: tuple[Event, ...] = EVENTS,
) -> list[EventResult]:
    registry = registry or load_registry()
    flags = rolls.load_flags() if flags is None else flags
    earliest = history["date"].min()
    return [
        check_event(history, e, registry, flags)
        for e in events
        if pd.Timestamp(e.date) >= earliest
    ]


def distribution(
    history: pd.DataFrame,
    registry: Registry | None = None,
    flags: pd.DataFrame | None = None,
    window: int = 60,
    min_obs: int = 30,
) -> pd.DataFrame:
    """Per-instrument z-score distribution statistics across all history.

    Expected on well-behaved data: mean near 0, sd near 1, and tails fatter than
    normal (|z|>3 on roughly 0.5-2% of sessions rather than the 0.27% a normal
    distribution implies). Financial returns are fat-tailed, so a tail count at
    or below the normal rate would suggest the scores are over-smoothed.
    """
    registry = registry or load_registry()
    flags = rolls.load_flags() if flags is None else flags
    wide = store.wide(history, "close")

    rows = []
    for inst in registry:
        if inst.id not in wide.columns:
            continue
        levels = wide[inst.id].dropna()
        if len(levels) < window + min_obs:
            continue
        changes = rolls.mask_changes(
            tf.change_series(levels, inst.change_unit), flags, inst.id
        )
        z = tf.rolling_zscore(changes, window, min_obs).dropna()
        if z.empty:
            continue
        rows.append(
            {
                "instrument_id": inst.id,
                "asset_class": inst.asset_class,
                "n": len(z),
                "mean": round(float(z.mean()), 3),
                "sd": round(float(z.std()), 3),
                "pct_gt2": round(float((z.abs() > 2).mean() * 100), 2),
                "pct_gt3": round(float((z.abs() > 3).mean() * 100), 2),
                "max_abs": round(float(z.abs().max()), 1),
            }
        )
    return pd.DataFrame(rows)


def distribution_flags(stats: pd.DataFrame) -> pd.DataFrame:
    """Instruments whose z distribution looks wrong."""
    if stats.empty:
        return stats

    problems = []
    for _, row in stats.iterrows():
        issues = []
        if abs(row["mean"]) > 0.25:
            issues.append(f"mean {row['mean']:+.2f} (drift; centring may be off)")
        if not 0.75 <= row["sd"] <= 1.35:
            issues.append(f"sd {row['sd']:.2f} (scaling may be off)")
        if row["pct_gt3"] > 3.0:
            issues.append(f"{row['pct_gt3']:.1f}% beyond 3 sigma (too many tails)")
        if row["max_abs"] > 12:
            issues.append(f"max |z| {row['max_abs']:.0f} (check for a quiet-vol period)")
        if issues:
            problems.append(
                {"instrument_id": row["instrument_id"], "issues": "; ".join(issues)}
            )
    return pd.DataFrame(problems)


def render_events(results: list[EventResult]) -> str:
    lines = [
        "=" * 84,
        "EVENT REPLAY - expected directions are a hypothesis, not ground truth",
        "=" * 84,
    ]
    for result in results:
        event = result.event
        status = "OK" if result.clean else "CHECK"
        conf = "" if event.confidence == "high" else f"  [confidence: {event.confidence}]"
        lines.append("")
        lines.append(
            f"[{status:5}] {result.session}  {event.name}{conf}"
        )
        lines.append(f"         gate: {result.gate_verdict}   "
                     f"matched {result.matched}/{result.total}")
        for check in result.checks:
            mark = "  ok " if check["match"] else "  XX "
            z = "--" if check["z"] is None else f"{check['z']:+.2f}"
            lines.append(
                f"       {mark}{check['instrument_id']:<10}"
                f"expected {check['expected']:<5}"
                f"observed {check['observed']:<5}"
                f"{check['change']:+8.2f}{check['unit']:<3} z={z}"
            )
        if result.unavailable:
            lines.append(f"         unavailable: {', '.join(result.unavailable)}")
        if event.note:
            lines.append(f"         {event.note}")

    clean = sum(1 for r in results if r.clean)
    lines.append("")
    lines.append("-" * 84)
    lines.append(f"{clean} of {len(results)} events matched every expected direction.")
    lines.append(
        "A mismatch means either the data is wrong or the expectation is. "
        "The harness cannot tell which - check the chart."
    )
    lines.append("=" * 84)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    default_store = Path(__file__).resolve().parent.parent / "data" / "history.parquet"

    parser = argparse.ArgumentParser(description="Replay the tape over known events")
    parser.add_argument("--store", default=str(default_store))
    parser.add_argument("--events", action="store_true", help="known-event signature check")
    parser.add_argument("--distribution", action="store_true", help="z-score statistics")
    parser.add_argument("--window", type=int, default=60)
    args = parser.parse_args(argv)

    if not (args.events or args.distribution):
        args.events = args.distribution = True

    history = store.read(Path(args.store))
    if history.empty:
        print("store is empty; run the backfill first")
        return 1

    registry = load_registry()
    flags = rolls.load_flags()
    failures = 0

    if args.events:
        results = replay_events(history, registry, flags)
        print(render_events(results))
        failures += sum(1 for r in results if not r.clean)

    if args.distribution:
        stats = distribution(history, registry, flags, args.window)
        print("\n" + "=" * 84)
        print("Z-SCORE DISTRIBUTION  (expect mean~0, sd~1, |z|>3 on 0.5-2% of sessions)")
        print("=" * 84 + "\n")
        print(stats.sort_values("sd").to_string(index=False))

        print("\nBY ASSET CLASS:\n")
        summary = stats.groupby("asset_class").agg(
            n=("n", "sum"), mean=("mean", "mean"),
            sd=("sd", "mean"), pct_gt3=("pct_gt3", "mean"),
        ).round(3)
        print(summary.to_string())

        problems = distribution_flags(stats)
        print("\nDISTRIBUTIONS WORTH A LOOK:\n")
        if problems.empty:
            print("  none")
        else:
            print(problems.to_string(index=False))
            failures += len(problems)

    return 0 if failures == 0 else 0  # diagnostics, never a hard failure


if __name__ == "__main__":
    raise SystemExit(main())
