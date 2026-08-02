"""The daily run.

    fetch -> validate -> tape -> classify -> write -> publish

Intended for 18:00 US/Eastern. Not 16:15: the Treasury par yield curve is
published in the early evening, and a 16:15 run would systematically miss the
authoritative rates data for the session it describes.

Two things this handles that the manual sequence did not:

  * TRADING-DAY GUARD. On a non-session the run exits cleanly without touching
    the store, so a holiday cannot be mistaken for a data outage.

  * ROLL FLAGS. Contract rolls were detected by a command run by hand, which is
    the kind of step that gets forgotten until a fabricated -47% move has been
    sitting in the volatility window for a month. The refresh now happens
    automatically, but only when a commodity actually moved enough for a roll to
    be plausible - fetching six ETF proxies every session to check for an event
    that occurs monthly would be wasteful.

Every stage is idempotent, so a re-run after a failure is safe.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import backfill, brief, calendars, classify, page, probe, rolls, store, tape
from .registry import load_registry

log = logging.getLogger("mb.daily")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORE = ROOT / "data" / "history.parquet"
DEFAULT_SITE = ROOT / "site"
DEFAULT_BRIEFS = ROOT / "data" / "briefs"

# How far back to re-fetch. Long enough to repair a missed run or absorb a
# vendor restatement, short enough to stay quick.
LOOKBACK_DAYS = 12

# Below this many stored sessions the incremental fetch is not enough: the
# z-score needs 60 sessions and the volatility floor 250, so a thin store
# produces a page of NaNs rather than an obvious failure. This is the state a
# runner lands in when its cache has been evicted, so the pipeline rebuilds
# instead of publishing something empty.
MIN_SESSIONS = 320
FULL_BACKFILL_START = "2019-01-01"

# A commodity move at least this large makes a contract roll plausible.
ROLL_TRIGGER_PCT = 8.0


@dataclass
class DailyResult:
    session: dt.date | None
    ran: bool
    verdict: str = ""
    regime: str | None = None
    rollcheck_ran: bool = False
    page_path: Path | None = None
    json_path: Path | None = None
    messages: list[str] = field(default_factory=list)

    def render(self) -> str:
        if not self.ran:
            return "\n".join(["DAILY RUN SKIPPED"] + [f"  {m}" for m in self.messages])
        lines = [
            f"DAILY RUN  {self.session}",
            f"  gate      : {self.verdict}",
            f"  regime    : {self.regime or 'none'}",
            f"  rollcheck : {'refreshed' if self.rollcheck_ran else 'not needed'}",
        ]
        if self.page_path:
            lines.append(f"  page      : {self.page_path}")
        if self.json_path:
            lines.append(f"  archive   : {self.json_path}")
        lines += [f"  {m}" for m in self.messages]
        return "\n".join(lines)


def commodity_moves(history: pd.DataFrame, session: pd.Timestamp) -> dict[str, float]:
    """Percent moves of the commodity contracts for one session."""
    from . import transforms as tf

    registry = load_registry()
    wide = store.wide(history[history["date"] <= session], "close")
    out = {}
    for inst in registry.by_asset_class("commodity"):
        if inst.id not in wide.columns:
            continue
        levels = wide[inst.id].dropna()
        if session not in levels.index or len(levels) < 2:
            continue
        change = tf.change_series(levels, inst.change_unit).get(session)
        if change is not None and not pd.isna(change):
            out[inst.id] = float(change)
    return out


def needs_rollcheck(history: pd.DataFrame, session: pd.Timestamp) -> bool:
    """Whether any commodity moved enough for a roll to be worth testing."""
    return any(
        abs(v) >= ROLL_TRIGGER_PCT for v in commodity_moves(history, session).values()
    )


def run(
    store_path: Path = DEFAULT_STORE,
    site_dir: Path = DEFAULT_SITE,
    briefs_dir: Path = DEFAULT_BRIEFS,
    session: dt.date | None = None,
    force: bool = False,
    skip_fetch: bool = False,
    call_model=None,
) -> DailyResult:
    registry = load_registry()
    today = session or dt.date.today()
    stamp = pd.Timestamp(today).normalize()

    if not force and not calendars.is_trading_day(stamp):
        return DailyResult(
            session=today, ran=False,
            messages=[f"{today} is not an NYSE session. Nothing to do."],
        )

    messages: list[str] = []

    if not skip_fetch:
        existing = store.read(store_path)
        sessions_held = existing["date"].nunique() if not existing.empty else 0
        if sessions_held < MIN_SESSIONS:
            start = FULL_BACKFILL_START
            messages.append(
                f"Only {sessions_held} sessions stored; rebuilding the full "
                "history rather than publishing an incomplete one."
            )
        else:
            # Reach back to whichever is earlier: the standard lookback, or a
            # few days before the last stored session. A fixed window would
            # leave a hole whenever a run is missed for longer than it - which
            # is exactly what happens after a cache eviction restores a store
            # that is weeks behind.
            last_held = existing["date"].max()
            start_ts = min(
                stamp - pd.Timedelta(days=LOOKBACK_DAYS),
                last_held - pd.Timedelta(days=3),
            )
            start = start_ts.date().isoformat()
            gap = (stamp - last_held).days
            if gap > LOOKBACK_DAYS:
                messages.append(
                    f"Store was {gap} days behind; fetching from {start} to close the gap."
                )
        report = backfill.backfill(
            registry, start, today.isoformat(), store_path,
            ROOT / "data" / "raw", os.environ.get("FRED_API_KEY"),
        )
        if report.failures:
            messages.append(
                f"{len(report.failures)} instrument(s) failed to fetch: "
                + ", ".join(sorted(report.failures)[:5])
            )

    history = store.read(store_path)
    if history.empty:
        return DailyResult(
            session=today, ran=False,
            messages=["Store is empty. Run the backfill first."],
        )

    latest = history["date"].max()
    if latest.date() < today and not force:
        messages.append(
            f"No data yet for {today}; using the latest stored session {latest.date()}."
        )
        stamp = latest

    # Refresh roll flags only when a commodity move makes a roll plausible.
    rollcheck_ran = False
    if needs_rollcheck(history, stamp):
        try:
            detected = probe.rollcheck(store_path, ROLL_TRIGGER_PCT)
            rolls.save_flags(detected)
            rollcheck_ran = True
            n = int((detected.get("label") == rolls.ROLL).sum()) if not detected.empty else 0
            messages.append(f"Roll flags refreshed: {n} artefact(s) across the history.")
        except Exception as exc:  # noqa: BLE001
            messages.append(f"Roll check failed, keeping existing flags: {exc}")

    flags = rolls.load_flags()

    built = tape.build_tape(history, registry, stamp, flags)
    result = classify.classify(history, registry, stamp, flags)
    sheet = brief.build_factsheet(history, registry, stamp, flags)
    written = brief.generate(sheet, call_model=call_model)

    if written.error:
        messages.append(written.error)
    if written.unverified_numbers:
        messages.append(
            "Narrative withheld: unverified figures "
            + ", ".join(str(n) for n in written.unverified_numbers)
        )

    briefs_dir = Path(briefs_dir)
    briefs_dir.mkdir(parents=True, exist_ok=True)
    json_path = briefs_dir / f"{built.session}.json"
    json_path.write_text(json.dumps({
        "factsheet": sheet,
        "headline": written.headline,
        "body": written.body,
        "generated": written.generated,
        "unverified_numbers": written.unverified_numbers,
    }, indent=2))

    archive = sorted(
        (p.stem for p in briefs_dir.glob("*.json") if p.stem != str(built.session)),
        reverse=True,
    )

    site_dir = Path(site_dir)
    dated = page.write_page(
        site_dir / f"{built.session}.html", built, result, written, registry, archive
    )
    page.write_page(
        site_dir / "index.html", built, result, written, registry, archive
    )

    return DailyResult(
        session=built.session, ran=True, verdict=built.verdict,
        regime=result.regime, rollcheck_ran=rollcheck_ran,
        page_path=dated, json_path=json_path, messages=messages,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the daily pipeline")
    parser.add_argument("--session", default=None, help="ISO date; default = today")
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    parser.add_argument("--site", default=str(DEFAULT_SITE))
    parser.add_argument("--briefs", default=str(DEFAULT_BRIEFS))
    parser.add_argument("--skip-fetch", action="store_true",
                        help="use the stored history without refetching")
    parser.add_argument("--force", action="store_true",
                        help="run even on a non-trading day")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    session = dt.date.fromisoformat(args.session) if args.session else None
    result = run(
        Path(args.store), Path(args.site), Path(args.briefs),
        session, args.force, args.skip_fetch,
    )
    print(result.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
