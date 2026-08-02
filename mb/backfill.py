"""Backfill the history store from all configured sources.

Usage:
    export FRED_API_KEY=...        # free key from fred.stlouisfed.org
    python -m mb.backfill --start 2019-01-01
    python -m mb.backfill --check-sources     # one probe per provider, no writes

Design notes:
  * Each instrument is fetched independently and failures are collected, not
    raised. One dead source must not abort a 45-instrument backfill.
  * Values outside the registry's declared bounds are DROPPED and counted, never
    stored. A bad tick that reaches the store poisons every z-score computed
    from that window afterwards.
  * Fallback sources are tried only when the primary yields nothing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import store
from .registry import Instrument, Registry, load_registry
from .sources import adapters
from .sources.http import FetchError

log = logging.getLogger("mb.backfill")

DEFAULT_STORE = Path(__file__).resolve().parent.parent / "data" / "history.parquet"
DEFAULT_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


@dataclass
class BackfillReport:
    fetched: dict[str, int] = field(default_factory=dict)
    dropped_bounds: dict[str, int] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    used_fallback: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = ["", "=" * 66, "BACKFILL REPORT", "=" * 66]
        ok = {k: v for k, v in self.fetched.items() if v > 0}
        lines.append(f"instruments with data : {len(ok)}")
        lines.append(f"instruments failed    : {len(self.failures)}")

        if self.used_fallback:
            lines.append("")
            lines.append("Fell back to a secondary source:")
            for iid in sorted(self.used_fallback):
                lines.append(f"  - {iid}")

        if self.dropped_bounds:
            lines.append("")
            lines.append("Rows dropped for failing bounds checks:")
            for iid, n in sorted(self.dropped_bounds.items(), key=lambda x: -x[1]):
                lines.append(f"  - {iid:<16} {n}")

        if self.failures:
            lines.append("")
            lines.append("FAILURES (these instruments have no data):")
            for iid, msg in sorted(self.failures.items()):
                lines.append(f"  - {iid:<16} {msg[:110]}")

        lines.append("=" * 66)
        return "\n".join(lines)


def _apply_bounds(
    frame: pd.DataFrame, inst: Instrument, report: BackfillReport
) -> pd.DataFrame:
    """Drop level values outside the instrument's declared sanity bounds.

    Only applied to price-like fields; volume has its own scale and is checked
    for non-negativity instead.
    """
    if frame.empty:
        return frame

    price_like = frame["field"] != "volume"
    in_bounds = frame["value"].between(inst.bounds_min, inst.bounds_max)
    volume_ok = (frame["field"] == "volume") & (frame["value"] >= 0)

    keep = (price_like & in_bounds) | volume_ok
    dropped = int((~keep).sum())
    if dropped:
        report.dropped_bounds[inst.id] = report.dropped_bounds.get(inst.id, 0) + dropped
        sample = frame.loc[~keep, "value"].head(3).tolist()
        log.warning(
            "%s: dropped %d row(s) outside bounds [%s, %s], e.g. %s",
            inst.id, dropped, inst.bounds_min, inst.bounds_max, sample,
        )
    return frame[keep]


def _fetch_one(
    inst: Instrument,
    source,
    start: str,
    end: str,
    treasury_cache: dict[str, pd.DataFrame],
    fred_key: str | None,
    raw_dir: Path,
) -> pd.DataFrame:
    """Fetch one instrument from one source, returned in store-record shape."""
    provider = source.provider

    if provider in adapters.TREASURY_TYPES:
        curve = treasury_cache[provider]
        rows = curve[curve["symbol"] == source.symbol]
        if rows.empty:
            available = sorted(curve["symbol"].unique())[:12]
            raise FetchError(
                f"tenor {source.symbol!r} not in Treasury curve; available: {available}"
            )
        # Treasury CSVs are whole calendar years, so the requested window has to
        # be applied here or --start/--end are silently ignored for rates.
        rows = rows[
            (rows["date"] >= pd.Timestamp(start)) & (rows["date"] <= pd.Timestamp(end))
        ]
        return store.to_records(rows["date"], rows["value"], inst.id, provider, "close")

    if provider == "fred":
        raw = adapters.fetch_fred([source.symbol], fred_key or "", start, end, raw_dir)
        return store.to_records(raw["date"], raw["value"], inst.id, provider, "close")

    if provider == "cboe":
        raw = adapters.fetch_cboe(source.symbol, raw_dir)
        raw = raw[(raw["date"] >= start) & (raw["date"] <= end)]
    elif provider == "yahoo":
        raw = adapters.fetch_yahoo(source.symbol, start, end)
    elif provider == "stooq":
        raw = adapters.fetch_stooq(source.symbol, raw_dir)
        raw = raw[(raw["date"] >= start) & (raw["date"] <= end)]
    else:
        raise ValueError(f"no adapter for provider {provider!r}")

    frames = [
        store.to_records(g["date"], g["value"], inst.id, provider, str(fname))
        for fname, g in raw.groupby("field")
    ]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else store.empty_frame()


def backfill(
    registry: Registry,
    start: str,
    end: str,
    store_path: Path,
    raw_dir: Path,
    fred_key: str | None,
) -> BackfillReport:
    report = BackfillReport()

    # Treasury CSVs are per-year and cover every tenor, so fetch once and reuse.
    treasury_cache: dict[str, pd.DataFrame] = {}
    years = list(range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1))
    for provider in adapters.TREASURY_TYPES:
        if not registry.by_provider(provider):
            continue
        try:
            treasury_cache[provider] = adapters.fetch_treasury(provider, years, raw_dir)
            log.info("%s: %d rows across %d years", provider, len(treasury_cache[provider]), len(years))
        except Exception as exc:  # noqa: BLE001
            log.error("%s failed wholesale: %s", provider, exc)
            treasury_cache[provider] = pd.DataFrame(columns=["date", "symbol", "value"])

    all_rows: list[pd.DataFrame] = []

    for inst in registry:
        rows = store.empty_frame()
        errors: list[str] = []

        for position, source in enumerate(inst.sources):
            try:
                rows = _fetch_one(
                    inst, source, start, end, treasury_cache, fred_key, raw_dir
                )
                rows = _apply_bounds(rows, inst, report)
                if not rows.empty:
                    if position > 0:
                        report.used_fallback.append(inst.id)
                        log.info("%s: using fallback %s", inst.id, source.provider)
                    break
                errors.append(f"{source.provider}: returned 0 usable rows")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{source.provider}: {exc}")
                log.warning("%s via %s failed: %s", inst.id, source.provider, exc)

        n_close = int((rows["field"] == "close").sum()) if not rows.empty else 0
        report.fetched[inst.id] = n_close

        if rows.empty:
            report.failures[inst.id] = " | ".join(errors) or "no data"
        else:
            all_rows.append(rows)
            log.info("%s: %d close observations", inst.id, n_close)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        result = store.upsert(store_path, combined)
        log.info(
            "store: +%d inserted, %d updated, %d total rows",
            result["inserted"], result["updated"], result["total"],
        )
    else:
        log.error("no rows fetched from any source; store unchanged")

    return report


def check_sources(registry: Registry, fred_key: str | None, raw_dir: Path) -> int:
    """Probe one instrument per provider to verify endpoints still parse."""
    end = dt.date.today()
    start = end - dt.timedelta(days=30)
    failures = 0

    for provider in sorted(
        {s.provider for inst in registry for s in inst.sources}
    ):
        candidates = [
            (inst, s)
            for inst in registry
            for s in inst.sources
            if s.provider == provider
        ]
        inst, source = candidates[0]
        try:
            cache: dict[str, pd.DataFrame] = {}
            if provider in adapters.TREASURY_TYPES:
                cache[provider] = adapters.fetch_treasury(provider, [end.year], raw_dir)
            rows = _fetch_one(
                inst, source, start.isoformat(), end.isoformat(), cache, fred_key, raw_dir
            )
            # Count close rows only: OHLCV sources return five rows per session,
            # single-value sources one, so raw row counts are not comparable.
            n_close = int((rows["field"] == "close").sum()) if not rows.empty else 0
            status = "OK " if n_close else "EMPTY"
            if not n_close:
                failures += 1
            print(
                f"[{status:5}] {provider:22} via {inst.id:<14} "
                f"{n_close:>4} sessions (last 30d)"
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[FAIL ] {provider:22} via {inst.id:<14} {exc}")

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill the market history store")
    parser.add_argument("--start", default="2019-01-01", help="ISO start date")
    parser.add_argument("--end", default=dt.date.today().isoformat(), help="ISO end date")
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--registry", default=None)
    parser.add_argument("--check-sources", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    registry = load_registry(args.registry)
    fred_key = os.environ.get("FRED_API_KEY")
    raw_dir = Path(args.raw_dir)

    if not fred_key:
        log.warning(
            "FRED_API_KEY not set. Credit spreads and rate fallbacks will fail. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    if args.check_sources:
        return 1 if check_sources(registry, fred_key, raw_dir) else 0

    report = backfill(
        registry, args.start, args.end, Path(args.store), raw_dir, fred_key
    )
    print(report.render())

    history = store.read(Path(args.store))
    print("\nCOVERAGE (lowest observation counts first):")
    print(store.coverage(history).head(15).to_string(index=False))

    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
