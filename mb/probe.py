"""Probe candidate data sources without touching the history store.

Two modes:

  Test whether a Yahoo ticker actually has usable history:
      python -m mb.probe --yahoo "CNH=X" "USDCNH=X" "CNY=X" --start 2019-01-01

  Audit what is already stored, including staleness:
      python -m mb.probe --audit

The first mode exists because a Yahoo symbol that resolves but returns almost no
history looks identical to a healthy one in the backfill logs. It only shows up
in the coverage report afterwards, by which point you have waited ten minutes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

from . import rolls, store
from .registry import load_registry
from .sources import adapters

DEFAULT_STORE = Path(__file__).resolve().parent.parent / "data" / "history.parquet"


def probe_yahoo(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch each candidate symbol and summarise what came back."""
    rows = []
    for symbol in symbols:
        try:
            frame = adapters.fetch_yahoo(symbol, start, end)
            close = frame[frame["field"] == "close"]
            if close.empty:
                rows.append(
                    {"symbol": symbol, "sessions": 0, "first": None, "last": None,
                     "gap_days": None, "status": "NO CLOSE DATA"}
                )
                continue

            first, last = close["date"].min(), close["date"].max()
            gap = (pd.Timestamp(end) - last).days

            # Measure against the window that was ASKED FOR. Comparing against
            # the returned range instead scores a single-row response as 100%
            # complete, which is how a dead symbol passes as healthy.
            requested = len(pd.bdate_range(start, end))
            covered = len(close) / requested if requested else 0.0
            starts_late = (first - pd.Timestamp(start)).days

            if len(close) < 30 or covered < 0.5:
                status = f"SPARSE ({len(close)} of ~{requested} requested sessions)"
            elif gap > 5:
                status = f"STALE ({gap}d behind)"
            elif starts_late > 90:
                status = f"PARTIAL (history starts {first.date()})"
            else:
                status = "OK"

            rows.append(
                {
                    "symbol": symbol,
                    "sessions": len(close),
                    "first": first.date(),
                    "last": last.date(),
                    "gap_days": gap,
                    "status": status,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {"symbol": symbol, "sessions": 0, "first": None, "last": None,
                 "gap_days": None, "status": f"FAIL: {str(exc)[:60]}"}
            )
    return pd.DataFrame(rows)


def audit_store(store_path: Path, as_of: dt.date | None = None) -> pd.DataFrame:
    """Coverage plus staleness for everything currently stored.

    Staleness is measured against the registry's stale_tolerance_days, so a
    T+1 series like hy_oas is not reported as stale for being one day behind.
    """
    as_of = as_of or dt.date.today()
    registry = load_registry()
    history = store.read(store_path)

    if history.empty:
        return pd.DataFrame(columns=["instrument_id", "status"])

    cov = store.coverage(history)
    newest = history["date"].max()

    rows = []
    for _, row in cov.iterrows():
        iid = row["instrument_id"]
        inst = registry.get(iid)
        last = pd.Timestamp(row["last_date"])
        behind = len(pd.bdate_range(last, newest)) - 1

        tolerance = inst.stale_tolerance_days if inst else 1
        span = pd.bdate_range(row["first_date"], row["last_date"])
        completeness = row["n_obs"] / len(span) if len(span) else 0.0

        problems = []
        if row["n_obs"] < 30:
            problems.append(f"only {row['n_obs']} obs")
        if behind > tolerance:
            problems.append(f"{behind}d stale")
        if completeness < 0.9:
            problems.append(f"{completeness:.0%} complete")

        rows.append(
            {
                "instrument_id": iid,
                "n_obs": row["n_obs"],
                "first": row["first_date"].date(),
                "last": row["last_date"].date(),
                "days_behind": behind,
                "complete": f"{completeness:.0%}",
                "status": "; ".join(problems) if problems else "ok",
            }
        )

    result = pd.DataFrame(rows)
    return result.sort_values(
        ["status", "n_obs"], key=lambda s: s.eq("ok") if s.name == "status" else s
    )


def quality_report(store_path: Path) -> dict[str, pd.DataFrame]:
    """Three checks the coverage audit cannot make.

    1. Sign crossings on percent-change instruments. A price that crosses zero
       makes percent change meaningless (WTI, April 2020).
    2. Changes exceeding the registry's declared max_abs_change. That field is
       declared but not yet enforced anywhere, so this surfaces what enforcing
       it would catch.
    3. Session counts by asset class. Different markets keep different holiday
       calendars, so a 'cross-asset' day is not always complete.
    """
    from . import transforms as tf

    registry = load_registry()
    history = store.read(store_path)
    wide = store.wide(history, "close")

    sign_rows, breach_rows = [], []

    for inst in registry:
        if inst.id not in wide.columns:
            continue
        levels = wide[inst.id].dropna()
        if len(levels) < 2:
            continue

        if inst.change_unit == "pct" and levels.min() < 0 < levels.max():
            crossings = levels[(levels.shift(1) * levels) < 0]
            sign_rows.append(
                {
                    "instrument_id": inst.id,
                    "min": round(levels.min(), 2),
                    "max": round(levels.max(), 2),
                    "crossings": len(crossings),
                    "first_crossing": crossings.index[0].date() if len(crossings) else None,
                }
            )

        changes = tf.change_series(levels, inst.change_unit)
        breaches = changes[changes.abs() > inst.max_abs_change].dropna()
        for date, value in breaches.items():
            breach_rows.append(
                {
                    "instrument_id": inst.id,
                    "date": date.date(),
                    "change": round(float(value), 1),
                    "unit": inst.change_unit,
                    "threshold": inst.max_abs_change,
                }
            )

    calendar_rows = []
    for asset_class in sorted({i.asset_class for i in registry}):
        ids = [i.id for i in registry.by_asset_class(asset_class) if i.id in wide.columns]
        if not ids:
            continue
        sessions = wide[ids].dropna(how="all")
        calendar_rows.append(
            {
                "asset_class": asset_class,
                "instruments": len(ids),
                "sessions": len(sessions),
                "first": sessions.index.min().date(),
                "last": sessions.index.max().date(),
            }
        )

    complete = wide.dropna(how="any")
    calendar = pd.DataFrame(calendar_rows).sort_values("sessions")

    # Which instruments block a complete cross-asset tape? Greedily drop the
    # worst offender and see how many aligned sessions that buys. This decides
    # whether the classifier can require a full tape or must tolerate gaps.
    missing = wide.isna().sum().sort_values(ascending=False)
    binding_rows = []
    for k in range(0, min(9, len(wide.columns))):
        excluded = list(missing.index[:k])
        kept = [c for c in wide.columns if c not in excluded]
        binding_rows.append(
            {
                "excluded": k,
                "dropped": excluded[-1] if excluded else "(none)",
                "missing_sessions": int(missing.iloc[k - 1]) if k else 0,
                "aligned_sessions": len(wide[kept].dropna(how="any")),
            }
        )

    return {
        "sign_crossings": pd.DataFrame(sign_rows),
        "threshold_breaches": pd.DataFrame(breach_rows).sort_values(
            "change", key=lambda s: s.abs(), ascending=False
        )
        if breach_rows
        else pd.DataFrame(),
        "calendar": calendar,
        "binding": pd.DataFrame(binding_rows),
        "fully_aligned_sessions": len(complete),
        "total_sessions": len(wide),
    }


# ETF proxies for the continuous futures contracts. GLD and SLV are physically
# backed (they hold bullion, never roll), so a divergence between them and our
# futures series is conclusive evidence of a roll artifact. USO, BNO, UNG and
# CPER are futures-based and roll themselves, but on published schedules that
# differ from front-month, so a large divergence is still diagnostic.
COMMODITY_PROXIES = {
    "gold": ("GLD", "physically backed - conclusive"),
    "silver": ("SLV", "physically backed - conclusive"),
    "wti": ("USO", "futures-based - indicative only"),
    "brent": ("BNO", "futures-based - indicative only"),
    "natgas": ("UNG", "futures-based - indicative only"),
    "copper": ("CPER", "futures-based - indicative only"),
}


def rollcheck(
    store_path: Path,
    min_abs_move: float = 8.0,
    start: str = "2019-01-01",
) -> pd.DataFrame:
    """Compare large futures moves against an ETF proxy on the same session.

    A genuine market move shows up in both. A continuous-contract roll shows up
    only in the futures series, because the level jumps by the calendar spread
    while nothing actually traded at a different price.
    """
    from . import transforms as tf

    registry = load_registry()
    history = store.read(store_path)
    wide = store.wide(history, "close")

    rows = []
    for iid, (proxy_symbol, reliability) in COMMODITY_PROXIES.items():
        if iid not in wide.columns:
            continue
        inst = registry[iid]
        futures = tf.change_series(wide[iid].dropna(), inst.change_unit)
        big = futures[futures.abs() >= min_abs_move].dropna()
        if big.empty:
            continue

        end = wide.index.max().date().isoformat()
        try:
            raw = adapters.fetch_yahoo(proxy_symbol, start, end)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {"instrument_id": iid, "date": None, "futures_pct": None,
                 "proxy": proxy_symbol, "proxy_pct": None, "ratio": None,
                 "verdict": f"proxy fetch failed: {str(exc)[:40]}",
                 "reliability": reliability}
            )
            continue

        close = raw[raw["field"] == "close"].set_index("date")["value"].sort_index()
        proxy_changes = tf.change_series(close, "pct")

        for date, fut_move in big.items():
            proxy_move = proxy_changes.get(date)
            if proxy_move is None or pd.isna(proxy_move):
                verdict, ratio = "no proxy data", None
            else:
                ratio = abs(fut_move) / max(abs(proxy_move), 0.01)
                label = rolls.classify(ratio)
                verdict = {
                    rolls.ROLL: "LIKELY ROLL ARTIFACT",
                    rolls.DIVERGENT: "divergent (likely settlement-time mismatch)",
                }.get(label, "real move")
            rows.append(
                {
                    "instrument_id": iid,
                    "date": date.date(),
                    "futures_pct": round(float(fut_move), 1),
                    "proxy": proxy_symbol,
                    "proxy_pct": None if proxy_move is None or pd.isna(proxy_move)
                    else round(float(proxy_move), 1),
                    "ratio": None if ratio is None else round(ratio, 1),
                    "verdict": verdict,
                    "label": rolls.classify(ratio),
                    "reliability": reliability,
                }
            )

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe data sources and audit the store")
    parser.add_argument("--yahoo", nargs="+", help="candidate Yahoo symbols to test")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=dt.date.today().isoformat())
    parser.add_argument("--audit", action="store_true", help="audit the stored history")
    parser.add_argument("--quality", action="store_true", help="data quality checks")
    parser.add_argument(
        "--rollcheck", action="store_true",
        help="cross-check large commodity moves against ETF proxies",
    )
    parser.add_argument(
        "--min-move", type=float, default=8.0,
        help="minimum absolute percent move to cross-check (default 8)",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="write detected flags to data/roll_flags.csv for the gate to consume",
    )
    parser.add_argument("--provider", metavar="ID", help="show sources used for an instrument")
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    args = parser.parse_args(argv)

    if not any([args.yahoo, args.audit, args.quality, args.provider, args.rollcheck]):
        parser.error(
            "pass --yahoo SYMBOL..., --audit, --quality, --rollcheck, or --provider ID"
        )

    if args.yahoo:
        print(f"\nProbing {len(args.yahoo)} Yahoo symbol(s) from {args.start}:\n")
        print(probe_yahoo(args.yahoo, args.start, args.end).to_string(index=False))

    if args.provider:
        history = store.read(Path(args.store))
        subset = history[history["instrument_id"] == args.provider]
        if subset.empty:
            print(f"\n{args.provider!r} not found in the store")
        else:
            print(f"\nSources actually used for {args.provider!r}:\n")
            print(subset.groupby("provider").size().to_string())

    if args.audit:
        print("\nSTORE AUDIT (problems first):\n")
        result = audit_store(Path(args.store))
        if result.empty:
            print("  store is empty")
        else:
            print(result.to_string(index=False))
            bad = result[result["status"] != "ok"]
            print(f"\n{len(bad)} of {len(result)} instruments need attention.")
            print(
                "\nNote: 'complete' compares against weekdays, not a trading "
                "calendar, so ~96% is normal for US markets (holidays) and FX "
                "can exceed 100%. It cannot detect a real gap yet."
            )

    if args.quality:
        report = quality_report(Path(args.store))

        print("\nSESSIONS BY ASSET CLASS:\n")
        print(report["calendar"].to_string(index=False))
        print(
            f"\nSessions where EVERY instrument has data: "
            f"{report['fully_aligned_sessions']} of {report['total_sessions']}"
        )

        print("\nWHAT BLOCKS A COMPLETE TAPE (drop worst offender, recount):\n")
        print(report["binding"].to_string(index=False))

        print("\nSIGN CROSSINGS (percent change is meaningless across zero):\n")
        signs = report["sign_crossings"]
        print("  none" if signs.empty else signs.to_string(index=False))

        breaches = report["threshold_breaches"]
        print(f"\nCHANGES EXCEEDING max_abs_change ({len(breaches)} total):\n")
        if breaches.empty:
            print("  none")
        else:
            print(breaches.head(15).to_string(index=False))

    if args.rollcheck:
        print(
            f"\nROLL CHECK: commodity moves >= {args.min_move}% vs ETF proxies"
            "\n(fetching proxies, this takes a moment)\n"
        )
        result = rollcheck(Path(args.store), args.min_move)
        if result.empty:
            print("  no moves above the threshold")
        else:
            print(result.to_string(index=False))
            n_roll = int((result["label"] == rolls.ROLL).sum())
            n_div = int((result["label"] == rolls.DIVERGENT).sum())
            print(
                f"\n{n_roll} roll artifact(s), {n_div} divergent. GLD and SLV are "
                "physically backed so their verdicts are conclusive; the "
                "futures-based proxies are indicative only."
            )
            if args.save:
                path = rolls.save_flags(result)
                print(f"\nWrote {n_roll + n_div} flag(s) to {path}")
                print(rolls.summary(rolls.load_flags(path)).to_string(index=False))
            else:
                print("\nRe-run with --save to write these flags for the gate.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
