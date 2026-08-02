"""Contract-roll flags for continuous futures series.

Yahoo's continuous front-month series (CL=F, NG=F, ...) switch contracts at
expiry. The level jumps by the calendar spread even though nothing traded at a
different price, so the computed change is fabricated. Natural gas is the worst
case: the contract terminates three business days before the first of the
delivery month, and the winter-to-spring spread produced roughly nineteen fake
moves over seven years, including a -47.5% print on 2026-01-29 while the UNG
proxy rose 3.6%.

Detection compares each large futures move against an ETF proxy on the same
session (see mb.probe.rollcheck). Two tiers:

  roll       ratio > 3.0. The change is treated as UNDEFINED, exactly as with a
             sign crossing. The level is kept; only the change is suppressed.
  divergent  ratio 1.8-3.0. Reported but still computed. Most of this band is
             settlement-time mismatch rather than a roll: COMEX metals settle at
             1:25pm ET while the ETFs close at 4pm, so volatile days diverge
             legitimately.

The test cannot separate "the series switched contracts" from "the front month
dislocated from the rest of the curve", so it over-flags. The January 2022 gas
squeeze and the July 2025 copper collapse were probably real front-month moves.
That is the correct side to err on: a fabricated move corrupts sixty sessions of
volatility estimates, whereas a missing one leaves an honest gap.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROLL = "roll"
DIVERGENT = "divergent"

ROLL_RATIO = 3.0
DIVERGENT_RATIO = 1.8

DEFAULT_FLAGS_PATH = Path(__file__).resolve().parent.parent / "data" / "roll_flags.csv"

COLUMNS = [
    "instrument_id", "date", "futures_pct", "proxy", "proxy_pct", "ratio", "label"
]


def classify(ratio: float | None) -> str | None:
    """Label a futures/proxy move ratio."""
    if ratio is None or pd.isna(ratio):
        return None
    if ratio > ROLL_RATIO:
        return ROLL
    if ratio >= DIVERGENT_RATIO:
        return DIVERGENT
    return None


def empty_flags() -> pd.DataFrame:
    frame = pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNS})
    frame["date"] = pd.Series(dtype="datetime64[ns]")
    return frame[COLUMNS]


def save_flags(frame: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Persist flags to CSV. Plain text so the file is auditable by hand."""
    path = Path(path) if path else DEFAULT_FLAGS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    out = frame[frame["label"].notna()].copy() if "label" in frame else empty_flags()
    for col in COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[COLUMNS].sort_values(["instrument_id", "date"])
    out.to_csv(path, index=False)
    return path


def load_flags(path: str | Path | None = None) -> pd.DataFrame:
    """Load flags, returning an empty frame when the file does not exist."""
    path = Path(path) if path else DEFAULT_FLAGS_PATH
    if not path.exists():
        return empty_flags()

    frame = pd.read_csv(path)
    if frame.empty:
        return empty_flags()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    for col in COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA
    return frame[COLUMNS]


def flagged_dates(
    flags: pd.DataFrame, instrument_id: str, label: str = ROLL
) -> set[pd.Timestamp]:
    """Dates flagged with `label` for one instrument."""
    if flags.empty:
        return set()
    subset = flags[
        (flags["instrument_id"] == instrument_id) & (flags["label"] == label)
    ]
    return set(pd.to_datetime(subset["date"]).dt.normalize())


def lookup(
    flags: pd.DataFrame, instrument_id: str, date: pd.Timestamp
) -> dict | None:
    """The flag record for one instrument-session, if any."""
    if flags.empty:
        return None
    date = pd.Timestamp(date).normalize()
    subset = flags[
        (flags["instrument_id"] == instrument_id)
        & (pd.to_datetime(flags["date"]).dt.normalize() == date)
    ]
    return None if subset.empty else subset.iloc[0].to_dict()


def mask_changes(
    changes: pd.Series, flags: pd.DataFrame, instrument_id: str
) -> pd.Series:
    """Set changes on roll dates to NaN, leaving every other value untouched.

    The level is never modified; only the derived change is suppressed. A roll
    means we do not know what the instrument did that day, and NaN says so
    honestly where a number would lie.
    """
    dates = flagged_dates(flags, instrument_id, ROLL)
    if not dates:
        return changes
    out = changes.copy()
    hits = out.index.isin(list(dates))
    out[hits] = float("nan")
    return out


def summary(flags: pd.DataFrame) -> pd.DataFrame:
    """Counts by instrument and label."""
    if flags.empty:
        return pd.DataFrame(columns=["instrument_id", "label", "n"])
    return (
        flags.groupby(["instrument_id", "label"], observed=True)
        .size()
        .reset_index(name="n")
        .sort_values(["label", "n"], ascending=[True, False])
    )
