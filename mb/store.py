"""History store.

Long ("tidy") format, one row per (date, instrument_id, field):

    date           date        session date the value belongs to
    instrument_id  str         registry id
    field          str         close | open | high | low | volume
    value          float64
    provider       str         which source actually supplied it
    ingested_at    timestamp   UTC, when we wrote it

Long format rather than wide because instruments come and go, and the available
fields differ by asset class (a Treasury yield has no volume). Adding an
instrument must never require a schema migration.

Writes are idempotent: re-running a backfill for a date range overwrites those
rows rather than duplicating them. The provenance columns exist so that when a
number looks wrong six months from now you can tell which source produced it.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

KEY_COLUMNS = ["date", "instrument_id", "field"]
COLUMNS = KEY_COLUMNS + ["value", "provider", "ingested_at"]

VALID_FIELDS = {"close", "open", "high", "low", "volume"}

_DTYPES = {
    "instrument_id": "string",
    "field": "string",
    "value": "float64",
    "provider": "string",
}


class StoreError(ValueError):
    """Raised when records violate the store schema."""


def empty_frame() -> pd.DataFrame:
    """An empty frame with the correct dtypes, safe to concat against."""
    frame = pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNS})
    frame = frame.astype(_DTYPES)
    frame["date"] = pd.Series(dtype="datetime64[ns]")
    frame["ingested_at"] = pd.Series(dtype="datetime64[ns, UTC]")
    return frame[COLUMNS]


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce and check a candidate frame against the schema.

    Rejects unknown fields, non-finite values, and duplicate keys within the
    batch. Does not apply per-instrument bounds; that is the ingest layer's job,
    since it needs the registry.
    """
    missing = set(KEY_COLUMNS + ["value"]) - set(frame.columns)
    if missing:
        raise StoreError(f"missing required columns: {sorted(missing)}")

    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["instrument_id"] = out["instrument_id"].astype("string")
    out["field"] = out["field"].astype("string")
    out["value"] = pd.to_numeric(out["value"], errors="coerce").astype("float64")

    if "provider" not in out.columns:
        out["provider"] = pd.Series([pd.NA] * len(out), dtype="string")
    out["provider"] = out["provider"].astype("string")

    if "ingested_at" not in out.columns:
        out["ingested_at"] = pd.Timestamp.now(tz="UTC")
    out["ingested_at"] = pd.to_datetime(out["ingested_at"], utc=True)

    bad_fields = set(out["field"].dropna().unique()) - VALID_FIELDS
    if bad_fields:
        raise StoreError(f"unknown field(s): {sorted(bad_fields)}")

    if out["value"].isna().any():
        n = int(out["value"].isna().sum())
        raise StoreError(
            f"{n} row(s) have null/non-numeric value. Drop them upstream rather "
            "than storing nulls; a missing value must be absent, not NaN."
        )

    import numpy as np

    if not np.isfinite(out["value"].to_numpy()).all():
        raise StoreError("non-finite values (inf/-inf) are not storable")

    dupes = out.duplicated(subset=KEY_COLUMNS, keep=False)
    if dupes.any():
        sample = out.loc[dupes, KEY_COLUMNS].head(5).to_dict("records")
        raise StoreError(f"duplicate keys within batch, e.g. {sample}")

    return out[COLUMNS]


def read(path: str | Path) -> pd.DataFrame:
    """Read the history store, returning an empty frame if it does not exist."""
    path = Path(path)
    if not path.exists():
        return empty_frame()
    frame = pd.read_parquet(path)
    for col, dtype in _DTYPES.items():
        if col in frame.columns:
            frame[col] = frame[col].astype(dtype)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["ingested_at"] = pd.to_datetime(frame["ingested_at"], utc=True)
    return frame[COLUMNS]


def upsert(path: str | Path, new_rows: pd.DataFrame) -> dict[str, int]:
    """Merge rows into the store, last write wins on key collision.

    Returns counts of inserted and updated rows.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    incoming = validate(new_rows)
    existing = read(path)

    if existing.empty:
        merged, inserted, updated = incoming, len(incoming), 0
    else:
        existing_keys = set(map(tuple, existing[KEY_COLUMNS].to_numpy()))
        incoming_keys = set(map(tuple, incoming[KEY_COLUMNS].to_numpy()))
        updated = len(existing_keys & incoming_keys)
        inserted = len(incoming_keys - existing_keys)

        merged = pd.concat([existing, incoming], ignore_index=True)
        merged = merged.drop_duplicates(subset=KEY_COLUMNS, keep="last")

    merged = merged.sort_values(KEY_COLUMNS).reset_index(drop=True)
    merged.to_parquet(path, index=False)
    return {"inserted": inserted, "updated": updated, "total": len(merged)}


def wide(
    frame: pd.DataFrame,
    field: str = "close",
    instrument_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Pivot to a date-indexed wide frame (one column per instrument).

    This is the shape every computation wants; long format is purely for storage.
    """
    if field not in VALID_FIELDS:
        raise StoreError(f"unknown field: {field!r}")

    subset = frame[frame["field"] == field]
    if instrument_ids is not None:
        subset = subset[subset["instrument_id"].isin(instrument_ids)]

    if subset.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))

    pivoted = subset.pivot_table(
        index="date", columns="instrument_id", values="value", aggfunc="last"
    )
    pivoted.columns.name = None
    return pivoted.sort_index()


def coverage(frame: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """Per-instrument coverage summary. Run after a backfill to spot silent gaps.

    A source that quietly returns nothing shows up here as a short history or a
    stale last_date, which is much easier to notice than a missing column.
    """
    subset = frame[frame["field"] == field]
    if subset.empty:
        return pd.DataFrame(
            columns=["instrument_id", "n_obs", "first_date", "last_date", "providers"]
        )

    grouped = subset.groupby("instrument_id", observed=True)
    summary = grouped.agg(
        n_obs=("value", "size"),
        first_date=("date", "min"),
        last_date=("date", "max"),
    ).reset_index()
    providers = (
        grouped["provider"]
        .agg(lambda s: ",".join(sorted(set(s.dropna().astype(str)))))
        .reset_index(name="providers")
    )
    return summary.merge(providers, on="instrument_id").sort_values("n_obs")


def to_records(
    dates,
    values,
    instrument_id: str,
    provider: str,
    field: str = "close",
) -> pd.DataFrame:
    """Build a store-shaped frame from parallel date/value sequences.

    Rows with null values are dropped here rather than stored, per the
    "missing means absent, not NaN" rule.
    """
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(pd.Series(list(dates))).dt.normalize(),
            "value": pd.to_numeric(pd.Series(list(values)), errors="coerce"),
        }
    )
    frame = frame.dropna(subset=["value"])
    frame["instrument_id"] = instrument_id
    frame["field"] = field
    frame["provider"] = provider
    frame["ingested_at"] = pd.Timestamp.now(tz="UTC")
    frame = frame.drop_duplicates(subset=["date"], keep="last")
    return frame[COLUMNS] if not frame.empty else empty_frame()


def last_session(frame: pd.DataFrame) -> dt.date | None:
    """Most recent date present anywhere in the store."""
    if frame.empty:
        return None
    return frame["date"].max().date()
