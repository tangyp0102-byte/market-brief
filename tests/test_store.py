"""Tests for mb.store: schema enforcement and idempotent upserts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mb import store


def rows(**overrides) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "instrument_id": ["spx", "spx"],
            "field": ["close", "close"],
            "value": [4700.0, 4750.0],
            "provider": ["yahoo", "yahoo"],
        }
    )
    for key, val in overrides.items():
        base[key] = val
    return base


# ------------------------------------------------------------------ validation

def test_validate_accepts_well_formed_rows():
    out = store.validate(rows())
    assert list(out.columns) == store.COLUMNS
    assert out["ingested_at"].dt.tz is not None


def test_validate_rejects_unknown_field():
    bad = rows(field=["settle", "close"])
    with pytest.raises(store.StoreError, match="unknown field"):
        store.validate(bad)


def test_validate_rejects_null_values():
    bad = rows()
    bad.loc[0, "value"] = None
    with pytest.raises(store.StoreError, match="null/non-numeric"):
        store.validate(bad)


def test_validate_rejects_infinities():
    bad = rows()
    bad.loc[0, "value"] = np.inf
    with pytest.raises(store.StoreError, match="non-finite"):
        store.validate(bad)


def test_validate_rejects_duplicate_keys_within_batch():
    bad = pd.concat([rows(), rows()], ignore_index=True)
    with pytest.raises(store.StoreError, match="duplicate keys"):
        store.validate(bad)


def test_validate_rejects_missing_columns():
    with pytest.raises(store.StoreError, match="missing required columns"):
        store.validate(pd.DataFrame({"date": [], "value": []}))


# ---------------------------------------------------------------------- upsert

def test_upsert_creates_store_and_reports_counts(tmp_path):
    path = tmp_path / "history.parquet"
    result = store.upsert(path, rows())
    assert result == {"inserted": 2, "updated": 0, "total": 2}
    assert path.exists()


def test_upsert_is_idempotent(tmp_path):
    """Re-running a backfill must not duplicate rows."""
    path = tmp_path / "history.parquet"
    store.upsert(path, rows())
    second = store.upsert(path, rows())
    assert second["inserted"] == 0
    assert second["updated"] == 2
    assert second["total"] == 2
    assert len(store.read(path)) == 2


def test_upsert_last_write_wins_on_correction(tmp_path):
    """A restated value must overwrite, not sit alongside, the original."""
    path = tmp_path / "history.parquet"
    store.upsert(path, rows())
    corrected = rows(value=[4701.5, 4750.0])
    store.upsert(path, corrected)

    history = store.read(path)
    assert len(history) == 2
    got = history[history["date"] == pd.Timestamp("2024-01-02")]["value"].iloc[0]
    assert got == pytest.approx(4701.5)


def test_upsert_appends_new_dates(tmp_path):
    path = tmp_path / "history.parquet"
    store.upsert(path, rows())
    later = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-04"]),
            "instrument_id": ["spx"],
            "field": ["close"],
            "value": [4800.0],
            "provider": ["yahoo"],
        }
    )
    result = store.upsert(path, later)
    assert result["inserted"] == 1
    assert result["total"] == 3


def test_read_missing_file_returns_typed_empty_frame(tmp_path):
    frame = store.read(tmp_path / "absent.parquet")
    assert frame.empty
    assert list(frame.columns) == store.COLUMNS


def test_stored_rows_are_sorted_by_key(tmp_path):
    path = tmp_path / "history.parquet"
    unsorted = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05", "2024-01-02"]),
            "instrument_id": ["vix", "spx"],
            "field": ["close", "close"],
            "value": [13.5, 4700.0],
            "provider": ["yahoo", "yahoo"],
        }
    )
    store.upsert(path, unsorted)
    history = store.read(path)
    assert history["date"].is_monotonic_increasing


# ------------------------------------------------------------------ projection

def test_wide_pivot_produces_one_column_per_instrument():
    frame = store.validate(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
                "instrument_id": ["spx", "vix", "spx"],
                "field": ["close", "close", "close"],
                "value": [4700.0, 13.2, 4750.0],
                "provider": ["yahoo"] * 3,
            }
        )
    )
    wide = store.wide(frame)
    assert sorted(wide.columns) == ["spx", "vix"]
    assert wide.loc[pd.Timestamp("2024-01-02"), "vix"] == pytest.approx(13.2)
    assert pd.isna(wide.loc[pd.Timestamp("2024-01-03"), "vix"])


def test_wide_separates_fields():
    frame = store.validate(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                "instrument_id": ["spy", "spy"],
                "field": ["close", "volume"],
                "value": [470.0, 8.5e7],
                "provider": ["yahoo", "yahoo"],
            }
        )
    )
    assert store.wide(frame, "close").loc[pd.Timestamp("2024-01-02"), "spy"] == 470.0
    assert store.wide(frame, "volume").loc[pd.Timestamp("2024-01-02"), "spy"] == 8.5e7


def test_wide_on_empty_selection_returns_empty_frame():
    assert store.wide(store.empty_frame()).empty


# -------------------------------------------------------------------- coverage

def test_coverage_surfaces_a_short_history():
    """A source that silently returns almost nothing must be visible."""
    frame = store.validate(
        pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-02"]
                ),
                "instrument_id": ["spx", "spx", "spx", "usdcnh"],
                "field": ["close"] * 4,
                "value": [4700.0, 4750.0, 4760.0, 7.25],
                "provider": ["yahoo"] * 4,
            }
        )
    )
    cov = store.coverage(frame)
    assert cov.iloc[0]["instrument_id"] == "usdcnh"  # sorted worst-first
    assert cov.iloc[0]["n_obs"] == 1
    assert cov.set_index("instrument_id").loc["spx", "n_obs"] == 3


# ------------------------------------------------------------------- to_records

def test_to_records_drops_nulls_rather_than_storing_them():
    frame = store.to_records(
        ["2024-01-02", "2024-01-03", "2024-01-04"],
        [4700.0, None, 4760.0],
        "spx",
        "yahoo",
    )
    assert len(frame) == 2
    assert frame["value"].notna().all()


def test_to_records_deduplicates_repeated_dates():
    frame = store.to_records(
        ["2024-01-02", "2024-01-02"], [4700.0, 4705.0], "spx", "yahoo"
    )
    assert len(frame) == 1
    assert frame["value"].iloc[0] == pytest.approx(4705.0)  # last wins


def test_to_records_all_null_returns_empty_typed_frame():
    frame = store.to_records(["2024-01-02"], [None], "spx", "yahoo")
    assert frame.empty
    assert list(frame.columns) == store.COLUMNS


def test_round_trip_through_parquet_preserves_values(tmp_path):
    path = tmp_path / "history.parquet"
    original = store.to_records(
        pd.date_range("2024-01-01", periods=100, freq="B"),
        np.linspace(4000, 4500, 100),
        "spx",
        "yahoo",
    )
    store.upsert(path, original)
    reloaded = store.read(path)
    assert len(reloaded) == 100
    np.testing.assert_allclose(
        reloaded.sort_values("date")["value"].to_numpy(),
        original.sort_values("date")["value"].to_numpy(),
    )
