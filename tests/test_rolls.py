"""Tests for mb.rolls and its integration with the validation gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mb import gate, rolls, store
from mb.registry import load_registry


def make_flags(**overrides) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "instrument_id": ["natgas", "silver"],
            "date": pd.to_datetime(["2026-01-29", "2026-01-26"]),
            "futures_pct": [-47.5, 14.0],
            "proxy": ["UNG", "SLV"],
            "proxy_pct": [3.6, 5.8],
            "ratio": [13.0, 2.4],
            "label": [rolls.ROLL, rolls.DIVERGENT],
        }
    )
    for k, v in overrides.items():
        base[k] = v
    return base


# ------------------------------------------------------------- classification

def test_classify_uses_two_tiers():
    assert rolls.classify(13.0) == rolls.ROLL
    assert rolls.classify(3.5) == rolls.ROLL
    assert rolls.classify(2.4) == rolls.DIVERGENT
    assert rolls.classify(1.8) == rolls.DIVERGENT
    assert rolls.classify(1.1) is None
    assert rolls.classify(None) is None
    assert rolls.classify(np.nan) is None


def test_roll_threshold_is_strictly_above_three():
    """Ratio 3.0 exactly is divergent, not a roll."""
    assert rolls.classify(3.0) == rolls.DIVERGENT
    assert rolls.classify(3.01) == rolls.ROLL


# ---------------------------------------------------------------- persistence

def test_flags_round_trip_through_csv(tmp_path):
    path = tmp_path / "flags.csv"
    rolls.save_flags(make_flags(), path)
    loaded = rolls.load_flags(path)

    assert len(loaded) == 2
    assert set(loaded["label"]) == {rolls.ROLL, rolls.DIVERGENT}
    assert loaded["date"].dtype.kind == "M"


def test_save_drops_unlabelled_rows(tmp_path):
    """Only flagged moves are persisted; 'real move' rows are not."""
    frame = make_flags()
    frame.loc[len(frame)] = ["wti", pd.Timestamp("2020-03-09"), -24.6, "USO", -25.3, 1.0, None]
    path = rolls.save_flags(frame, tmp_path / "flags.csv")
    assert len(rolls.load_flags(path)) == 2


def test_missing_flag_file_is_not_an_error(tmp_path):
    loaded = rolls.load_flags(tmp_path / "absent.csv")
    assert loaded.empty
    assert list(loaded.columns) == rolls.COLUMNS


def test_flagged_dates_filters_by_label():
    flags = make_flags()
    assert rolls.flagged_dates(flags, "natgas", rolls.ROLL) == {
        pd.Timestamp("2026-01-29")
    }
    assert rolls.flagged_dates(flags, "natgas", rolls.DIVERGENT) == set()
    assert rolls.flagged_dates(flags, "silver", rolls.DIVERGENT) == {
        pd.Timestamp("2026-01-26")
    }


def test_lookup_returns_the_record():
    flags = make_flags()
    hit = rolls.lookup(flags, "natgas", pd.Timestamp("2026-01-29"))
    assert hit["ratio"] == 13.0
    assert hit["proxy"] == "UNG"
    assert rolls.lookup(flags, "natgas", pd.Timestamp("2026-01-28")) is None


# -------------------------------------------------------------------- masking

def test_mask_sets_roll_dates_to_nan_and_leaves_the_rest():
    idx = pd.to_datetime(["2026-01-28", "2026-01-29", "2026-01-30"])
    changes = pd.Series([2.0, -47.5, 11.1], index=idx)

    masked = rolls.mask_changes(changes, make_flags(), "natgas")
    assert masked.iloc[0] == pytest.approx(2.0)
    assert pd.isna(masked.iloc[1])          # the fabricated move
    assert masked.iloc[2] == pytest.approx(11.1)


def test_mask_does_not_touch_divergent_dates():
    """Divergent is informational; the change is still computed."""
    idx = pd.to_datetime(["2026-01-26"])
    changes = pd.Series([14.0], index=idx)
    masked = rolls.mask_changes(changes, make_flags(), "silver")
    assert masked.iloc[0] == pytest.approx(14.0)


def test_mask_is_a_copy_not_in_place():
    idx = pd.to_datetime(["2026-01-29"])
    changes = pd.Series([-47.5], index=idx)
    rolls.mask_changes(changes, make_flags(), "natgas")
    assert changes.iloc[0] == pytest.approx(-47.5)


def test_mask_with_no_flags_is_a_no_op():
    idx = pd.to_datetime(["2026-01-29"])
    changes = pd.Series([-47.5], index=idx)
    out = rolls.mask_changes(changes, rolls.empty_flags(), "natgas")
    assert out.equals(changes)


def test_summary_counts_by_instrument_and_label():
    counts = rolls.summary(make_flags())
    assert set(counts["label"]) == {rolls.ROLL, rolls.DIVERGENT}
    assert int(counts["n"].sum()) == 2


# -------------------------------------------------------------- gate wiring

def _history_with(session: pd.Timestamp, instrument: str, pct: float) -> pd.DataFrame:
    """Realistic history with one engineered move on `session`."""
    import sys
    sys.path.insert(0, "tests")
    from test_gate import build_history

    history = build_history(session)
    prior = history[
        (history.instrument_id == instrument) & (history.date < session)
    ]["value"]
    base = float(prior.iloc[-1])
    history.loc[
        (history.instrument_id == instrument) & (history.date == session), "value"
    ] = base * (1 + pct / 100)
    return history


def test_gate_reports_roll_as_undefined_not_as_a_move():
    session = pd.Timestamp("2024-10-15")
    history = _history_with(session, "natgas", -47.5)

    flags = make_flags(
        instrument_id=["natgas", "silver"],
        date=[session, pd.Timestamp("2026-01-26")],
    )
    result = gate.run_gate(history, session=session, flags=flags)

    assert any(f.instrument_id == "natgas" for f in result.undefined)
    assert not any(f.instrument_id == "natgas" for f in result.review)
    assert "contract roll" in result.render().lower()


def test_gate_reports_divergent_as_review_and_still_computes():
    session = pd.Timestamp("2024-10-15")
    history = _history_with(session, "silver", 14.0)

    flags = make_flags(
        instrument_id=["natgas", "silver"],
        date=[pd.Timestamp("2026-01-29"), session],
    )
    result = gate.run_gate(history, session=session, flags=flags)

    assert any(f.instrument_id == "silver" for f in result.review)
    assert not any(f.instrument_id == "silver" for f in result.undefined)


def test_gate_without_flag_file_still_runs():
    import sys
    sys.path.insert(0, "tests")
    from test_gate import build_history, NORMAL_SESSION

    history = build_history(NORMAL_SESSION)
    result = gate.run_gate(history, session=NORMAL_SESSION, flags=rolls.empty_flags())
    assert result.verdict == gate.PASS


def test_masked_roll_does_not_pollute_the_volatility_window():
    """The point of the whole exercise: a fake move must not inflate sigma."""
    from mb import transforms as tf

    idx = pd.bdate_range("2024-01-01", periods=120)
    values = np.full(120, 3.0)
    values[60] = 1.5          # a -50% "move" that never happened
    values[61] = 3.0
    levels = pd.Series(values, index=idx)

    flags = pd.DataFrame(
        {
            "instrument_id": ["natgas"] * 2,
            "date": [idx[60], idx[61]],
            "futures_pct": [-50.0, 100.0],
            "proxy": ["UNG"] * 2,
            "proxy_pct": [0.5, 0.5],
            "ratio": [100.0, 200.0],
            "label": [rolls.ROLL, rolls.ROLL],
        }
    )

    raw = tf.change_series(levels, "pct")
    masked = rolls.mask_changes(raw, flags, "natgas")

    assert raw.abs().max() > 49          # the fabrication is present
    assert masked.abs().max() < 1e-9     # and removed
    assert masked.notna().sum() == raw.notna().sum() - 2
