"""Tests for mb.calendars and mb.gate."""

from __future__ import annotations

import pandas as pd
import pytest

from mb import calendars, gate, store
from mb.registry import load_registry

# Columbus Day 2024: NYSE trades, the bond market is shut.
BOND_HOLIDAY = pd.Timestamp("2024-10-14")
NORMAL_SESSION = pd.Timestamp("2024-10-15")
HALF_DAY = pd.Timestamp("2024-11-29")       # day after Thanksgiving
NYSE_CLOSED = pd.Timestamp("2024-12-25")


# ------------------------------------------------------------------ calendars

def test_calendars_match_the_observed_history_exactly():
    """The counts that validated the backfill: NYSE 1905, SIFMA 1896."""
    nyse = calendars.sessions("NYSE", "2019-01-01", "2026-07-31")
    sifma = calendars.sessions("SIFMA_US", "2019-01-01", "2026-07-31")
    assert len(nyse) == 1905
    assert len(sifma) == 1896


def test_nyse_open_while_bond_market_closed():
    assert calendars.is_trading_day(BOND_HOLIDAY, "NYSE")
    assert not calendars.is_trading_day(BOND_HOLIDAY, "SIFMA_US")


def test_christmas_closes_nyse():
    assert not calendars.is_trading_day(NYSE_CLOSED, "NYSE")


def test_early_closes_identifies_half_days():
    flagged = calendars.early_closes("2024-01-01", "2024-12-31")
    dates = {d.date().isoformat() for d in flagged}
    assert "2024-11-29" in dates    # day after Thanksgiving
    assert "2024-12-24" in dates    # Christmas Eve
    assert "2024-10-15" not in dates


def test_fx_pseudo_calendar_covers_every_weekday():
    fx = calendars.sessions("FX", "2024-12-23", "2024-12-27")
    assert BOND_HOLIDAY not in fx
    assert pd.Timestamp("2024-12-25") in fx  # FX does not observe US holidays


def test_expected_on_distinguishes_holiday_from_outage():
    """The central distinction: a missing yield on Columbus Day is correct."""
    reg = load_registry()
    assert not calendars.expected_on(reg["ust_10y"], BOND_HOLIDAY)
    assert calendars.expected_on(reg["spx"], BOND_HOLIDAY)
    assert calendars.expected_on(reg["eurusd"], BOND_HOLIDAY)
    assert calendars.expected_on(reg["ust_10y"], NORMAL_SESSION)


def test_staleness_counted_in_sessions_not_calendar_days():
    """Friday to Monday is one session, not three days."""
    friday = pd.Timestamp("2024-10-18")
    monday = pd.Timestamp("2024-10-21")
    assert calendars.sessions_between(friday, monday) == 1


def test_previous_session_skips_weekend():
    assert calendars.previous_session(pd.Timestamp("2024-10-21")) == pd.Timestamp("2024-10-18")


# ----------------------------------------------------------------------- gate

def build_history(session: pd.Timestamp, drop: list[str] | None = None) -> pd.DataFrame:
    """Realistic history honouring each instrument's own calendar."""
    reg = load_registry()
    drop = drop or []
    frames = []
    start = (session - pd.Timedelta(days=200)).date().isoformat()
    end = session.date().isoformat()

    for inst in reg:
        if inst.id in drop:
            continue
        days = calendars.sessions(inst.calendar, start, end)
        days = days[days <= session]
        if inst.quote_unit == "percent":
            values = [4.0 + 0.001 * i for i in range(len(days))]
        elif inst.quote_unit == "fx_rate":
            values = [1.1 + 0.0001 * i for i in range(len(days))]
        else:
            values = [100.0 + 0.05 * i for i in range(len(days))]
        frames.append(store.to_records(days, values, inst.id, "test", "close"))

    return store.validate(pd.concat(frames, ignore_index=True))


def test_gate_passes_on_a_clean_session():
    history = build_history(NORMAL_SESSION)
    result = gate.run_gate(history, session=NORMAL_SESSION)
    assert result.verdict == gate.PASS
    assert result.publishable
    assert not result.missing


def test_gate_blocks_when_nyse_is_closed():
    history = build_history(NORMAL_SESSION)
    result = gate.run_gate(history, session=NYSE_CLOSED)
    assert result.verdict == gate.BLOCKED
    assert not result.is_trading_day
    assert "closed" in result.render().lower()


def test_gate_does_not_penalise_a_bond_holiday():
    """Columbus Day: rates legitimately absent, brief still publishable."""
    history = build_history(BOND_HOLIDAY)
    result = gate.run_gate(history, session=BOND_HOLIDAY)
    assert result.verdict == gate.PASS
    assert "ust_10y" in result.not_expected
    assert not any(f.instrument_id == "ust_10y" for f in result.missing)


def test_gate_blocks_when_a_core_instrument_is_missing():
    history = build_history(NORMAL_SESSION, drop=["spx"])
    result = gate.run_gate(history, session=NORMAL_SESSION)
    assert result.verdict == gate.BLOCKED
    assert not result.publishable
    assert any(f.instrument_id == "spx" for f in result.missing)
    assert result.core_problems


def test_gate_degrades_when_only_confirmations_are_missing():
    history = build_history(NORMAL_SESSION, drop=["xlre", "xlc"])
    result = gate.run_gate(history, session=NORMAL_SESSION)
    assert result.verdict == gate.DEGRADED
    assert result.publishable  # still worth publishing, with a named gap
    assert not result.core_problems


def test_gate_flags_half_session():
    history = build_history(HALF_DAY)
    result = gate.run_gate(history, session=HALF_DAY)
    assert result.is_early_close
    assert "half session" in result.render().lower()


def test_gate_detects_staleness_beyond_tolerance():
    history = build_history(NORMAL_SESSION)
    stale_from = NORMAL_SESSION - pd.Timedelta(days=20)
    history = history[~((history.instrument_id == "vix3m") & (history.date > stale_from))]
    result = gate.run_gate(history, session=NORMAL_SESSION)
    assert any(f.instrument_id == "vix3m" for f in result.stale)
    assert result.verdict == gate.DEGRADED


def test_gate_reviews_but_never_drops_a_large_move():
    """A real 20% energy move must be surfaced, not filtered away."""
    history = build_history(NORMAL_SESSION)
    mask = (history.instrument_id == "xle") & (history.date == NORMAL_SESSION)
    history.loc[mask, "value"] = history.loc[mask, "value"] * 0.75  # -25%
    result = gate.run_gate(history, session=NORMAL_SESSION)

    assert any(f.instrument_id == "xle" for f in result.review)
    assert result.verdict == gate.PASS          # flagged, not blocked
    assert "review" in result.render().lower()


def test_gate_reports_undefined_change_across_zero():
    history = build_history(NORMAL_SESSION)
    prev = calendars.previous_session(NORMAL_SESSION, "CMEGlobex_EnergyAndMetals")
    history.loc[
        (history.instrument_id == "wti") & (history.date == prev), "value"
    ] = -37.63
    result = gate.run_gate(history, session=NORMAL_SESSION)
    assert any(f.instrument_id == "wti" for f in result.undefined)


def test_registry_requires_at_least_one_core_instrument():
    reg = load_registry()
    assert len(reg.core) >= 5
    assert "spx" in {i.id for i in reg.core}
    assert "ust_10y" in {i.id for i in reg.core}


def test_percent_change_is_undefined_across_zero():
    from mb import transforms as tf

    idx = pd.date_range("2020-04-17", periods=3, freq="B")
    levels = pd.Series([18.27, -37.63, 10.01], index=idx)

    unsafe = tf.change_series(levels, "pct", undefined_across_zero=False)
    assert unsafe.iloc[1] < -100          # the nonsense value

    safe = tf.change_series(levels, "pct")
    assert pd.isna(safe.iloc[1])
    assert pd.isna(safe.iloc[2])


def test_sign_crossings_located():
    from mb import transforms as tf

    idx = pd.date_range("2020-04-17", periods=4, freq="B")
    levels = pd.Series([18.27, -37.63, -5.0, 10.01], index=idx)
    crossings = tf.sign_crossings(levels)
    assert len(crossings) == 2


def test_is_early_close_works_on_a_single_date():
    """A one-day window makes the half day its own mode; must widen internally."""
    assert calendars.is_early_close(HALF_DAY)
    assert not calendars.is_early_close(NORMAL_SESSION)


def test_stress_session_list_covers_each_code_path():
    """The curated list must actually exercise every branch, or it is decoration."""
    labels = " ".join(why.lower() for _, why in gate.STRESS_SESSIONS)
    assert "bonds closed" in labels        # calendar divergence
    assert "half session" in labels        # early close
    assert "nyse closed" in labels         # blocked
    assert "negative" in labels            # sign crossing
    assert "roll" in labels                # unverified extremes

    dates = [d for d, _ in gate.STRESS_SESSIONS]
    assert len(dates) == len(set(dates)), "duplicate stress dates"

    # Every bond-holiday entry must genuinely diverge from NYSE.
    for day, why in gate.STRESS_SESSIONS:
        if "bonds closed" in why:
            ts = pd.Timestamp(day)
            assert calendars.is_trading_day(ts, "NYSE"), day
            assert not calendars.is_trading_day(ts, "SIFMA_US"), day
        if "half session" in why:
            assert calendars.is_early_close(pd.Timestamp(day)), day
        if why == "NYSE closed":
            assert not calendars.is_trading_day(pd.Timestamp(day), "NYSE"), day


def test_scan_helper_reports_per_session_counts():
    history = build_history(NORMAL_SESSION)
    days = [NORMAL_SESSION - pd.Timedelta(days=1), NORMAL_SESSION]
    frame = gate._scan(history, load_registry(), days)
    assert list(frame.columns)[:3] == ["session", "verdict", "missing"]
    assert len(frame) == 2
