"""Tests for mb.replay."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "tests")

from mb import replay, rolls, store
from mb.registry import load_registry


@pytest.fixture(scope="module")
def history():
    from test_pipeline_smoke import synthetic_history
    return store.validate(synthetic_history())


# ------------------------------------------------------------ event catalogue

def test_every_event_names_only_real_instruments():
    """An expectation on a non-existent id would silently never be checked."""
    valid = set(load_registry().ids)
    for event in replay.EVENTS:
        for iid in event.expect:
            assert iid in valid, f"{event.date}: unknown instrument {iid!r}"


def test_events_use_direction_only():
    for event in replay.EVENTS:
        for iid, direction in event.expect.items():
            assert direction in (replay.UP, replay.DOWN), (event.date, iid)


def test_event_dates_are_valid_and_unique():
    dates = [e.date for e in replay.EVENTS]
    assert len(dates) == len(set(dates))
    for d in dates:
        pd.Timestamp(d)


def test_lower_confidence_events_are_labelled():
    """Events near the edge of reliable knowledge must say so."""
    conf = {e.date: e.confidence for e in replay.EVENTS}
    assert conf["2025-04-03"] == "medium"
    assert conf["2020-03-09"] == "high"
    assert all(e.confidence in ("high", "medium") for e in replay.EVENTS)


# ------------------------------------------------------------- event checking

def _engineer(history, session, moves: dict[str, float]) -> pd.DataFrame:
    """Force specific percent moves on a session."""
    out = history.copy()
    for iid, pct in moves.items():
        prior = out[(out.instrument_id == iid) & (out.date < session)]["value"]
        base = float(prior.iloc[-1])
        out.loc[
            (out.instrument_id == iid) & (out.date == session), "value"
        ] = base * (1 + pct / 100)
    return out


def test_matching_directions_report_clean(history):
    session = history["date"].max()
    event = replay.Event(
        session.date().isoformat(), "engineered",
        {"spx": replay.DOWN, "vix": replay.UP},
    )
    engineered = _engineer(history, session, {"spx": -3.0, "vix": 20.0})
    result = replay.check_event(engineered, event, flags=rolls.empty_flags())
    assert result.total == 2
    assert result.clean


def test_wrong_direction_is_flagged_not_hidden(history):
    session = history["date"].max()
    event = replay.Event(
        session.date().isoformat(), "engineered",
        {"spx": replay.DOWN, "vix": replay.UP},
    )
    engineered = _engineer(history, session, {"spx": +3.0, "vix": 20.0})
    result = replay.check_event(engineered, event, flags=rolls.empty_flags())
    assert not result.clean
    assert result.matched == 1
    bad = [c for c in result.checks if not c["match"]]
    assert bad[0]["instrument_id"] == "spx"
    assert bad[0]["expected"] == "down" and bad[0]["observed"] == "up"


def test_missing_instrument_reported_as_unavailable(history):
    session = history["date"].max()
    trimmed = history[history.instrument_id != "vix"]
    event = replay.Event(
        session.date().isoformat(), "engineered", {"vix": replay.UP}
    )
    result = replay.check_event(trimmed, event, flags=rolls.empty_flags())
    assert "vix" in result.unavailable
    assert result.total == 0


def test_render_states_that_expectations_are_hypotheses(history):
    session = history["date"].max()
    event = replay.Event(session.date().isoformat(), "x", {"spx": replay.DOWN})
    text = replay.render_events(
        [replay.check_event(history, event, flags=rolls.empty_flags())]
    ).lower()
    assert "hypothesis" in text
    assert "cannot tell which" in text


# -------------------------------------------------------------- distribution

def test_distribution_stats_are_well_behaved(history):
    stats = replay.distribution(history, flags=rolls.empty_flags())
    assert len(stats) > 30
    # Synthetic data is a clean random walk, so these should be tight.
    assert stats["sd"].between(0.7, 1.4).mean() > 0.9
    assert stats["mean"].abs().max() < 0.5


def test_distribution_flags_a_broken_scale():
    stats = pd.DataFrame([
        {"instrument_id": "ok", "asset_class": "fx", "n": 100, "mean": 0.01,
         "sd": 1.02, "pct_gt2": 4.0, "pct_gt3": 0.8, "max_abs": 4.1},
        {"instrument_id": "bad_sd", "asset_class": "fx", "n": 100, "mean": 0.0,
         "sd": 3.10, "pct_gt2": 30.0, "pct_gt3": 12.0, "max_abs": 9.0},
        {"instrument_id": "drifting", "asset_class": "fx", "n": 100, "mean": 0.8,
         "sd": 1.0, "pct_gt2": 5.0, "pct_gt3": 1.0, "max_abs": 4.0},
    ])
    problems = replay.distribution_flags(stats)
    flagged = set(problems["instrument_id"])
    assert "bad_sd" in flagged
    assert "drifting" in flagged
    assert "ok" not in flagged


def test_distribution_flags_quiet_vol_artefact():
    stats = pd.DataFrame([
        {"instrument_id": "usdjpy", "asset_class": "fx", "n": 1900, "mean": 0.0,
         "sd": 1.05, "pct_gt2": 5.0, "pct_gt3": 1.0, "max_abs": 14.0},
    ])
    problems = replay.distribution_flags(stats)
    assert "quiet-vol" in problems["issues"].iloc[0]


def test_replay_skips_events_before_the_history_starts(history):
    ancient = replay.Event("1999-01-04", "before the data", {"spx": replay.UP})
    results = replay.replay_events(
        history, flags=rolls.empty_flags(), events=(ancient,)
    )
    assert results == []
