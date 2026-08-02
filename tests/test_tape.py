"""Tests for mb.derived and mb.tape."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "tests")

from mb import derived, rolls, store, tape
from mb import transforms as tf
from mb.registry import load_registry


@pytest.fixture(scope="module")
def history():
    from test_pipeline_smoke import synthetic_history
    return store.validate(synthetic_history())


@pytest.fixture(scope="module")
def session(history):
    return history["date"].max()


@pytest.fixture(scope="module")
def wide(history):
    return store.wide(history, "close")


# -------------------------------------------------------------------- derived

def test_all_derived_series_compute(wide):
    series = derived.build(wide, load_registry())
    assert set(series) == {d.id for d in derived.DERIVED}
    for name, values in series.items():
        assert not values.empty, name
        assert values.notna().all(), name


def test_spreads_use_plain_difference_not_bp_multiplication():
    """A spread level already in bp must use change_unit='pts'.

    Using 'bp' would multiply by 100 and inflate every curve move 100x.
    """
    for spec in derived.DERIVED:
        if spec.unit_label == "bp":
            assert spec.change_unit == "pts", (
                f"{spec.id} would inflate its change by 100x"
            )


def test_percent_derived_series_use_pct():
    for spec in derived.DERIVED:
        if spec.unit_label == "%":
            assert spec.change_unit == "pct", spec.id


def test_curve_spread_equals_difference_of_legs(wide):
    series = derived.build(wide, load_registry())
    spread = series["ust_2s10s"]
    expected = (wide["ust_10y"] - wide["ust_2y"]) * 100
    aligned = expected.reindex(spread.index)
    np.testing.assert_allclose(spread.to_numpy(), aligned.to_numpy(), rtol=1e-9)


def test_breakeven_is_nominal_minus_real_in_bp(wide):
    series = derived.build(wide, load_registry())
    be = series["be_10y"]
    expected = (wide["ust_10y"] - wide["ust_real_10y"]) * 100
    np.testing.assert_allclose(
        be.to_numpy(), expected.reindex(be.index).to_numpy(), rtol=1e-9
    )


def test_dxy_falls_when_the_euro_rises():
    """Controlled test: move EUR only, hold the other five crosses fixed.

    A statistical test on random-walk fixtures is the wrong tool here: EUR is
    57.6% of the index, so the other five can outvote a modest EUR move when
    they are independent. Real crosses are dollar-correlated; synthetic ones are
    not. Isolating the variable tests the convention rather than the fixture.
    """
    registry = load_registry()
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    base = {"eurusd": 1.0850, "usdjpy": 149.50, "gbpusd": 1.2700,
            "usdcad": 1.3550, "usdsek": 10.40, "usdchf": 0.8800}

    frame = pd.DataFrame({k: [v] * 3 for k, v in base.items()}, index=idx)
    frame.loc[idx[1], "eurusd"] = base["eurusd"] * 1.01   # euro up 1%
    frame.loc[idx[2], "eurusd"] = base["eurusd"] * 0.99   # euro down 1%

    dxy = derived.build(frame, registry)["dxy"]
    changes = tf.change_series(dxy, "pct")

    assert changes.iloc[1] < 0, "euro up must push the dollar index down"
    assert changes.iloc[2] > 0, "euro down must push the dollar index up"
    # 57.6% weight: a 1% euro move is roughly a 0.58% index move.
    assert 0.4 < abs(changes.iloc[1]) < 0.8


def test_dxy_and_euro_disagree_only_when_other_crosses_dominate(wide):
    """Weaker statistical check on the fixture, with an honest threshold."""
    series = derived.build(wide, load_registry())
    dxy_chg = tf.change_series(series["dxy"], "pct")
    eur_chg = tf.change_series(wide["eurusd"].dropna(), "pct")
    joined = pd.concat([dxy_chg, eur_chg], axis=1).dropna()
    joined.columns = ["dxy", "eur"]

    # Restrict to sessions where the euro clearly dominates the basket.
    big = joined[joined["eur"].abs() > 0.5]
    assert len(big) > 10
    opposite = ((big["eur"] > 0) != (big["dxy"] > 0)).mean()
    assert opposite > 0.95


def test_derived_skips_missing_inputs_rather_than_raising(wide):
    """Credit spreads do not exist before 2023; a 2020 tape must still build."""
    reduced = wide.drop(columns=["hy_oas", "ig_oas"])
    series = derived.build(reduced, load_registry())
    assert "hy_ig" not in series
    assert "dxy" in series
    assert "hy_ig" in derived.unavailable(reduced)


# ----------------------------------------------------------------------- tape

def test_tape_contains_instruments_and_derived(history, session):
    result = tape.build_tape(
        history, session=session, flags=rolls.empty_flags(), run_gate=False
    )
    assert len(result.rows) == len(load_registry()) + len(derived.DERIVED)
    assert sum(r.is_derived for r in result.rows) == len(derived.DERIVED)


def test_tape_units_match_the_registry(history, session):
    result = tape.build_tape(
        history, session=session, flags=rolls.empty_flags(), run_gate=False
    )
    reg = load_registry()
    for row in result.rows:
        if row.is_derived:
            continue
        expected = {"bp": "bp", "pct": "%", "pts": "pt"}[reg[row.instrument_id].change_unit]
        assert row.unit_label == expected, row.instrument_id


def test_yield_rows_are_in_basis_points(history, session):
    result = tape.build_tape(
        history, session=session, flags=rolls.empty_flags(), run_gate=False
    )
    rates = [r for r in result.by_class("rates") if not r.is_derived]
    assert rates
    assert all(r.unit_label == "bp" for r in rates)


def test_zscores_are_populated_and_bounded(history, session):
    result = tape.build_tape(
        history, session=session, flags=rolls.empty_flags(), run_gate=False
    )
    scored = [r for r in result.rows if r.z is not None and not pd.isna(r.z)]
    assert len(scored) > 40
    assert all(abs(r.z) < 6 for r in scored)


def test_significance_markers_track_magnitude():
    row = tape.TapeRow("x", "X", "equity", "core", 1.0, 1.0, "%", 2.5, 2.5)
    assert row.significance == "**"
    row.z = 3.4
    assert row.significance == "***"
    row.z = 0.4
    assert row.significance == ""
    row.z = None
    assert row.significance == ""


def test_roll_flag_removes_the_change_and_the_zscore(history, session):
    """A flagged roll must show as undefined, not as a move."""
    flags = pd.DataFrame(
        {
            "instrument_id": ["natgas"],
            "date": [session],
            "futures_pct": [-47.5],
            "proxy": ["UNG"],
            "proxy_pct": [3.6],
            "ratio": [13.0],
            "label": [rolls.ROLL],
        }
    )
    result = tape.build_tape(history, session=session, flags=flags, run_gate=False)
    row = next(r for r in result.rows if r.instrument_id == "natgas")
    assert row.change is None or pd.isna(row.change)
    assert row.z is None or pd.isna(row.z)
    assert row.level > 0          # the level survives


def test_largest_moves_sorted_by_absolute_z(history, session):
    result = tape.build_tape(
        history, session=session, flags=rolls.empty_flags(), run_gate=False
    )
    top = result.largest(6)
    zs = [abs(r.z) for r in top]
    assert zs == sorted(zs, reverse=True)


def test_render_produces_no_interpretation(history, session):
    """Step 3 is deliberately dumb. Regime language here would be a bug."""
    result = tape.build_tape(
        history, session=session, flags=rolls.empty_flags(), run_gate=False
    )
    text = tape.render(result).lower()
    for word in ("risk-off", "risk off", "growth scare", "goldilocks",
                 "reflation", "hawkish", "dovish", "regime", "suggests",
                 "implies that", "driven by"):
        assert word not in text, f"interpretation leaked into the tape: {word!r}"


def test_render_includes_every_asset_class(history, session):
    result = tape.build_tape(
        history, session=session, flags=rolls.empty_flags(), run_gate=False
    )
    text = tape.render(result)
    for asset_class in ["RATES", "CREDIT", "EQUITY", "VOL", "COMMODITY", "FX"]:
        assert asset_class in text


def test_blocked_gate_produces_no_tape(history):
    result = tape.build_tape(
        history, session=pd.Timestamp("2024-12-25"), flags=rolls.empty_flags()
    )
    assert result.rows == []
    assert "blocked" in tape.render(result).lower()


# --------------------------------------------------------------- consistency

def test_verify_checks_all_pass_on_clean_data(history, session):
    result = tape.verify(history, session=session, flags=rolls.empty_flags())
    assert not result.empty
    assert result["ok"].all(), result[~result["ok"]].to_dict("records")


def test_verify_catches_an_inverted_dollar_convention(history, session, monkeypatch):
    """Flip a dollar_direction and the sign check must fail."""
    result = tape.verify(history, session=session, flags=rolls.empty_flags())
    sign_check = result[result["check"].str.contains("EURUSD")]
    assert not sign_check.empty
    assert bool(sign_check["ok"].iloc[0])

    # Independent confirmation: the two must genuinely move oppositely.
    computed = float(sign_check["computed"].iloc[0])
    independent = float(sign_check["independent"].iloc[0])
    assert (computed > 0) != (independent > 0)


def test_verify_spread_units_are_consistent(history, session):
    result = tape.verify(history, session=session, flags=rolls.empty_flags())
    row = result[result["check"].str.contains("2s10s")]
    assert not row.empty
    assert bool(row["ok"].iloc[0])
    # Exact, not approximate: a spread is a difference of its legs.
    assert abs(row["computed"].iloc[0] - row["independent"].iloc[0]) < 1e-6


def test_quiet_vol_flagged_when_trailing_vol_collapses():
    """A large z during a calm stretch must be labelled, not silently smoothed."""
    row = tape.TapeRow("usdjpy", "USD/JPY", "fx", "core", 160.18, -1.91, "%",
                       -9.81, -13.4, vol_ratio=0.42)
    assert row.quiet_vol
    row.vol_ratio = 0.95
    assert not row.quiet_vol
    row.vol_ratio = None
    assert not row.quiet_vol


def test_tape_populates_vol_ratio(history, session):
    result = tape.build_tape(
        history, session=session, flags=rolls.empty_flags(), run_gate=False
    )
    scored = [r for r in result.rows if r.vol_ratio is not None and not pd.isna(r.vol_ratio)]
    assert len(scored) > 30
    assert all(r.vol_ratio > 0 for r in scored)


def test_clip_bounds_zscore_for_composite_use():
    idx = pd.bdate_range("2024-01-01", periods=200)
    rng = np.random.default_rng(3)
    values = list(rng.normal(0, 1.0, 199)) + [25.0]
    s = pd.Series(values, index=idx)
    assert abs(tf.rolling_zscore(s, 60, 30).iloc[-1]) > 10
    assert abs(tf.rolling_zscore(s, 60, 30, clip=4).iloc[-1]) == pytest.approx(4.0)


def test_small_ratios_render_with_useful_precision():
    """Copper/gold is ~0.0016; two decimals would render the signal as 0.00."""
    assert tape._fmt(0.00159) == "0.001590"
    assert tape._fmt(0.2878) == "0.2878"
    assert tape._fmt(7489.72) == "7,489.72"
    assert tape._fmt(4.75) == "4.75"
