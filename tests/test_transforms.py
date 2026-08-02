"""Tests for mb.transforms. These are the highest-value tests in the project."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mb import transforms as tf


# ------------------------------------------------------------------ unit maths

def test_bp_change_from_percent_yields():
    # 4.25% -> 4.31% is +6bp, not +0.06 and not +6000.
    assert tf.compute_change(4.25, 4.31, "bp") == pytest.approx(6.0)
    assert tf.compute_change(4.25, 4.19, "bp") == pytest.approx(-6.0)


def test_pct_change_from_prices():
    assert tf.compute_change(100.0, 101.5, "pct") == pytest.approx(1.5)
    assert tf.compute_change(50.0, 45.0, "pct") == pytest.approx(-10.0)


def test_pts_change_for_vix():
    # VIX 14.2 -> 18.7 is +4.5 points. Expressing this as a percent would be
    # technically true and analytically useless.
    assert tf.compute_change(14.2, 18.7, "pts") == pytest.approx(4.5)


def test_unknown_change_unit_raises():
    with pytest.raises(ValueError, match="unknown change_unit"):
        tf.compute_change(1.0, 2.0, "basis_points")


def test_pct_change_from_zero_base_raises():
    with pytest.raises(ZeroDivisionError):
        tf.compute_change(0.0, 1.0, "pct")


def test_change_series_matches_scalar_version():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    levels = pd.Series([4.20, 4.25, 4.18, 4.30, 4.31], index=idx)
    changes = tf.change_series(levels, "bp")

    assert math.isnan(changes.iloc[0])  # no prior observation
    assert changes.iloc[1] == pytest.approx(5.0)
    assert changes.iloc[2] == pytest.approx(-7.0)
    for i in range(1, len(levels)):
        expected = tf.compute_change(levels.iloc[i - 1], levels.iloc[i], "bp")
        assert changes.iloc[i] == pytest.approx(expected)


def test_change_series_sorts_unordered_input():
    idx = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"])
    levels = pd.Series([4.30, 4.10, 4.20], index=idx)
    changes = tf.change_series(levels, "bp")
    assert changes.index.is_monotonic_increasing
    assert changes.iloc[1] == pytest.approx(10.0)
    assert changes.iloc[2] == pytest.approx(10.0)


# ------------------------------------------------------------ dollar direction

def test_dollar_normalise_makes_positive_mean_stronger_usd():
    # EUR/USD -1% and USD/JPY +1% are the same dollar event.
    assert tf.dollar_normalise(-1.0, -1) == pytest.approx(1.0)
    assert tf.dollar_normalise(1.0, 1) == pytest.approx(1.0)
    # And the reverse.
    assert tf.dollar_normalise(0.8, -1) == pytest.approx(-0.8)


def test_dollar_normalise_rejects_bad_direction():
    with pytest.raises(ValueError):
        tf.dollar_normalise(1.0, 0)


# ------------------------------------------------------------------- z-scores

def _deterministic_changes(n: int = 200) -> pd.Series:
    rng = np.random.default_rng(20240101)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.0, 5.0, n), index=idx, name="chg")


def test_zscore_window_excludes_the_current_observation():
    """The score for day t must be computed from days t-60..t-1 only."""
    s = _deterministic_changes()
    z = tf.rolling_zscore(s, window=60, min_obs=30)

    i = 150
    prior = s.iloc[i - 60 : i]  # strictly before position i
    expected = (s.iloc[i] - prior.mean()) / prior.std(ddof=1)
    assert z.iloc[i] == pytest.approx(expected)


def test_zscore_is_immune_to_future_data():
    """Mutating a later observation must not change an earlier score."""
    s = _deterministic_changes()
    z_before = tf.rolling_zscore(s, window=60, min_obs=30)

    mutated = s.copy()
    mutated.iloc[180] = 500.0  # a crisis day, well after position 150
    z_after = tf.rolling_zscore(mutated, window=60, min_obs=30)

    assert z_before.iloc[150] == pytest.approx(z_after.iloc[150])
    assert z_before.iloc[:180].equals(z_after.iloc[:180]) or np.allclose(
        z_before.iloc[:180].dropna(), z_after.iloc[:180].dropna()
    )


def test_zscore_flags_an_outlier_that_a_contaminated_window_would_shrink():
    idx = pd.date_range("2020-01-01", periods=80, freq="B")
    values = [1.0, -1.0] * 39 + [0.0, 40.0]  # calm, then a large jump
    s = pd.Series(values, index=idx)
    z = tf.rolling_zscore(s, window=60, min_obs=30)
    # With the spike excluded from its own window, it should score enormous.
    assert z.iloc[-1] > 10


def test_zscore_returns_nan_before_min_obs():
    s = _deterministic_changes(50)
    z = tf.rolling_zscore(s, window=60, min_obs=30)
    assert z.iloc[:29].isna().all()
    assert z.iloc[35:].notna().any()


def test_zscore_handles_degenerate_scale_without_infinities():
    idx = pd.date_range("2020-01-01", periods=80, freq="B")
    s = pd.Series([0.0] * 79 + [5.0], index=idx)  # zero variance prior window
    z = tf.rolling_zscore(s, window=60, min_obs=30)
    assert not np.isinf(z.to_numpy()).any()
    assert pd.isna(z.iloc[-1])


def test_robust_zscore_less_sensitive_to_a_prior_outlier():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2020-01-01", periods=120, freq="B")
    values = rng.normal(0, 1, 120)
    values[60] = 60.0   # one crisis day inside the trailing window
    values[-1] = 4.0    # a genuinely notable day afterwards
    s = pd.Series(values, index=idx)

    z_std = tf.rolling_zscore(s, window=60, min_obs=30, robust=False)
    z_rob = tf.rolling_zscore(s, window=60, min_obs=30, robust=True)

    # The std-based window is inflated by the crisis day, so it understates.
    assert abs(z_rob.iloc[-1]) > abs(z_std.iloc[-1])


def test_zscore_argument_validation():
    s = _deterministic_changes(100)
    with pytest.raises(ValueError):
        tf.rolling_zscore(s, window=1)
    with pytest.raises(ValueError):
        tf.rolling_zscore(s, window=60, min_obs=90)


# ------------------------------------------------------------------------ DXY

DXY_WEIGHTS = {
    "eurusd": -0.576,
    "usdjpy": 0.136,
    "gbpusd": -0.119,
    "usdcad": 0.091,
    "usdsek": 0.042,
    "usdchf": 0.036,
}

PLAUSIBLE_QUOTES = {
    "eurusd": 1.0850,
    "usdjpy": 149.50,
    "gbpusd": 1.2700,
    "usdcad": 1.3550,
    "usdsek": 10.40,
    "usdchf": 0.8800,
}


def test_dxy_matches_independent_formula():
    got = tf.dxy_from_components(PLAUSIBLE_QUOTES, DXY_WEIGHTS)
    expected = 50.14348112
    for iid, weight in DXY_WEIGHTS.items():
        expected *= PLAUSIBLE_QUOTES[iid] ** weight
    assert got == pytest.approx(expected, rel=1e-12)


def test_dxy_lands_in_a_plausible_range_for_realistic_quotes():
    """External sanity check: these rates should produce an index near 100-110."""
    got = tf.dxy_from_components(PLAUSIBLE_QUOTES, DXY_WEIGHTS)
    assert 95.0 < got < 115.0


def test_dxy_rises_when_the_euro_falls():
    base = tf.dxy_from_components(PLAUSIBLE_QUOTES, DXY_WEIGHTS)
    weaker_eur = {**PLAUSIBLE_QUOTES, "eurusd": 1.0650}
    assert tf.dxy_from_components(weaker_eur, DXY_WEIGHTS) > base


def test_dxy_rises_when_the_yen_falls():
    base = tf.dxy_from_components(PLAUSIBLE_QUOTES, DXY_WEIGHTS)
    weaker_jpy = {**PLAUSIBLE_QUOTES, "usdjpy": 155.0}  # higher USD/JPY = weaker yen
    assert tf.dxy_from_components(weaker_jpy, DXY_WEIGHTS) > base


def test_dxy_euro_dominates_swiss_franc():
    """A 1% EUR move must move the index far more than a 1% CHF move."""
    base = tf.dxy_from_components(PLAUSIBLE_QUOTES, DXY_WEIGHTS)
    eur_move = tf.dxy_from_components(
        {**PLAUSIBLE_QUOTES, "eurusd": PLAUSIBLE_QUOTES["eurusd"] * 0.99}, DXY_WEIGHTS
    )
    chf_move = tf.dxy_from_components(
        {**PLAUSIBLE_QUOTES, "usdchf": PLAUSIBLE_QUOTES["usdchf"] * 1.01}, DXY_WEIGHTS
    )
    assert abs(eur_move - base) > 10 * abs(chf_move - base)


def test_dxy_rejects_missing_component():
    incomplete = {k: v for k, v in PLAUSIBLE_QUOTES.items() if k != "usdchf"}
    with pytest.raises(KeyError, match="usdchf"):
        tf.dxy_from_components(incomplete, DXY_WEIGHTS)


def test_dxy_rejects_weights_that_do_not_sum_to_one():
    bad = {**DXY_WEIGHTS, "usdchf": 0.5}
    with pytest.raises(ValueError, match="sum to 1.0"):
        tf.dxy_from_components(PLAUSIBLE_QUOTES, bad)


def test_dxy_rejects_non_positive_quote():
    with pytest.raises(ValueError, match="positive"):
        tf.dxy_from_components({**PLAUSIBLE_QUOTES, "eurusd": 0.0}, DXY_WEIGHTS)


def test_dxy_series_matches_scalar_row_by_row():
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    frame = pd.DataFrame(
        {k: [v, v * 1.001, v * 0.999, v] for k, v in PLAUSIBLE_QUOTES.items()},
        index=idx,
    )
    series = tf.dxy_series(frame, DXY_WEIGHTS)
    assert len(series) == 4
    for date in idx:
        expected = tf.dxy_from_components(frame.loc[date].to_dict(), DXY_WEIGHTS)
        assert series.loc[date] == pytest.approx(expected, rel=1e-10)


def test_dxy_series_drops_rows_with_a_missing_component():
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    frame = pd.DataFrame({k: [v] * 3 for k, v in PLAUSIBLE_QUOTES.items()}, index=idx)
    frame.loc[idx[1], "usdjpy"] = np.nan
    series = tf.dxy_series(frame, DXY_WEIGHTS)
    assert len(series) == 2
    assert idx[1] not in series.index


# ----------------------------------------------------------- spreads & ratios

def test_curve_spread_is_in_basis_points():
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    ten = pd.Series([4.30, 4.25], index=idx)
    two = pd.Series([4.80, 4.60], index=idx)
    spread = tf.curve_spread(ten, two)
    assert spread.iloc[0] == pytest.approx(-50.0)   # inverted by 50bp
    assert spread.iloc[1] == pytest.approx(-35.0)


def test_breakeven_is_nominal_minus_real():
    idx = pd.date_range("2024-01-01", periods=1, freq="B")
    nominal = pd.Series([4.30], index=idx)
    real = pd.Series([1.90], index=idx)
    assert tf.breakeven(nominal, real).iloc[0] == pytest.approx(240.0)


def test_spread_aligns_on_common_dates_only():
    ten = pd.Series([4.3, 4.2], index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    two = pd.Series([4.8], index=pd.to_datetime(["2024-01-02"]))
    spread = tf.curve_spread(ten, two)
    assert len(spread) == 1
    assert spread.iloc[0] == pytest.approx(-60.0)


def test_ratio_series_for_relative_performance():
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    rsp = pd.Series([160.0, 162.0], index=idx)
    spy = pd.Series([480.0, 490.0], index=idx)
    ratio = tf.ratio_series(rsp, spy, "rsp_spy")
    assert ratio.name == "rsp_spy"
    assert ratio.iloc[0] == pytest.approx(1 / 3)
