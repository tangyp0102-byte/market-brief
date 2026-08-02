"""End-to-end smoke test with synthetic data.

Proves the pieces compose: registry -> store -> wide pivot -> unit-aware changes
-> z-scores -> DXY, using every instrument in the real registry. Catches the
integration bugs that pass unit tests individually (a registry id that no
transform handles, a change_unit the pivot cannot serve).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mb import store
from mb import transforms as tf
from mb.registry import load_registry

SESSIONS = 400


def synthetic_history(seed: int = 42) -> pd.DataFrame:
    """Random-walk history for every registry instrument, within its bounds."""
    reg = load_registry()
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=SESSIONS)

    frames = []
    for inst in reg:
        # Start at the centre of the declared bounds so a 400-step walk never
        # reaches an edge. Clipping the levels would create zero-variance
        # stretches, and the z-score of the first move out of one would be a
        # fixture artefact rather than a real signal.
        if inst.bounds_min > 0:
            start = float(np.sqrt(inst.bounds_min * inst.bounds_max))
        else:
            start = (inst.bounds_min + inst.bounds_max) / 2.0

        # Sized so a 400-step walk stays >4 sigma clear of the nearest bound
        # for the tightest-bounded instrument in the registry.
        vol = 0.015 if inst.asset_class == "vol" else 0.004
        steps = rng.normal(0.0, vol, SESSIONS)
        levels = start * np.exp(np.cumsum(steps))

        # The generator must not need clipping; assert rather than clamp.
        assert levels.min() > inst.bounds_min, f"{inst.id} walked below bounds"
        assert levels.max() < inst.bounds_max, f"{inst.id} walked above bounds"

        frames.append(store.to_records(dates, levels, inst.id, "synthetic", "close"))

    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def history(tmp_path_factory) -> pd.DataFrame:
    path = tmp_path_factory.mktemp("store") / "history.parquet"
    store.upsert(path, synthetic_history())
    return store.read(path)


def test_every_registry_instrument_survives_a_round_trip(history):
    reg = load_registry()
    stored = set(history["instrument_id"].unique())
    assert stored == set(reg.ids)


def test_wide_pivot_covers_all_instruments(history):
    reg = load_registry()
    wide = store.wide(history, "close")
    assert len(wide) == SESSIONS
    assert set(wide.columns) == set(reg.ids)
    assert wide.index.is_monotonic_increasing


def test_unit_aware_changes_computed_for_every_instrument(history):
    """Each instrument's change must use its own declared unit."""
    reg = load_registry()
    wide = store.wide(history, "close")

    for inst in reg:
        changes = tf.change_series(wide[inst.id], inst.change_unit)
        assert len(changes) == SESSIONS
        assert changes.iloc[1:].notna().all(), inst.id

        # Synthetic vol is ~0.4%/day, so changes must land in the right order of
        # magnitude for their unit. This catches a bp/pct mix-up.
        typical = changes.iloc[1:].abs().median()
        if inst.change_unit == "bp":
            assert 0.01 < typical < 200, f"{inst.id}: {typical} bp is implausible"
        elif inst.change_unit == "pct":
            assert 0.01 < typical < 20, f"{inst.id}: {typical}% is implausible"


def test_zscores_are_produced_and_bounded(history):
    reg = load_registry()
    wide = store.wide(history, "close")

    for inst in reg:
        changes = tf.change_series(wide[inst.id], inst.change_unit)
        z = tf.rolling_zscore(changes, reg.zscore_window, reg.zscore_min_obs)

        settled = z.iloc[reg.zscore_window + 5 :]
        assert settled.notna().all(), inst.id
        # A random walk should almost never exceed 6 sigma.
        assert settled.abs().max() < 6.0, inst.id


def test_zscore_distribution_is_roughly_standard(history):
    """Sanity check on the scaling: a random walk should give sd(z) near 1."""
    wide = store.wide(history, "close")
    changes = tf.change_series(wide["spx"], "pct")
    z = tf.rolling_zscore(changes, 60, 30).dropna()
    assert 0.7 < z.std() < 1.4
    assert abs(z.mean()) < 0.4


def test_dxy_reconstructed_from_stored_components(history):
    reg = load_registry()
    wide = store.wide(history, "close")
    weights = {c.id: c.dxy_weight for c in reg.dxy_components()}

    dxy = tf.dxy_series(wide, weights)
    assert len(dxy) == SESSIONS
    assert dxy.notna().all()
    assert (dxy > 0).all()

    changes = tf.change_series(dxy, "pct")
    z = tf.rolling_zscore(changes, 60, 30)
    assert z.iloc[70:].notna().all()


def test_dollar_normalisation_agrees_across_pairs(history):
    """A synthetic dollar move must read the same sign through any FX pair."""
    reg = load_registry()
    wide = store.wide(history, "close")
    date = wide.index[-1]
    prev = wide.index[-2]

    for inst in reg.by_asset_class("fx"):
        raw = tf.compute_change(wide[inst.id].loc[prev], wide[inst.id].loc[date], "pct")
        normalised = tf.dollar_normalise(raw, inst.dollar_direction)
        assert normalised == pytest.approx(raw * inst.dollar_direction)


def test_curve_spreads_and_breakevens_compute(history):
    wide = store.wide(history, "close")

    twos_tens = tf.curve_spread(wide["ust_10y"], wide["ust_2y"])
    assert len(twos_tens) == SESSIONS
    assert twos_tens.abs().max() < 2000  # sane bp magnitude

    be10 = tf.breakeven(wide["ust_10y"], wide["ust_real_10y"])
    assert len(be10) == SESSIONS


def test_relative_performance_ratio_computes(history):
    wide = store.wide(history, "close")
    ratio = tf.ratio_series(wide["rsp"], wide["spy"], "rsp_spy")
    assert ratio.notna().all()
    changes = tf.change_series(ratio, "pct")
    assert changes.iloc[1:].notna().all()


def test_backfill_rerun_does_not_duplicate(tmp_path):
    path = tmp_path / "history.parquet"
    data = synthetic_history()
    first = store.upsert(path, data)
    second = store.upsert(path, data)

    assert second["inserted"] == 0
    assert second["total"] == first["total"]


def test_coverage_report_is_complete(history):
    reg = load_registry()
    cov = store.coverage(history)
    assert len(cov) == len(reg)
    assert (cov["n_obs"] == SESSIONS).all()
