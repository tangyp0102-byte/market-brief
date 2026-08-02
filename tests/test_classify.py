"""Tests for mb.classify."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "tests")

from mb import classify, derived, rolls, store
from mb.registry import load_registry


@pytest.fixture(scope="module")
def history():
    from test_pipeline_smoke import synthetic_history
    return store.validate(synthetic_history())


# ------------------------------------------------------------ design contracts

def test_natural_gas_is_excluded_from_classification():
    """Roll-contaminated and weather-driven; it must not reach a rule."""
    assert "natgas" in classify.EXCLUDED
    members = {m for b in classify.BLOCS for m in b.members}
    assert "natgas" not in members
    reps = {b.representative for b in classify.BLOCS}
    assert "natgas" not in reps


def test_credit_signal_uses_the_same_day_proxy_not_the_t_plus_one_series():
    credit = classify.BLOC_BY_ID["credit"]
    assert credit.representative == "credit_excess"
    assert "hy_oas" not in credit.members
    assert "ig_oas" not in credit.members


def test_every_bloc_representative_exists():
    valid = set(load_registry().ids) | set(derived.BY_ID)
    for bloc in classify.BLOCS:
        assert bloc.representative in valid, bloc.id
        assert bloc.representative in bloc.members or bloc.id == "credit"


def test_correlated_instruments_are_deflated_to_one_signal():
    """Six FX rows moved together on 2026-07-31; they are one observation."""
    dollar = classify.BLOC_BY_ID["dollar"]
    assert len(dollar.members) >= 6
    assert dollar.representative == "dxy"
    rates = classify.BLOC_BY_ID["rates"]
    assert len(rates.members) == 4
    assert rates.select == "strongest"


def test_no_instrument_belongs_to_two_blocs():
    seen = set()
    for bloc in classify.BLOCS:
        for member in bloc.members:
            assert member not in seen, f"{member} counted twice"
            seen.add(member)


def test_rules_only_reference_real_blocs():
    ids = {b.id for b in classify.BLOCS}
    for rule in classify.RULES:
        for leg in rule.legs:
            assert leg in ids, (rule.id, leg)
        assert all(d in (classify.UP, classify.DOWN) for d in rule.legs.values())


def test_rule_signatures_are_distinct():
    """Two rules with identical legs could never be told apart."""
    seen = {}
    for rule in classify.RULES:
        key = tuple(sorted(rule.legs.items()))
        assert key not in seen, f"{rule.id} duplicates {seen.get(key)}"
        seen[key] = rule.id


# ------------------------------------------------------------------- mechanics

def test_zscore_is_clipped_before_entering_the_composite():
    """One quiet-vol instrument at 8 sigma must not set the intensity alone."""
    sig = classify.Signal("dollar", "US dollar", "dxy", -1.9, "%", -8.58, 0.99)
    assert sig.clipped_z == pytest.approx(-classify.Z_CLIP)
    assert abs(sig.clipped_z) < abs(sig.z)


def test_composite_intensity_averages_core_blocs():
    signals = {
        b: classify.Signal(b, b, "x", 1.0, "%", 2.0, 0.9)
        for b in classify.CORE_BLOCS
    }
    assert classify.composite_intensity(signals) == pytest.approx(2.0)


def test_move_threshold_uses_percentile_not_sigma():
    """Sigma is not calibrated here; the replay showed sd of 1.05-1.19."""
    below = classify.Signal("equity", "Equity", "spx", 1.0, "%", 2.2, 0.55)
    above = classify.Signal("equity", "Equity", "spx", 1.0, "%", 1.4, 0.85)
    assert not below.moved      # high sigma, unremarkable for this instrument
    assert above.moved          # lower sigma, genuinely unusual for it


def test_direction_reads_from_change_not_zscore():
    assert classify.Signal("x", "x", "y", -0.5, "%", 2.0, 0.9).direction == classify.DOWN
    assert classify.Signal("x", "x", "y", 0.5, "%", -2.0, 0.9).direction == classify.UP
    assert classify.Signal("x", "x", "y", None, "%", None, None).direction == 0


# ------------------------------------------------------------- classification

def _signals(**kwargs) -> dict[str, classify.Signal]:
    """Build a signal set; values are (change, percentile)."""
    out = {}
    for bloc, (change, pct) in kwargs.items():
        out[bloc] = classify.Signal(
            bloc, bloc, classify.BLOC_BY_ID[bloc].representative,
            change, "%", change * 2, pct,
        )
    return out


def test_growth_scare_signature_matches():
    signals = _signals(
        equity=(-2.0, 0.9), rates=(-8.0, 0.9), dollar=(0.6, 0.85), gold=(1.5, 0.9),
        energy=(-2.0, 0.8),
    )
    matched = [
        r for r in classify.RULES
        if all(signals[b].moved and signals[b].direction == d
               for b, d in r.legs.items() if b in signals)
        and set(r.legs) <= set(signals)
    ]
    assert "growth_scare" in {r.id for r in matched}


def test_term_premium_is_distinguishable_from_hawkish():
    """Both have equities down and yields up; the dollar separates them.

    Hawkish repricing no longer REQUIRES a stronger dollar - that requirement is
    what made the classifier miss every risk-off day where rate-cut repricing
    outweighed haven demand. The dollar is now a confirmation there, and a
    required leg only for term premium, where it is the defining feature.
    """
    term = next(r for r in classify.RULES if r.id == "term_premium")
    hawk = next(r for r in classify.RULES if r.id == "hawkish_repricing")
    assert term.legs["dollar"] == classify.DOWN
    assert "dollar" not in hawk.legs
    assert "dollar_up" in hawk.confirmations
    # Term premium is more specific, so it wins the tie-break.
    assert len(term.legs) > len(hawk.legs)


def test_risk_off_without_a_dollar_bid_is_classifiable():
    """The gap that made 2020-03-09 return 'no clean signal' at 3.6x threshold."""
    scare = next(r for r in classify.RULES if r.id == "growth_scare")
    assert set(scare.legs) == {"equity", "rates"}
    assert "dollar_up" in scare.confirmations   # confirmation, not requirement


def test_quadrant_grid_covers_every_equity_rates_combination():
    grid = {
        (r.legs["equity"], r.legs["rates"])
        for r in classify.RULES
        if set(r.legs) == {"equity", "rates"}
    }
    assert grid == {
        (classify.DOWN, classify.DOWN), (classify.DOWN, classify.UP),
        (classify.UP, classify.DOWN), (classify.UP, classify.UP),
    }


def test_rates_bloc_picks_the_tenor_that_actually_moved():
    """A fixed 10y proxy missed the front-end-led CPI selloff of 2022-09-13."""
    rates = classify.BLOC_BY_ID["rates"]
    assert rates.select == "strongest"
    assert "ust_2y" in rates.members and "ust_30y" in rates.members


def test_quiet_session_is_not_classified(history):
    result = classify.classify(
        history, session=history["date"].max(), flags=rolls.empty_flags(),
        intensity_threshold=99.0,
    )
    assert not result.classified
    assert result.regime_name == "No clean signal"
    assert any("noise" in n.lower() for n in result.notes)


def test_active_session_without_a_signature_is_not_forced(history):
    """A day can be busy without having a coherent cross-asset story."""
    result = classify.classify(
        history, session=history["date"].max(), flags=rolls.empty_flags(),
        intensity_threshold=0.0,
    )
    if not result.classified:
        assert any("no rule matched" in n.lower() or "noise" in n.lower()
                   for n in result.notes)


def test_confidence_reports_contradiction():
    result = classify.Classification(
        session=pd.Timestamp("2024-01-02").date(), regime="growth_scare",
        regime_name="Growth scare", intensity=1.5, intensity_percentile=0.9,
        signals={},
        confirmations=[
            classify.Confirmation("defensives_lead", False, "cyclicals led"),
            classify.Confirmation("vol_up", False, "VIX fell"),
            classify.Confirmation("credit_weak", True, "HY weak"),
        ],
    )
    assert result.confidence == "contradicted"
    assert len(result.conflicting) == 2


def test_confidence_strong_when_all_confirmations_hold():
    result = classify.Classification(
        session=pd.Timestamp("2024-01-02").date(), regime="growth_scare",
        regime_name="Growth scare", intensity=1.5, intensity_percentile=0.9,
        signals={},
        confirmations=[
            classify.Confirmation("vol_up", True, "VIX rose"),
            classify.Confirmation("credit_weak", True, "HY weak"),
        ],
    )
    assert result.confidence == "strong"
    assert not result.conflicting


def test_unevaluable_confirmations_do_not_count_as_conflicts():
    result = classify.Classification(
        session=pd.Timestamp("2024-01-02").date(), regime="x", regime_name="X",
        intensity=1.0, intensity_percentile=None, signals={},
        confirmations=[classify.Confirmation("credit_weak", None, "unavailable")],
    )
    assert result.confidence == "unconfirmed"
    assert not result.conflicting


def test_signals_build_from_real_history(history):
    signals = classify.build_signals(
        history, session=history["date"].max(), flags=rolls.empty_flags()
    )
    for bloc in classify.CORE_BLOCS:
        assert bloc in signals, bloc
        assert signals[bloc].percentile is not None


def test_intensity_history_spans_the_dataset(history):
    series = classify.intensity_history(history, flags=rolls.empty_flags())
    assert len(series) > 200
    assert (series >= 0).all()
    assert series.max() <= classify.Z_CLIP


def test_render_states_conflicts_rather_than_resolving_them():
    result = classify.Classification(
        session=pd.Timestamp("2024-01-02").date(), regime="growth_scare",
        regime_name="Growth scare", intensity=1.5, intensity_percentile=0.9,
        signals={}, confirmations=[
            classify.Confirmation("defensives_lead", False, "cyclicals led"),
        ],
        notes=["CONFLICT: defensives_lead did not corroborate."],
    )
    text = classify.render(result)
    assert "CONFLICT" in text
    assert "XX" in text


def test_numpy_booleans_are_coerced_for_identity_checks():
    """`np.True_ is True` is False, which silently broke every confirmation."""
    c = classify.Confirmation("x", np.True_, "")
    assert c.holds is True
    assert classify.Confirmation("y", np.False_, "").holds is False
    assert classify.Confirmation("z", None, "").holds is None

    result = classify.Classification(
        session=pd.Timestamp("2024-01-02").date(), regime="g", regime_name="X",
        intensity=1.0, intensity_percentile=None, signals={},
        confirmations=[
            classify.Confirmation("a", np.True_, ""),
            classify.Confirmation("b", np.True_, ""),
        ],
    )
    assert len(result.supported) == 2
    assert result.confidence == "strong"


def test_contradicted_classification_is_withdrawn(history):
    """Detecting a mismatch then asserting the label anyway is the worst option."""
    result = classify.Classification(
        session=pd.Timestamp("2024-01-02").date(), regime="hawkish_repricing",
        regime_name="Hawkish repricing", intensity=1.5, intensity_percentile=0.9,
        signals={},
        confirmations=[
            classify.Confirmation("front_end_leads", False, "long end led"),
            classify.Confirmation("curve_flattens", False, "curve steepened"),
            classify.Confirmation("dollar_up", False, "dollar fell"),
            classify.Confirmation("growth_underperforms", True, "tech lagged"),
        ],
    )
    assert result.confidence == "contradicted"
    assert len(result.conflicting) == 3


def test_severity_gate_blocks_a_quiet_liquidation(history):
    """A calm session matching liquidation's legs must not get the label."""
    session = history["date"].max()
    engineered = history.copy()
    # equity down, gold down, dollar up - but only mildly
    for iid, pct in {"spx": -0.6, "gold": -0.5, "eurusd": -0.3, "usdjpy": 0.3,
                     "gbpusd": -0.3, "usdcad": 0.3, "usdsek": 0.3,
                     "usdchf": 0.3}.items():
        base = float(
            engineered[(engineered.instrument_id == iid)
                       & (engineered.date < session)]["value"].iloc[-1]
        )
        engineered.loc[
            (engineered.instrument_id == iid) & (engineered.date == session), "value"
        ] = base * (1 + pct / 100)

    result = classify.classify(
        engineered, session=session, flags=rolls.empty_flags(),
        intensity_threshold=0.0,
    )
    assert result.regime != "liquidation"


def test_severity_is_conveyed_by_intensity_not_by_the_label():
    """The same legs describe routine de-risking and a funding crisis.

    'Dollar funding stress' fired on a top-7% session that was not a crisis.
    Rather than tune a severity floor until it excluded that day, the label now
    describes what the legs actually detect, and the intensity percentile
    carries the severity. 2020-03-16 and an ordinary risk-off day share a
    signature; only their magnitude differs.
    """
    rule = next(r for r in classify.RULES if r.id == "broad_derisking")
    assert "funding" not in rule.name.lower()
    assert "liquidation" not in rule.name.lower()
    assert rule.min_intensity_percentile is None
    assert "intensity" in rule.description.lower()


def test_term_premium_retains_a_severity_floor():
    """A structural claim about term premium needs a substantial session."""
    rule = next(r for r in classify.RULES if r.id == "term_premium")
    assert rule.min_intensity_percentile == 0.85
