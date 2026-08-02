"""Tests for mb.probe: source probing, store audit, and quality checks."""

from __future__ import annotations

import pandas as pd
import pytest

from mb import probe, store


@pytest.fixture
def qc_store(tmp_path):
    """Store with the two real pathologies injected: a WTI sign crossing and a
    session that rates missed but equities traded."""
    import sys
    sys.path.insert(0, "tests")
    from test_pipeline_smoke import synthetic_history

    h = synthetic_history()
    dates = sorted(h.date.unique())

    mask = (h.instrument_id == "wti") & (h.date.isin(dates[100:103]))
    h.loc[mask, "value"] = [-15.0, -37.63, 8.0]
    h = h[~(h.instrument_id.str.startswith("ust_") & (h.date == dates[200]))]

    path = tmp_path / "qc.parquet"
    store.upsert(path, h)
    return path


def test_quality_detects_sign_crossing(qc_store):
    """WTI crossing zero must be flagged; percent change is undefined there."""
    report = probe.quality_report(qc_store)
    signs = report["sign_crossings"]
    assert not signs.empty
    assert "wti" in set(signs["instrument_id"])
    assert signs[signs.instrument_id == "wti"]["min"].iloc[0] < 0


def test_quality_flags_absurd_changes_from_sign_crossing(qc_store):
    """The breach list must contain the impossible percent moves."""
    report = probe.quality_report(qc_store)
    breaches = report["threshold_breaches"]
    wti = breaches[breaches.instrument_id == "wti"]
    assert not wti.empty
    assert wti["change"].abs().max() > 100  # nonsense magnitude


def test_quality_detects_calendar_divergence(qc_store):
    """Rates missing a session equities traded must show up."""
    report = probe.quality_report(qc_store)
    cal = report["calendar"].set_index("asset_class")
    assert cal.loc["rates", "sessions"] < cal.loc["equity", "sessions"]
    assert report["fully_aligned_sessions"] < report["total_sessions"]


def test_quality_clean_store_reports_nothing(tmp_path):
    import sys
    sys.path.insert(0, "tests")
    from test_pipeline_smoke import synthetic_history

    path = tmp_path / "clean.parquet"
    store.upsert(path, synthetic_history())
    report = probe.quality_report(path)
    assert report["sign_crossings"].empty
    assert report["fully_aligned_sessions"] == report["total_sessions"]


def test_probe_flags_single_row_symbol_as_sparse(monkeypatch):
    """The bug that reported CNH=X (1 session over 7 years) as OK."""
    from mb.sources import adapters

    def fake(symbol, start, end):
        dates = pd.bdate_range(end=pd.Timestamp("2026-07-31"), periods=1)
        return pd.DataFrame(
            {"date": dates, "symbol": symbol, "field": "close", "value": 7.2}
        )

    monkeypatch.setattr(adapters, "fetch_yahoo", fake)
    result = probe.probe_yahoo(["CNH=X"], "2019-01-01", "2026-08-01")
    assert "SPARSE" in result["status"].iloc[0]


def test_audit_empty_store(tmp_path):
    assert probe.audit_store(tmp_path / "nothing.parquet").empty


def test_binding_diagnostic_identifies_the_blocking_instruments(tmp_path):
    """A short-history series should show up as the constraint on a full tape."""
    import sys
    sys.path.insert(0, "tests")
    from test_pipeline_smoke import synthetic_history

    h = synthetic_history()
    dates = sorted(h.date.unique())
    h = h[~(h.instrument_id.isin(["hy_oas", "ig_oas"]) & (h.date < dates[-160]))]

    path = tmp_path / "bind.parquet"
    store.upsert(path, h)
    report = probe.quality_report(path)

    binding = report["binding"]
    dropped_first_two = set(binding["dropped"].iloc[1:3])
    assert dropped_first_two == {"hy_oas", "ig_oas"}

    # Excluding them must recover most of the history.
    assert binding["aligned_sessions"].iloc[2] > 2 * binding["aligned_sessions"].iloc[0]


def test_rollcheck_flags_futures_move_absent_from_physical_proxy(tmp_path, monkeypatch):
    """A 31% futures drop with a 3% move in physically-backed SLV is an artifact."""
    import sys
    sys.path.insert(0, "tests")
    from test_pipeline_smoke import synthetic_history
    from mb.sources import adapters

    h = synthetic_history()
    dates = sorted(h.date.unique())
    event = dates[-5]
    prior = dates[-6]

    # engineer a -31% silver move in the futures series only
    base = float(h[(h.instrument_id == "silver") & (h.date == prior)]["value"].iloc[0])
    h.loc[(h.instrument_id == "silver") & (h.date == event), "value"] = base * 0.687

    path = tmp_path / "roll.parquet"
    store.upsert(path, h)

    def fake_yahoo(symbol, start, end):
        idx = pd.DatetimeIndex(sorted(h.date.unique()))
        values = pd.Series(100.0, index=idx)
        values.loc[event] = 97.0          # proxy barely moved
        return pd.DataFrame(
            {"date": idx, "symbol": symbol, "field": "close", "value": values.values}
        )

    monkeypatch.setattr(adapters, "fetch_yahoo", fake_yahoo)
    result = probe.rollcheck(path, min_abs_move=8.0)

    silver = result[(result.instrument_id == "silver") & (result.date == event.date())]
    assert not silver.empty
    assert silver["verdict"].iloc[0] == "LIKELY ROLL ARTIFACT"
    assert silver["ratio"].iloc[0] > 3.0


def test_rollcheck_accepts_a_move_confirmed_by_the_proxy(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, "tests")
    from test_pipeline_smoke import synthetic_history
    from mb.sources import adapters

    h = synthetic_history()
    dates = sorted(h.date.unique())
    event, prior = dates[-5], dates[-6]

    base = float(h[(h.instrument_id == "silver") & (h.date == prior)]["value"].iloc[0])
    h.loc[(h.instrument_id == "silver") & (h.date == event), "value"] = base * 0.88

    path = tmp_path / "real.parquet"
    store.upsert(path, h)

    def fake_yahoo(symbol, start, end):
        idx = pd.DatetimeIndex(sorted(h.date.unique()))
        values = pd.Series(100.0, index=idx)
        values.loc[event] = 88.0          # proxy moved the same amount
        return pd.DataFrame(
            {"date": idx, "symbol": symbol, "field": "close", "value": values.values}
        )

    monkeypatch.setattr(adapters, "fetch_yahoo", fake_yahoo)
    result = probe.rollcheck(path, min_abs_move=8.0)
    silver = result[(result.instrument_id == "silver") & (result.date == event.date())]
    assert silver["verdict"].iloc[0] == "real move"
