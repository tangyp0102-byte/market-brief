"""Tests for mb.registry, including validation of the shipped config."""

from __future__ import annotations

import textwrap

import pytest

from mb.registry import RegistryError, load_registry

MINIMAL = """
meta:
  version: 1
instruments:
  - id: ust_10y
    name: US 10Y
    tier: core
    asset_class: rates
    quote_unit: percent
    change_unit: bp
    sources:
      - {provider: treasury_par_nominal, symbol: "10 Yr"}
    bounds: {min: -1.0, max: 20.0}
    max_abs_change: 50
"""


def write(tmp_path, body: str):
    path = tmp_path / "instruments.yaml"
    path.write_text(textwrap.dedent(body))
    return path


# ---------------------------------------------------------- the real registry

def test_shipped_registry_loads_and_validates():
    reg = load_registry()
    assert len(reg) > 30
    assert reg["spx"].change_unit == "pct"
    assert reg["ust_10y"].change_unit == "bp"
    assert reg["vix"].change_unit == "pts"


def test_every_fx_instrument_declares_dollar_direction():
    reg = load_registry()
    fx = reg.by_asset_class("fx")
    assert fx, "expected FX instruments"
    for inst in fx:
        assert inst.dollar_direction in (1, -1), inst.id


def test_yahoo_short_form_tickers_have_correct_dollar_direction():
    """The 'JPY=X means USD/JPY' trap. Wrong here and every FX read is inverted."""
    reg = load_registry()
    assert reg["usdjpy"].dollar_direction == 1
    assert reg["usdjpy"].primary.symbol == "JPY=X"
    assert reg["eurusd"].dollar_direction == -1
    assert reg["eurusd"].primary.symbol == "EURUSD=X"
    assert reg["audusd"].dollar_direction == -1


def test_dxy_components_are_complete_and_weighted_correctly():
    reg = load_registry()
    comps = reg.dxy_components()
    assert len(comps) == 6
    assert sum(abs(c.dxy_weight) for c in comps) == pytest.approx(1.0)
    weights = {c.id: c.dxy_weight for c in comps}
    assert weights["eurusd"] == pytest.approx(-0.576)
    assert weights["usdjpy"] == pytest.approx(0.136)


def test_all_yield_instruments_use_basis_points():
    reg = load_registry()
    for inst in list(reg.by_asset_class("rates")) + list(reg.by_asset_class("credit")):
        if inst.quote_unit == "percent":
            assert inst.change_unit == "bp", f"{inst.id} should change in bp"


def test_t_plus_one_series_have_relaxed_staleness():
    reg = load_registry()
    assert reg["hy_oas"].stale_tolerance_days == 2
    assert reg["spx"].stale_tolerance_days == 1


def test_tradingview_urls_are_built_for_deep_linking():
    reg = load_registry()
    url = reg["spx"].tradingview_url()
    assert url is not None and url.startswith("https://www.tradingview.com/chart/")
    assert "%3A" in url  # the colon in SP:SPX must be encoded


# ------------------------------------------------------------ validation rules

def test_bp_change_on_a_price_instrument_is_rejected(tmp_path):
    body = MINIMAL.replace("quote_unit: percent", "quote_unit: usd")
    with pytest.raises(RegistryError, match="incompatible"):
        load_registry(write(tmp_path, body))


def test_fx_without_dollar_direction_is_rejected(tmp_path):
    body = """
    instruments:
      - id: eurusd
        name: EUR/USD
        asset_class: fx
        quote_unit: fx_rate
        change_unit: pct
        sources: [{provider: yahoo, symbol: "EURUSD=X"}]
        bounds: {min: 0.5, max: 2.0}
        max_abs_change: 8
    """
    with pytest.raises(RegistryError, match="dollar_direction"):
        load_registry(write(tmp_path, body))


def test_dxy_weight_sign_must_agree_with_dollar_direction(tmp_path):
    body = """
    instruments:
      - id: eurusd
        name: EUR/USD
        asset_class: fx
        quote_unit: fx_rate
        change_unit: pct
        dollar_direction: -1
        dxy_component: {weight: 0.576}
        sources: [{provider: yahoo, symbol: "EURUSD=X"}]
        bounds: {min: 0.5, max: 2.0}
        max_abs_change: 8
    """
    with pytest.raises(RegistryError, match="disagrees"):
        load_registry(write(tmp_path, body))


def test_duplicate_instrument_id_is_rejected(tmp_path):
    body = MINIMAL + (
        "  - id: ust_10y\n"
        "    name: Duplicate\n"
        "    asset_class: rates\n"
        "    quote_unit: percent\n"
        "    change_unit: bp\n"
        "    sources: [{provider: fred, symbol: DGS10}]\n"
        "    bounds: {min: -1.0, max: 20.0}\n"
        "    max_abs_change: 50\n"
    )
    with pytest.raises(RegistryError, match="duplicate instrument id"):
        load_registry(write(tmp_path, body))


def test_shared_primary_source_is_rejected(tmp_path):
    body = MINIMAL + (
        "  - id: ust_ten_year_copy\n"
        "    name: Copy paste error\n"
        "    asset_class: rates\n"
        "    quote_unit: percent\n"
        "    change_unit: bp\n"
        '    sources: [{provider: treasury_par_nominal, symbol: "10 Yr"}]\n'
        "    bounds: {min: -1.0, max: 20.0}\n"
        "    max_abs_change: 50\n"
    )
    with pytest.raises(RegistryError, match="shares a primary source"):
        load_registry(write(tmp_path, body))


def test_unknown_provider_is_rejected(tmp_path):
    body = MINIMAL.replace("treasury_par_nominal", "bloomberg")
    with pytest.raises(RegistryError, match="provider"):
        load_registry(write(tmp_path, body))


def test_inverted_bounds_are_rejected(tmp_path):
    body = MINIMAL.replace("{min: -1.0, max: 20.0}", "{min: 20.0, max: -1.0}")
    with pytest.raises(RegistryError, match="bounds.min"):
        load_registry(write(tmp_path, body))


def test_missing_file_raises(tmp_path):
    with pytest.raises(RegistryError, match="not found"):
        load_registry(tmp_path / "nope.yaml")


def test_lookup_helpers(tmp_path):
    reg = load_registry(write(tmp_path, MINIMAL))
    assert reg["ust_10y"].name == "US 10Y"
    assert reg.get("missing") is None
    with pytest.raises(KeyError):
        reg["missing"]
    assert reg.by_provider("treasury_par_nominal")


def test_cny_substitution_is_registered_with_a_fallback():
    """CNH was unavailable; CNY is the documented substitute."""
    reg = load_registry()
    assert reg.get("usdcnh") is None, "usdcnh should have been replaced by usdcny"
    cny = reg["usdcny"]
    assert cny.dollar_direction == 1
    assert cny.primary.symbol == "CNY=X"
    assert len(cny.sources) == 2  # USDCNY=X fallback


def test_vix3m_prefers_cboe_over_stale_yahoo():
    reg = load_registry()
    vix3m = reg["vix3m"]
    assert vix3m.primary.provider == "cboe"
    assert vix3m.fallbacks[0].provider == "yahoo"


def test_wti_bounds_admit_negative_prices():
    """April 2020 settled at -$37.63. Excluding that discards real history."""
    reg = load_registry()
    assert reg["wti"].bounds_min < -37.63


def test_stooq_is_no_longer_referenced():
    """Stooq consistently serves HTML to this client; a fallback that never
    works creates false confidence about redundancy."""
    reg = load_registry()
    assert not reg.by_provider("stooq")


def test_fallback_coverage_is_honestly_reported():
    """Most instruments are now single-sourced. Cross-source reconciliation in
    the step-2 validation gate can only apply where a real fallback exists."""
    reg = load_registry()
    multi = [i.id for i in reg if len(i.sources) > 1]
    # rates (treasury+fred), usdcny (two yahoo tickers), vix3m (cboe+yahoo)
    assert "ust_10y" in multi
    assert "vix3m" in multi
    assert "usdcny" in multi


def test_registry_without_a_core_instrument_is_rejected(tmp_path):
    """A registry with no core tape would let the gate publish on an empty tape."""
    body = MINIMAL.replace("    tier: core\n", "")
    with pytest.raises(RegistryError, match="no core instruments"):
        load_registry(write(tmp_path, body))


def test_bond_series_follow_the_sifma_calendar():
    reg = load_registry()
    assert reg["ust_10y"].calendar == "SIFMA_US"
    assert reg["hy_oas"].calendar == "SIFMA_US"   # FRED OAS, not an ETF
    assert reg["hyg"].calendar == "NYSE"          # ETF, trades on NYSE
    assert reg["spx"].calendar == "NYSE"
    assert reg["wti"].calendar == "CMEGlobex_EnergyAndMetals"
    assert reg["eurusd"].calendar == "FX"
