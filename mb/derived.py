"""Derived series computed from stored instruments.

These carry most of the cross-asset signal, and none of them is fetchable: the
dollar index is reconstructed from its components, curve slopes and breakevens
are differences between yields, and the relative-value ratios are quotients.

UNITS. A spread level is naturally quoted in basis points (2s10s at -50bp), but
change_unit='bp' in transforms means "multiply the difference by 100" because it
assumes the input is in percent. A spread already expressed in bp therefore uses
change_unit='pts' (a plain difference) with unit_label='bp' for display. Getting
this backwards inflates every curve move by 100x, which is exactly the kind of
error a plain table is meant to expose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from . import transforms as tf
from .registry import Registry


@dataclass(frozen=True)
class Derived:
    id: str
    name: str
    asset_class: str
    change_unit: str          # how the change is computed: bp | pct | pts
    unit_label: str           # how the change is displayed: bp | % | pt
    inputs: tuple[str, ...]
    compute: Callable[[pd.DataFrame, Registry], pd.Series]
    tier: str = "confirmation"
    note: str = ""

    def available(self, wide: pd.DataFrame) -> bool:
        return all(i in wide.columns for i in self.inputs)


def _dxy(wide: pd.DataFrame, registry: Registry) -> pd.Series:
    weights = {c.id: c.dxy_weight for c in registry.dxy_components()}
    return tf.dxy_series(wide, weights)


def _spread(long_leg: str, short_leg: str):
    def go(wide: pd.DataFrame, registry: Registry) -> pd.Series:
        return tf.curve_spread(wide[long_leg], wide[short_leg])
    return go


def _breakeven(nominal: str, real: str):
    def go(wide: pd.DataFrame, registry: Registry) -> pd.Series:
        return tf.breakeven(wide[nominal], wide[real])
    return go


def _ratio(numerator: str, denominator: str, name: str):
    def go(wide: pd.DataFrame, registry: Registry) -> pd.Series:
        return tf.ratio_series(wide[numerator], wide[denominator], name)
    return go


# iShares reports HYG effective duration around 3.2 years. The exact figure
# drifts with the index, so this is approximate by construction.
HYG_DURATION = 3.2


def _credit_excess(wide: pd.DataFrame, registry: Registry) -> pd.Series:
    """Cumulative index of HYG's return with its rate exposure removed.

    The authoritative credit signal is the ICE BofA OAS from FRED, but that
    publishes T+1 and the brief runs at 6pm ET, so it is structurally never
    available on the day it describes.

    HYG's raw return is not a substitute: it is a bond fund, so a large part of
    any move is duration, not credit. If the 5y yield rises 10bp, HYG falls
    about 0.32% with no change whatsoever in credit risk. Reading that as risk
    aversion would systematically mistake hawkish repricing for credit stress.

    Removing the rate component leaves the part attributable to spreads:

        excess = hyg_return + duration x (5y yield change in percent)

    Yields are forward-filled onto the equity calendar first, so a bond-market
    holiday contributes a zero rate change rather than dropping the session.
    """
    hyg = wide["hyg"].dropna()
    yields = wide["ust_5y"].reindex(hyg.index).ffill()

    hyg_return = hyg.pct_change() * 100.0
    rate_change_pct = yields.diff()          # already in percent
    excess = hyg_return + HYG_DURATION * rate_change_pct

    excess = excess.reindex(hyg.index)
    if excess.dropna().empty:
        return pd.Series(dtype=float)

    # Base the index at the first session so the first excess return survives.
    # Compounding from the first non-null value instead would silently discard it.
    factors = 1 + excess.fillna(0.0) / 100.0
    return (factors.cumprod() * 100.0).rename("credit_excess")


DERIVED: tuple[Derived, ...] = (
    Derived(
        id="dxy",
        name="US Dollar Index (reconstructed)",
        asset_class="fx",
        change_unit="pct",
        unit_label="%",
        inputs=("eurusd", "usdjpy", "gbpusd", "usdcad", "usdsek", "usdchf"),
        compute=_dxy,
        tier="core",
        note="computed from components; free DXY quotes go stale silently",
    ),
    Derived(
        id="ust_2s10s",
        name="2s10s Curve Slope",
        asset_class="rates",
        change_unit="pts",       # level already in bp
        unit_label="bp",
        inputs=("ust_10y", "ust_2y"),
        compute=_spread("ust_10y", "ust_2y"),
        tier="core",
        note="negative = inverted; bear-flattening implies hawkish repricing",
    ),
    Derived(
        id="ust_3m10y",
        name="3m10y Curve Slope",
        asset_class="rates",
        change_unit="pts",
        unit_label="bp",
        inputs=("ust_10y", "ust_3m"),
        compute=_spread("ust_10y", "ust_3m"),
    ),
    Derived(
        id="ust_5s30s",
        name="5s30s Curve Slope",
        asset_class="rates",
        change_unit="pts",
        unit_label="bp",
        inputs=("ust_30y", "ust_5y"),
        compute=_spread("ust_30y", "ust_5y"),
        note="long-end steepening without a dollar bid suggests term premium",
    ),
    Derived(
        id="be_10y",
        name="10y Inflation Breakeven",
        asset_class="rates",
        change_unit="pts",
        unit_label="bp",
        inputs=("ust_10y", "ust_real_10y"),
        compute=_breakeven("ust_10y", "ust_real_10y"),
        note="nominal minus real; separates inflation from real-rate moves",
    ),
    Derived(
        id="be_5y",
        name="5y Inflation Breakeven",
        asset_class="rates",
        change_unit="pts",
        unit_label="bp",
        inputs=("ust_5y", "ust_real_5y"),
        compute=_breakeven("ust_5y", "ust_real_5y"),
    ),
    Derived(
        id="hy_ig",
        name="HY minus IG Spread",
        asset_class="credit",
        change_unit="pts",
        unit_label="bp",
        inputs=("hy_oas", "ig_oas"),
        compute=_spread("hy_oas", "ig_oas"),
        note="credit quality discrimination; widens on genuine risk aversion",
    ),
    Derived(
        id="credit_excess",
        name="HY Credit Excess (duration-hedged)",
        asset_class="credit",
        change_unit="pct",
        unit_label="%",
        inputs=("hyg", "ust_5y"),
        compute=_credit_excess,
        note="same-day credit proxy; FRED OAS is authoritative but T+1",
    ),
    Derived(
        id="rsp_spy",
        name="Equal Weight / Cap Weight",
        asset_class="equity",
        change_unit="pct",
        unit_label="%",
        inputs=("rsp", "spy"),
        compute=_ratio("rsp", "spy", "rsp_spy"),
        note="breadth proxy; falling = narrow mega-cap leadership",
    ),
    Derived(
        id="vix_term",
        name="VIX / VIX3M Term Structure",
        asset_class="vol",
        change_unit="pct",
        unit_label="%",
        inputs=("vix", "vix3m"),
        compute=_ratio("vix", "vix3m", "vix_term"),
        note="above 1.0 = inverted = acute near-term stress",
    ),
    Derived(
        id="copper_gold",
        name="Copper / Gold Ratio",
        asset_class="commodity",
        change_unit="pct",
        unit_label="%",
        inputs=("copper", "gold"),
        compute=_ratio("copper", "gold", "copper_gold"),
        note="growth versus haven demand; tends to track real yields",
    ),
)

BY_ID = {d.id: d for d in DERIVED}


def build(wide: pd.DataFrame, registry: Registry) -> dict[str, pd.Series]:
    """Compute every derived series whose inputs are present.

    Silently skips those with missing inputs rather than raising: credit spreads
    do not exist before August 2023, and a 2020 replay must still produce a tape.
    """
    out: dict[str, pd.Series] = {}
    for spec in DERIVED:
        if not spec.available(wide):
            continue
        try:
            series = spec.compute(wide, registry).dropna()
        except (KeyError, ValueError, ZeroDivisionError):
            continue
        if not series.empty:
            out[spec.id] = series
    return out


def unavailable(wide: pd.DataFrame) -> list[str]:
    """Derived ids that cannot be computed from the supplied levels."""
    return [d.id for d in DERIVED if not d.available(wide)]
