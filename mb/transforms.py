"""Unit-aware transforms. This module is pure: no I/O, no network.

Everything here is unit-tested, because a sign error or a look-ahead bug in
this file would produce a confident, plausible, wrong brief every single day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 1.4826 rescales the median absolute deviation to be a consistent estimator
# of the standard deviation for normally distributed data.
_MAD_TO_SIGMA = 1.4826

# Official ICE US Dollar Index scaling constant.
DXY_CONSTANT = 50.14348112


def compute_change(prev: float, curr: float, change_unit: str) -> float:
    """Day-over-day change in the instrument's declared change_unit.

    bp  : (curr - prev) * 100    inputs are in percent, e.g. 4.25 -> 4.25%
    pct : (curr / prev - 1) * 100
    pts : (curr - prev)
    """
    if change_unit == "bp":
        return (curr - prev) * 100.0
    if change_unit == "pct":
        if prev == 0:
            raise ZeroDivisionError("cannot compute percent change from a zero base")
        return (curr / prev - 1.0) * 100.0
    if change_unit == "pts":
        return curr - prev
    raise ValueError(f"unknown change_unit: {change_unit!r}")


def change_series(
    levels: pd.Series, change_unit: str, undefined_across_zero: bool = True
) -> pd.Series:
    """Vectorised compute_change over a level series indexed by date.

    The series is sorted by index first. The first observation yields NaN.

    undefined_across_zero (pct only): return NaN where either endpoint is
    non-positive. WTI settled at -$37.63 on 2020-04-20, and computing a percent
    change through zero yields -306%, which is not a number anyone can act on.
    Worse, it sits in the trailing volatility window for three months and
    suppresses every genuine signal after it. Undefined is the honest answer.
    """
    s = levels.sort_index().astype(float)
    if change_unit == "bp":
        out = s.diff() * 100.0
    elif change_unit == "pct":
        out = s.pct_change() * 100.0
        if undefined_across_zero:
            invalid = (s <= 0) | (s.shift(1) <= 0)
            out = out.where(~invalid)
    elif change_unit == "pts":
        out = s.diff()
    else:
        raise ValueError(f"unknown change_unit: {change_unit!r}")
    return out.rename(s.name)


def sign_crossings(levels: pd.Series) -> pd.Series:
    """Dates where the level crossed zero, in either direction."""
    s = levels.sort_index().astype(float).dropna()
    return s[(s.shift(1) * s) < 0]


def dollar_normalise(pct_change: float, dollar_direction: int) -> float:
    """Re-sign an FX percent change so that positive always means a stronger USD.

    EUR/USD falling 1% and USD/JPY rising 1% are both 'dollar up 1%'.
    """
    if dollar_direction not in (1, -1):
        raise ValueError(f"dollar_direction must be +1 or -1, got {dollar_direction!r}")
    return pct_change * dollar_direction


def rolling_zscore(
    changes: pd.Series,
    window: int = 60,
    min_obs: int = 30,
    robust: bool = False,
    vol_floor_window: int | None = 250,
    vol_floor_frac: float = 0.5,
    clip: float | None = None,
) -> pd.Series:
    """Z-score each change against the distribution of the PRECEDING `window` changes.

    Critically the window is shifted by one, so the score for day t never sees
    day t's own value. Without this the scores are contaminated by look-ahead
    and every large move gets shrunk toward the mean.

    robust=True uses median/MAD instead of mean/std. Daily financial changes are
    fat-tailed, so a single crisis day inflates a trailing std and suppresses the
    scores of the days immediately after it. Robust scaling is usually the
    better default for anomaly detection; std is kept for comparability.

    VOLATILITY FLOOR. A 60-session window can collapse during a quiet stretch,
    and then any ordinary move scores as extraordinary. USD/JPY realised 0.19%
    daily vol into 2026-07-31 and a -1.91% move scored -9.8 sigma. The move was
    genuinely notable, but the scale was a sample artefact of a calm period
    rather than a belief about the pair's actual volatility.

    The floor says: do not accept that volatility has fallen below
    `vol_floor_frac` of its `vol_floor_window` level. It is a safety net for a
    genuine vol collapse, not a cure: a moderate quieting still produces a large
    score, because the score is arithmetically correct.

    `clip` caps the magnitude symmetrically. Leave it None for reporting, where
    the honest number belongs, and set it when building a composite across
    instruments, where one quiet-period reading would otherwise dominate every
    other signal. See vol_ratio() for recognising the artefact.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    if min_obs < 2:
        raise ValueError("min_obs must be >= 2")
    if min_obs > window:
        raise ValueError("min_obs cannot exceed window")
    if not 0 < vol_floor_frac <= 1:
        raise ValueError("vol_floor_frac must be in (0, 1]")

    s = changes.sort_index().astype(float)
    prior = s.shift(1)  # exclude the current observation from its own window

    if robust:
        centre = prior.rolling(window, min_periods=min_obs).median()
        abs_dev = (prior - centre).abs()
        scale = abs_dev.rolling(window, min_periods=min_obs).median() * _MAD_TO_SIGMA
    else:
        centre = prior.rolling(window, min_periods=min_obs).mean()
        scale = prior.rolling(window, min_periods=min_obs).std(ddof=1)

    if vol_floor_window and vol_floor_window > window:
        if robust:
            long_centre = prior.rolling(vol_floor_window, min_periods=window).median()
            long_scale = (
                (prior - long_centre).abs()
                .rolling(vol_floor_window, min_periods=window)
                .median()
                * _MAD_TO_SIGMA
            )
        else:
            long_scale = prior.rolling(
                vol_floor_window, min_periods=window
            ).std(ddof=1)
        floor = long_scale * vol_floor_frac
        scale = pd.concat([scale, floor], axis=1).max(axis=1, skipna=True)

    scale = scale.where(scale > 1e-12)  # degenerate scale -> NaN, never inf
    z = ((s - centre) / scale).rename(s.name)
    return z if clip is None else z.clip(-clip, clip)


def vol_ratio(
    changes: pd.Series,
    window: int = 60,
    long_window: int = 250,
) -> pd.Series:
    """Short-window volatility as a fraction of its long-window level.

    Well below 1 means the instrument has been unusually quiet, which inflates
    every z-score computed against that window. Reported alongside the score so
    an extreme reading can be recognised as a vol-regime artefact rather than
    silently smoothed away: the arithmetic is correct, and the honest response is
    to say why it is large, not to shrink it.
    """
    s = changes.sort_index().astype(float)
    prior = s.shift(1)
    short = prior.rolling(window, min_periods=window // 2).std(ddof=1)
    long = prior.rolling(long_window, min_periods=window).std(ddof=1)
    return (short / long.where(long > 1e-12)).rename("vol_ratio")


def dxy_from_components(
    quotes: dict[str, float],
    weights: dict[str, float],
    constant: float = DXY_CONSTANT,
) -> float:
    """Reconstruct the US Dollar Index from its six component crosses.

    quotes and weights are keyed by instrument id. Weight signs follow the
    quote convention: negative for XXX/USD pairs (EUR, GBP), positive for
    USD/XXX pairs (JPY, CAD, SEK, CHF).

    We compute this ourselves rather than pulling a vendor DXY quote because
    free DXY feeds go stale silently, and here every input is independently
    bounds-checked.
    """
    missing = set(weights) - set(quotes)
    if missing:
        raise KeyError(f"missing DXY component quotes: {sorted(missing)}")

    total = abs(sum(abs(w) for w in weights.values()))
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"DXY weights must sum to 1.0 in absolute value, got {total:.6f}")

    index = constant
    for iid, weight in weights.items():
        q = quotes[iid]
        if q <= 0:
            raise ValueError(f"DXY component {iid!r} must be positive, got {q}")
        index *= q**weight
    return index


def dxy_series(
    quote_frame: pd.DataFrame,
    weights: dict[str, float],
    constant: float = DXY_CONSTANT,
) -> pd.Series:
    """Vectorised DXY over a wide frame of component quotes (columns = instrument ids).

    Rows with any missing component are dropped rather than partially computed.
    """
    missing = set(weights) - set(quote_frame.columns)
    if missing:
        raise KeyError(f"missing DXY component columns: {sorted(missing)}")

    frame = quote_frame[list(weights)].dropna(how="any")
    if (frame <= 0).any().any():
        bad = frame.columns[(frame <= 0).any()].tolist()
        raise ValueError(f"DXY components must be positive; non-positive values in {bad}")

    log_index = np.log(constant) + sum(
        weights[col] * np.log(frame[col]) for col in weights
    )
    return np.exp(log_index).rename("dxy")


def curve_spread(long_leg: pd.Series, short_leg: pd.Series) -> pd.Series:
    """Yield curve spread in basis points, aligned on common dates.

    Inputs are yields in percent. Positive means the long leg yields more.
    """
    aligned = pd.concat([long_leg, short_leg], axis=1, join="inner").dropna()
    return ((aligned.iloc[:, 0] - aligned.iloc[:, 1]) * 100.0).rename("spread_bp")


def breakeven(nominal: pd.Series, real: pd.Series) -> pd.Series:
    """Inflation breakeven in basis points: nominal yield minus real yield."""
    return curve_spread(nominal, real).rename("breakeven_bp")


def ratio_series(numerator: pd.Series, denominator: pd.Series, name: str) -> pd.Series:
    """Aligned ratio, used for relative-performance pairs such as RSP/SPY."""
    aligned = pd.concat([numerator, denominator], axis=1, join="inner").dropna()
    if (aligned.iloc[:, 1] == 0).any():
        raise ZeroDivisionError(f"zero denominator in ratio {name!r}")
    return (aligned.iloc[:, 0] / aligned.iloc[:, 1]).rename(name)
