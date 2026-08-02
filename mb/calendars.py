"""Trading calendars.

NYSE is the spine. The brief is a US market-close summary, so a "session" means
an NYSE session: if NYSE is shut there is no brief, regardless of what FX or
futures did overnight.

Within an NYSE session, each instrument is governed by its own calendar. On
Columbus Day and Veterans Day the equity market trades while the bond market is
shut, so a missing Treasury yield on those days is correct behaviour rather than
a data failure. Conflating the two is what makes a naive gate cry wolf ~13 times
per decade and get ignored.

Verified against the stored history: NYSE gives exactly 1905 sessions and
SIFMA_US exactly 1896 for 2019-01-01 to 2026-07-31, matching the equity and
rates observation counts precisely.
"""

from __future__ import annotations

import datetime as dt
import functools

import pandas as pd

from .registry import Instrument

SPINE = "NYSE"


class CalendarError(RuntimeError):
    """Raised when a calendar cannot be resolved."""


@functools.lru_cache(maxsize=None)
def _calendar(name: str):
    try:
        import pandas_market_calendars as mcal
    except ImportError as exc:  # pragma: no cover
        raise CalendarError(
            "pandas_market_calendars is required; pip install pandas_market_calendars"
        ) from exc

    try:
        return mcal.get_calendar(name)
    except Exception as exc:  # noqa: BLE001
        raise CalendarError(f"unknown calendar {name!r}: {exc}") from None


@functools.lru_cache(maxsize=None)
def sessions(name: str, start: str, end: str) -> pd.DatetimeIndex:
    """Trading sessions for a named calendar, as tz-naive normalised dates.

    The FX pseudo-calendar is every weekday: spot FX trades continuously from
    Sunday evening to Friday evening and does not observe US holidays.
    """
    if name == "FX":
        return pd.bdate_range(start, end)

    days = _calendar(name).valid_days(start, end)
    return pd.DatetimeIndex(pd.to_datetime(days).tz_localize(None).normalize())


def spine_sessions(start: str, end: str) -> pd.DatetimeIndex:
    """NYSE sessions: the days on which a brief can exist at all."""
    return sessions(SPINE, start, end)


def is_trading_day(date: dt.date | pd.Timestamp, name: str = SPINE) -> bool:
    """Whether `name` is open on `date`."""
    ts = pd.Timestamp(date).normalize()
    iso = ts.date().isoformat()
    return ts in sessions(name, iso, iso)


@functools.lru_cache(maxsize=None)
def early_closes(start: str, end: str, name: str = SPINE) -> tuple[pd.Timestamp, ...]:
    """Sessions that close before the usual time (1pm ET half days for NYSE).

    The "usual time" is the modal close across the requested window, so the
    window must be wide enough to contain mostly normal sessions. Use
    is_early_close() to test a single date; it widens the window for you.

    Worth surfacing rather than correcting: a half day has thin volume and muted
    ranges, so a z-score computed against full-day volatility understates the
    move. The brief should say "half session" instead of pretending otherwise.
    """
    if name == "FX":
        return ()

    schedule = _calendar(name).schedule(start, end)
    if schedule.empty or "market_close" not in schedule:
        return ()

    local = schedule["market_close"].dt.tz_convert("America/New_York")
    modal = local.dt.time.mode()
    if modal.empty:
        return ()

    flagged = local[local.dt.time != modal[0]]
    return tuple(pd.Timestamp(d).normalize() for d in flagged.index)


def is_early_close(date: dt.date | pd.Timestamp, name: str = SPINE) -> bool:
    """Whether `date` is a half session.

    Widens to the surrounding calendar year before taking the modal close: on a
    one-day window the half day IS the mode, so nothing would ever be flagged.
    """
    ts = pd.Timestamp(date).normalize()
    year = ts.year
    return ts in early_closes(f"{year}-01-01", f"{year}-12-31", name)


def expected_on(instrument: Instrument, session: pd.Timestamp) -> bool:
    """Should this instrument have an observation for this NYSE session?

    False means absence is legitimate, not a failure. This is the single most
    important distinction in the validation gate.
    """
    session = pd.Timestamp(session).normalize()
    if instrument.calendar == SPINE or instrument.calendar == "FX":
        return True
    iso = session.date().isoformat()
    return session in sessions(instrument.calendar, iso, iso)


def expected_instruments(registry, session: pd.Timestamp) -> tuple[Instrument, ...]:
    """Every instrument that should report on this session."""
    return tuple(i for i in registry if expected_on(i, session))


def previous_session(
    session: pd.Timestamp, name: str = SPINE, lookback_days: int = 14
) -> pd.Timestamp | None:
    """The trading session immediately before `session` on the given calendar."""
    session = pd.Timestamp(session).normalize()
    start = (session - dt.timedelta(days=lookback_days)).date().isoformat()
    prior = sessions(name, start, session.date().isoformat())
    prior = prior[prior < session]
    return prior[-1] if len(prior) else None


def sessions_between(
    earlier: pd.Timestamp, later: pd.Timestamp, name: str = SPINE
) -> int:
    """Count trading sessions strictly after `earlier`, up to and including `later`.

    Staleness is measured in trading sessions, not calendar days, so a Friday
    close read on Monday morning is one session old rather than three days old.
    """
    earlier = pd.Timestamp(earlier).normalize()
    later = pd.Timestamp(later).normalize()
    if later <= earlier:
        return 0
    span = sessions(name, earlier.date().isoformat(), later.date().isoformat())
    return int((span > earlier).sum())
