"""Source adapters. Each returns a tidy DataFrame: date, tenor/symbol, value.

Parsing is separated from fetching throughout, so the parsers can be unit-tested
against fixture payloads without any network access.

ENDPOINT CAVEAT: the Treasury.gov CSV URLs below are stable but have changed
format before. Run `python -m mb.backfill --check-sources` after cloning to
confirm they still parse; the error messages will tell you what broke.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd

from .http import FetchError, fetch_text

log = logging.getLogger(__name__)

TREASURY_CSV = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all"
)
TREASURY_TYPES = {
    "treasury_par_nominal": "daily_treasury_yield_curve",
    "treasury_par_real": "daily_treasury_real_yield_curve",
}

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
STOOQ_URL = "https://stooq.com/q/d/l/"


# --------------------------------------------------------------------- Treasury

def parse_treasury_csv(body: str) -> pd.DataFrame:
    """Parse a Treasury daily rates CSV into long format.

    Nominal columns look like: Date, "1 Mo", "3 Mo", "2 Yr", "10 Yr", "30 Yr"...
    Real columns look like:    Date, "5 YR", "10 YR", "20 YR", "30 YR"
    Tenor labels are preserved verbatim as the registry `symbol`.
    """
    frame = pd.read_csv(io.StringIO(body))

    if "Date" not in frame.columns:
        raise FetchError(
            f"Treasury CSV has no 'Date' column; got {list(frame.columns)[:8]}. "
            "The endpoint format has changed."
        )

    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"])

    tenor_cols = [c for c in frame.columns if c != "Date"]
    if not tenor_cols:
        raise FetchError("Treasury CSV contained no tenor columns")

    long = frame.melt(
        id_vars=["Date"], value_vars=tenor_cols, var_name="symbol", value_name="value"
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    long = long.rename(columns={"Date": "date"})
    long["date"] = long["date"].dt.normalize()
    return long[["date", "symbol", "value"]].sort_values(["date", "symbol"])


def fetch_treasury(
    provider: str,
    years: list[int],
    raw_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch Treasury par yield curves for the given calendar years."""
    if provider not in TREASURY_TYPES:
        raise ValueError(f"unknown treasury provider: {provider!r}")

    curve_type = TREASURY_TYPES[provider]
    frames = []
    for year in sorted(set(years)):
        body = fetch_text(
            TREASURY_CSV.format(year=year),
            params={
                "type": curve_type,
                "field_tdr_date_value": str(year),
                "page": "",
                "_format": "csv",
            },
            raw_dir=raw_dir,
            label=f"{provider}_{year}",
        )
        frames.append(parse_treasury_csv(body))

    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "value"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["date", "symbol"], keep="last"
    )


# ------------------------------------------------------------------------ FRED

def parse_fred_json(payload: dict, series_id: str) -> pd.DataFrame:
    """Parse a FRED observations payload. FRED encodes missing values as '.'."""
    if "observations" not in payload:
        raise FetchError(
            f"FRED response for {series_id} has no 'observations' key "
            f"(got {sorted(payload)[:5]}). Check the API key."
        )

    rows = payload["observations"]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["date", "symbol", "value"])

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["value"] = pd.to_numeric(
        frame["value"].replace(".", pd.NA), errors="coerce"
    )
    frame = frame.dropna(subset=["date", "value"])
    frame["symbol"] = series_id
    return frame[["date", "symbol", "value"]].sort_values("date")


# Values people paste by mistake instead of a real key.
_FRED_PLACEHOLDERS = {
    "your_key", "your_key_here", "paste_your_key_here", "your_api_key",
    "yourkey", "api_key", "xxx", "changeme", "todo", "none", "null",
}

_FRED_KEY_HELP = (
    "Get a free key at https://fred.stlouisfed.org -> My Account -> API Keys. "
    "It is a 32-character lowercase alphanumeric string. Set it with:\n"
    '  PowerShell : $env:FRED_API_KEY = "your32charkey"\n'
    "  macOS/Linux: export FRED_API_KEY=your32charkey"
)


def validate_fred_key(api_key: str | None) -> str:
    """Fail fast on a missing or obviously-placeholder key.

    Worth doing up front: FRED answers a bad key with HTTP 400, not 401, and
    without this check a whole backfill burns three retries per series against
    a key that was never going to work.
    """
    if not api_key or not api_key.strip():
        raise ValueError(f"FRED_API_KEY is not set.\n{_FRED_KEY_HELP}")

    key = api_key.strip()
    if key.lower() in _FRED_PLACEHOLDERS:
        raise ValueError(
            f"FRED_API_KEY is set to the placeholder text {key!r}, not a real key.\n"
            f"{_FRED_KEY_HELP}"
        )
    if len(key) != 32:
        raise ValueError(
            f"FRED_API_KEY is {len(key)} characters; FRED keys are exactly 32. "
            "Check for a partial paste or stray quotes.\n"
            f"{_FRED_KEY_HELP}"
        )
    return key


def fetch_fred(
    series_ids: list[str],
    api_key: str,
    start: str,
    end: str,
    raw_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch FRED series. Requires a free API key from fred.stlouisfed.org."""
    import json

    key = validate_fred_key(api_key)

    frames = []
    for series_id in series_ids:
        try:
            body = fetch_text(
                FRED_URL,
                params={
                    "series_id": series_id,
                    "api_key": key,
                    "file_type": "json",
                    "observation_start": start,
                    "observation_end": end,
                },
                raw_dir=raw_dir,
                label=f"fred_{series_id}",
            )
        except FetchError as exc:
            if "HTTP 400" in str(exc):
                raise FetchError(
                    f"FRED rejected the request for {series_id} with HTTP 400. "
                    "FRED returns 400 (not 401) for an invalid API key, so this "
                    f"almost always means the key itself is wrong.\n{_FRED_KEY_HELP}"
                ) from None
            raise
        frames.append(parse_fred_json(json.loads(body), series_id))

    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "value"])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------- Stooq

def parse_stooq_csv(body: str, symbol: str) -> pd.DataFrame:
    """Parse Stooq daily CSV (Date,Open,High,Low,Close,Volume) into OHLCV long form."""
    if body.strip().lower().startswith("no data"):
        raise FetchError(f"Stooq returned no data for {symbol!r}")

    frame = pd.read_csv(io.StringIO(body))
    if "Date" not in frame.columns or "Close" not in frame.columns:
        raise FetchError(
            f"Stooq CSV for {symbol!r} missing expected columns; got {list(frame.columns)}"
        )

    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["Date"])

    field_map = {
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    }
    present = {k: v for k, v in field_map.items() if k in frame.columns}

    long = frame.melt(
        id_vars=["Date"], value_vars=list(present),
        var_name="raw_field", value_name="value",
    )
    long["field"] = long["raw_field"].map(present)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    long = long.rename(columns={"Date": "date"})
    long["symbol"] = symbol
    return long[["date", "symbol", "field", "value"]].sort_values(["date", "field"])


def fetch_stooq(symbol: str, raw_dir: str | Path | None = None) -> pd.DataFrame:
    """Fetch full daily history for one Stooq symbol (e.g. 'spy.us').

    Stooq serves this CSV endpoint to browsers but applies a daily download quota
    and sometimes answers automated clients with an HTML notice instead. It is a
    fallback source only, so a failure here is not fatal: every instrument that
    lists stooq also lists yahoo. If it stays blocked, drop stooq from the
    registry rather than escalating around the block.
    """
    try:
        body = fetch_text(
            STOOQ_URL,
            params={"s": symbol, "i": "d"},
            headers={"Accept": "text/csv,text/plain,*/*"},
            raw_dir=raw_dir,
            label=f"stooq_{symbol}",
        )
    except FetchError as exc:
        if "HTML page" in str(exc):
            raise FetchError(
                f"Stooq served HTML rather than CSV for {symbol!r}. This is usually "
                "the daily download quota or a block on automated clients, not a "
                "format change. Check data/raw/ for the archived page, and confirm "
                f"in a browser: https://stooq.com/q/d/l/?s={symbol}&i=d"
            ) from None
        raise
    return parse_stooq_csv(body, symbol)


# ----------------------------------------------------------------------- CBOE

# CBOE publishes daily history for its own indices. Preferred over Yahoo for the
# VIX family for the same reason Treasury.gov is preferred for yields: it is the
# index publisher rather than a redistributor.
CBOE_HISTORY = (
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"
)


def parse_cboe_csv(body: str, symbol: str) -> pd.DataFrame:
    """Parse a CBOE index history CSV into OHLC long form.

    Expected columns are DATE, OPEN, HIGH, LOW, CLOSE, but CBOE has varied the
    capitalisation over time, so matching is case-insensitive.
    """
    frame = pd.read_csv(io.StringIO(body))
    lookup = {str(c).strip().upper(): c for c in frame.columns}

    if "DATE" not in lookup:
        raise FetchError(
            f"CBOE CSV for {symbol!r} has no DATE column; got {list(frame.columns)[:8]}. "
            "The endpoint format has changed."
        )
    if "CLOSE" not in lookup:
        raise FetchError(
            f"CBOE CSV for {symbol!r} has no CLOSE column; got {list(frame.columns)[:8]}"
        )

    frame["_date"] = pd.to_datetime(frame[lookup["DATE"]], errors="coerce")
    frame = frame.dropna(subset=["_date"])
    frame["_date"] = frame["_date"].dt.normalize()

    wanted = {"OPEN": "open", "HIGH": "high", "LOW": "low", "CLOSE": "close"}
    present = {lookup[k]: v for k, v in wanted.items() if k in lookup}

    long = frame.melt(
        id_vars=["_date"], value_vars=list(present),
        var_name="raw_field", value_name="value",
    )
    long["field"] = long["raw_field"].map(present)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    long = long.rename(columns={"_date": "date"})
    long["symbol"] = symbol
    return long[["date", "symbol", "field", "value"]].sort_values(["date", "field"])


def fetch_cboe(symbol: str, raw_dir: str | Path | None = None) -> pd.DataFrame:
    """Fetch full daily history for a CBOE index (e.g. 'VIX3M', 'VIX')."""
    body = fetch_text(
        CBOE_HISTORY.format(symbol=symbol),
        raw_dir=raw_dir,
        label=f"cboe_{symbol}",
    )
    return parse_cboe_csv(body, symbol)


# ---------------------------------------------------------------------- Yahoo

def normalise_yahoo_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Convert a yfinance OHLCV frame to long form.

    Deliberately uses raw Close, not Adj Close. Adjusted prices are restated
    retroactively on every dividend, which would make a stored history mutate
    under you and silently change past z-scores.
    """
    if frame is None or frame.empty:
        raise FetchError(f"Yahoo returned no rows for {symbol!r}")

    df = frame.copy()

    # yfinance returns a MultiIndex when several tickers are requested.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df = df.dropna(subset=[date_col])
    df[date_col] = df[date_col].dt.tz_convert(None).dt.normalize()

    field_map = {
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    }
    present = {k: v for k, v in field_map.items() if k in df.columns}
    if "close" not in present.values():
        raise FetchError(f"Yahoo frame for {symbol!r} has no Close column")

    long = df.melt(
        id_vars=[date_col], value_vars=list(present),
        var_name="raw_field", value_name="value",
    )
    long["field"] = long["raw_field"].map(present)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    long = long.rename(columns={date_col: "date"})
    long["symbol"] = symbol

    # Yahoo reports volume 0 for indices and FX; drop rather than store a fake zero.
    long = long[~((long["field"] == "volume") & (long["value"] <= 0))]
    return long[["date", "symbol", "field", "value"]].sort_values(["date", "field"])


def fetch_yahoo(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLCV for one Yahoo symbol via yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise FetchError("yfinance is not installed; pip install yfinance") from exc

    raw = yf.download(
        symbol,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return normalise_yahoo_frame(raw, symbol)
