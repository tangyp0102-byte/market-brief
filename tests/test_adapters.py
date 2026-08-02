"""Parser tests using fixture payloads shaped like the real source responses.

No network. These cover the parsing half of each adapter, which is where format
drift actually shows up.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mb.sources import adapters
from mb.sources.http import FetchError

TREASURY_NOMINAL_CSV = (
    'Date,"1 Mo","3 Mo","6 Mo","1 Yr","2 Yr","5 Yr","10 Yr","20 Yr","30 Yr"\n'
    "01/02/2024,5.60,5.51,5.32,4.80,4.33,3.93,3.95,4.24,4.08\n"
    "01/03/2024,5.59,5.50,5.31,4.79,4.33,3.91,3.91,4.21,4.05\n"
)

TREASURY_REAL_CSV = (
    'Date,"5 YR","7 YR","10 YR","20 YR","30 YR"\n'
    "01/02/2024,1.79,1.79,1.79,2.00,2.02\n"
    "01/03/2024,1.77,1.77,1.76,1.97,1.99\n"
)

STOOQ_CSV = (
    "Date,Open,High,Low,Close,Volume\n"
    "2024-01-02,472.16,473.67,470.49,472.65,123273000\n"
    "2024-01-03,470.43,471.19,468.17,468.79,103585000\n"
)

FRED_JSON = {
    "observations": [
        {"date": "2024-01-02", "value": "3.95"},
        {"date": "2024-01-03", "value": "."},      # FRED's missing-value marker
        {"date": "2024-01-04", "value": "3.98"},
    ]
}


# -------------------------------------------------------------------- Treasury

def test_parse_treasury_nominal_curve():
    frame = adapters.parse_treasury_csv(TREASURY_NOMINAL_CSV)
    assert set(frame.columns) == {"date", "symbol", "value"}

    ten_year = frame[(frame["symbol"] == "10 Yr")].set_index("date")["value"]
    assert ten_year.loc[pd.Timestamp("2024-01-02")] == pytest.approx(3.95)
    assert ten_year.loc[pd.Timestamp("2024-01-03")] == pytest.approx(3.91)


def test_treasury_tenor_labels_match_the_registry():
    """Registry symbols are the CSV headers verbatim; a mismatch means no data."""
    from mb.registry import load_registry

    reg = load_registry()
    nominal = set(adapters.parse_treasury_csv(TREASURY_NOMINAL_CSV)["symbol"])
    real = set(adapters.parse_treasury_csv(TREASURY_REAL_CSV)["symbol"])

    for inst in reg:
        for src in inst.sources:
            if src.provider == "treasury_par_nominal":
                assert src.symbol in nominal, f"{inst.id}: {src.symbol!r} not a CSV column"
            if src.provider == "treasury_par_real":
                assert src.symbol in real, f"{inst.id}: {src.symbol!r} not a CSV column"


def test_parse_treasury_real_curve_uses_uppercase_yr():
    """Nominal uses '10 Yr', real uses '10 YR'. Easy to get wrong."""
    frame = adapters.parse_treasury_csv(TREASURY_REAL_CSV)
    symbols = set(frame["symbol"])
    assert "10 YR" in symbols
    assert "10 Yr" not in symbols


def test_treasury_csv_without_date_column_raises_clearly():
    with pytest.raises(FetchError, match="no 'Date' column"):
        adapters.parse_treasury_csv("Foo,Bar\n1,2\n")


def test_treasury_parser_skips_blank_cells():
    body = TREASURY_NOMINAL_CSV + "01/04/2024,5.58,,5.30,4.78,4.31,3.90,3.90,4.20,4.04\n"
    frame = adapters.parse_treasury_csv(body)
    three_month = frame[frame["symbol"] == "3 Mo"]
    assert pd.Timestamp("2024-01-04") not in set(three_month["date"])
    ten_year = frame[frame["symbol"] == "10 Yr"]
    assert pd.Timestamp("2024-01-04") in set(ten_year["date"])


# ------------------------------------------------------------------------ FRED

def test_parse_fred_json_drops_the_dot_placeholder():
    frame = adapters.parse_fred_json(FRED_JSON, "DGS10")
    assert len(frame) == 2
    assert pd.Timestamp("2024-01-03") not in set(frame["date"])
    assert set(frame["symbol"]) == {"DGS10"}


def test_parse_fred_json_without_observations_raises():
    with pytest.raises(FetchError, match="no 'observations'"):
        adapters.parse_fred_json({"error_message": "Bad API key"}, "DGS10")


def test_parse_fred_json_empty_observations():
    frame = adapters.parse_fred_json({"observations": []}, "DGS10")
    assert frame.empty


# ----------------------------------------------------------------------- Stooq

def test_parse_stooq_csv_yields_ohlcv():
    frame = adapters.parse_stooq_csv(STOOQ_CSV, "spy.us")
    assert set(frame["field"]) == {"open", "high", "low", "close", "volume"}
    close = frame[(frame["field"] == "close")].set_index("date")["value"]
    assert close.loc[pd.Timestamp("2024-01-02")] == pytest.approx(472.65)


def test_parse_stooq_no_data_response_raises():
    with pytest.raises(FetchError, match="no data"):
        adapters.parse_stooq_csv("No data", "bogus.us")


# ----------------------------------------------------------------------- Yahoo

def _yahoo_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        pd.to_datetime(["2024-01-02", "2024-01-03"]), name="Date"
    )
    return pd.DataFrame(
        {
            "Open": [472.16, 470.43],
            "High": [473.67, 471.19],
            "Low": [470.49, 468.17],
            "Close": [472.65, 468.79],
            "Adj Close": [471.10, 467.25],
            "Volume": [123273000, 103585000],
        },
        index=idx,
    )


def test_normalise_yahoo_uses_raw_close_not_adjusted():
    """Adj Close is restated on every dividend; storing it would mutate history."""
    frame = adapters.normalise_yahoo_frame(_yahoo_frame(), "SPY")
    close = frame[frame["field"] == "close"].set_index("date")["value"]
    assert close.loc[pd.Timestamp("2024-01-02")] == pytest.approx(472.65)
    assert 471.10 not in set(frame["value"])


def test_normalise_yahoo_flattens_multiindex_columns():
    raw = _yahoo_frame()
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["SPY"]])
    frame = adapters.normalise_yahoo_frame(raw, "SPY")
    assert "close" in set(frame["field"])


def test_normalise_yahoo_drops_placeholder_zero_volume():
    """Yahoo reports volume 0 for indices and FX; a stored zero is a lie."""
    raw = _yahoo_frame()
    raw["Volume"] = [0, 0]
    frame = adapters.normalise_yahoo_frame(raw, "^GSPC")
    assert "volume" not in set(frame["field"])
    assert "close" in set(frame["field"])


def test_normalise_yahoo_strips_timezone_to_session_date():
    raw = _yahoo_frame()
    raw.index = raw.index.tz_localize("America/New_York")
    frame = adapters.normalise_yahoo_frame(raw, "SPY")
    assert set(frame["date"]) == {
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    }


def test_normalise_yahoo_empty_raises():
    with pytest.raises(FetchError, match="no rows"):
        adapters.normalise_yahoo_frame(pd.DataFrame(), "SPY")


def test_normalise_yahoo_without_close_raises():
    raw = _yahoo_frame().drop(columns=["Close"])
    with pytest.raises(FetchError, match="no Close column"):
        adapters.normalise_yahoo_frame(raw, "SPY")


# --------------------------------------------------- credential handling

def test_redact_masks_api_key_in_urls():
    from mb.sources.http import redact

    url = "https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key=abc123&file_type=json"
    masked = redact(url)
    assert "abc123" not in masked
    assert "<REDACTED>" in masked
    assert "series_id=DGS10" in masked  # non-secret params survive


def test_redact_masks_key_inside_an_exception_message():
    from mb.sources.http import redact

    msg = "400 Client Error for url: https://x.com/a?api_key=deadbeef&b=1"
    assert "deadbeef" not in redact(msg)


def test_fred_key_placeholder_is_rejected():
    with pytest.raises(ValueError, match="placeholder"):
        adapters.validate_fred_key("your_key")


def test_fred_key_missing_is_rejected():
    with pytest.raises(ValueError, match="not set"):
        adapters.validate_fred_key(None)
    with pytest.raises(ValueError, match="not set"):
        adapters.validate_fred_key("   ")


def test_fred_key_wrong_length_is_rejected():
    with pytest.raises(ValueError, match="32"):
        adapters.validate_fred_key("tooshort")


def test_fred_key_valid_shape_is_accepted():
    key = "a" * 32
    assert adapters.validate_fred_key(f"  {key}  ") == key


# ------------------------------------------------------------------ CBOE

CBOE_CSV = (
    "DATE,OPEN,HIGH,LOW,CLOSE\n"
    "1/2/2024,14.35,14.90,13.98,14.12\n"
    "1/3/2024,14.20,15.60,14.11,15.44\n"
)


def test_parse_cboe_csv_yields_ohlc():
    frame = adapters.parse_cboe_csv(CBOE_CSV, "VIX3M")
    assert set(frame["field"]) == {"open", "high", "low", "close"}
    close = frame[frame["field"] == "close"].set_index("date")["value"]
    assert close.loc[pd.Timestamp("2024-01-02")] == pytest.approx(14.12)
    assert close.loc[pd.Timestamp("2024-01-03")] == pytest.approx(15.44)


def test_parse_cboe_csv_is_case_insensitive_on_headers():
    lowered = CBOE_CSV.replace("DATE,OPEN,HIGH,LOW,CLOSE", "Date,Open,High,Low,Close")
    frame = adapters.parse_cboe_csv(lowered, "VIX3M")
    assert len(frame[frame["field"] == "close"]) == 2


def test_parse_cboe_csv_without_close_raises():
    with pytest.raises(FetchError, match="no CLOSE column"):
        adapters.parse_cboe_csv("DATE,OPEN\n1/2/2024,14.35\n", "VIX3M")


def test_parse_cboe_csv_without_date_raises():
    with pytest.raises(FetchError, match="no DATE column"):
        adapters.parse_cboe_csv("FOO,CLOSE\n1,2\n", "VIX3M")
