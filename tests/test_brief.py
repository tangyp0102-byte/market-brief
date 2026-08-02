"""Tests for mb.brief, especially the numeral validator."""

from __future__ import annotations

import datetime as dt
import sys

import pandas as pd
import pytest

sys.path.insert(0, "tests")

from mb import brief, rolls, store


SHEET = {
    "session": "2026-07-31",
    "gate": {"verdict": "pass", "half_session": False, "missing": [],
             "stale": [], "undefined": [], "review": []},
    "regime": {"id": "growth_scare", "name": "Growth scare / flight to quality",
               "confidence": "mixed", "intensity": 3.20,
               "intensity_percentile": 0.99},
    "signals": [
        {"bloc": "equity", "instrument": "spx", "change": -7.60, "unit": "%",
         "z": -16.42, "percentile": 1.0, "moved": True, "quiet_vol": False},
        {"bloc": "rates", "instrument": "ust_10y", "change": -20.0, "unit": "bp",
         "z": -3.51, "percentile": 1.0, "moved": True, "quiet_vol": False},
    ],
    "confirmations": [
        {"id": "vol_up", "holds": True, "detail": "VIX +12.52pt"},
        {"id": "dollar_up", "holds": False, "detail": "dxy -1.34%"},
    ],
    "largest_moves": [
        {"name": "S&P 500", "change": -7.60, "unit": "%", "z": -16.42,
         "quiet_vol": False},
    ],
    "notes": ["equities and yields fall together"],
}


def quiet_sheet() -> dict:
    sheet = {k: (v.copy() if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
             for k, v in SHEET.items()}
    sheet["regime"] = {"id": None, "name": "No clean signal", "confidence": "none",
                       "intensity": 0.55, "intensity_percentile": 0.31}
    sheet["confirmations"] = []
    return sheet


# ------------------------------------------------------------- number extraction

def test_extract_handles_signs_decimals_and_separators():
    found = brief.extract_numbers("SPX -7.60%, 10y +20bp, index 7,489.72, flat 0")
    assert -7.60 in found
    assert 20.0 in found
    assert 7489.72 in found
    assert 0.0 in found


def test_extract_ignores_words():
    assert brief.extract_numbers("all four confirmations held") == []


# ------------------------------------------------------------------ validation

def test_figures_from_the_factsheet_pass():
    text = ("The S&P 500 fell 7.60% while the 10-year yield dropped 20 basis "
            "points. Intensity reached 3.20.")
    assert brief.unverified_numbers(text, SHEET) == []


def test_an_invented_figure_is_caught():
    """The failure mode that matters: fluent prose with a fabricated number."""
    text = "The 10-year yield fell 12 basis points."   # it fell 20
    bad = brief.unverified_numbers(text, SHEET)
    assert 12.0 in bad


def test_a_plausible_but_wrong_number_is_caught():
    text = "The S&P 500 fell 7.85%."                   # it fell 7.60
    assert 7.85 in brief.unverified_numbers(text, SHEET)


def test_rounding_within_tolerance_is_accepted():
    text = "The S&P fell 7.6% and intensity was 3.2."
    assert brief.unverified_numbers(text, SHEET) == []


def test_percentiles_may_be_written_as_percentages():
    """0.99 in the sheet is naturally written as 99% in prose."""
    text = "Intensity sat above 99% of sessions."
    assert brief.unverified_numbers(text, SHEET) == []


def test_the_session_date_is_allowed():
    text = "On 2026-07-31 the tape was ugly."
    assert brief.unverified_numbers(text, SHEET) == []


def test_numbers_inside_confirmation_details_are_allowed():
    text = "VIX rose 12.52 points while the dollar index fell 1.34%."
    assert brief.unverified_numbers(text, SHEET) == []


# --------------------------------------------------------------- quiet days

def test_quiet_session_never_calls_the_model():
    called = []

    def spy(prompt):
        called.append(prompt)
        return "should not happen"

    result = brief.generate(quiet_sheet(), call_model=spy)
    assert called == []
    assert not result.generated
    assert "No clean signal" in result.headline


def test_quiet_template_contains_no_interpretation():
    _, body = brief.quiet_template(quiet_sheet())
    lowered = body.lower()
    for word in ("suggests", "driven by", "because", "risk-off", "investors",
                 "sentiment", "fear"):
        assert word not in lowered


def test_quiet_template_states_that_most_days_are_noise():
    _, body = brief.quiet_template(quiet_sheet())
    assert "noise" in body.lower()


def test_blocked_gate_produces_no_brief():
    sheet = quiet_sheet()
    sheet["gate"]["verdict"] = "blocked"
    result = brief.generate(sheet, call_model=lambda p: "x")
    assert not result.generated
    assert "No brief" in result.headline


# ------------------------------------------------------------------ narrative

def test_clean_narrative_is_published():
    text = ("WHAT MOVED\nThe S&P 500 fell 7.60% and the 10-year yield dropped "
            "20 basis points.\n\nLIKELY DRIVER\nThe pattern is consistent with "
            "a flight to quality.\n\nWHAT TO WATCH\nWhether the dollar follows.")
    result = brief.generate(SHEET, call_model=lambda p: text)
    assert result.generated
    assert result.valid
    assert "7.60%" in result.body


def test_narrative_with_an_invented_number_is_withheld():
    text = "WHAT MOVED\nThe S&P 500 fell 7.60% and oil collapsed 31.40%."
    result = brief.generate(SHEET, call_model=lambda p: text)
    assert not result.generated
    assert not result.valid
    assert 31.40 in result.unverified_numbers
    assert "oil collapsed" not in result.body      # the prose is not published
    assert "WITHHELD" in result.render()


def test_model_failure_falls_back_to_the_template():
    def boom(prompt):
        raise RuntimeError("network down")

    result = brief.generate(SHEET, call_model=boom)
    assert not result.generated
    assert result.error is not None
    assert "fell back" in result.error
    assert result.body                              # something is still produced


def test_prompt_forbids_inventing_numbers_and_asserting_cause():
    prompt = brief.build_prompt(SHEET)
    lowered = prompt.lower()
    assert "must appear in the fact sheet" in lowered
    assert "cannot know why" in lowered
    assert "hypothesis" in lowered
    assert "no investment advice" in lowered
    assert "counts as words" in lowered


def test_prompt_carries_the_whole_factsheet():
    prompt = brief.build_prompt(SHEET)
    assert "growth_scare" in prompt
    assert "-7.6" in prompt
    assert "dollar_up" in prompt


# ---------------------------------------------------------- end to end shape

@pytest.fixture(scope="module")
def history():
    from test_pipeline_smoke import synthetic_history
    return store.validate(synthetic_history())


def test_factsheet_builds_from_real_history(history):
    sheet = brief.build_factsheet(
        history, session=history["date"].max(), flags=rolls.empty_flags()
    )
    assert sheet["session"]
    assert "regime" in sheet and "signals" in sheet
    assert isinstance(sheet["largest_moves"], list)
    # Must be JSON-serialisable: it is pasted straight into the prompt.
    import json
    json.dumps(sheet)


def test_factsheet_numbers_are_all_traceable(history):
    """Whatever the model quotes, it must be able to find here."""
    sheet = brief.build_factsheet(
        history, session=history["date"].max(), flags=rolls.empty_flags()
    )
    assert len(brief.allowed_values(sheet)) > 10


def test_validator_checks_magnitude_not_direction():
    """A stated limitation, asserted so it cannot be forgotten.

    Prose carries direction in words, so the numeral cannot be sign-matched.
    A reversed direction passes this check; the prompt is the only defence.
    """
    reversed_direction = "The S&P 500 rose 7.60%."     # it fell
    assert brief.unverified_numbers(reversed_direction, SHEET) == []

    wrong_magnitude = "The S&P 500 fell 8.60%."
    assert brief.unverified_numbers(wrong_magnitude, SHEET) == [8.60]


def test_withheld_narrative_keeps_the_classification():
    """Withholding the prose must not also deny the regime that was found."""
    text = "WHAT MOVED\nThe S&P fell 7.60% and crude collapsed 31.40%."
    result = brief.generate(SHEET, call_model=lambda p: text)
    assert "Growth scare" in result.headline
    assert "No cross-asset signature" not in result.body
    assert "S&P 500" in result.body


def test_quiet_day_body_does_say_no_signature():
    _, body = brief.quiet_template(quiet_sheet())
    assert "No cross-asset signature" in body


def test_factual_summary_makes_no_regime_claim():
    body = brief.factual_summary(SHEET)
    assert "signature" not in body.lower()
    assert "noise" not in body.lower()


def test_default_model_is_overridable_by_environment():
    """Model IDs go stale; the default must not be the only way to set one."""
    import os
    assert brief.DEFAULT_MODEL
    assert "MB_MODEL" in open(brief.__file__).read()


# ------------------------------------------------- identifiers vs quantities

def test_index_names_are_not_treated_as_quoted_figures():
    """'S&P 500' withheld a correct brief because 500 is in the name."""
    for name in ["The S&P 500 fell sharply.",
                 "The Nasdaq 100 led losses.",
                 "The Russell 2000 lagged.",
                 "VIX3M rose."]:
        assert brief.unverified_numbers(name, SHEET) == [], name


def test_tenor_references_are_not_treated_as_quoted_figures():
    for text in ["The 10-year yield fell 20 basis points.",
                 "The 2y led the move.",
                 "The 30-year underperformed.",
                 "3-month bills were steady.",
                 "The 2s10s curve steepened."]:
        assert brief.unverified_numbers(text, SHEET) == [], text


def test_a_real_quantity_of_500_is_still_checked():
    """Stripping names must not create a hole for genuine numbers."""
    assert 500.0 in brief.unverified_numbers("Yields rose 500 basis points.", SHEET)
    assert 250.0 in brief.unverified_numbers("The index gained 250 points.", SHEET)
    # 100 is legitimately allowed here: a percentile of 1.0 reads as "100%".
    assert brief.unverified_numbers("It sat above 100% of sessions.", SHEET) == []


def test_identifier_stripping_preserves_surrounding_figures():
    text = "The S&P 500 fell 7.60% and the 10-year yield dropped 20 basis points."
    assert brief.unverified_numbers(text, SHEET) == []
    stripped = brief.strip_identifiers(text)
    assert "7.60" in stripped and "20" in stripped
    assert "500" not in stripped


def test_prompt_forbids_inventing_concepts_and_moving_figures():
    """Two failures the numeral validator is structurally blind to."""
    lowered = brief.build_prompt(SHEET).lower()
    assert "attach every figure to the instrument it belongs to" in lowered
    assert "do not transfer a number from one instrument to another" in lowered
    assert "term premium" in lowered          # named as a forbidden example
    assert "only discuss what you were given" in lowered


def test_module_documents_the_validator_blind_spots():
    """These limits must stay visible to anyone reading the module."""
    source = open(brief.__file__).read().lower()
    for gap in ("direction", "attribution", "invented concepts"):
        assert gap in source, gap
