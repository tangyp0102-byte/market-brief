"""Tests for mb.page and mb.daily."""

from __future__ import annotations

import datetime as dt
import sys

import pandas as pd
import pytest

sys.path.insert(0, "tests")

from mb import brief, classify, daily, page, rolls, store, tape
from mb.registry import load_registry


@pytest.fixture(scope="module")
def history():
    from test_pipeline_smoke import synthetic_history
    return store.validate(synthetic_history())


@pytest.fixture(scope="module")
def pieces(history):
    session = history["date"].max()
    flags = rolls.empty_flags()
    built = tape.build_tape(history, session=session, flags=flags, run_gate=False)
    result = classify.classify(history, session=session, flags=flags,
                               intensity_threshold=0.0)
    sheet = brief.build_factsheet(history, session=session, flags=flags)
    written = brief.generate(sheet, call_model=lambda p: (
        "WHAT MOVED\nThings moved.\n\nLIKELY DRIVER\nThe pattern is consistent "
        "with something.\n\nWHAT TO WATCH\nMore of the same."
    ))
    return built, result, written


# ------------------------------------------------------------------- the page

def test_page_renders_valid_standalone_html(pieces):
    html = page.render_page(*pieces)
    assert html.startswith("<!doctype html>")
    assert html.rstrip().endswith("</html>")
    assert "<script" not in html          # no JavaScript by design


def test_tape_appears_before_the_commentary(pieces):
    """The checkable layer must be read first."""
    html = page.render_page(*pieces)
    boundary = html.index("Below: commentary, not record")
    assert html.index("Signals &mdash; one per bloc") < boundary
    assert html.index("Rates") < boundary
    assert html.index("class='commentary'") > boundary


def test_boundary_label_is_present(pieces):
    html = page.render_page(*pieces)
    assert "Below: commentary, not record" in html


def test_commentary_carries_its_own_caveat(pieces):
    """A model-written narrative must carry the caveat; a template need not."""
    built, result, _ = pieces
    written = brief.Brief(
        session=built.session, headline="X", body="WHAT MOVED\nThings moved.",
        generated=True,
    )
    html = page.render_page(built, result, written)
    assert "Read it as commentary" in html
    assert "no news, no events, no policy" in html


def test_template_output_says_no_model_was_used(pieces):
    built, result, _ = pieces
    written = brief.Brief(
        session=built.session, headline="X", body="Quiet.", generated=False,
    )
    html = page.render_page(built, result, written)
    assert "without a language model" in html


def test_instrument_names_link_to_tradingview(pieces):
    html = page.render_page(*pieces)
    assert "tradingview.com/chart" in html
    assert 'rel="noopener"' in html


def test_withheld_narrative_is_shown_as_withheld(pieces):
    """Withholding must be visible on the page, and must not hide the tape."""
    built, result, _ = pieces
    written = brief.Brief(
        session=built.session, headline="X",
        body="Largest moves by z-score:", generated=False,
        unverified_numbers=[99.87],
    )
    html = page.render_page(built, result, written)
    assert "withheld" in html.lower()
    assert "99.87" in html
    assert "The tape above is unaffected" in html
    assert "Signals &mdash; one per bloc" in html


def test_page_escapes_content(pieces):
    built, result, written = pieces
    written.body = "<script>alert('x')</script>"
    html = page.render_page(built, result, written)
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_page_is_responsive_and_respects_reduced_motion(pieces):
    html = page.render_page(*pieces)
    assert "@media (max-width: 620px)" in html
    assert "prefers-reduced-motion" in html
    assert "focus-visible" in html


def test_write_page_creates_the_file(tmp_path, pieces):
    path = page.write_page(tmp_path / "site" / "index.html", *pieces)
    assert path.exists()
    assert "<!doctype html>" in path.read_text(encoding="utf-8")


# ------------------------------------------------------------------ the run

def test_non_trading_day_does_nothing(tmp_path):
    result = daily.run(
        store_path=tmp_path / "none.parquet",
        session=dt.date(2024, 12, 25),      # Christmas
    )
    assert not result.ran
    assert "not an NYSE session" in result.messages[0]


def test_rollcheck_triggers_only_on_a_large_commodity_move(history):
    session = history["date"].max()
    assert not daily.needs_rollcheck(history, session)

    moved = history.copy()
    base = float(
        moved[(moved.instrument_id == "wti") & (moved.date < session)]["value"].iloc[-1]
    )
    moved.loc[
        (moved.instrument_id == "wti") & (moved.date == session), "value"
    ] = base * 0.85                          # -15%
    assert daily.needs_rollcheck(moved, session)


def test_commodity_moves_reports_each_contract(history):
    moves = daily.commodity_moves(history, history["date"].max())
    assert "wti" in moves and "gold" in moves
    assert "natgas" in moves                 # collected, though not classified


def test_daily_run_writes_page_and_archive(tmp_path, history):
    store_path = tmp_path / "history.parquet"
    store.upsert(store_path, history)

    result = daily.run(
        store_path=store_path,
        site_dir=tmp_path / "site",
        briefs_dir=tmp_path / "briefs",
        session=history["date"].max().date(),
        force=True, skip_fetch=True,
        call_model=lambda p: "WHAT MOVED\nQuiet.\n",
    )
    assert result.ran
    assert result.page_path.exists()
    assert result.json_path.exists()
    assert (tmp_path / "site" / "index.html").exists()


def test_daily_run_is_idempotent(tmp_path, history):
    store_path = tmp_path / "history.parquet"
    store.upsert(store_path, history)
    kwargs = dict(
        store_path=store_path, site_dir=tmp_path / "site",
        briefs_dir=tmp_path / "briefs",
        session=history["date"].max().date(),
        force=True, skip_fetch=True, call_model=lambda p: "WHAT MOVED\nQuiet.\n",
    )
    first = daily.run(**kwargs)
    second = daily.run(**kwargs)
    assert first.page_path == second.page_path
    assert len(list((tmp_path / "briefs").glob("*.json"))) == 1


def test_workflow_guards_publication_with_the_test_suite():
    """A broken pipeline must not be able to publish."""
    from pathlib import Path
    wf = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "daily.yml"
    text = wf.read_text()
    assert "pytest" in text
    tests_at = text.index("pytest")
    run_at = text.index("python -m mb.daily")
    assert tests_at < run_at, "tests must run before the pipeline"
    assert "workflow_dispatch" in text        # manual re-run after a failure


def test_workflow_pins_the_timezone_correctly():
    """timezone must be a sibling of cron INSIDE the list item, not under on.schedule."""
    import yaml
    from pathlib import Path
    wf = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "daily.yml"
    doc = yaml.safe_load(wf.read_text())
    schedule = doc[True]["schedule"]        # YAML parses bare `on:` as True
    assert len(schedule) == 1
    entry = schedule[0]
    assert entry["timezone"] == "America/New_York"
    assert entry["cron"] == "0 18 * * 1-5"


def test_workflow_caches_the_history_store():
    """Without this the runner starts empty and every z-score is NaN."""
    from pathlib import Path
    wf = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "daily.yml"
    text = wf.read_text()
    assert "actions/cache" in text
    assert "data/history.parquet" in text
    assert "restore-keys" in text


def test_history_store_is_not_gitignored():
    """It was, and the first scheduled run would have published a page of NaNs."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    ignored = (root / ".gitignore").read_text()
    assert "data/*.parquet" not in ignored
    assert "data/roll_flags.csv" not in ignored.split("# NOT ignored")[0]


def test_daily_rebuilds_when_the_store_is_too_thin():
    """Cache eviction must trigger a rebuild, not an empty page."""
    assert daily.MIN_SESSIONS > 250          # the volatility floor window
    source = open(daily.__file__).read()
    assert "rebuilding the full" in source


def test_lookback_stretches_to_cover_a_stale_store(tmp_path, history, monkeypatch):
    """A fixed window leaves a hole after a missed run or a cache eviction.

    Asserted by capturing the start date actually handed to the backfill,
    rather than by matching source text: the previous version of this test
    checked for a literal `pd.Timedelta` call and broke the moment that was
    swapped for the stdlib equivalent, without the behaviour changing at all.
    """
    from mb import backfill

    store_path = tmp_path / "history.parquet"
    store.upsert(store_path, history)
    last_held = history["date"].max()

    captured = {}

    def spy(registry, start, end, *args, **kwargs):
        captured["start"] = start
        return backfill.BackfillReport()

    monkeypatch.setattr(backfill, "backfill", spy)

    # Pretend the run is happening well after the store was last updated.
    stale_by = 40
    session = (last_held + dt.timedelta(days=stale_by)).date()
    daily.run(
        store_path=store_path, site_dir=tmp_path / "site",
        briefs_dir=tmp_path / "briefs", session=session, force=True,
        call_model=lambda p: "WHAT MOVED\nQuiet.\n",
    )

    start = dt.date.fromisoformat(captured["start"])
    # Must reach back past the last stored session, not just LOOKBACK_DAYS.
    assert start < last_held.date()
    assert start > (last_held - dt.timedelta(days=10)).date()


def test_blocked_session_does_not_overwrite_the_index(tmp_path, history):
    """A holiday or forced weekend run must not replace the last good page."""
    store_path = tmp_path / "history.parquet"
    store.upsert(store_path, history)
    site = tmp_path / "site"
    kwargs = dict(
        store_path=store_path, site_dir=site, briefs_dir=tmp_path / "briefs",
        force=True, skip_fetch=True, call_model=lambda p: "WHAT MOVED\nQuiet.\n",
    )

    good = daily.run(session=history["date"].max().date(), **kwargs)
    assert good.ran
    original = (site / "index.html").read_text(encoding="utf-8")

    # Christmas: the gate blocks, so the index must be left alone.
    blocked = daily.run(session=dt.date(2024, 12, 25), **kwargs)
    assert (site / "index.html").read_text(encoding="utf-8") == original
    assert any("left pointing at the previous" in m for m in blocked.messages)


def test_workflow_allows_a_manual_session_and_force():
    """Testing the full pipeline should not require waiting for a weekday."""
    import yaml
    from pathlib import Path
    wf = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "daily.yml"
    doc = yaml.safe_load(wf.read_text())
    inputs = doc[True]["workflow_dispatch"]["inputs"]
    assert "session" in inputs
    assert "force" in inputs


# ---------------------------------------------------------------- navigation

ARCHIVE = [
    {"date": "2026-08-03", "regime": "hawkish_repricing",
     "regime_name": "Hawkish repricing", "intensity": 1.67},
    {"date": "2026-07-31", "regime": None,
     "regime_name": "No clean signal", "intensity": 1.28},
    {"date": "2026-07-30", "regime": None,
     "regime_name": "No clean signal", "intensity": 1.23},
    {"date": "2026-06-23", "regime": "growth_scare",
     "regime_name": "Growth scare", "intensity": 0.98},
]


def _with_archive(pieces, session="2026-07-31"):
    built, result, written = pieces
    built.session = dt.date.fromisoformat(session)
    return page.render_page(built, result, written, archive=ARCHIVE)


def test_archive_uses_no_javascript(pieces):
    """A select needs JS to navigate; fetch fails on file:// URLs. Neither works."""
    html = _with_archive(pieces)
    assert "<script" not in html
    assert "details class=\"archive\"" in html


def test_archive_lists_every_analysed_session(pieces):
    html = _with_archive(pieces)
    assert "All 4 analysed sessions" in html
    for entry in ARCHIVE:
        if entry["date"] != "2026-07-31":
            assert f'href="{entry["date"]}.html"' in html


def test_archive_groups_by_month(pieces):
    html = _with_archive(pieces)
    assert "August 2026" in html
    assert "July 2026" in html
    assert "June 2026" in html


def test_archive_shows_the_regime_for_each_session(pieces):
    """A wall of dates is unscannable; the label is what makes it browsable."""
    html = _with_archive(pieces)
    assert "Hawkish repricing" in html
    assert "Growth scare" in html
    assert "no signal" in html


def test_current_session_is_marked_not_linked(pieces):
    html = _with_archive(pieces)
    assert 'href="2026-07-31.html"' not in html
    assert "viewing" in html


def test_prev_next_navigation_points_at_neighbours(pieces):
    html = _with_archive(pieces)
    nav = html[html.index('class="sessions"'):html.index('class="eyebrow"')]
    assert "2026-07-30" in nav      # older
    assert "2026-08-03" in nav      # newer


def test_newest_session_links_forward_to_index(pieces):
    html = _with_archive(pieces, session="2026-08-03")
    nav = html[html.index('class="sessions"'):html.index('class="eyebrow"')]
    assert 'href="index.html"' in nav


def test_page_records_when_it_was_generated(pieces):
    html = _with_archive(pieces)
    assert "Generated" in html and "UTC" in html


def test_archive_accepts_bare_date_strings(pieces):
    """Callers holding only filenames should still get working navigation."""
    built, result, written = pieces
    html = page.render_page(built, result, written,
                            archive=["2026-07-30", "2026-07-29"])
    assert 'href="2026-07-30.html"' in html


def test_build_archive_reads_regime_from_stored_briefs(tmp_path):
    import json
    for date, rid, name in [("2026-07-30", None, "No clean signal"),
                            ("2026-07-31", "hawkish_repricing", "Hawkish repricing")]:
        (tmp_path / f"{date}.json").write_text(json.dumps(
            {"factsheet": {"regime": {"id": rid, "name": name, "intensity": 1.5}}}))

    entries = daily.build_archive(tmp_path)
    assert [e["date"] for e in entries] == ["2026-07-31", "2026-07-30"]
    assert entries[0]["regime"] == "hawkish_repricing"


def test_build_archive_skips_unreadable_briefs(tmp_path):
    """A half-written file must not become a dead link."""
    (tmp_path / "2026-07-31.json").write_text("{not json")
    (tmp_path / "2026-07-30.json").write_text(
        '{"factsheet": {"regime": {"id": null, "name": "No clean signal"}}}')
    entries = daily.build_archive(tmp_path)
    assert [e["date"] for e in entries] == ["2026-07-30"]
