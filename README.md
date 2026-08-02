# market-brief

Daily cross-asset market close summary. Personal tool, free data sources.

**Status: step 1 of 8 complete** — instrument registry, history store, transforms,
and backfill. No live pipeline, classifier, or web page yet.

\---

## Setup

```bash
pip install -r requirements.txt

# Free key: https://fred.stlouisfed.org/docs/api/api\_key.html
export FRED\_API\_KEY=39c2c1e23974acdc8476c0bac7269b0b
# Verify every source endpoint still parses before committing to a long backfill
python -m mb.backfill --check-sources

# Backfill. \~5 years gives a comfortable margin over the 60-day z-score window.
python -m mb.backfill --start 2019-01-01

pytest -q
```

`--check-sources` first is worth the thirty seconds. Free endpoints change format
without notice, and a failure there is far easier to read than a half-finished
backfill.

\---

## Conventions

### Snapshot time

"Market close" is not one moment. This project fixes the convention as:

|Asset class|Source of the daily mark|
|-|-|
|Rates|Treasury.gov par yield curve, struck from \~3:30pm ET quotes|
|Equities, vol, credit ETFs|4:00pm ET close (4:15pm for VIX)|
|Commodities|Front-month settlement|
|FX|Daily bar close from the vendor|
|Credit OAS|FRED, T+1|

Because the pipeline should run after Treasury publishes, the intended cron is
**6:00pm ET**, not 4:15pm.

### Units

Declared per instrument in `config/instruments.yaml` and enforced by the registry
validator:

* `bp` — yields and spreads. `(curr - prev) \* 100`, inputs in percent.
* `pct` — prices, indices, FX.
* `pts` — VIX and other point-quoted vols. A percent change on VIX is technically
true and analytically useless.

`dollar\_direction` is mandatory on every FX instrument and never inferred from the
ticker. Yahoo's short-form symbols are a trap: `EURUSD=X` is EUR/USD but `JPY=X`
is USD/JPY. Getting this wrong inverts every FX read in the brief.

\---

## Design decisions worth knowing

**Long-format store.** One row per `(date, instrument\_id, field)`. Instruments come
and go, and available fields differ by asset class — a Treasury yield has no volume.
Adding an instrument must never require a schema migration. Everything pivots to
wide in memory via `store.wide()`.

**Raw close, never adjusted close.** Adjusted prices are restated retroactively on
every dividend. Storing them would make the history mutate underneath you and
silently change past z-scores.

**Missing means absent, not NaN.** Nulls are dropped before storage, never written.
Out-of-bounds values are dropped and counted in the backfill report. A bad tick that
reaches the store poisons every z-score computed from that window afterwards.

**DXY is reconstructed, not fetched.** Free DXY quotes go stale silently. Computing
it from the six component crosses means every input is independently bounds-checked,
and the weights are validated to sum to 1.

**Z-scores exclude the current observation.** The window for day `t` is `t-60..t-1`.
Without the shift, large moves are contaminated by their own presence in the window
and get shrunk toward the mean — which would defeat the entire purpose of layer B.

**Raw payloads archived.** Every response lands in `data/raw/` before parsing. When
a number looks wrong in six months, that is the only way to tell whether the source
lied or the parser did.

\---

## Layout

```
config/instruments.yaml   registry: ids, units, sources, bounds
mb/registry.py            loader + strict validation
mb/transforms.py          pure numerics (changes, z-scores, DXY, spreads)
mb/store.py               parquet history, idempotent upsert
mb/sources/http.py        retry, timeout, raw archiving
mb/sources/adapters.py    Treasury / FRED / Yahoo / Stooq
mb/backfill.py            orchestration CLI
tests/                    93 tests, no network required
```

\---

## Known gaps

* **Treasury endpoints have changed format before.** `parse\_treasury\_csv` fails loudly
with the observed columns rather than returning empty. Run `--check-sources` after
any long gap.
* **Futures roll artifacts.** Yahoo continuous contracts (`CL=F`, `GC=F`) jump at
contract expiry. Roll-gap detection belongs in step 2's validation gate; until then
a roll may read as a large move.
* **`max\_abs\_change` is declared but not yet enforced.** It is consumed by the
validation gate in step 2, not by the backfill.
* **No trading-calendar guard yet.** Also step 2.
* **Bounds are hand-set.** Sanity ranges, not tight ones. Review after the first real
backfill using the coverage report.

\---

## Next: step 2, the validation gate

Freshness against `stale\_tolerance\_days`, `max\_abs\_change` enforcement, cross-source
reconciliation where a fallback exists, roll-gap detection, and the NYSE holiday and
half-day calendar. After that, step 3 (layers A/B rendered as a plain table) and step 4
(the replay harness) — and the classifier stays unwritten until replay says it behaves
sensibly on days you already know the answer to.

