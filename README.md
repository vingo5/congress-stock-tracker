# Congressional Stock Trading Analysis

A quantitative research pipeline analyzing US Senate stock disclosures (STOCK Act filings),
built to demonstrate data engineering, stochastic modeling, and machine learning end to end.

## Project roadmap

- [x] **Phase 1 — Data engineering foundation**: normalized SQLite schema, idempotent ETL pipeline
- [x] **Phase 1b — Filing date backfill**: joins per-filing report files back to transactions (~74% coverage)
- [x] **Phase 2 — Ornstein-Uhlenbeck model**: mean-reversion z-score signal per trade
- [ ] **Phase 3 — ML features & models**: pull historical prices, engineer features, train LightGBM/XGBoost
- [ ] **Dashboard**: Streamlit front-end for exploring senators, tickers, and model signals

## Data source

Data comes from the [senate-stock-watcher-data](https://github.com/timothycarambat/senate-stock-watcher-data)
GitHub mirror (Timothy Carambat), rather than scraping raw PDFs from
[efdsearch.senate.gov](https://efdsearch.senate.gov) directly.

This is a deliberate engineering tradeoff, not laziness:
- The mirror has already solved PDF parsing, a genuinely hard problem (inconsistent formats,
  scanned images, OCR errors) that isn't the focus of this project.
- It lets the project focus on the parts being demonstrated: schema design, ETL correctness,
  stochastic modeling, and ML — rather than re-solving a parsing problem others have solved.
- The tradeoff is explicit and documented (see Known Limitations) rather than hidden.

In a production setting, the next step would be building a scraper against the primary source
directly, since the mirror can go stale (as it has — see below).

## Known limitations

- **The mirror stopped updating in 2020.** Data covers 2012–2020 disclosures only. The pipeline
  (`fetch_raw_records()` in `src/ingest.py`) is isolated to one function, so pointing it at a live
  source later is a small, contained change.
- **Filing dates cover ~74% of transactions, not 100%.** `src/backfill_filing_dates.py` joins the
  aggregate transaction file against a second part of the same source repo — one JSON file per
  filing date (`data/transaction_report_for_MM_DD_YYYY.json`) — matched via the same field-hash
  logic used for deduplication in `ingest.py`. The ~26% gap is not silently absorbed; it comes from
  two confirmed causes: (1) some filing-report files cover dates outside the aggregate file's own
  coverage window — an inconsistency between two files in the same source repo, not a bug in this
  pipeline; (2) a minority of `asset_description` fields in the per-filing files contain embedded
  HTML (e.g. nested option/strike-price details) that isn't present in the cleaned aggregate file,
  so the two representations of the same trade don't string-match. This ceiling was measured, not
  assumed, and is left as a known limitation rather than chased with fuzzy matching, which would
  trade one explainable gap for an unverifiable one.
- **The backfill join itself required a correctness fix worth documenting.** An early version took
  the first matching filing_date it found per transaction; because filing files are named
  `transaction_report_for_MM_DD_YYYY.json` and sorted alphabetically (grouping by month-then-year,
  not chronologically), this occasionally attached an earlier-sorted-but-later-actual-date filing
  to a transaction, producing a handful of impossible `filing_date < transaction_date` rows. Fixed
  by gathering all candidate filing dates per transaction and selecting the earliest one that is
  actually on or after the trade date. Caught via a validation query
  (`filing_date < transaction_date` should always return 0 rows), not by inspection.
- **~4.5% of matched trades were disclosed more than 45 days after the transaction** — the STOCK
  Act's legal disclosure deadline. This is a genuine finding in the data, not a pipeline artifact.
- **Two distinct junk-ticker placeholders exist in the source (`'--'` and `'N/A'`)**, together
  ~25% of all rows. Both are normalized to `NULL` in `ticker` during ingestion — see
  `clean_ticker()` in `src/ingest.py`. This was caught via a `GROUP BY ticker ORDER BY COUNT(*)`
  inspection, not assumed; the fix is left visible in code rather than silently patched.
- **~7% of source rows are exact duplicate disclosures** (identical senator/date/ticker/amount/
  type/description). The ingestion pipeline dedupes these via a deterministic content hash
  (`source_hash`), since the source provides no stable transaction ID.

## Schema design

Three core design decisions worth discussing in interviews:

1. **Normalized, not denormalized.** `politicians` and `transactions` are separate tables with a
   foreign key, rather than repeating the senator's name on every row. Standard 3NF practice, but
   deliberate here given the analytical (not transactional) workload.
2. **SQLite locally, Postgres-compatible SQL.** No SQLite-specific syntax beyond `AUTOINCREMENT`,
   which maps directly to Postgres's `SERIAL`/`GENERATED ALWAYS AS IDENTITY`. Raw SQL rather than
   an ORM, to keep the schema and query logic transparent and portable.
3. **Forward-looking placeholder tables.** `price_history` and `trade_signals` are defined in
   Phase 1, empty, so that Phase 2 and 3 are additive (`INSERT`s into existing tables) rather than
   requiring schema migrations later.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/ingest.py
python src/backfill_filing_dates.py
python src/test_ou_model.py       # validates the OU estimator before trusting it
python src/fetch_prices.py        # pulls real prices for top 30 tickers (~1-2 min)
python src/generate_ou_signals.py
```

This creates `data/congress_trades.db` (gitignored — it's fully reproducible from `ingest.py`,
so there's no reason to version a derived binary artifact).

## Phase 2: Ornstein-Uhlenbeck model

**What it models.** Raw stock prices are closer to a random walk than a mean-reverting process, so
fitting an OU process directly to price would be the wrong regime. Instead, `src/ou_model.py` fits
OU to the *deviation of log-price from its own trailing 20-day moving average* — a standard
stat-arb technique, since that deviation is mean-reverting by construction. For each ticker, the
fit produces a mean-reversion speed (`theta`), an implied half-life, and a stationary standard
deviation. `src/generate_ou_signals.py` then computes a z-score for the specific trading day of
each disclosed trade: how many stationary standard deviations the price was from its recent trend
when that trade happened. Output lands in `trade_signals` (`model_name='ou_zscore'`).

**Validated before trusting it on real data.** `src/test_ou_model.py` fits the estimator against
simulated OU paths with known ground-truth parameters and confirms it recovers them (theta within
~1–5% in tests, exact z-score arithmetic) before the model is pointed at real prices. It also
caught a genuine subtlety worth knowing: R² is *not* a valid mean-reversion diagnostic — a pure
random walk (no reversion at all) still produces R² > 0.99 in an AR(1) fit, since today still
predicts tomorrow closely either way. `theta`/`phi` are the correct diagnostics, not R².

**Known simplifying assumption.** The current implementation fits *one* OU model per ticker using
its full available price history. A stock's mean-reversion regime can drift across years (2013
market conditions differ from 2020's), so a rolling/local fit near each trade date would be more
faithful — documented here as the natural next iteration rather than silently assumed away.

**Scope note.** `src/fetch_prices.py` pulls real daily prices via `yfinance`, scoped by default to
the top 30 most-traded tickers (generalizes to all 1,007 tickers unchanged — just widen the limit
— which is the natural bridge into Phase 3's full price ingestion).

## Project structure

```
congress-stock-tracker/
├── src/
│   ├── schema.sql                  # table definitions
│   ├── ingest.py                   # Phase 1: ETL pipeline
│   ├── backfill_filing_dates.py    # Phase 1b: filing date join
│   ├── ou_model.py                 # Phase 2: OU fitting logic
│   ├── test_ou_model.py            # Phase 2: validation against simulated data
│   ├── fetch_prices.py             # Phase 2: real price ingestion (top tickers)
│   └── generate_ou_signals.py      # Phase 2: writes z-scores to trade_signals
├── data/                # gitignored; generated by ingest.py / fetch_prices.py
├── requirements.txt
└── README.md
```
