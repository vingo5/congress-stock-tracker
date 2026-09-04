# Congressional Stock Trading Analysis

A quantitative research pipeline analyzing US Senate stock disclosures (STOCK Act filings),
built to demonstrate data engineering, stochastic modeling, and machine learning end to end.

## Project roadmap

- [x] **Phase 1 — Data engineering foundation**: normalized SQLite schema, idempotent ETL pipeline
- [x] **Phase 1b — Filing date backfill**: joins per-filing report files back to transactions (~74% coverage)
- [x] **Phase 2 — Ornstein-Uhlenbeck model**: mean-reversion z-score signal per trade
- [x] **Phase 3 — ML features & models**: leak-free features, LightGBM classifier vs. market benchmark
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

**Real-run result.** Of the top 30 tickers, 6 failed to fetch price data — not a bug, but a real
data-source limitation confirmed by inspection: `DISCA`, `FDC`, `FEYE`, `HBI`, and `WPX` reflect
genuine corporate actions (mergers, acquisitions, and delistings) that occurred after the trades
in this dataset were made, and `FB`'s ticker was later renamed to `META`. The pipeline logs each
failure and continues rather than crashing. The remaining 24 tickers produced **1,237 OU signals**,
with fitted mean-reversion half-lives ranging ~5.5–10.7 trading days — a sane range for daily-price
deviation from a 20-day trend, not the near-zero or near-infinite half-lives that would indicate a
broken fit.

## Phase 3: ML features & model

**Target.** Binary classification: did the ticker's return over the 30 trading days *following the
filing date* exceed SPY's return over the same window (positive excess return)? Anchored on
`filing_date`, not `transaction_date` — the filing date is the earliest point the trade was public
information; anchoring on the (earlier, non-public) transaction date would itself be a subtle
look-ahead bias, predicting off information nobody could have acted on yet.

**Fixing a real look-ahead bias from Phase 2.** `generate_ou_signals.py` fits one OU model per
ticker using that ticker's *entire* price history — fine for describing a stock's general
reversion behavior, but invalid as an ML feature meant to predict a forward outcome, since a 2014
trade's z-score would then be computed partly from 2020 data nobody had yet. `build_features.py`
fixes this: it refits OU per *transaction*, using only price history up to and including that
trade's filing date. More expensive (one fit per transaction instead of per ticker), but it's the
only version of this feature that's valid to train a supervised model on.

**Time-based train/test split.** `train_model.py` sorts by `filing_date` and trains on the earliest
80%, tests on the most recent 20% — never a random shuffle, which would let the model train on
trades that happened after some of its own test trades and produce a meaningless accuracy number.
This is the same constraint a live trading signal would actually face.

**Pipeline validated against a known injected effect before trusting real data.** Synthetic price
series were generated with real mean-reversion parameters embedded (the same `simulate_ou()` used
to validate Phase 2), and the leak-free feature pipeline correctly recovered a negative correlation
between `ou_zscore` and forward excess return (mean-reverting: below-trend prices → positive
subsequent excess return), and the trained model correctly ranked `ou_zscore` as its most important
feature, with walk-forward AUC clearly above 0.5 in most folds (0.55–0.76). This confirms the
pipeline surfaces a real signal when one exists, before it's pointed at real data where the true
effect (if any) is unknown.

**Real-data result: no detectable predictive edge.** Walk-forward cross-validation across 5
time-based folds spanning 2016–2020 gives AUC = 0.497 ± 0.032 (mean ± std), with every individual
fold landing within noise of 0.5 (range 0.449–0.546) — including folds entirely within calm,
pre-2020 market conditions. This rules out "the 2020 COVID regime shift alone explains it": in the
synthetic validation run, non-2020 folds showed clear signal (AUC 0.55–0.76) while only the 2020
fold degraded; in the real data, *every* fold is flat. `ou_zscore` remains the model's most
important feature by a wide margin, meaning the model consistently tries to use the mean-reversion
signal — it just doesn't find that signal predictive of which stocks beat the market over a 30-day
window following a congressional trade's filing date, at least not via this feature set.

This is treated as a genuine empirical finding, not a failed build: the estimator was validated
end-to-end against synthetic data with a known ground-truth effect before ever touching real
prices, so a null result on real data reads as an honest "no edge found here" rather than "the
pipeline is broken." It's also consistent with a meaningful chunk of the academic literature on
congressional trading, which finds decidedly mixed evidence for a market-beating edge depending on
sample period, aggregation level, and holding horizon.

**To run for real:**
```bash
python src/fetch_prices.py 100 --forward --benchmark   # widen ticker scope + pull SPY
python src/build_features.py
python src/train_model.py
```
The default scope (top 30 tickers, no forward buffer, no benchmark) from Phase 2 isn't enough for
Phase 3 — `--forward` extends the price pull past the last trade date (needed for the 30-day
forward-return label) and `--benchmark` pulls SPY. Widening past the top 30 tickers (e.g. `100`)
gives the model more training rows; expect some tickers to fail (see the Phase 2 note on corporate
actions/delistings) — the script logs and continues rather than crashing.

## Project structure

```
congress-stock-tracker/
├── src/
│   ├── schema.sql                  # table definitions
│   ├── ingest.py                   # Phase 1: ETL pipeline
│   ├── backfill_filing_dates.py    # Phase 1b: filing date join
│   ├── ou_model.py                 # Phase 2: OU fitting logic
│   ├── test_ou_model.py            # Phase 2: validation against simulated data
│   ├── fetch_prices.py             # Phase 2/3: real price ingestion (+ forward buffer, SPY)
│   ├── generate_ou_signals.py      # Phase 2: writes z-scores to trade_signals
│   ├── build_features.py           # Phase 3: leak-free features + forward-return labels
│   └── train_model.py              # Phase 3: time-split LightGBM training + evaluation
├── data/                # gitignored; generated by ingest.py / fetch_prices.py
├── requirements.txt
└── README.md
```
