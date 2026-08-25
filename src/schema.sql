-- schema.sql
-- Congressional Stock Trading Analysis — normalized schema.
-- Written in plain SQL (no ORM) so it's portable to Postgres later:
-- only sqlite-specific bit is AUTOINCREMENT, which has a direct Postgres
-- equivalent (SERIAL / GENERATED ALWAYS AS IDENTITY).

-- One row per unique senator. Normalizing this out avoids storing the
-- same name string thousands of times and gives us a stable FK target.
CREATE TABLE IF NOT EXISTS politicians (
    politician_id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name     TEXT NOT NULL UNIQUE,
    chamber       TEXT NOT NULL DEFAULT 'Senate'
);

-- One row per disclosed transaction.
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    politician_id      INTEGER NOT NULL REFERENCES politicians(politician_id),
    ticker             TEXT,                 -- NULL when source had no valid ticker
    asset_description  TEXT NOT NULL,
    asset_type         TEXT,
    transaction_type    TEXT NOT NULL,         -- e.g. 'Purchase', 'Sale (Full)', 'Sale (Partial)', 'Exchange'
    owner               TEXT,                  -- 'Self' / 'Spouse' / 'Joint' / 'Child'
    amount_min          INTEGER,               -- parsed lower bound of disclosed range, in USD
    amount_max          INTEGER,               -- parsed upper bound of disclosed range, in USD
    transaction_date    TEXT NOT NULL,         -- ISO 8601 (YYYY-MM-DD)
    filing_date         TEXT,                  -- NULL for now; backfilled in Phase 1b
    ptr_link            TEXT,
    source_hash         TEXT NOT NULL UNIQUE,  -- dedupe key, see ingest.py
    ingested_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_transactions_ticker ON transactions(ticker);
CREATE INDEX IF NOT EXISTS idx_transactions_politician ON transactions(politician_id);
CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(transaction_date);

-- Placeholder for Phase 3 (historical prices pulled per ticker).
-- Defined now so Phase 3 is an INSERT-only addition, not a migration.
CREATE TABLE IF NOT EXISTS price_history (
    price_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT NOT NULL,
    price_date  TEXT NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      INTEGER,
    UNIQUE(ticker, price_date)
);

-- Placeholder for Phase 2/3 outputs (OU model signals, ML predictions).
CREATE TABLE IF NOT EXISTS trade_signals (
    signal_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id  INTEGER NOT NULL REFERENCES transactions(transaction_id),
    model_name      TEXT NOT NULL,   -- e.g. 'ou_zscore', 'lgbm_v1'
    signal_value    REAL,
    generated_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
