"""
ingest.py — ETL pipeline for Senate stock disclosure data.

Design notes (for interview talking points):
- Source-agnostic: fetch_raw_records() is the only function that knows about
  the specific mirror. Swapping data sources later means rewriting one
  function, not the pipeline.
- Idempotent: re-running this script does not create duplicate rows. Each
  transaction gets a deterministic source_hash (senator+date+ticker+amount+
  type) used as an upsert key, since the source has no stable transaction ID.
- Data quality is handled explicitly, not silently: see clean_ticker().
"""

import hashlib
import json
import sqlite3
import urllib.request
from datetime import datetime
from pathlib import Path

DATA_URL = (
    "https://raw.githubusercontent.com/timothycarambat/"
    "senate-stock-watcher-data/master/aggregate/all_transactions.json"
)
DB_PATH = Path(__file__).parent.parent / "data" / "congress_trades.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# Discovered via manual inspection (GROUP BY ticker, ORDER BY count DESC):
# the source uses TWO different placeholder strings for "no ticker",
# not just one. Both must be treated as NULL or they silently pollute
# every ticker-level aggregation downstream.
JUNK_TICKERS = {"--", "N/A", ""}


def fetch_raw_records() -> list[dict]:
    """Pulls the aggregate JSON from the GitHub mirror."""
    req = urllib.request.Request(DATA_URL, headers={"User-Agent": "congress-stock-tracker"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def clean_ticker(raw_ticker: str | None) -> str | None:
    """Normalizes junk ticker placeholders to NULL so downstream queries
    don't accidentally treat '--' or 'N/A' as a real, tradeable symbol."""
    if raw_ticker is None:
        return None
    stripped = raw_ticker.strip()
    return None if stripped in JUNK_TICKERS else stripped


def parse_amount_range(amount_str: str) -> tuple[int | None, int | None]:
    """Parses strings like '$50,001 - $100,000' into (50001, 100000).
    Handles the '$1,000,000 +' style top-bracket string too."""
    if not amount_str:
        return None, None
    cleaned = amount_str.replace("$", "").replace(",", "").strip()
    if "+" in cleaned:
        low = cleaned.replace("+", "").strip()
        return (int(low) if low.isdigit() else None), None
    parts = [p.strip() for p in cleaned.split("-")]
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        return int(parts[0]), int(parts[1])
    return None, None


def parse_date(raw_date: str) -> str | None:
    """Source dates are 'MM/DD/YYYY'; we normalize to ISO 8601 for
    consistent sorting and compatibility with Postgres DATE columns later."""
    if not raw_date:
        return None
    try:
        return datetime.strptime(raw_date, "%m/%d/%Y").strftime("%Y-%m-%d")
    except ValueError:
        return None


def make_source_hash(record: dict) -> str:
    """Deterministic dedupe key. The source has no transaction ID, so we
    hash the fields that together uniquely identify a disclosed trade."""
    key = "|".join([
        record.get("senator", ""),
        record.get("transaction_date", ""),
        record.get("ticker", ""),
        record.get("amount", ""),
        record.get("type", ""),
        record.get("asset_description", ""),
    ])
    return hashlib.sha256(key.encode()).hexdigest()


def get_or_create_politician(cur: sqlite3.Cursor, full_name: str) -> int:
    cur.execute("SELECT politician_id FROM politicians WHERE full_name = ?", (full_name,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("INSERT INTO politicians (full_name) VALUES (?)", (full_name,))
    return cur.lastrowid


def run_ingestion():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    cur = conn.cursor()

    print("Fetching source data...")
    raw_records = fetch_raw_records()
    print(f"Fetched {len(raw_records)} raw records.")

    inserted, skipped_dupe, skipped_bad = 0, 0, 0

    for record in raw_records:
        senator = record.get("senator")
        tx_date = parse_date(record.get("transaction_date"))
        if not senator or not tx_date:
            skipped_bad += 1
            continue

        politician_id = get_or_create_politician(cur, senator)
        ticker = clean_ticker(record.get("ticker"))
        amount_min, amount_max = parse_amount_range(record.get("amount", ""))
        source_hash = make_source_hash(record)

        try:
            cur.execute(
                """
                INSERT INTO transactions (
                    politician_id, ticker, asset_description, asset_type,
                    transaction_type, owner, amount_min, amount_max,
                    transaction_date, ptr_link, source_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_hash) DO NOTHING
                """,
                (
                    politician_id, ticker, record.get("asset_description"),
                    record.get("asset_type"), record.get("type"),
                    record.get("owner"), amount_min, amount_max,
                    tx_date, record.get("ptr_link"), source_hash,
                ),
            )
            if cur.rowcount:
                inserted += 1
            else:
                skipped_dupe += 1
        except sqlite3.IntegrityError:
            skipped_dupe += 1

    conn.commit()

    # Data quality summary — printed every run so drift is visible immediately.
    cur.execute("SELECT COUNT(*) FROM transactions WHERE ticker IS NULL")
    null_ticker_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM transactions")
    total = cur.fetchone()[0]

    print(f"\nInserted: {inserted} | Duplicates skipped: {skipped_dupe} | Bad rows skipped: {skipped_bad}")
    print(f"Total transactions in DB: {total}")
    print(f"Transactions with NULL ticker (was junk placeholder): {null_ticker_count}")

    conn.close()


if __name__ == "__main__":
    run_ingestion()
