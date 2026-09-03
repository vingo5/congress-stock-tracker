"""
fetch_prices.py — pulls daily OHLCV price history for traded tickers.

This is a deliberate, scoped pull-forward of part of Phase 3 (full price
ingestion + ML features), done now because the OU model needs real price
data to fit against. Scope is limited to the top N most-traded tickers by
default so this stays fast and demoable; the same function generalizes to
all tickers later without changes (just pass a larger/full ticker list).

Requires: pip install yfinance
"""

import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "congress_trades.db"

# Extra trading days pulled before the earliest transaction date, so the
# OU model has a trailing window to compute a moving average/deviation
# even for trades near the start of the dataset.
LOOKBACK_BUFFER_DAYS = 120


def get_top_tickers(conn: sqlite3.Connection, limit: int = 30) -> list[str]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ticker, COUNT(*) as n FROM transactions
        WHERE ticker IS NOT NULL
        GROUP BY ticker ORDER BY n DESC LIMIT ?
        """,
        (limit,),
    )
    return [row[0] for row in cur.fetchall()]


def get_date_range(conn: sqlite3.Connection, ticker: str) -> tuple[str, str]:
    cur = conn.cursor()
    cur.execute(
        "SELECT MIN(transaction_date), MAX(transaction_date) FROM transactions WHERE ticker = ?",
        (ticker,),
    )
    min_date, max_date = cur.fetchone()
    start = (datetime.strptime(min_date, "%Y-%m-%d") - timedelta(days=LOOKBACK_BUFFER_DAYS)).strftime("%Y-%m-%d")
    return start, max_date


def fetch_and_store_ticker(conn: sqlite3.Connection, ticker: str) -> int:
    import yfinance as yf

    start, end = get_date_range(conn, ticker)
    # yfinance's `end` is exclusive, so pad by a day to include the last trade date
    end_padded = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    hist = yf.download(ticker, start=start, end=end_padded, progress=False, auto_adjust=False)
    if hist.empty:
        print(f"  {ticker}: no data returned")
        return 0

    # yfinance sometimes returns a MultiIndex column header even for a single ticker
    if isinstance(hist.columns, __import__("pandas").MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    cur = conn.cursor()
    inserted = 0
    for date, row in hist.iterrows():
        try:
            cur.execute(
                """
                INSERT INTO price_history (ticker, price_date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, price_date) DO NOTHING
                """,
                (
                    ticker, date.strftime("%Y-%m-%d"),
                    float(row["Open"]), float(row["High"]), float(row["Low"]),
                    float(row["Close"]), int(row["Volume"]),
                ),
            )
            if cur.rowcount:
                inserted += 1
        except (KeyError, ValueError) as e:
            print(f"  {ticker} {date}: skipped row ({e})")
    conn.commit()
    return inserted


def run(limit: int = 30):
    conn = sqlite3.connect(DB_PATH)
    tickers = get_top_tickers(conn, limit=limit)
    print(f"Fetching price history for {len(tickers)} tickers: {', '.join(tickers)}\n")

    total = 0
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}...")
        try:
            n = fetch_and_store_ticker(conn, ticker)
            print(f"  inserted {n} price rows")
            total += n
        except Exception as e:
            print(f"  ERROR fetching {ticker}: {e}")
        time.sleep(0.5)  # be polite to the data source

    print(f"\nTotal price rows inserted: {total}")
    conn.close()


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    run(limit=limit)
