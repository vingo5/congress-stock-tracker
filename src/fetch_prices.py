"""
fetch_prices.py — pulls daily OHLCV price history for traded tickers.

This is a deliberate, scoped pull-forward of part of Phase 3 (full price
ingestion + ML features), done now because the OU model needs real price
data to fit against. Scope is limited to the top N most-traded tickers by
default so this stays fast and demoable; the same function generalizes to
all tickers later without changes (just pass a larger/full ticker list).

Phase 3 addition: also fetches a market benchmark (SPY) and extends the
date range FORWARD past the last trade date, not just backward. Phase 2
only needed history up to each trade; Phase 3's ML labels need N trading
days of price data AFTER the filing date too, to compute forward returns.

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

# Extra calendar days pulled AFTER the latest transaction date, so Phase 3
# can compute a forward return label (e.g. 30 trading days ~ 45 calendar
# days including weekends/holidays) even for trades near the end of the
# dataset. Only relevant once forward-looking labels are needed (Phase 3).
FORWARD_BUFFER_DAYS = 60

BENCHMARK_TICKER = "SPY"


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


def get_date_range(conn: sqlite3.Connection, ticker: str, include_forward_buffer: bool = False) -> tuple[str, str]:
    cur = conn.cursor()
    cur.execute(
        "SELECT MIN(transaction_date), MAX(transaction_date) FROM transactions WHERE ticker = ?",
        (ticker,),
    )
    min_date, max_date = cur.fetchone()
    start = (datetime.strptime(min_date, "%Y-%m-%d") - timedelta(days=LOOKBACK_BUFFER_DAYS)).strftime("%Y-%m-%d")
    end = max_date
    if include_forward_buffer:
        end = (datetime.strptime(max_date, "%Y-%m-%d") + timedelta(days=FORWARD_BUFFER_DAYS)).strftime("%Y-%m-%d")
    return start, end


def get_global_date_range(conn: sqlite3.Connection) -> tuple[str, str]:
    """Full range needed for the benchmark: earliest transaction minus
    lookback, latest transaction plus forward buffer."""
    cur = conn.cursor()
    cur.execute("SELECT MIN(transaction_date), MAX(transaction_date) FROM transactions")
    min_date, max_date = cur.fetchone()
    start = (datetime.strptime(min_date, "%Y-%m-%d") - timedelta(days=LOOKBACK_BUFFER_DAYS)).strftime("%Y-%m-%d")
    end = (datetime.strptime(max_date, "%Y-%m-%d") + timedelta(days=FORWARD_BUFFER_DAYS)).strftime("%Y-%m-%d")
    return start, end


def _download_and_store(conn: sqlite3.Connection, ticker: str, start: str, end: str) -> int:
    import yfinance as yf

    end_padded = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    hist = yf.download(ticker, start=start, end=end_padded, progress=False, auto_adjust=False)
    if hist.empty:
        print(f"  {ticker}: no data returned")
        return 0

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


def fetch_and_store_ticker(conn: sqlite3.Connection, ticker: str, include_forward_buffer: bool = False) -> int:
    start, end = get_date_range(conn, ticker, include_forward_buffer=include_forward_buffer)
    return _download_and_store(conn, ticker, start, end)


def fetch_benchmark(conn: sqlite3.Connection) -> int:
    """Fetches SPY over the full range needed by any transaction, so
    forward-return labels can be computed against the market regardless
    of which ticker's window is being evaluated."""
    start, end = get_global_date_range(conn)
    print(f"Fetching benchmark {BENCHMARK_TICKER} ({start} to {end})...")
    n = _download_and_store(conn, BENCHMARK_TICKER, start, end)
    print(f"  inserted {n} benchmark price rows")
    return n


def run(limit: int = 30, with_forward_buffer: bool = False, with_benchmark: bool = False):
    conn = sqlite3.connect(DB_PATH)
    tickers = get_top_tickers(conn, limit=limit)
    print(f"Fetching price history for {len(tickers)} tickers: {', '.join(tickers)}\n")

    total = 0
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}...")
        try:
            n = fetch_and_store_ticker(conn, ticker, include_forward_buffer=with_forward_buffer)
            print(f"  inserted {n} price rows")
            total += n
        except Exception as e:
            print(f"  ERROR fetching {ticker}: {e}")
        time.sleep(0.5)  # be polite to the data source

    if with_benchmark:
        fetch_benchmark(conn)

    print(f"\nTotal price rows inserted: {total}")
    conn.close()


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    # Phase 3 usage: python src/fetch_prices.py 100 --forward --benchmark
    with_forward = "--forward" in sys.argv
    with_bench = "--benchmark" in sys.argv
    run(limit=limit, with_forward_buffer=with_forward, with_benchmark=with_bench)
