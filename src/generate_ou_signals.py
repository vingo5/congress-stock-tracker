"""
generate_ou_signals.py — Phase 2 output.

For each ticker with price history, fits one OU model to its full available
deviation-from-trailing-SMA series, then computes a z-score for the specific
trading day of each disclosed transaction on that ticker: how many
stationary standard deviations the price was from its recent trend when the
trade happened.

Simplifying assumption, stated plainly: this fits ONE OU model per ticker
using its full history, rather than a rolling/local fit near each trade
date. A single stock's mean-reversion regime can drift over years (2013 vs
2020 are different markets), so a rolling-window fit would be more
faithful - noted here as a concrete Phase 2 enhancement rather than quietly
assumed to be fine.
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from ou_model import fit_ou, z_score

DB_PATH = Path(__file__).parent.parent / "data" / "congress_trades.db"
SMA_WINDOW = 20  # trailing days used to detrend price before fitting OU


def load_price_series(conn: sqlite3.Connection, ticker: str):
    cur = conn.cursor()
    cur.execute(
        "SELECT price_date, close FROM price_history WHERE ticker = ? ORDER BY price_date",
        (ticker,),
    )
    rows = cur.fetchall()
    if len(rows) < SMA_WINDOW + 10:
        return None, None
    dates = [r[0] for r in rows]
    closes = np.array([r[1] for r in rows])
    log_close = np.log(closes)

    sma = np.convolve(log_close, np.ones(SMA_WINDOW) / SMA_WINDOW, mode="valid")
    deviation = log_close[SMA_WINDOW - 1:] - sma
    aligned_dates = dates[SMA_WINDOW - 1:]
    return aligned_dates, deviation


def nearest_trading_day_on_or_before(dates: list[str], target_date: str):
    """Trade dates can fall on weekends/holidays with no price row; use
    the most recent available trading day at or before the trade date."""
    candidates = [d for d in dates if d <= target_date]
    return candidates[-1] if candidates else None


def run():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT ticker FROM price_history")
    tickers = [row[0] for row in cur.fetchall()]
    print(f"Tickers with price history: {len(tickers)}")

    total_signals, skipped_thin_data, skipped_no_trading_day = 0, 0, 0

    for ticker in tickers:
        dates, deviation = load_price_series(conn, ticker)
        if dates is None:
            skipped_thin_data += 1
            continue

        fit = fit_ou(deviation)
        date_to_dev = dict(zip(dates, deviation))

        cur.execute(
            "SELECT transaction_id, transaction_date FROM transactions WHERE ticker = ?",
            (ticker,),
        )
        for transaction_id, tx_date in cur.fetchall():
            trading_day = nearest_trading_day_on_or_before(dates, tx_date)
            if trading_day is None:
                skipped_no_trading_day += 1
                continue

            dev_value = date_to_dev[trading_day]
            z = z_score(dev_value, fit)
            if np.isnan(z):
                continue

            cur.execute(
                """
                INSERT INTO trade_signals (transaction_id, model_name, signal_value)
                VALUES (?, 'ou_zscore', ?)
                """,
                (transaction_id, float(z)),
            )
            total_signals += 1

        print(f"  {ticker}: theta={fit.theta:.4f} half_life={fit.half_life_days:.1f}d "
              f"stationary_std={fit.stationary_std:.4f}")

    conn.commit()
    print(f"\nTotal OU signals written: {total_signals}")
    print(f"Tickers skipped (insufficient price history): {skipped_thin_data}")
    print(f"Transactions skipped (no trading day on/before trade date): {skipped_no_trading_day}")
    conn.close()


if __name__ == "__main__":
    run()
