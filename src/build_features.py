"""
build_features.py — Phase 3 feature engineering.

CRITICAL DESIGN RULE: no feature may use information that would not have
been available at the trade's filing_date. This script exists partly to
FIX a look-ahead bias problem in Phase 2's output: generate_ou_signals.py
fits one OU model per ticker using that ticker's FULL price history
(including years after some trades), which is fine for characterizing a
stock's general reversion behavior but is invalid as an ML feature meant
to predict a FORWARD outcome - it would let the model see the future.

Here, ou_zscore is recomputed per transaction using ONLY price history up
to and including the filing_date (a trailing window ending at filing_date,
not the ticker's whole history). This is more expensive (one OU fit per
transaction instead of one per ticker) but is the correct approach for a
supervised learning feature.

Label: does the ticker's N-trading-day forward return (measured from
filing_date) exceed SPY's return over the same window? Binary, 1 = beat
the market, 0 = did not. Using filing_date (not transaction_date) as the
anchor because that's the earliest point the trade was public information
- anchoring on transaction_date would be a second, more subtle look-ahead
bias (predicting off information nobody had yet).
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from ou_model import fit_ou, z_score

DB_PATH = Path(__file__).parent.parent / "data" / "congress_trades.db"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "ml_features.csv"

SMA_WINDOW = 20
MIN_TRAILING_OBS = SMA_WINDOW + 30  # need enough history for a stable OU fit
FORWARD_WINDOW_TRADING_DAYS = 30
BENCHMARK_TICKER = "SPY"


def load_price_series(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    df = pd.read_sql(
        "SELECT price_date, close FROM price_history WHERE ticker = ? ORDER BY price_date",
        conn, params=(ticker,),
    )
    df["log_close"] = np.log(df["close"])
    return df


def leak_free_ou_zscore(price_df: pd.DataFrame, as_of_date: str) -> float | None:
    """Fits OU using only rows with price_date <= as_of_date (the trade's
    filing_date), then returns the z-score of the LAST such observation.
    Returns None if there isn't enough trailing history yet."""
    trailing = price_df[price_df["price_date"] <= as_of_date]
    if len(trailing) < MIN_TRAILING_OBS:
        return None

    log_close = trailing["log_close"].values
    sma = pd.Series(log_close).rolling(SMA_WINDOW).mean().values
    deviation = log_close[SMA_WINDOW - 1:] - sma[SMA_WINDOW - 1:]
    deviation = deviation[~np.isnan(deviation)]
    if len(deviation) < 10:
        return None

    fit = fit_ou(deviation)
    return z_score(deviation[-1], fit)


def forward_return(price_df: pd.DataFrame, as_of_date: str, n_trading_days: int) -> float | None:
    """N-trading-day forward return starting from the first available
    trading day on/after as_of_date. Returns None if there isn't enough
    forward price history yet (e.g. trade too close to the data cutoff)."""
    forward = price_df[price_df["price_date"] >= as_of_date].reset_index(drop=True)
    if len(forward) <= n_trading_days:
        return None
    start_price = forward.loc[0, "close"]
    end_price = forward.loc[n_trading_days, "close"]
    return (end_price / start_price) - 1


def build_dataset():
    conn = sqlite3.connect(DB_PATH)

    transactions = pd.read_sql(
        """
        SELECT t.transaction_id, p.full_name AS politician, t.ticker,
               t.transaction_type, t.owner, t.asset_type,
               t.amount_min, t.amount_max,
               t.transaction_date, t.filing_date
        FROM transactions t JOIN politicians p ON t.politician_id = p.politician_id
        WHERE t.filing_date IS NOT NULL AND t.ticker IS NOT NULL
        ORDER BY t.filing_date
        """,
        conn,
    )
    print(f"Candidate transactions (have filing_date + ticker): {len(transactions)}")

    available_tickers = pd.read_sql("SELECT DISTINCT ticker FROM price_history", conn)["ticker"].tolist()
    if BENCHMARK_TICKER not in available_tickers:
        raise RuntimeError(
            f"{BENCHMARK_TICKER} price history not found. Run: "
            f"python src/fetch_prices.py <N> --forward --benchmark"
        )

    spy_df = load_price_series(conn, BENCHMARK_TICKER)
    price_cache: dict[str, pd.DataFrame] = {}

    rows = []
    skipped_no_price, skipped_thin_history, skipped_no_forward = 0, 0, 0

    for _, tx in transactions.iterrows():
        ticker = tx["ticker"]
        if ticker not in available_tickers:
            skipped_no_price += 1
            continue

        if ticker not in price_cache:
            price_cache[ticker] = load_price_series(conn, ticker)
        price_df = price_cache[ticker]

        filing_date = tx["filing_date"]

        ou_z = leak_free_ou_zscore(price_df, filing_date)
        if ou_z is None or np.isnan(ou_z):
            skipped_thin_history += 1
            continue

        stock_fwd = forward_return(price_df, filing_date, FORWARD_WINDOW_TRADING_DAYS)
        spy_fwd = forward_return(spy_df, filing_date, FORWARD_WINDOW_TRADING_DAYS)
        if stock_fwd is None or spy_fwd is None:
            skipped_no_forward += 1
            continue

        excess_return = stock_fwd - spy_fwd
        beat_market = int(excess_return > 0)

        amount_min = tx["amount_min"] or 0
        amount_max = tx["amount_max"] or amount_min
        amount_mid = (amount_min + amount_max) / 2 if (amount_min or amount_max) else np.nan

        disclosure_delay_days = (
            pd.Timestamp(tx["filing_date"]) - pd.Timestamp(tx["transaction_date"])
        ).days

        rows.append({
            "transaction_id": tx["transaction_id"],
            "politician": tx["politician"],
            "ticker": ticker,
            "transaction_type": tx["transaction_type"],
            "owner": tx["owner"],
            "asset_type": tx["asset_type"],
            "amount_mid": amount_mid,
            "log_amount_mid": np.log1p(amount_mid) if pd.notna(amount_mid) else np.nan,
            "disclosure_delay_days": disclosure_delay_days,
            "ou_zscore": ou_z,
            "filing_date": filing_date,
            "excess_return_30d": excess_return,
            "beat_market_30d": beat_market,
        })

    df = pd.DataFrame(rows)
    print(f"\nSkipped - ticker has no price history at all: {skipped_no_price}")
    print(f"Skipped - insufficient trailing history for leak-free OU fit: {skipped_thin_history}")
    print(f"Skipped - insufficient forward price history for label: {skipped_no_forward}")
    print(f"Final dataset size: {len(df)}")
    if len(df) > 0:
        print(f"Positive label rate (beat market over 30d): {df['beat_market_30d'].mean():.1%}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {OUTPUT_PATH}")
    conn.close()
    return df


if __name__ == "__main__":
    build_dataset()
