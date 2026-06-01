"""
Generate today's momentum signal.

Downloads the last ~400 calendar days of price data (enough to compute
12-month momentum + 200-day MA), runs the same composite_signal logic
used in the backtest, and returns today's target positions.

Usage (standalone):
    python src/live_signal.py              # print signal to terminal
    python src/live_signal.py --save       # also save output/live_signal.csv
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import download_close
from src.indicators import momentum_score, volatility
from src.signals import composite_signal

# Must match run_backtest.py universe and parameters
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM",
    "V", "UNH", "XOM", "LLY", "JNJ", "WMT", "MA", "PG", "HD", "MRK",
    "AVGO", "CVX", "ABBV", "COST", "PEP", "KO", "ADBE", "CRM", "TMO",
    "ACN", "MCD", "BAC",
]
TOP_N = 10
USE_TREND = True
USE_VOL_FILTER = True

# 400 calendar days → ~280 trading days, comfortably covers 252-day lookback
LOOKBACK_CALENDAR_DAYS = 400


def get_live_signal(tickers: list[str] = None) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Download recent data and compute today's target positions.

    Returns:
        signal_df : DataFrame with columns
                    [ticker, hold, weight, price, momentum_score, ann_vol]
                    sorted by momentum_score descending.
        as_of     : The date of the last available price bar.
    """
    if tickers is None:
        tickers = TICKERS

    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=LOOKBACK_CALENDAR_DAYS)).strftime("%Y-%m-%d")

    close = download_close(tickers, start=start, end=end)
    close = close.dropna(how="all", axis=1)

    signal = composite_signal(
        close,
        n=TOP_N,
        use_trend_filter=USE_TREND,
        use_vol_filter=USE_VOL_FILTER,
    )

    today_signal = signal.iloc[-1]
    today_close = close.iloc[-1]
    today_scores = momentum_score(close).iloc[-1]
    today_vol = volatility(close).iloc[-1]

    holdings = today_signal[today_signal == 1].index.tolist()
    weight = 1.0 / len(holdings) if holdings else 0.0

    rows = []
    for ticker in close.columns:
        rows.append({
            "ticker": ticker,
            "hold": bool(today_signal[ticker]),
            "weight": weight if today_signal[ticker] else 0.0,
            "price": round(today_close[ticker], 2),
            "momentum_score": round(today_scores.get(ticker, float("nan")), 4),
            "ann_vol": round(today_vol.get(ticker, float("nan")), 4),
        })

    df = (
        pd.DataFrame(rows)
        .sort_values("momentum_score", ascending=False)
        .reset_index(drop=True)
    )
    return df, close.index[-1]


def print_signal(df: pd.DataFrame, as_of: pd.Timestamp) -> None:
    hold = df[df["hold"]]
    out = df[~df["hold"]]
    n = len(hold)
    w = 1 / n if n > 0 else 0

    print(f"\n{'='*55}")
    print(f"  MOMENTUM SIGNAL  —  as of {as_of.date()}")
    print(f"{'='*55}")
    print(f"\nHOLD  ({n} positions, {w:.1%} each)\n")
    print(f"  {'Ticker':<8} {'Price':>8} {'Mom.Score':>11} {'Ann.Vol':>9}")
    print(f"  {'-'*40}")
    for _, row in hold.iterrows():
        print(f"  {row['ticker']:<8} ${row['price']:>7.2f} {row['momentum_score']:>11.4f} {row['ann_vol']:>8.1%}")

    print(f"\nOUT  (top 5 by score, excluded by filters)\n")
    print(f"  {'Ticker':<8} {'Price':>8} {'Mom.Score':>11} {'Ann.Vol':>9}")
    print(f"  {'-'*40}")
    for _, row in out.head(5).iterrows():
        print(f"  {row['ticker']:<8} ${row['price']:>7.2f} {row['momentum_score']:>11.4f} {row['ann_vol']:>8.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save signal to output/live_signal.csv")
    args = parser.parse_args()

    print("Fetching data and computing signal...")
    df, as_of = get_live_signal()
    print_signal(df, as_of)

    if args.save:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "live_signal.csv")
        df.to_csv(path, index=False)
        print(f"\nSaved to {path}")
