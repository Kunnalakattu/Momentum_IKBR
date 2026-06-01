"""
Data loading utilities.

Re-exports the core download helpers from src/data.py and adds a single
convenience function that returns the split universe + benchmark series
used by every runner in this project.
"""

from __future__ import annotations

import pandas as pd
from src.data import download_close, download_prices, get_returns   # noqa: F401


def load_universe(
    tickers: list[str],
    benchmark_ticker: str,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Download and clean prices for the stock universe and benchmark.

    Returns
    -------
    close : pd.DataFrame  — adjusted close prices, one column per ticker.
    benchmark_close : pd.Series — adjusted close for the benchmark (e.g. SPY).

    Rows with all-NaN are dropped; tickers with any NaN for the full period
    are removed from the universe.
    """
    all_tickers = tickers + [benchmark_ticker]
    raw = download_close(all_tickers, start=start, end=end)
    raw = raw.dropna(how="all")

    close           = raw[tickers].dropna(how="all", axis=1)
    benchmark_close = raw[benchmark_ticker]

    print(
        f"Universe: {len(close.columns)} tickers  |  "
        f"{close.index[0].date()} → {close.index[-1].date()}"
    )
    return close, benchmark_close
