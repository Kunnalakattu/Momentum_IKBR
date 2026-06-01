import yfinance as yf
import pandas as pd
from typing import Union


def download_prices(
    tickers: Union[str, list[str]],
    start: str,
    end: str = None,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download OHLCV data for one or more tickers from Yahoo Finance."""
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
    )
    return data


def download_close(
    tickers: Union[str, list[str]],
    start: str,
    end: str = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Return only adjusted close prices as a DataFrame with tickers as columns."""
    data = download_prices(tickers, start=start, end=end, interval=interval)
    if isinstance(tickers, str):
        return data[["Close"]].rename(columns={"Close": tickers})
    return data["Close"]


def get_returns(close: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Compute simple period returns from a close price DataFrame."""
    return close.pct_change(periods).dropna()
