import pandas as pd
import numpy as np


def total_return(close: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Raw price return over a lookback window (in trading days)."""
    return close.pct_change(lookback)


def momentum_12_1(close: pd.DataFrame) -> pd.DataFrame:
    """Classic 12-1 momentum: 12-month return skipping the most recent month.

    Avoids short-term reversal by excluding the last ~21 trading days.
    """
    ret_12m = close.pct_change(252)
    ret_1m = close.pct_change(21)
    return (1 + ret_12m) / (1 + ret_1m) - 1


def rate_of_change(close: pd.DataFrame, period: int) -> pd.DataFrame:
    """Rate of change: (price_now / price_n_periods_ago) - 1."""
    return close / close.shift(period) - 1


def rsi(close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def volatility(close: pd.DataFrame, period: int = 21) -> pd.DataFrame:
    """Rolling annualised volatility of daily returns."""
    return close.pct_change().rolling(period).std() * np.sqrt(252)


def risk_adjusted_momentum(close: pd.DataFrame, lookback: int = 252, vol_period: int = 21) -> pd.DataFrame:
    """Momentum score divided by rolling volatility (Sharpe-style ranking signal)."""
    mom = total_return(close, lookback)
    vol = volatility(close, vol_period)
    return mom / vol


def momentum_score(close: pd.DataFrame) -> pd.DataFrame:
    """Composite momentum score: average of 3, 6, and 12-month return ranks.

    Returns a cross-sectional rank score (higher = stronger momentum).
    """
    ret_3m = total_return(close, 63).rank(axis=1, pct=True)
    ret_6m = total_return(close, 126).rank(axis=1, pct=True)
    ret_12m = total_return(close, 252).rank(axis=1, pct=True)
    return (ret_3m + ret_6m + ret_12m) / 3


def dual_momentum(close: pd.DataFrame, benchmark: pd.Series, lookback: int = 252) -> pd.DataFrame:
    """Gary Antonacci's dual momentum: absolute + relative momentum filter.

    Returns a boolean DataFrame — True where the asset beats both the benchmark
    and a risk-free proxy (0 return) over the lookback window.
    """
    asset_ret = total_return(close, lookback)
    bench_ret = benchmark.pct_change(lookback)
    absolute_filter = asset_ret > 0
    relative_filter = asset_ret.gt(bench_ret, axis=0)
    return absolute_filter & relative_filter


def residual_momentum_score(
    close: pd.DataFrame,
    market_close: pd.Series,
    lookback: int = 252,
    beta_window: int = 252,
) -> pd.DataFrame:
    """
    Idiosyncratic (beta-adjusted) momentum score.

    For each stock, subtract the market-explained return from its total return
    over `lookback` days.  Stocks with high residual return outperformed on
    alpha, not just because the market went up.

        residual_return_i = total_return_i - beta_i × market_return

    Beta is estimated with a rolling `beta_window`-day OLS regression.

    Returns a cross-sectional percentile-rank DataFrame (higher = stronger
    idiosyncratic momentum), aligned to close.index.
    """
    stock_ret  = close.pct_change()
    market_ret = market_close.pct_change()
    market_var = market_ret.rolling(beta_window).var()

    # Rolling beta: Cov(stock, market) / Var(market)
    betas = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for ticker in close.columns:
        cov = stock_ret[ticker].rolling(beta_window).cov(market_ret)
        betas[ticker] = (cov / market_var).clip(-3.0, 3.0)

    # Residual return over lookback
    total_r  = close.pct_change(lookback)
    market_r = market_close.pct_change(lookback)
    residual = total_r.sub(betas.mul(market_r, axis=0))

    # Cross-sectional percentile rank (same convention as momentum_score)
    return residual.rank(axis=1, pct=True)


def vol_adjusted_momentum_score(
    close: pd.DataFrame,
    lookback: int = 252,
    vol_period: int = 21,
) -> pd.DataFrame:
    """
    Momentum divided by trailing realised volatility (Sharpe-style ranking).

    Rewards stocks that rose without excessive volatility — a cleaner alpha
    signal than raw return in high-vol environments.

    Returns cross-sectional percentile ranks.
    """
    ret = close.pct_change(lookback)
    vol = close.pct_change().rolling(vol_period).std() * np.sqrt(252)
    raw = ret / vol.replace(0, np.nan)
    return raw.rank(axis=1, pct=True)
