import pandas as pd
import numpy as np
from src.indicators import (
    momentum_score,
    risk_adjusted_momentum,
    rsi,
    volatility,
    dual_momentum,
    residual_momentum_score,
    vol_adjusted_momentum_score,
)


# ---------------------------------------------------------------------------
# Cross-sectional signals (rank-based, multi-asset)
# ---------------------------------------------------------------------------

def top_n_signal(close: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Buy the top-N momentum stocks each period, sell everything else.

    Returns a boolean DataFrame — True = hold, False = out.
    Rebalance by applying this signal at your chosen frequency.
    """
    scores = momentum_score(close)
    ranked = scores.rank(axis=1, ascending=False)
    return ranked <= n


def percentile_signal(close: pd.DataFrame, top_pct: float = 0.2, bottom_pct: float = 0.2) -> pd.DataFrame:
    """Long top percentile, short bottom percentile (long/short momentum).

    Returns a DataFrame with values: 1 (long), -1 (short), 0 (neutral).
    """
    scores = momentum_score(close)
    signal = pd.DataFrame(0, index=scores.index, columns=scores.columns)
    upper = scores.quantile(1 - top_pct, axis=1)
    lower = scores.quantile(bottom_pct, axis=1)
    signal[scores.gt(upper, axis=0)] = 1
    signal[scores.lt(lower, axis=0)] = -1
    return signal


def risk_adjusted_signal(close: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top-N signal ranked by risk-adjusted momentum (momentum / volatility)."""
    scores = risk_adjusted_momentum(close)
    ranked = scores.rank(axis=1, ascending=False)
    return ranked <= n


# ---------------------------------------------------------------------------
# Time-series / absolute momentum signals (single asset)
# ---------------------------------------------------------------------------

def trend_signal(close: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """Golden/death cross: buy when fast MA crosses above slow MA.

    Returns 1 (long) or 0 (flat).
    """
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    return (fast_ma > slow_ma).astype(int)


def rsi_signal(close: pd.DataFrame, period: int = 14, oversold: int = 30, overbought: int = 70) -> pd.DataFrame:
    """RSI mean-reversion signal: buy on oversold, sell on overbought.

    Returns 1 (long), -1 (short), 0 (neutral).
    """
    r = rsi(close, period)
    signal = pd.DataFrame(0, index=r.index, columns=r.columns)
    signal[r < oversold] = 1
    signal[r > overbought] = -1
    return signal


def absolute_momentum_signal(close: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Hold asset only when its own trailing return is positive (time-series momentum).

    Returns 1 (long) or 0 (flat).
    """
    ret = close.pct_change(lookback)
    return (ret > 0).astype(int)


def breakout_signal(close: pd.DataFrame, period: int = 52) -> pd.DataFrame:
    """52-week high breakout: buy when price makes a new high over the window.

    Returns 1 (long) or 0 (flat).
    """
    rolling_high = close.rolling(period).max()
    at_high = close >= rolling_high.shift(1)
    signal = at_high.astype(int)
    # Stay long until price falls below the rolling high
    return signal.replace(0, np.nan).ffill().fillna(0).astype(int)


# ---------------------------------------------------------------------------
# Combined / composite signals
# ---------------------------------------------------------------------------

def composite_signal(
    close: pd.DataFrame,
    n: int = 10,
    use_trend_filter: bool = True,
    use_vol_filter: bool = True,
    vol_threshold: float = 0.40,
) -> pd.DataFrame:
    """Composite buy signal combining cross-sectional rank, trend, and vol filter.

    Logic:
      1. Select top-N momentum stocks (cross-sectional rank).
      2. Optionally filter out stocks below their 200-day MA (trend filter).
      3. Optionally filter out stocks with annualised vol > threshold (vol filter).

    Returns 1 (buy/hold) or 0 (out).
    """
    signal = top_n_signal(close, n=n).astype(int)

    if use_trend_filter:
        above_ma200 = (close > close.rolling(200).mean()).astype(int)
        signal = signal & above_ma200

    if use_vol_filter:
        low_vol = (volatility(close) < vol_threshold).astype(int)
        signal = signal & low_vol

    return signal.astype(int)


def dual_momentum_signal(
    close: pd.DataFrame,
    benchmark: pd.Series,
    lookback: int = 252,
) -> pd.DataFrame:
    """Signal based on Antonacci's dual momentum — absolute + relative filter.

    Returns 1 (long) or 0 (flat).
    """
    return dual_momentum(close, benchmark, lookback).astype(int)


# ---------------------------------------------------------------------------
# Improved composite signal (multi-factor ranking)
# ---------------------------------------------------------------------------

def improved_composite_signal(
    close: pd.DataFrame,
    market_close: pd.Series | None = None,
    n: int = 10,
    use_trend_filter: bool = True,
    use_vol_filter: bool = True,
    vol_threshold: float = 0.40,
    weights_standard: float = 0.50,
    weights_vol_adj: float  = 0.25,
    weights_residual: float = 0.25,
) -> pd.DataFrame:
    """
    Multi-factor momentum signal combining three ranking scores:

      1. Standard momentum   (avg 3m/6m/12m percentile rank)   — 50 % weight
      2. Vol-adjusted score  (momentum / realised vol)           — 25 % weight
      3. Residual momentum   (beta-adjusted idiosyncratic alpha) — 25 % weight
         (requires market_close; falls back to equal split if None)

    Each component is cross-sectionally percentile-ranked so they are on
    the same scale before combining.  The weighted average rank is used to
    select the top-N stocks each period.

    Trend and vol filters (same as composite_signal) are applied last.
    """
    # ── Factor scores (all in [0, 1] cross-sectional percentile rank) ─
    score_standard = momentum_score(close)            # already ranked

    score_vol_adj  = vol_adjusted_momentum_score(close)

    if market_close is not None and not market_close.empty:
        score_residual = residual_momentum_score(close, market_close)
        w_s = weights_standard
        w_v = weights_vol_adj
        w_r = weights_residual
    else:
        # No market data — fall back to equal split between two factors
        score_residual = score_vol_adj   # dummy, will be zero-weighted
        total = weights_standard + weights_vol_adj
        w_s = weights_standard / total
        w_v = weights_vol_adj  / total
        w_r = 0.0

    composite = w_s * score_standard + w_v * score_vol_adj + w_r * score_residual

    # ── Select top-N ─────────────────────────────────────────────────
    ranked = composite.rank(axis=1, ascending=False)
    signal = (ranked <= n).astype(int)

    # ── Trend filter: stock price > its own 200-day MA ────────────────
    if use_trend_filter:
        above_ma200 = (close > close.rolling(200).mean()).astype(int)
        signal = signal & above_ma200

    # ── Volatility filter: annualised vol < threshold ─────────────────
    if use_vol_filter:
        ann_vol  = close.pct_change().rolling(21).std() * np.sqrt(252)
        low_vol  = (ann_vol < vol_threshold).astype(int)
        signal   = signal & low_vol

    return signal.astype(int)


# ---------------------------------------------------------------------------
# Entry / exit event generation
# ---------------------------------------------------------------------------

def entry_exit_events(signal: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Convert a 0/1 signal DataFrame into explicit entry and exit event DataFrames.

    Returns:
        {
            "entries": DataFrame — True on the bar where signal flips from 0 to 1,
            "exits":   DataFrame — True on the bar where signal flips from 1 to 0,
        }
    """
    entries = (signal.diff() == 1)
    exits = (signal.diff() == -1)
    return {"entries": entries, "exits": exits}
