"""
Simple, rules-based market regime filters.

Two filters are provided:
  vol_regime_multiplier   — reduce exposure when realised vol is elevated
  trend_regime_multiplier — reduce exposure when market is in a downtrend

Both return a pd.Series of multipliers in (0, 1] that can be combined
with a volatility-targeting scaler in RiskManager.  All values are
shifted forward by one day to avoid look-ahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def vol_regime_multiplier(
    market_returns: pd.Series,
    vol_window: int   = 20,
    regime_window: int = 252,
    threshold_pct: float = 0.75,
    reduced_mult: float  = 0.5,
) -> pd.Series:
    """
    Compare 20-day realised vol to its rolling percentile rank.

    When vol is above `threshold_pct` of its trailing `regime_window`-day
    distribution, halve exposure.  This systematically avoids being fully
    invested during volatility spikes (e.g. Feb 2020, Q4 2018, 2022).

    Returns 1.0 (normal) or `reduced_mult` (high-vol), lagged 1 day.
    """
    realised_vol = market_returns.rolling(vol_window).std() * np.sqrt(252)
    # Percentile rank within the rolling history
    pct_rank = realised_vol.rolling(regime_window).rank(pct=True)

    mult = pd.Series(1.0, index=market_returns.index, name="vol_regime")
    mult[pct_rank > threshold_pct] = reduced_mult

    # Lag 1 day: today's filter is based on yesterday's vol observation
    return mult.shift(1).fillna(1.0)


def trend_regime_multiplier(
    market_close: pd.Series,
    ma_window: int  = 200,
    reduced_mult: float = 0.5,
) -> pd.Series:
    """
    SPY above its 200-day MA → full exposure.
    SPY below its 200-day MA → halve exposure.

    The 200-day MA is one of the most robust trend filters in the
    academic and practitioner literature.  It captures major bear
    markets (2008, 2020 crash, 2022) without excessive whipsawing.

    Returns 1.0 or `reduced_mult`, lagged 1 day.
    """
    ma = market_close.rolling(ma_window).mean()
    above_ma = (market_close > ma).astype(float)
    above_ma[above_ma == 0] = reduced_mult

    return above_ma.shift(1).fillna(1.0).rename("trend_regime")


def combined_vol_trend_multiplier(
    market_close: pd.Series,
    market_returns: pd.Series,
    vol_window: int          = 20,
    vol_regime_window: int   = 252,
    vol_threshold: float     = 0.75,
    ma_window: int           = 200,
    below_ma_mult: float     = 0.25,
    high_vol_above_mult: float = 0.6,
    persistence_days: int    = 5,
) -> pd.Series:
    """
    4-state conditional exposure grid:

        above MA + low vol   → 1.0                    (best regime)
        above MA + high vol  → high_vol_above_mult     (0.6 default)
        below MA + low vol   → below_ma_mult           (0.25 default — stay partial)
        below MA + high vol  → 0.0                    (worst regime — flat)

    Persistence filter (persistence_days > 0):
        Regime changes only "confirm" after N consecutive days in the new state.
        This prevents whipsaw exposure cuts on brief MA crossings (e.g. a
        3-day dip below the 200d MA during an otherwise intact uptrend).

    All values lagged 1 day to avoid look-ahead.
    """
    ma       = market_close.rolling(ma_window).mean()
    above_ma = (market_close > ma)

    rv       = market_returns.rolling(vol_window).std() * np.sqrt(252)
    pct_rank = rv.rolling(vol_regime_window).rank(pct=True)
    high_vol = (pct_rank > vol_threshold)

    # Apply persistence filter: only confirm "below MA" after N consecutive days.
    # Until confirmed, the position is treated as still above MA (avoiding premature cuts).
    # We do NOT apply persistence to high_vol — vol changes are smoother and
    # less prone to whipsaw.
    if persistence_days > 0:
        # Rolling sum: True when last persistence_days are ALL below MA
        below_ma_confirmed = (
            (~above_ma).astype(int).rolling(persistence_days).sum() >= persistence_days
        )
        effective_above_ma = ~below_ma_confirmed
    else:
        effective_above_ma = above_ma

    # 4-state grid assignment
    mult = pd.Series(0.0, index=market_close.index, name="combined_regime")
    mult[~effective_above_ma & ~high_vol] = below_ma_mult        # downtrend, calm
    mult[ effective_above_ma & ~high_vol] = 1.0                  # uptrend, calm
    mult[ effective_above_ma &  high_vol] = high_vol_above_mult  # uptrend, turbulent
    # ~effective_above_ma & high_vol stays 0.0 (worst case)

    return mult.shift(1).fillna(1.0)


def combined_regime_multiplier(
    market_close: pd.Series,
    use_vol_regime:   bool  = True,
    use_trend_regime: bool  = True,
    vol_window:       int   = 20,
    regime_window:    int   = 252,
    threshold_pct:    float = 0.75,
    ma_window:        int   = 200,
    vol_reduced_mult: float = 0.5,
    trend_reduced_mult: float = 0.5,
) -> pd.Series:
    """
    Combine vol and trend regime filters multiplicatively.

    Neither fires  → 1.00  (full exposure)
    One fires      → 0.50  (half exposure)
    Both fire      → 0.25  (quarter exposure — strong protective signal)

    Already lagged 1 day (from the individual filters).
    """
    market_returns = market_close.pct_change()
    mult = pd.Series(1.0, index=market_close.index, name="regime")

    if use_vol_regime:
        mult = mult * vol_regime_multiplier(
            market_returns, vol_window, regime_window,
            threshold_pct, vol_reduced_mult,
        )

    if use_trend_regime:
        mult = mult * trend_regime_multiplier(
            market_close, ma_window, trend_reduced_mult,
        )

    return mult
