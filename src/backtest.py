"""
Backtesting engine.

Responsibilities:
  - Convert a signal DataFrame into daily portfolio weights.
  - Apply rebalancing frequency and transaction costs.
  - Compute portfolio return series and equity curve.
  - Return a BacktestResult that any runner can consume.
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class BacktestResult:
    returns: pd.Series          # daily portfolio returns
    equity: pd.Series           # cumulative equity curve (starts at 1.0)
    weights: pd.DataFrame       # end-of-day weights per asset
    turnover: pd.Series         # daily one-way turnover


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def equal_weight(signal: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight the active positions from a 0/1 or boolean signal."""
    active = signal.astype(float)
    row_sum = active.sum(axis=1).replace(0, np.nan)
    return active.div(row_sum, axis=0).fillna(0.0)


def volatility_parity(signal: pd.DataFrame, close: pd.DataFrame, vol_period: int = 21) -> pd.DataFrame:
    """Weight each active position inverse to its recent volatility."""
    daily_ret = close.pct_change()
    vol = daily_ret.rolling(vol_period).std()
    inv_vol = (1.0 / vol).where(signal.astype(bool), other=0.0)
    row_sum = inv_vol.sum(axis=1).replace(0, np.nan)
    return inv_vol.div(row_sum, axis=0).fillna(0.0)


# ---------------------------------------------------------------------------
# Rebalancing
# ---------------------------------------------------------------------------

def _rebalance_weights(target: pd.DataFrame, freq: int) -> pd.DataFrame:
    """Hold weights constant between rebalance dates.

    Args:
        target: desired weights on every bar
        freq:   rebalance every N trading days (e.g. 21 = monthly)
    """
    held = target.copy()
    rebalance_mask = pd.Series(False, index=target.index)
    rebalance_mask.iloc[::freq] = True
    last = held.iloc[0].copy()
    for i, (idx, _) in enumerate(held.iterrows()):
        if rebalance_mask.iloc[i]:
            last = target.loc[idx].copy()
        held.loc[idx] = last
    return held


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

def run(
    close: pd.DataFrame,
    signal: pd.DataFrame,
    sizing: str = "equal",      # "equal" | "vol_parity"
    rebalance_freq: int = 21,   # trading days between rebalances
    cost_bps: float = 10.0,    # one-way transaction cost in basis points
    vol_period: int = 21,
) -> BacktestResult:
    """Run a vectorised backtest.

    Args:
        close:          Adjusted close prices, shape (dates, tickers).
        signal:         0/1 or boolean signal aligned to close, same shape.
        sizing:         Position sizing method: "equal" or "vol_parity".
        rebalance_freq: Hold weights for this many days before rebalancing.
        cost_bps:       One-way cost per trade in basis points (10 bps = 0.10%).
        vol_period:     Lookback for volatility parity sizing.

    Returns:
        BacktestResult with returns, equity curve, weights, and daily turnover.
    """
    signal = signal.reindex(close.index).fillna(0).astype(float)

    # 1. Compute target weights on every bar
    if sizing == "vol_parity":
        target_weights = volatility_parity(signal, close, vol_period)
    else:
        target_weights = equal_weight(signal)

    # 2. Apply rebalancing schedule — hold weights between rebalance dates
    weights = _rebalance_weights(target_weights, rebalance_freq)

    # 3. Forward-shift weights to avoid look-ahead: trade on close, earn next day
    weights_lagged = weights.shift(1).fillna(0.0)

    # 4. Daily asset returns
    asset_returns = close.pct_change()

    # 5. Gross portfolio return
    gross_returns = (weights_lagged * asset_returns).sum(axis=1)

    # 6. Transaction costs: turnover * cost
    turnover = weights_lagged.diff().abs().sum(axis=1)
    cost = turnover * (cost_bps / 10_000)
    net_returns = gross_returns - cost

    equity = (1 + net_returns).cumprod()

    return BacktestResult(
        returns=net_returns,
        equity=equity,
        weights=weights_lagged,
        turnover=turnover,
    )
