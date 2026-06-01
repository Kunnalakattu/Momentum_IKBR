"""
Enhanced backtesting engine.

Wraps the original src/backtest.py logic and adds:
  - RiskManager overlay (vol targeting + regime + drawdown control)
  - Richer transaction-cost model  (commission + bid-ask + slippage)
  - Extended BacktestResult with gross returns, cost series, and turnover

The public API mirrors src/backtest.run() so the walk-forward engine
can swap between original and improved without restructuring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Reuse the sizing and rebalance helpers from the original engine
from src.backtest import equal_weight, volatility_parity, _rebalance_weights


@dataclass
class BacktestResult:
    returns:       pd.Series     # net daily portfolio returns
    equity:        pd.Series     # cumulative equity curve (starts at 1.0)
    weights:       pd.DataFrame  # end-of-day weights per asset
    turnover:      pd.Series     # daily one-way turnover
    gross_returns: pd.Series     # before transaction costs
    costs:         pd.Series     # daily cost drag


def run(
    close:          pd.DataFrame,
    signal:         pd.DataFrame,
    market_close:   pd.Series,
    risk_manager,                        # RiskManager | None
    sizing:         str   = "equal",     # "equal" | "vol_parity"
    rebalance_freq: int   = 21,
    cost_bps:       float = 20.0,        # all-in one-way cost in basis points
    vol_period:     int   = 21,
) -> BacktestResult:
    """
    Run an improved backtest with an optional RiskManager overlay.

    Pipeline
    --------
    1. Compute base weights from signal (equal or vol-parity).
    2. Apply rebalancing schedule (hold N days between trades).
    3. Lag weights by 1 day — signals on day T, trade at close T, earn T+1.
    4. Compute static multiplier (vol targeting × regime) from market data.
    5. Apply drawdown-control sequentially (if enabled).
    6. Compute gross return, turnover, and net return after costs.

    cost_bps covers commission (~10 bps) + bid-ask (~5 bps) + slippage (~5 bps).
    For large-cap US equities, 20 bps one-way is a realistic all-in estimate.
    """
    signal = signal.reindex(close.index).fillna(0).astype(float)

    # 1 — Base weights
    if sizing == "vol_parity":
        target_weights = volatility_parity(signal, close, vol_period)
    else:
        target_weights = equal_weight(signal)

    # 2 — Rebalancing schedule (hold weights constant between rebalance days)
    weights = _rebalance_weights(target_weights, rebalance_freq)

    # 3 — Lag by 1 trading day (no look-ahead)
    weights_lagged = weights.shift(1).fillna(0.0)

    # 4–5 — Apply risk management overlay
    if risk_manager is not None:
        static_mult = risk_manager.compute_static_multiplier(market_close)
        static_mult = static_mult.reindex(weights_lagged.index).fillna(1.0)

        # apply_drawdown_control handles both the static mult AND the DD logic
        # in one sequential pass so the DD state tracks actual portfolio equity
        weights_lagged = risk_manager.apply_drawdown_control(
            weights_lagged, close, static_mult
        )
    # (if risk_manager is None the weights_lagged are used as-is)

    # 6 — Returns and costs
    asset_returns = close.pct_change()
    gross_returns = (weights_lagged * asset_returns).sum(axis=1)

    # One-way turnover: sum of absolute daily weight changes per asset
    turnover = weights_lagged.diff().abs().sum(axis=1)
    costs    = turnover * (cost_bps / 10_000)

    net_returns = gross_returns - costs
    equity      = (1 + net_returns).cumprod()

    return BacktestResult(
        returns=net_returns,
        equity=equity,
        weights=weights_lagged,
        turnover=turnover,
        gross_returns=gross_returns,
        costs=costs,
    )
