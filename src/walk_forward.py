"""
Walk-forward engine for the improved (risk-managed) momentum strategy.

Walk-forward purity rules
--------------------------
1. TOP_N is optimised ONLY on the training window using the BASE strategy
   (no risk management) so risk-management parameters never bleed into
   the parameter-selection loop.
2. Risk management is applied only on the out-of-sample test window.
3. The test signal is computed on close[train_start : test_end] so that
   indicators (12-month momentum, 200-day MA) have their full history on
   the very first day of the test window — without using any future test
   observations.
4. No test-period data is touched during training or optimisation.

This file is intentionally named walk_forward.py (underscore) to
co-exist with the original walkforward.py (no underscore).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.signals import composite_signal, improved_composite_signal
from src.backtester import run as _run_improved
from src.backtest import run as _run_base          # original, for train-opt only
from src.metrics import sharpe_ratio, cagr, max_drawdown


@dataclass
class FoldResult:
    fold:          int
    train_start:   pd.Timestamp
    train_end:     pd.Timestamp
    test_start:    pd.Timestamp
    test_end:      pd.Timestamp
    best_top_n:    int
    train_sharpe:  float
    oos_returns:   pd.Series
    oos_sharpe:    float
    oos_cagr:      float
    oos_max_dd:    float
    avg_turnover:  float        # mean daily one-way turnover


def run(
    close:               pd.DataFrame,
    market_close:        pd.Series,
    risk_manager,                               # RiskManager instance or None
    train_years:         int        = 3,
    test_years:          int        = 1,
    top_n_grid:          list[int]  = None,
    use_trend:           bool       = True,
    use_vol_filter:      bool       = True,
    sizing:              str        = "equal",
    rebalance_freq:      int        = 21,
    cost_bps:            float      = 20.0,
    use_improved_signal: bool       = False,    # use multi-factor signal (residual + vol-adj)
) -> tuple[pd.Series, list[FoldResult]]:
    """
    Walk-forward validation for the risk-managed momentum strategy.

    Parameters
    ----------
    close          : Full adjusted close prices for the whole date range.
    market_close   : Benchmark (SPY) close, used by the risk manager.
    risk_manager   : RiskManager instance.  Pass None to run base strategy.
    train_years    : Training window length in years.
    test_years     : Test (OOS) window length in years.
    top_n_grid     : Candidate TOP_N values to optimise on training folds.
    use_trend      : 200-day MA filter in composite_signal.
    use_vol_filter : Annualised vol < 40 % filter in composite_signal.
    sizing         : "equal" or "vol_parity".
    rebalance_freq : Trading days between rebalances.
    cost_bps       : All-in one-way transaction cost in basis points.

    Returns
    -------
    oos_returns  : pd.Series  — concatenated OOS daily returns.
    fold_results : list[FoldResult]
    """
    if top_n_grid is None:
        top_n_grid = [5, 7, 10, 12, 15]

    dates      = close.index
    data_start = dates[0]
    data_end   = dates[-1]

    # ── Build rolling fold boundaries ─────────────────────────────────
    folds: list[tuple] = []
    fold_start = data_start
    while True:
        train_end = fold_start + pd.DateOffset(years=train_years)
        test_end  = train_end  + pd.DateOffset(years=test_years)
        if test_end > data_end:
            break
        folds.append((fold_start, train_end, train_end, test_end))
        fold_start = fold_start + pd.DateOffset(years=test_years)

    if not folds:
        raise ValueError(
            f"Not enough data for one fold.  "
            f"Need {train_years + test_years}+ years; "
            f"data covers {data_start.date()} → {data_end.date()}."
        )

    fold_results:    list[FoldResult] = []
    oos_return_list: list[pd.Series]  = []

    for idx, (train_start, train_end, test_start, test_end) in enumerate(folds, 1):
        close_train = close.loc[train_start : train_end]
        close_test  = close.loc[test_start  : test_end]
        mkt_test    = market_close.loc[test_start : test_end]

        if len(close_train) < 252 or len(close_test) < 20:
            print(f"  Fold {idx}: skipped — insufficient rows.")
            continue

        # ── Step 1: optimise TOP_N on training window ─────────────────
        # Always use standard composite_signal for optimisation (fast, no market data).
        # Risk management and signal upgrades are applied only in Step 2/3.
        mkt_train = market_close.loc[train_start : train_end]
        best_n, best_train_sh = top_n_grid[0], -np.inf
        for n in top_n_grid:
            sig = composite_signal(
                close_train, n=n,
                use_trend_filter=use_trend,
                use_vol_filter=use_vol_filter,
            )
            res = _run_base(
                close=close_train, signal=sig,
                sizing=sizing, rebalance_freq=rebalance_freq, cost_bps=cost_bps,
            )
            sh = sharpe_ratio(res.returns.dropna())
            if sh > best_train_sh:
                best_train_sh, best_n = sh, n

        # ── Step 2: compute OOS signal with full training lookback ─────
        close_full = close.loc[train_start : test_end]
        mkt_full   = market_close.loc[train_start : test_end]

        if use_improved_signal:
            sig_full = improved_composite_signal(
                close_full, market_close=mkt_full, n=best_n,
                use_trend_filter=use_trend,
                use_vol_filter=use_vol_filter,
            )
        else:
            sig_full = composite_signal(
                close_full, n=best_n,
                use_trend_filter=use_trend,
                use_vol_filter=use_vol_filter,
            )
        sig_test = (
            sig_full.loc[test_start : test_end]
            .reindex(close_test.index)
            .fillna(0)
        )

        # ── Step 3: improved backtest on test window ───────────────────
        res_test = _run_improved(
            close=close_test,
            signal=sig_test,
            market_close=mkt_test,
            risk_manager=risk_manager,
            sizing=sizing,
            rebalance_freq=rebalance_freq,
            cost_bps=cost_bps,
        )

        oos_ret = res_test.returns.dropna()
        oos_sh  = sharpe_ratio(oos_ret)
        oos_cg  = cagr(oos_ret)
        oos_mdd = max_drawdown(oos_ret)
        avg_to  = float(res_test.turnover.mean())

        print(
            f"  Fold {idx:>2}  "
            f"train {train_start.date()} → {train_end.date()}  "
            f"test {test_start.date()} → {test_end.date()}  │  "
            f"N={best_n}  trSh={best_train_sh:.2f}  │  "
            f"OOS Sh={oos_sh:.2f}  CAGR={oos_cg:.1%}  MDD={oos_mdd:.1%}  "
            f"TO={avg_to:.3f}"
        )

        fold_results.append(FoldResult(
            fold=idx,
            train_start=train_start, train_end=train_end,
            test_start=test_start,   test_end=test_end,
            best_top_n=best_n,       train_sharpe=best_train_sh,
            oos_returns=oos_ret,
            oos_sharpe=oos_sh,       oos_cagr=oos_cg,
            oos_max_dd=oos_mdd,      avg_turnover=avg_to,
        ))
        oos_return_list.append(oos_ret)

    if not oos_return_list:
        raise RuntimeError("All folds were skipped — check data length.")

    oos_returns = pd.concat(oos_return_list).sort_index()
    # Drop boundary-date duplicates that arise when fold i ends on the same
    # calendar date that fold i+1 begins (inclusive pandas slicing).
    oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")]
    return oos_returns, fold_results
