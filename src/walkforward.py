"""
Walk-forward out-of-sample validation engine.

For each fold:
  1. Train window: find the TOP_N (from a grid) that maximises Sharpe.
  2. Test window:  run the strategy with that TOP_N, collect OOS returns.

Concatenating the test-window returns gives a fully out-of-sample equity
curve with no parameter look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.signals import composite_signal
from src.backtest import run as _run_backtest
from src.metrics import sharpe_ratio, cagr, max_drawdown


@dataclass
class FoldResult:
    fold:        int
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp
    best_top_n:  int
    train_sharpe: float
    oos_returns: pd.Series
    oos_sharpe:  float
    oos_cagr:    float
    oos_max_dd:  float


def run(
    close: pd.DataFrame,
    train_years:    int        = 3,
    test_years:     int        = 1,
    top_n_grid:     list[int]  = None,
    use_trend:      bool       = True,
    use_vol_filter: bool       = True,
    sizing:         str        = "equal",
    rebalance_freq: int        = 21,
    cost_bps:       float      = 10.0,
) -> tuple[pd.Series, list[FoldResult]]:
    """
    Walk-forward validation for the momentum strategy.

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
            f"Not enough data for one fold. "
            f"Need at least {train_years + test_years} years; "
            f"data covers {data_start.date()} → {data_end.date()}."
        )

    fold_results:   list[FoldResult] = []
    oos_return_list: list[pd.Series]  = []

    for idx, (train_start, train_end, test_start, test_end) in enumerate(folds, 1):
        close_train = close.loc[train_start : train_end]
        close_test  = close.loc[test_start  : test_end]

        if len(close_train) < 252 or len(close_test) < 20:
            print(f"  Fold {idx}: skipped — insufficient rows.")
            continue

        # Step 1: optimise TOP_N on training window
        best_n, best_train_sharpe = top_n_grid[0], -np.inf
        for n in top_n_grid:
            sig = composite_signal(
                close_train, n=n,
                use_trend_filter=use_trend,
                use_vol_filter=use_vol_filter,
            )
            res = _run_backtest(
                close=close_train, signal=sig,
                sizing=sizing, rebalance_freq=rebalance_freq, cost_bps=cost_bps,
            )
            sh = sharpe_ratio(res.returns.dropna())
            if sh > best_train_sharpe:
                best_train_sharpe, best_n = sh, n

        # Step 2: compute OOS signal with full training lookback
        close_full = close.loc[train_start : test_end]
        sig_full   = composite_signal(
            close_full, n=best_n,
            use_trend_filter=use_trend,
            use_vol_filter=use_vol_filter,
        )
        sig_test = sig_full.loc[test_start : test_end].reindex(close_test.index).fillna(0)

        # Step 3: backtest on test window only
        res_test = _run_backtest(
            close=close_test, signal=sig_test,
            sizing=sizing, rebalance_freq=rebalance_freq, cost_bps=cost_bps,
        )

        oos_ret = res_test.returns.dropna()
        oos_sh  = sharpe_ratio(oos_ret)
        oos_cg  = cagr(oos_ret)
        oos_mdd = max_drawdown(oos_ret)

        print(
            f"  Fold {idx:>2}  "
            f"train {train_start.date()} → {train_end.date()}  "
            f"test {test_start.date()} → {test_end.date()}  │  "
            f"best N={best_n}  train Sh={best_train_sharpe:.2f}  │  "
            f"OOS Sh={oos_sh:.2f}  CAGR={oos_cg:.1%}  MDD={oos_mdd:.1%}"
        )

        fold_results.append(FoldResult(
            fold=idx,
            train_start=train_start, train_end=train_end,
            test_start=test_start,   test_end=test_end,
            best_top_n=best_n,       train_sharpe=best_train_sharpe,
            oos_returns=oos_ret,
            oos_sharpe=oos_sh,       oos_cagr=oos_cg,  oos_max_dd=oos_mdd,
        ))
        oos_return_list.append(oos_ret)

    if not oos_return_list:
        raise RuntimeError("All folds were skipped — check data length.")

    oos_returns = pd.concat(oos_return_list).sort_index()
    oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")]
    return oos_returns, fold_results
