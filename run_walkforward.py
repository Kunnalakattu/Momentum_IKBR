"""
Walk-forward + out-of-sample validation runner.

Splits 2015-2024 into rolling folds (3yr train / 1yr test by default).
On each fold, optimises TOP_N on the training window, then evaluates on
the test window.  Concatenated test returns form a fully out-of-sample
equity curve.

Usage:
    python run_walkforward.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from src.data import download_close
from src.walkforward import run as run_walkforward
from src.metrics import summary, drawdown_series

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM",
    "V", "UNH", "XOM", "LLY", "JNJ", "WMT", "MA", "PG", "HD", "MRK",
    "AVGO", "CVX", "ABBV", "COST", "PEP", "KO", "ADBE", "CRM", "TMO",
    "ACN", "MCD", "BAC",
]
BENCHMARK_TICKER = "SPY"
START = "2015-01-01"
END   = "2024-12-31"

TRAIN_YEARS    = 3
TEST_YEARS     = 1
TOP_N_GRID     = [5, 7, 10, 12, 15]
USE_TREND      = True
USE_VOL_FILTER = True
SIZING         = "equal"
REBALANCE_FREQ = 21
COST_BPS       = 10.0

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Downloading price data...")
    all_tickers = TICKERS + [BENCHMARK_TICKER]
    close_all = download_close(all_tickers, start=START, end=END)
    close_all = close_all.dropna(how="all")

    close           = close_all[TICKERS].dropna(how="all", axis=1)
    benchmark_close = close_all[BENCHMARK_TICKER]

    print(
        f"Universe: {len(close.columns)} tickers  |  "
        f"{close.index[0].date()} → {close.index[-1].date()}\n"
    )
    print(
        f"Walk-forward config:  train={TRAIN_YEARS}yr  test={TEST_YEARS}yr  "
        f"TOP_N grid={TOP_N_GRID}\n"
    )

    # ------------------------------------------------------------------
    # Run walk-forward
    # ------------------------------------------------------------------
    oos_returns, folds = run_walkforward(
        close=close,
        train_years=TRAIN_YEARS,
        test_years=TEST_YEARS,
        top_n_grid=TOP_N_GRID,
        use_trend=USE_TREND,
        use_vol_filter=USE_VOL_FILTER,
        sizing=SIZING,
        rebalance_freq=REBALANCE_FREQ,
        cost_bps=COST_BPS,
    )

    # ------------------------------------------------------------------
    # Per-fold table
    # ------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("  WALK-FORWARD FOLD SUMMARY")
    print("=" * 88)
    header = (
        f"  {'Fold':>4}  {'Train period':<23}  {'Test period':<23}  "
        f"{'Best N':>6}  {'Train Sh':>8}  {'OOS Sh':>7}  {'OOS CAGR':>9}  {'OOS MDD':>8}"
    )
    print(header)
    print("  " + "-" * 86)
    for f in folds:
        print(
            f"  {f.fold:>4}  "
            f"{str(f.train_start.date())+' → '+str(f.train_end.date()):<23}  "
            f"{str(f.test_start.date())+' → '+str(f.test_end.date()):<23}  "
            f"{f.best_top_n:>6}  "
            f"{f.train_sharpe:>8.2f}  "
            f"{f.oos_sharpe:>7.2f}  "
            f"{f.oos_cagr:>8.1%}  "
            f"{f.oos_max_dd:>7.1%}"
        )

    # ------------------------------------------------------------------
    # Overall OOS metrics
    # ------------------------------------------------------------------
    oos_start = oos_returns.index[0].date()
    oos_end   = oos_returns.index[-1].date()
    bench_returns = benchmark_close.pct_change().reindex(oos_returns.index)

    print(f"\n{'='*55}")
    print(f"  OOS PERFORMANCE  ({oos_start} → {oos_end})")
    print(f"{'='*55}")
    print(summary(oos_returns, benchmark=bench_returns).to_string())
    print()

    bench_summary = summary(bench_returns)
    bench_summary.name = "SPY (B&H)"
    print(f"  BENCHMARK  (same OOS period)")
    print(f"{'='*55}")
    print(bench_summary.to_string())

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    oos_returns.to_csv(os.path.join(OUTPUT_DIR, "wf_oos_returns.csv"), header=["returns"])
    print(f"\nOOS returns saved → output/wf_oos_returns.csv")

    # ------------------------------------------------------------------
    # Plot: OOS equity curve + drawdown, fold boundaries shaded
    # ------------------------------------------------------------------
    oos_equity    = (1 + oos_returns).cumprod()
    bench_equity  = (1 + bench_returns).cumprod()
    dd            = drawdown_series(oos_returns)

    fig, axes = plt.subplots(
        2, 1, figsize=(13, 8),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle(
        f"Walk-Forward OOS  —  {TRAIN_YEARS}yr train / {TEST_YEARS}yr test  "
        f"(TOP_N optimised per fold)",
        fontsize=13,
    )

    # Shade alternating folds for readability
    colors = ["#e8f4f8", "#f8e8e8"]
    for i, f in enumerate(folds):
        for ax in axes:
            ax.axvspan(f.test_start, f.test_end, alpha=0.35, color=colors[i % 2], zorder=0)

    axes[0].plot(oos_equity,   label="Momentum (OOS)", linewidth=1.5)
    axes[0].plot(bench_equity, label="SPY (Buy & Hold)", linewidth=1.5, linestyle="--", color="grey")
    axes[0].set_ylabel("Equity (normalised)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(dd.index, dd.values, 0, color="red", alpha=0.4)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "wf_equity_curve.png")
    plt.savefig(chart_path, dpi=150)
    print(f"Chart saved → output/wf_equity_curve.png")
    plt.show()


if __name__ == "__main__":
    main()
