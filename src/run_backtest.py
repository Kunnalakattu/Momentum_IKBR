"""
Entry point for running the momentum strategy backtest.

This script wires together every module:
  data  →  indicators (via signals)  →  backtest engine  →  metrics output

Tweak CONFIG to experiment with different universes, date ranges, or parameters.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt

from src.data import download_close
from src.signals import composite_signal
from src.backtest import run as run_backtest
from src.metrics import summary, drawdown_series

# ---------------------------------------------------------------------------
# CONFIG — change these to run different experiments
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

# Signal parameters
TOP_N          = 10       # hold top-N momentum stocks
USE_TREND      = True     # filter: must be above 200-day MA
USE_VOL_FILTER = True     # filter: annualised vol < 40%

# Backtest parameters
SIZING         = "equal"  # "equal" or "vol_parity"
REBALANCE_FREQ = 21       # rebalance every ~1 month (trading days)
COST_BPS       = 10.0     # one-way transaction cost (10 bps)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Downloading price data...")
    all_tickers = TICKERS + [BENCHMARK_TICKER]
    close_all = download_close(all_tickers, start=START, end=END)
    close_all = close_all.dropna(how="all")

    close = close_all[TICKERS].dropna(how="all", axis=1)
    benchmark_close = close_all[BENCHMARK_TICKER]

    print(f"Universe: {len(close.columns)} tickers | {close.index[0].date()} → {close.index[-1].date()}")

    # ------------------------------------------------------------------
    # Generate signals
    # ------------------------------------------------------------------
    print("Computing signals...")
    signal = composite_signal(
        close,
        n=TOP_N,
        use_trend_filter=USE_TREND,
        use_vol_filter=USE_VOL_FILTER,
    )

    # ------------------------------------------------------------------
    # Run backtest
    # ------------------------------------------------------------------
    print("Running backtest...")
    result = run_backtest(
        close=close,
        signal=signal,
        sizing=SIZING,
        rebalance_freq=REBALANCE_FREQ,
        cost_bps=COST_BPS,
    )

    # ------------------------------------------------------------------
    # Benchmark returns (buy-and-hold SPY)
    # ------------------------------------------------------------------
    bench_returns = benchmark_close.pct_change().reindex(result.returns.index)

    # ------------------------------------------------------------------
    # Print metrics
    # ------------------------------------------------------------------
    print("\n" + "=" * 45)
    print("  MOMENTUM STRATEGY — PERFORMANCE SUMMARY")
    print("=" * 45)
    print(summary(result.returns, benchmark=bench_returns).to_string())
    print()

    bench_summary = summary(bench_returns)
    bench_summary.name = "SPY (B&H)"
    print("  BENCHMARK (SPY buy-and-hold)")
    print("=" * 45)
    print(bench_summary.to_string())

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    result.returns.to_csv(os.path.join(OUTPUT_DIR, "portfolio_returns.csv"), header=["returns"])
    result.equity.to_csv(os.path.join(OUTPUT_DIR, "equity_curve.csv"), header=["equity"])
    result.weights.to_csv(os.path.join(OUTPUT_DIR, "weights.csv"))
    print(f"\nResults saved to {OUTPUT_DIR}/")

    # ------------------------------------------------------------------
    # Plot equity curve + drawdown
    # ------------------------------------------------------------------
    bench_equity = (1 + bench_returns).cumprod()
    dd = drawdown_series(result.returns)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Momentum Strategy vs SPY", fontsize=14)

    axes[0].plot(result.equity, label="Momentum Strategy", linewidth=1.5)
    axes[0].plot(bench_equity, label="SPY (Buy & Hold)", linewidth=1.5, linestyle="--")
    axes[0].set_ylabel("Equity (normalised)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(dd.index, dd.values, 0, color="red", alpha=0.4)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "equity_curve.png")
    plt.savefig(chart_path, dpi=150)
    print(f"Chart saved to {chart_path}")
    plt.show()


if __name__ == "__main__":
    main()
