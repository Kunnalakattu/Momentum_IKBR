"""
Improved momentum strategy — main runner.

Adds on top of the base strategy:
  - Volatility targeting (scale to 15 % annualised)
  - Vol-regime filter   (reduce to 50 % when vol > 75th percentile)
  - 200-day trend filter (reduce to 50 % when SPY < 200d MA)
  - Drawdown circuit-breaker (soft -15 %, hard -20 %, 10-day cooldown)
  - Improved transaction costs (20 bps all-in vs 10 bps original)

Outputs saved to output/:
  improved_oos_returns.csv
  improved_equity_curve.png
  improved_diagnostics.png
  improved_monte_carlo.png
  fold_comparison.csv
  comparison_table.csv
  parameter_sensitivity.csv

Usage:
    python run_improved_strategy.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from src.data_loader import load_universe
from src.risk_management import RiskManager, RiskParams
from src.walk_forward import run as run_improved_wf
from src.walkforward import run as run_original_wf    # original baseline
from src.signals import composite_signal
from src.metrics import summary
import src.diagnostics as diag

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM",
    "V", "UNH", "XOM", "LLY", "JNJ", "WMT", "MA", "PG", "HD", "MRK",
    "AVGO", "CVX", "ABBV", "COST", "PEP", "KO", "ADBE", "CRM", "TMO",
    "ACN", "MCD", "BAC",
]
BENCHMARK_TICKER = "SPY"
START = "2015-01-01"
END   = "2024-12-31"

# Walk-forward settings (same for both runs so comparison is fair)
TRAIN_YEARS    = 3
TEST_YEARS     = 1
TOP_N_GRID     = [5, 7, 10, 12, 15]
USE_TREND      = True
USE_VOL_FILTER = True
SIZING         = "equal"
REBALANCE_FREQ = 21

ORIGINAL_COST_BPS = 10.0   # original strategy
IMPROVED_COST_BPS = 20.0   # improved (adds spread + slippage)

# Risk management — tune these here, not in the walk-forward loop
RISK_PARAMS = RiskParams(
    # Volatility targeting
    target_vol   = 0.15,    # scale portfolio to ~15 % annualised vol
    vol_lookback = 20,      # 20-day realised vol window
    min_vol      = 0.05,    # floor to prevent extreme position inflation
    max_leverage = 1.0,     # no leverage — hard cap at 100 % invested

    # Regime filters
    use_vol_regime    = True,
    use_trend_regime  = True,
    vol_regime_pct    = 0.75,   # reduce when vol > 75th percentile of trailing year
    vol_regime_window = 252,
    trend_ma_window   = 200,

    # Drawdown circuit-breaker
    use_dd_control    = True,
    dd_soft_threshold = -0.15,  # portfolio DD ≤ -15 % → halve exposure
    dd_hard_threshold = -0.20,  # portfolio DD ≤ -20 % → go flat
    dd_recovery       = -0.10,  # re-enter only after DD recovers to -10 %
    cooldown_days     = 10,     # forced cash days after hard stop
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Download data ─────────────────────────────────────────────────
    print("Downloading price data...")
    close, market_close = load_universe(TICKERS, BENCHMARK_TICKER, START, END)
    bench_returns = market_close.pct_change().dropna()

    # ── Run original walk-forward (baseline) ──────────────────────────
    _print_section("ORIGINAL STRATEGY — WALK-FORWARD (baseline)")
    orig_oos, orig_folds = run_original_wf(
        close=close,
        train_years=TRAIN_YEARS,    test_years=TEST_YEARS,
        top_n_grid=TOP_N_GRID,
        use_trend=USE_TREND,        use_vol_filter=USE_VOL_FILTER,
        sizing=SIZING,              rebalance_freq=REBALANCE_FREQ,
        cost_bps=ORIGINAL_COST_BPS,
    )

    # ── Run improved walk-forward ─────────────────────────────────────
    _print_section("IMPROVED STRATEGY — WALK-FORWARD")
    risk_manager = RiskManager(RISK_PARAMS)
    impr_oos, impr_folds = run_improved_wf(
        close=close,               market_close=market_close,
        risk_manager=risk_manager,
        train_years=TRAIN_YEARS,   test_years=TEST_YEARS,
        top_n_grid=TOP_N_GRID,
        use_trend=USE_TREND,       use_vol_filter=USE_VOL_FILTER,
        sizing=SIZING,             rebalance_freq=REBALANCE_FREQ,
        cost_bps=IMPROVED_COST_BPS,
    )

    # ── Align benchmark to OOS period ─────────────────────────────────
    bench_oos = bench_returns.reindex(impr_oos.index).dropna()

    # ── Comparison table ──────────────────────────────────────────────
    _print_section("ORIGINAL vs IMPROVED — OOS METRICS")
    comp = diag.comparison_table(orig_oos, impr_oos, bench_oos)
    print(comp.to_string())

    # ── Fold-by-fold comparison ───────────────────────────────────────
    _print_section("FOLD-BY-FOLD COMPARISON")
    fold_rows = []
    for of, imf in zip(orig_folds, impr_folds):
        fold_rows.append({
            "Fold":      of.fold,
            "Test":      f"{of.test_start.date()} → {of.test_end.date()}",
            "Orig N":    of.best_top_n,
            "Orig Sh":   round(of.oos_sharpe, 2),
            "Orig CAGR": f"{of.oos_cagr:.1%}",
            "Orig MDD":  f"{of.oos_max_dd:.1%}",
            "Impr N":    imf.best_top_n,
            "Impr Sh":   round(imf.oos_sharpe, 2),
            "Impr CAGR": f"{imf.oos_cagr:.1%}",
            "Impr MDD":  f"{imf.oos_max_dd:.1%}",
            "Avg TO":    f"{imf.avg_turnover:.3f}",
        })
    fold_df = pd.DataFrame(fold_rows)
    print(fold_df.to_string(index=False))

    # ── Regime performance ────────────────────────────────────────────
    _print_section("IMPROVED STRATEGY — REGIME PERFORMANCE")
    rp = diag.regime_performance(impr_oos, market_close, market_close.pct_change())
    print(rp.to_string())

    # ── Parameter sensitivity (run on in-sample 2015-2020 only) ──────
    _print_section("PARAMETER SENSITIVITY (in-sample 2015–2020)")
    close_is  = close.loc["2015":"2020"]
    mkt_is    = market_close.loc["2015":"2020"]

    def _sig_fn(c: pd.DataFrame, n: int) -> pd.DataFrame:
        return composite_signal(c, n=n,
                                use_trend_filter=USE_TREND,
                                use_vol_filter=USE_VOL_FILTER)

    sens = diag.parameter_sensitivity(
        close=close_is, market_close=mkt_is,
        signal_fn=_sig_fn, top_n=10,
        param_grid={
            "target_vol":        [0.10, 0.15, 0.20],
            "dd_soft_threshold": [-0.10, -0.15, -0.20],
        },
        cost_bps=IMPROVED_COST_BPS,
    )
    print(sens.to_string(index=False))

    # ── Save CSVs ────────────────────────────────────────────────────
    impr_oos.to_csv(
        os.path.join(OUTPUT_DIR, "improved_oos_returns.csv"), header=["returns"]
    )
    comp.to_csv(os.path.join(OUTPUT_DIR, "comparison_table.csv"))
    fold_df.to_csv(os.path.join(OUTPUT_DIR, "fold_comparison.csv"), index=False)
    sens.to_csv(os.path.join(OUTPUT_DIR, "parameter_sensitivity.csv"), index=False)
    print(f"\nCSV files saved to {OUTPUT_DIR}/")

    # ── Charts ───────────────────────────────────────────────────────
    fold_dates = [(f.test_start, f.test_end) for f in impr_folds]

    print("Generating charts...")

    diag.plot_equity_comparison(
        original_returns  = orig_oos,
        improved_returns  = impr_oos,
        benchmark_returns = bench_oos,
        original_label    = "Original (10 bps)",
        improved_label    = "Improved (VolTgt + Regime + DD)",
        title=(
            f"Original vs Improved — OOS  "
            f"({impr_oos.index[0].date()} → {impr_oos.index[-1].date()})"
        ),
        fold_dates = fold_dates,
        save_path  = os.path.join(OUTPUT_DIR, "improved_equity_curve.png"),
    )

    diag.plot_rolling_diagnostics(
        returns   = impr_oos,
        benchmark = bench_oos,
        title     = "Improved Strategy — Rolling Diagnostics (OOS)",
        save_path = os.path.join(OUTPUT_DIR, "improved_diagnostics.png"),
    )

    diag.plot_monte_carlo(
        returns       = impr_oos,
        n_simulations = 500,
        title         = "Monte Carlo (Block Bootstrap) — Improved OOS",
        save_path     = os.path.join(OUTPUT_DIR, "improved_monte_carlo.png"),
    )

    print(f"Charts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
