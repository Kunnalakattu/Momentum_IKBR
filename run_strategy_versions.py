"""
Strategy version comparison runner.

Versions tested
---------------
Original  — base momentum, no risk overlay, 10 bps
D         — 4-state conditional exposure + persistence filter + soft DD scaling
E         — Version D + improved multi-factor signal (residual + vol-adjusted)

4-state exposure grid (D and E):
    above MA + low vol   → 1.0   full exposure
    above MA + high vol  → 0.6   profitable but turbulent — reduce
    below MA + low vol   → 0.25  downtrend but calm — stay partial
    below MA + high vol  → 0.0   worst regime — flat

Persistence filter: regime confirmed only after 5 consecutive days → no whipsaw.
Soft DD scaling: if portfolio drawdown < -10 %, multiply exposure by 0.7 (no hard stop).

Usage:
    python run_strategy_versions.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_universe
from src.risk_management import RiskManager, RiskParams
from src.walk_forward import run as run_wf
from src.walkforward import run as run_original_wf
from src.metrics import (
    cagr, annualised_volatility, sharpe_ratio, sortino_ratio,
    max_drawdown, calmar_ratio, beta, alpha,
)
import src.diagnostics as diag

# ─────────────────────────────────────────────────────────────────────
# CONFIG  (shared across all versions)
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

TRAIN_YEARS    = 3
TEST_YEARS     = 1
TOP_N_GRID     = [5, 7, 10, 12, 15]
USE_TREND      = True
USE_VOL_FILTER = True
SIZING         = "equal"
REBALANCE_FREQ = 21

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# ─────────────────────────────────────────────────────────────────────
# Risk parameter configs
# ─────────────────────────────────────────────────────────────────────

# D — 4-state exposure + persistence + soft DD scaling
PARAMS_D = RiskParams(
    use_vol_targeting   = True,
    target_vol          = 0.15,
    max_leverage        = 1.0,        # no leverage; rely on signal quality
    use_combined_regime = True,
    below_ma_mult       = 0.25,       # below MA + low vol  → 25 % exposure
    high_vol_above_mult = 0.60,       # above MA + high vol → 60 % exposure
    persistence_days    = 5,          # 5 consecutive days to confirm regime
    use_dd_control      = True,
    dd_scale_threshold  = -0.10,      # soft trigger at -10 % drawdown
    dd_scale_factor     = 0.70,       # scale to 70 % when triggered
)

# E — same risk overlay as D, improved multi-factor signal
PARAMS_E = RiskParams(
    use_vol_targeting   = True,
    target_vol          = 0.15,
    max_leverage        = 1.0,
    use_combined_regime = True,
    below_ma_mult       = 0.25,
    high_vol_above_mult = 0.60,
    persistence_days    = 5,
    use_dd_control      = True,
    dd_scale_threshold  = -0.10,
    dd_scale_factor     = 0.70,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def _fold_table(label: str, folds: list) -> None:
    print(f"\n  {label}")
    print(f"  {'Fold':>4}  {'Test period':<23}  {'N':>4}  "
          f"{'Sharpe':>7}  {'CAGR':>8}  {'MDD':>8}")
    print("  " + "-" * 62)
    for f in folds:
        print(f"  {f.fold:>4}  "
              f"{str(f.test_start.date())+' → '+str(f.test_end.date()):<23}  "
              f"{f.best_top_n:>4}  "
              f"{f.oos_sharpe:>7.2f}  "
              f"{f.oos_cagr:>8.1%}  "
              f"{f.oos_max_dd:>7.1%}")


def _metrics_row(name: str, r: pd.Series, b: pd.Series) -> dict:
    return {
        "Version":  name,
        "CAGR":     f"{cagr(r):.1%}",
        "Vol":      f"{annualised_volatility(r):.1%}",
        "Sharpe":   f"{sharpe_ratio(r):.2f}",
        "Sortino":  f"{sortino_ratio(r):.2f}",
        "Max DD":   f"{max_drawdown(r):.1%}",
        "Calmar":   f"{calmar_ratio(r):.2f}",
        "Beta":     f"{beta(r, b):.2f}",
        "Alpha":    f"{alpha(r, b):.1%}",
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Downloading price data...")
    close, market_close = load_universe(TICKERS, BENCHMARK_TICKER, START, END)
    bench_returns = market_close.pct_change().dropna()

    # ── Original baseline (10 bps, no risk overlay) ───────────────────
    _section("ORIGINAL — baseline (10 bps, standard signal)")
    orig_oos, orig_folds = run_original_wf(
        close=close,
        train_years=TRAIN_YEARS,    test_years=TEST_YEARS,
        top_n_grid=TOP_N_GRID,
        use_trend=USE_TREND,        use_vol_filter=USE_VOL_FILTER,
        sizing=SIZING,              rebalance_freq=REBALANCE_FREQ,
        cost_bps=10.0,
    )
    _fold_table("Original", orig_folds)

    # ── Version D: 4-state exposure + persistence + soft DD ──────────
    _section("D — 4-state regime + persistence(5d) + soft DD scaling (standard signal)")
    d_oos, d_folds = run_wf(
        close=close,                market_close=market_close,
        risk_manager=RiskManager(PARAMS_D),
        train_years=TRAIN_YEARS,    test_years=TEST_YEARS,
        top_n_grid=TOP_N_GRID,
        use_trend=USE_TREND,        use_vol_filter=USE_VOL_FILTER,
        sizing=SIZING,              rebalance_freq=REBALANCE_FREQ,
        cost_bps=20.0,
        use_improved_signal=False,
    )
    _fold_table("D", d_folds)

    # ── Version E: same as D + improved multi-factor signal ───────────
    _section("E — same as D + improved signal (residual + vol-adjusted momentum)")
    e_oos, e_folds = run_wf(
        close=close,                market_close=market_close,
        risk_manager=RiskManager(PARAMS_E),
        train_years=TRAIN_YEARS,    test_years=TEST_YEARS,
        top_n_grid=TOP_N_GRID,
        use_trend=USE_TREND,        use_vol_filter=USE_VOL_FILTER,
        sizing=SIZING,              rebalance_freq=REBALANCE_FREQ,
        cost_bps=20.0,
        use_improved_signal=True,
    )
    _fold_table("E", e_folds)

    # ── Summary table ─────────────────────────────────────────────────
    _section("SUMMARY — ALL VERSIONS vs BENCHMARK (OOS)")
    bench_oos = bench_returns.reindex(orig_oos.index).dropna()
    b_aligned = bench_oos.reindex

    rows = []
    for name, oos in [("Original", orig_oos), ("D: 4-state+persist+softDD", d_oos),
                      ("E: D + improved signal", e_oos)]:
        r = oos.dropna()
        b = bench_oos.reindex(r.index).dropna()
        rows.append(_metrics_row(name, r, b))

    b = bench_oos.dropna()
    rows.append({
        "Version": "Benchmark (SPY)",
        "CAGR":    f"{cagr(b):.1%}",   "Vol":    f"{annualised_volatility(b):.1%}",
        "Sharpe":  f"{sharpe_ratio(b):.2f}", "Sortino": f"{sortino_ratio(b):.2f}",
        "Max DD":  f"{max_drawdown(b):.1%}", "Calmar": f"{calmar_ratio(b):.2f}",
        "Beta":    "1.00",              "Alpha":  "0.0%",
    })

    summary_df = pd.DataFrame(rows)
    print(f"\n{summary_df.to_string(index=False)}")

    # ── Fold-by-fold D vs E ───────────────────────────────────────────
    _section("FOLD DETAIL — D vs E (signal improvement per year)")
    fold_rows = []
    for df, ef in zip(d_folds, e_folds):
        fold_rows.append({
            "Fold":    df.fold,
            "Test":    f"{df.test_start.date()} → {df.test_end.date()}",
            "D Sh":    round(df.oos_sharpe, 2),
            "D CAGR":  f"{df.oos_cagr:.1%}",
            "D MDD":   f"{df.oos_max_dd:.1%}",
            "E Sh":    round(ef.oos_sharpe, 2),
            "E CAGR":  f"{ef.oos_cagr:.1%}",
            "E MDD":   f"{ef.oos_max_dd:.1%}",
            "ΔSharpe": round(ef.oos_sharpe - df.oos_sharpe, 2),
        })
    print(pd.DataFrame(fold_rows).to_string(index=False))

    # ── Save CSVs ────────────────────────────────────────────────────
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "version_comparison.csv"), index=False)
    orig_oos.to_csv(os.path.join(OUTPUT_DIR, "oos_original.csv"), header=["returns"])
    d_oos.to_csv(os.path.join(OUTPUT_DIR, "oos_D.csv"), header=["returns"])
    e_oos.to_csv(os.path.join(OUTPUT_DIR, "oos_E.csv"), header=["returns"])
    pd.DataFrame(fold_rows).to_csv(
        os.path.join(OUTPUT_DIR, "fold_D_vs_E.csv"), index=False
    )
    print(f"\nCSVs saved → {OUTPUT_DIR}/")

    # ── Chart ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(13, 9),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Strategy Versions — OOS Equity Curves", fontsize=13)

    bench_eq = (1 + bench_oos.fillna(0)).cumprod()
    axes[0].plot(bench_eq, color="dimgray", linewidth=1.2,
                 linestyle="--", label="SPY B&H", zorder=1)

    palette = {
        "Original":                ("steelblue",  1.4),
        "D: 4-state+persist+softDD": ("darkorange", 1.5),
        "E: D + improved signal":  ("seagreen",   1.6),
    }
    for name, oos in [("Original", orig_oos),
                      ("D: 4-state+persist+softDD", d_oos),
                      ("E: D + improved signal",  e_oos)]:
        eq   = (1 + oos.fillna(0)).cumprod()
        dd   = diag.rolling_drawdown(oos) * 100
        col, lw = palette[name]
        axes[0].plot(eq, label=name, linewidth=lw, color=col)
        axes[1].plot(dd, linewidth=0.9, color=col, alpha=0.85)

    # Fold boundary lines
    for f in d_folds:
        axes[0].axvline(f.test_start, color="grey", alpha=0.25,
                        linewidth=0.8, linestyle=":")

    axes[0].set_ylabel("Equity (normalised to 1.0)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("Drawdown %")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "version_comparison.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Chart saved → {chart_path}")


if __name__ == "__main__":
    main()
