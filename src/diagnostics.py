"""
Strategy diagnostics and visualisation.

Provides:
  Rolling metrics  — Sharpe, volatility, drawdown, beta
  Regime breakdown — performance table split by trend/vol regime
  Parameter sensitivity — grid search over key risk parameters
  Monte Carlo — block-bootstrap resampling to stress-test OOS returns
  Comparison table  — side-by-side original vs improved vs benchmark
  Chart functions   — save PNG files for each diagnostic
"""

from __future__ import annotations

import itertools
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────
# Rolling metrics
# ─────────────────────────────────────────────────────────────────────

def rolling_sharpe(returns: pd.Series, window: int = 63) -> pd.Series:
    """Rolling annualised Sharpe ratio (window trading days)."""
    r = returns.rolling(window)
    return (r.mean() / r.std() * np.sqrt(252)).rename("rolling_sharpe")


def rolling_vol(returns: pd.Series, window: int = 63) -> pd.Series:
    """Rolling annualised volatility."""
    return (returns.rolling(window).std() * np.sqrt(252)).rename("rolling_vol")


def rolling_drawdown(returns: pd.Series) -> pd.Series:
    """Full drawdown time series (negative values, 0 at peaks)."""
    cum  = (1 + returns).cumprod()
    peak = cum.cummax()
    return ((cum - peak) / peak).rename("drawdown")


def rolling_beta(
    returns: pd.Series,
    benchmark: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Rolling beta vs benchmark (rolling covariance / rolling variance)."""
    aligned = pd.concat([returns, benchmark], axis=1).dropna()
    r = aligned.iloc[:, 0]
    b = aligned.iloc[:, 1]
    cov = r.rolling(window).cov(b)
    var = b.rolling(window).var()
    return (cov / var).rename("rolling_beta")


# ─────────────────────────────────────────────────────────────────────
# Regime performance table
# ─────────────────────────────────────────────────────────────────────

def regime_performance(
    returns: pd.Series,
    market_close: pd.Series,
    market_returns: pd.Series,
    vol_window: int        = 20,
    vol_regime_window: int = 252,
    vol_threshold: float   = 0.75,
    ma_window: int         = 200,
) -> pd.DataFrame:
    """
    Break down strategy performance by market regime.

    Regimes reported:
      Trend  — above / below 200-day MA (1-day lag)
      Vol    — low / high realised volatility (1-day lag)
    """
    # Trend regime labels (lagged)
    ma = market_close.rolling(ma_window).mean().shift(1)
    trend = (market_close > ma).map({True: "above MA", False: "below MA"})

    # Vol regime labels (lagged)
    rv        = market_returns.rolling(vol_window).std() * np.sqrt(252)
    pct_rank  = rv.rolling(vol_regime_window).rank(pct=True).shift(1)
    vol_regime = (pct_rank <= vol_threshold).map({True: "low vol", False: "high vol"})

    rows: dict[str, dict] = {}
    for regime_name, labels in [("Trend", trend), ("Vol", vol_regime)]:
        for label in labels.dropna().unique():
            mask = (labels == label).reindex(returns.index, fill_value=False)
            r = returns[mask].dropna()
            if len(r) < 20:
                continue
            ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
            ann_vol = r.std() * np.sqrt(252)
            sh      = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
            cum     = (1 + r).cumprod()
            mdd     = ((cum / cum.cummax()) - 1).min()
            rows[f"{regime_name}: {label}"] = {
                "Days":   len(r),
                "CAGR":   f"{ann_ret:.1%}",
                "Vol":    f"{ann_vol:.1%}",
                "Sharpe": f"{sh:.2f}",
                "MaxDD":  f"{mdd:.1%}",
            }

    return pd.DataFrame(rows).T


# ─────────────────────────────────────────────────────────────────────
# Parameter sensitivity
# ─────────────────────────────────────────────────────────────────────

def parameter_sensitivity(
    close: pd.DataFrame,
    market_close: pd.Series,
    signal_fn: Callable,
    top_n: int = 10,
    param_grid: dict | None = None,
    sizing: str = "equal",
    rebalance_freq: int = 21,
    cost_bps: float = 20.0,
) -> pd.DataFrame:
    """
    Grid search over key risk-management parameters on a fixed dataset.

    Run on the *training* portion of your data only — never on the test set.

    param_grid keys must be valid RiskParams field names.
    Example:
        {
            "target_vol":        [0.10, 0.15, 0.20],
            "dd_soft_threshold": [-0.10, -0.15, -0.20],
        }
    """
    from src.risk_management import RiskManager, RiskParams
    from src.backtester import run as run_bt
    from src.metrics import sharpe_ratio, cagr, max_drawdown

    if param_grid is None:
        param_grid = {
            "target_vol":        [0.10, 0.15, 0.20],
            "dd_soft_threshold": [-0.10, -0.15, -0.20],
        }

    signal = signal_fn(close, top_n)
    keys   = list(param_grid.keys())
    rows   = []

    for combo in itertools.product(*param_grid.values()):
        kwargs = dict(zip(keys, combo))
        params = RiskParams(**kwargs)
        rm     = RiskManager(params)

        res = run_bt(
            close=close, signal=signal,
            market_close=market_close, risk_manager=rm,
            sizing=sizing, rebalance_freq=rebalance_freq, cost_bps=cost_bps,
        )
        r    = res.returns.dropna()
        row  = {**kwargs}
        row["Sharpe"] = round(sharpe_ratio(r), 2)
        row["CAGR"]   = f"{cagr(r):.1%}"
        row["MaxDD"]  = f"{max_drawdown(r):.1%}"
        rows.append(row)

    return pd.DataFrame(rows).sort_values("Sharpe", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────
# Monte Carlo
# ─────────────────────────────────────────────────────────────────────

def monte_carlo_equity(
    returns: pd.Series,
    n_simulations: int = 500,
    block_size: int    = 21,
) -> pd.DataFrame:
    """
    Block-bootstrap resampling to stress-test the return series.

    Sampling blocks of `block_size` days preserves autocorrelation and
    volatility clustering better than i.i.d. resampling.

    Returns a DataFrame of shape (len(returns), n_simulations) where
    each column is a simulated cumulative equity path.
    """
    n       = len(returns)
    ret_arr = returns.values
    matrix  = np.zeros((n, n_simulations))

    rng = np.random.default_rng(42)
    for s in range(n_simulations):
        sample: list[float] = []
        while len(sample) < n:
            start = rng.integers(0, max(1, n - block_size))
            sample.extend(ret_arr[start : start + block_size])
        matrix[:, s] = sample[:n]

    equity = np.cumprod(1.0 + matrix, axis=0)
    return pd.DataFrame(equity, index=returns.index)


# ─────────────────────────────────────────────────────────────────────
# Comparison table
# ─────────────────────────────────────────────────────────────────────

def comparison_table(
    original_returns:  pd.Series,
    improved_returns:  pd.Series,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    """
    Side-by-side performance metrics for original, improved, and benchmark.
    Aligned to the common OOS date range.
    """
    from src.metrics import (
        cagr, annualised_volatility, sharpe_ratio, sortino_ratio,
        max_drawdown, calmar_ratio, win_rate, profit_factor, beta, alpha,
    )

    def _metrics(r: pd.Series, bench: pd.Series) -> dict:
        return {
            "Total Return":  f"{(1 + r).prod() - 1:.1%}",
            "CAGR":          f"{cagr(r):.1%}",
            "Ann. Vol":      f"{annualised_volatility(r):.1%}",
            "Sharpe":        f"{sharpe_ratio(r):.2f}",
            "Sortino":       f"{sortino_ratio(r):.2f}",
            "Max Drawdown":  f"{max_drawdown(r):.1%}",
            "Calmar":        f"{calmar_ratio(r):.2f}",
            "Win Rate":      f"{win_rate(r):.1%}",
            "Profit Factor": f"{profit_factor(r):.2f}",
            "Beta":          f"{beta(r, bench):.2f}",
            "Alpha (ann.)":  f"{alpha(r, bench):.1%}",
        }

    # Align to the intersection of all three series
    idx   = original_returns.index.intersection(improved_returns.index)
    bench = benchmark_returns.reindex(idx).dropna()
    orig  = original_returns.reindex(idx).dropna()
    impr  = improved_returns.reindex(idx).dropna()

    return pd.DataFrame({
        "Original":  _metrics(orig, bench),
        "Improved":  _metrics(impr, bench),
        "Benchmark": _metrics(bench, bench),
    })


# ─────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────

def plot_rolling_diagnostics(
    returns: pd.Series,
    benchmark: pd.Series,
    title: str     = "Strategy — Rolling Diagnostics",
    save_path: str = None,
) -> None:
    """4-panel rolling diagnostics chart (Sharpe, vol, drawdown, beta)."""
    rs = rolling_sharpe(returns)
    rv = rolling_vol(returns) * 100
    dd = rolling_drawdown(returns) * 100
    rb = rolling_beta(returns, benchmark)

    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
    fig.suptitle(title, fontsize=13)

    axes[0].plot(rs, color="steelblue", linewidth=1)
    axes[0].axhline(0,        color="grey",   linewidth=0.8, linestyle="--")
    axes[0].axhline(rs.mean(), color="orange", linewidth=0.8, linestyle="--",
                    label=f"mean = {rs.mean():.2f}")
    axes[0].set_ylabel("Rolling Sharpe (63d)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rv, color="darkorange", linewidth=1)
    axes[1].set_ylabel("Rolling Vol % (63d)")
    axes[1].grid(True, alpha=0.3)

    axes[2].fill_between(dd.index, dd.values, 0, color="red", alpha=0.4)
    axes[2].set_ylabel("Drawdown %")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(rb, color="purple", linewidth=1)
    axes[3].axhline(1.0,       color="grey",   linewidth=0.8, linestyle="--")
    axes[3].axhline(rb.mean(), color="orange", linewidth=0.8, linestyle="--",
                    label=f"mean = {rb.mean():.2f}")
    axes[3].set_ylabel("Rolling Beta (63d)")
    axes[3].legend(fontsize=8)
    axes[3].set_xlabel("Date")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_equity_comparison(
    original_returns:  pd.Series,
    improved_returns:  pd.Series,
    benchmark_returns: pd.Series,
    original_label:   str        = "Original",
    improved_label:   str        = "Improved",
    title:            str        = "Strategy Comparison",
    fold_dates:       list | None = None,
    save_path:        str        = None,
) -> None:
    """
    Three-panel chart: equity curves + drawdown comparison.
    Alternating fold shading is drawn if fold_dates is provided.
    """
    fig, axes = plt.subplots(
        3, 1, figsize=(13, 10),
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )
    fig.suptitle(title, fontsize=13)

    orig_eq  = (1 + original_returns).cumprod()
    impr_eq  = (1 + improved_returns).cumprod()
    bench_eq = (1 + benchmark_returns.reindex(orig_eq.index).fillna(0)).cumprod()

    shade_colors = ["#dce8f5", "#f5dce0"]
    if fold_dates:
        for i, (ts, te) in enumerate(fold_dates):
            for ax in axes:
                ax.axvspan(ts, te, alpha=0.25,
                           color=shade_colors[i % 2], zorder=0)

    axes[0].plot(orig_eq,  label=original_label, linewidth=1.5, color="steelblue")
    axes[0].plot(impr_eq,  label=improved_label, linewidth=1.5, color="seagreen")
    axes[0].plot(bench_eq, label="SPY B&H",      linewidth=1.5, color="dimgray",
                 linestyle="--")
    axes[0].set_ylabel("Equity (normalised to 1.0)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    orig_dd = rolling_drawdown(original_returns) * 100
    impr_dd = rolling_drawdown(improved_returns) * 100

    axes[1].fill_between(orig_dd.index, orig_dd.values, 0,
                         color="steelblue", alpha=0.4)
    axes[1].set_ylabel(f"{original_label} DD %")
    axes[1].grid(True, alpha=0.3)

    axes[2].fill_between(impr_dd.index, impr_dd.values, 0,
                         color="seagreen", alpha=0.4)
    axes[2].set_ylabel(f"{improved_label} DD %")
    axes[2].set_xlabel("Date")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_monte_carlo(
    returns: pd.Series,
    n_simulations: int = 500,
    title: str     = "Monte Carlo — Block Bootstrap",
    save_path: str = None,
) -> None:
    """Simulated equity paths with 5/25/50/75/95th percentile bands."""
    sim    = monte_carlo_equity(returns, n_simulations=n_simulations)
    actual = (1 + returns).cumprod()

    p5,  p25 = sim.quantile(0.05, axis=1), sim.quantile(0.25, axis=1)
    p50, p75 = sim.quantile(0.50, axis=1), sim.quantile(0.75, axis=1)
    p95       = sim.quantile(0.95, axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(returns.index, p5,  p95, alpha=0.12, color="steelblue",
                    label="5–95 %ile")
    ax.fill_between(returns.index, p25, p75, alpha=0.22, color="steelblue",
                    label="25–75 %ile")
    ax.plot(p50,    color="steelblue", linewidth=1.5, label="Median sim.")
    ax.plot(actual, color="orange",    linewidth=1.8, linestyle="--",
            label="Actual OOS")
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Equity (normalised to 1.0)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
