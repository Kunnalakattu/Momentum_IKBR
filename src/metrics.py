import pandas as pd
import numpy as np


def total_return(returns: pd.Series) -> float:
    """Cumulative return over the full period."""
    return (1 + returns).prod() - 1


def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate."""
    n_years = len(returns) / periods_per_year
    return (1 + total_return(returns)) ** (1 / n_years) - 1


def annualised_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised standard deviation of returns."""
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio."""
    excess = returns - risk_free / periods_per_year
    return (excess.mean() / excess.std()) * np.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """Sortino ratio using downside deviation."""
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0].std() * np.sqrt(periods_per_year)
    return (excess.mean() * periods_per_year) / downside


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    return drawdown.min()


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """CAGR divided by absolute max drawdown."""
    mdd = abs(max_drawdown(returns))
    return cagr(returns, periods_per_year) / mdd if mdd != 0 else np.nan


def win_rate(returns: pd.Series) -> float:
    """Fraction of periods with positive return."""
    return (returns > 0).mean()


def profit_factor(returns: pd.Series) -> float:
    """Gross profit divided by gross loss."""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return gains / losses if losses != 0 else np.nan


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR at given confidence level (positive = loss magnitude)."""
    return -returns.quantile(1 - confidence)


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """CVaR / Expected Shortfall: mean loss beyond VaR threshold."""
    var = value_at_risk(returns, confidence)
    return -returns[returns <= -var].mean()


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """Beta relative to a benchmark."""
    aligned = pd.concat([returns, benchmark], axis=1).dropna()
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return cov[0, 1] / cov[1, 1]


def alpha(returns: pd.Series, benchmark: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """Jensen's alpha (annualised)."""
    b = beta(returns, benchmark)
    ann = periods_per_year
    port_ann = returns.mean() * ann
    bench_ann = benchmark.mean() * ann
    return port_ann - (risk_free + b * (bench_ann - risk_free))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Full drawdown time series."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    return (cum - peak) / peak


def summary(
    returns: pd.Series,
    benchmark: pd.Series = None,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    """Return a single-column summary of all key metrics."""
    metrics = {
        "Total Return": f"{total_return(returns):.2%}",
        "CAGR": f"{cagr(returns, periods_per_year):.2%}",
        "Annualised Vol": f"{annualised_volatility(returns, periods_per_year):.2%}",
        "Sharpe Ratio": f"{sharpe_ratio(returns, risk_free, periods_per_year):.2f}",
        "Sortino Ratio": f"{sortino_ratio(returns, risk_free, periods_per_year):.2f}",
        "Max Drawdown": f"{max_drawdown(returns):.2%}",
        "Calmar Ratio": f"{calmar_ratio(returns, periods_per_year):.2f}",
        "Win Rate": f"{win_rate(returns):.2%}",
        "Profit Factor": f"{profit_factor(returns):.2f}",
        "VaR (95%)": f"{value_at_risk(returns):.2%}",
        "CVaR (95%)": f"{conditional_var(returns):.2%}",
    }
    if benchmark is not None:
        metrics["Beta"] = f"{beta(returns, benchmark):.2f}"
        metrics["Alpha (ann.)"] = f"{alpha(returns, benchmark, risk_free, periods_per_year):.2%}"
    return pd.Series(metrics, name="Strategy")
