"""
Risk management overlay: volatility targeting, regime filtering, and
drawdown circuit-breaker with cooldown.

Design
------
RiskManager is the single interface consumed by backtester.py.
It produces two things:

  1. compute_static_multiplier()
       A daily scalar in [0, max_leverage] derived from market data ONLY
       (vol targeting × regime filters).  Because it uses only SPY data,
       there is no circular dependency with portfolio returns.

  2. apply_drawdown_control()
       A sequential simulation that tracks the *actual* portfolio equity
       day-by-day and further scales weights when drawdown exceeds
       configured thresholds.  This is the one component that cannot be
       vectorised — it depends on the running cumulative return.

All computations use information lagged by at least one day.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.regime import combined_regime_multiplier, combined_vol_trend_multiplier


@dataclass
class RiskParams:
    # ── Volatility targeting ───────────────────────────────────────────
    use_vol_targeting: bool  = True    # False → skip scaling, use raw weights
    target_vol:   float = 0.15
    vol_lookback: int   = 20
    min_vol:      float = 0.05
    max_leverage: float = 1.0          # set 1.25 to allow modest leverage

    # ── Regime filters ─────────────────────────────────────────────────
    use_combined_regime: bool  = True  # True → 4-state vol+trend rule
                                       # False → independent vol/trend multipliers
    # 4-state combined rule parameters
    below_ma_mult:       float = 0.25  # below MA, low vol  → partial exposure
    high_vol_above_mult: float = 0.60  # above MA, high vol → reduced exposure
    persistence_days:    int   = 5     # consecutive days to confirm regime change

    # Independent filter parameters (use_combined_regime=False)
    use_vol_regime:    bool  = True
    use_trend_regime:  bool  = True
    vol_regime_pct:    float = 0.75
    vol_regime_window: int   = 252
    trend_ma_window:   int   = 200

    # ── Drawdown soft scaling ──────────────────────────────────────────
    # Replaces the hard circuit-breaker with smooth scaling.
    # When portfolio DD < dd_scale_threshold, multiply exposure by dd_scale_factor.
    # No hard exit, no cooldown — avoids forced exits near bottoms.
    use_dd_control:      bool  = True
    dd_scale_threshold:  float = -0.10   # DD level that triggers soft scaling
    dd_scale_factor:     float = 0.70    # multiply exposure by this when triggered
    # Legacy hard-stop fields (kept for backward compatibility, not used by default)
    dd_soft_threshold:   float = -0.15
    dd_hard_threshold:   float = -0.20
    dd_recovery:         float = -0.10
    cooldown_days:       int   = 10


class RiskManager:
    """
    Combines volatility targeting, regime filters, and drawdown control
    into a daily position-sizing multiplier.

    Parameters
    ----------
    params : RiskParams
        All risk management configuration in one dataclass.
    """

    def __init__(self, params: RiskParams | None = None) -> None:
        self.params = params or RiskParams()

    # ------------------------------------------------------------------
    # Static multiplier (vectorised, market-data only)
    # ------------------------------------------------------------------

    def compute_static_multiplier(self, market_close: pd.Series) -> pd.Series:
        """
        Combine vol-targeting scaler and regime filters into one daily scalar.

        Uses only SPY close prices → no circular dependency with portfolio
        returns.  The result is lagged 1 day (ready to multiply into weights).

        Returns a Series in [0, max_leverage] aligned to market_close.index.
        """
        p       = self.params
        mkt_ret = market_close.pct_change()

        # 1. Volatility-targeting scaler (optional)
        if p.use_vol_targeting:
            realised_vol = (
                mkt_ret.rolling(p.vol_lookback)
                .std()
                .mul(np.sqrt(252))
                .clip(lower=p.min_vol)
            )
            vol_scaler = (p.target_vol / realised_vol).clip(upper=p.max_leverage)
            vol_scaler = vol_scaler.shift(1).fillna(1.0)
        else:
            vol_scaler = pd.Series(1.0, index=market_close.index)

        # 2. Regime multiplier — two modes
        if p.use_combined_regime:
            regime_mult = combined_vol_trend_multiplier(
                market_close, mkt_ret,
                vol_window=p.vol_lookback,
                vol_regime_window=p.vol_regime_window,
                vol_threshold=p.vol_regime_pct,
                ma_window=p.trend_ma_window,
                below_ma_mult=p.below_ma_mult,
                high_vol_above_mult=p.high_vol_above_mult,
                persistence_days=p.persistence_days,
            )
        else:
            # Legacy: independent vol and trend multipliers
            regime_mult = combined_regime_multiplier(
                market_close,
                use_vol_regime=p.use_vol_regime,
                use_trend_regime=p.use_trend_regime,
                vol_window=p.vol_lookback,
                regime_window=p.vol_regime_window,
                threshold_pct=p.vol_regime_pct,
                ma_window=p.trend_ma_window,
            )
        regime_mult = regime_mult.reindex(market_close.index).fillna(1.0)

        # Combine: vol scaler first, then regime discount on top
        combined = (vol_scaler * regime_mult).clip(0.0, p.max_leverage)
        return combined.rename("static_mult")

    # ------------------------------------------------------------------
    # Sequential drawdown control
    # ------------------------------------------------------------------

    def apply_drawdown_control(
        self,
        weights: pd.DataFrame,
        close: pd.DataFrame,
        static_mult: pd.Series,
    ) -> pd.DataFrame:
        """
        Sequential simulation tracking actual portfolio drawdown day-by-day.

        Soft scaling approach (replaces previous hard circuit-breaker):
            When portfolio DD < dd_scale_threshold:
                exposure *= dd_scale_factor   (e.g. 0.7x — smooth, not a cliff)
            Otherwise:
                full exposure

        This avoids forced exits near bottoms and missed recoveries.
        The scaling applies multiplicatively ON TOP of the static_mult.
        """
        if not self.params.use_dd_control:
            return weights.multiply(
                static_mult.reindex(weights.index).fillna(1.0), axis=0
            )

        p             = self.params
        asset_returns = close.pct_change()
        adjusted      = weights.copy().mul(0.0)

        equity = 1.0
        peak   = 1.0

        for date in weights.index:
            w  = weights.loc[date]
            sm = static_mult.at[date] if date in static_mult.index else 1.0

            dd = (equity - peak) / peak if peak > 0 else 0.0

            # Simple soft scale: no hard stop, no cooldown, no recovery wait
            dd_mult = p.dd_scale_factor if dd < p.dd_scale_threshold else 1.0

            final_w = w * sm * dd_mult
            adjusted.loc[date] = final_w

            if date in asset_returns.index:
                day_ret  = asset_returns.loc[date].fillna(0.0)
                port_ret = float((final_w * day_ret).sum())
                equity   = equity * (1.0 + port_ret)
                peak     = max(peak, equity)

        return adjusted
