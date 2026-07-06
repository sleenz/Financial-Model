"""
Mixed-period Sharpe estimate for the Portfolio Builder — DISPLAY ONLY.

Per explicit instruction (refining the original Phase 4 spec): the return
leg and volatility leg of this Sharpe estimate come from two different
reference periods, and that mismatch is disclosed in the UI rather than
papered over:

- Return leg: REALIZED historical returns of the exact constructed
  portfolio (actual input share counts) over a trailing lookback_days
  window — not a modeled/CAPM/expected return.
- Volatility leg: the existing DCC-GARCH model's CURRENT fitted
  conditional covariance only (current_correlation + the latest row of
  conditional_volatilities) — reused, not refit. forecast_correlation()
  at any horizon is deliberately never used here: projecting a horizon
  would compound the return/volatility period mismatch instead of
  containing it to the one disclosed spot.

Isolation (absolute constraint 3): compute_realized_return(),
compute_dcc_garch_volatility_current(), and compute_sharpe() are
DISPLAY-ONLY, same treatment as expected_return_estimate() in the
original spec. Phase 4's CHECK greps ranking.py to confirm zero
references to any of the three.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SharpeConfig:
    lookback_days: int = 252
    # Annual rate (e.g. 0.045 for 4.5%) — must match the ~annual scale of
    # both compute_realized_return's cumulative window and
    # compute_dcc_garch_volatility_current's annualized volatility.
    # Deliberately Optional with no numeric default: FRED-sourced
    # risk-free rate is broken upstream (see Phase 0 audit), so this must
    # be set explicitly as a manual constant until that's fixed.
    # compute_sharpe() raises ValueError if this is still None — it must
    # never silently default to 0.0, which would quietly overstate Sharpe.
    risk_free_rate: Optional[float] = None
    # Closed choice: "dcc_garch_current" is the only implemented/allowed
    # value. This field exists so the choice is an explicit, named,
    # auditable config value rather than an unstated assumption — not
    # because a different value actually works. compute_sharpe() raises
    # if this is changed to anything else.
    volatility_source: str = "dcc_garch_current"

    def __post_init__(self) -> None:
        if self.lookback_days <= 0:
            raise ValueError(f"lookback_days must be > 0, got {self.lookback_days}")


def compute_realized_return(portfolio_returns: pd.Series, lookback_days: int) -> float:
    """
    Cumulative (compounded) realized return of the ACTUAL constructed
    portfolio over the trailing lookback_days daily returns — "trailing
    12-month realized" for the default lookback_days=252. Not a modeled or
    CAPM-implied return: this is what the specific share-count portfolio
    the user actually built would have returned.

    Raises ValueError rather than silently truncating or ignoring gaps —
    a Sharpe number quietly built on a short or gappy window is worse than
    an explicit failure.
    """
    if len(portfolio_returns) < lookback_days:
        raise ValueError(
            f"compute_realized_return: only {len(portfolio_returns)} days of "
            f"returns available, need at least lookback_days={lookback_days}"
        )

    window = portfolio_returns.iloc[-lookback_days:]
    if window.isna().any():
        raise ValueError(
            "compute_realized_return: NaN value(s) in the lookback window — "
            "refusing to silently drop or fill them"
        )

    cumulative = float((1.0 + window).prod() - 1.0)
    return cumulative


def compute_dcc_garch_volatility_current(
    dcc_result,
    weights: pd.Series,
    annualization_days: int = 252,
) -> float:
    """
    w' Sigma w using ONLY the DCC-GARCH model's latest fitted conditional
    covariance — current_correlation (R) combined with the last row of
    conditional_volatilities (D) as Sigma = D R D. Never
    forecast_correlation() at any horizon: a projected horizon would
    compound the return/volatility period mismatch this whole module
    exists to disclose, not fix it.

    `weights` must be SECTOR weights aligned to dcc_result.sector_names —
    this DCC-GARCH engine is fit at sector level (see Phase 0 audit /
    network.py's docstring for why it was never extended to ticker
    level), so a ticker-level weights Series would silently misalign here
    without the reindex check below catching it.

    annualization_days converts the model's native daily volatility to an
    annual-scale figure so the ratio in compute_sharpe() is comparable to
    compute_realized_return's ~annual cumulative return — this is a unit
    conversion (standard sqrt-time scaling), not a business threshold, so
    it's a plain parameter with a standard default rather than a config
    dataclass field.
    """
    sectors = list(dcc_result.sector_names)

    w = weights.reindex(sectors)
    if w.isna().any():
        missing = w.index[w.isna()].tolist()
        raise ValueError(
            f"compute_dcc_garch_volatility_current: weights missing for sector(s) "
            f"{missing} (expected sector-level weights aligned to dcc_result.sector_names)"
        )

    sigma_daily = dcc_result.conditional_volatilities.iloc[-1].reindex(sectors)
    if sigma_daily.isna().any():
        missing = sigma_daily.index[sigma_daily.isna()].tolist()
        raise ValueError(
            f"compute_dcc_garch_volatility_current: conditional_volatilities "
            f"missing sector(s) {missing}"
        )

    correlation = dcc_result.current_correlation.reindex(index=sectors, columns=sectors)
    if correlation.isna().any().any():
        raise ValueError(
            "compute_dcc_garch_volatility_current: current_correlation has "
            "missing sector(s) after aligning to dcc_result.sector_names"
        )

    D = np.diag(sigma_daily.values)
    sigma_matrix_daily = D @ correlation.values @ D

    w_arr = w.values
    portfolio_variance_daily = float(w_arr @ sigma_matrix_daily @ w_arr)
    # Guard against tiny negative numerical noise from a near-singular
    # correlation matrix — a real negative variance is impossible.
    portfolio_variance_daily = max(portfolio_variance_daily, 0.0)

    daily_vol = np.sqrt(portfolio_variance_daily)
    annualized_vol = float(daily_vol * np.sqrt(annualization_days))
    return annualized_vol


def compute_sharpe(realized_return: float, volatility: float, config: SharpeConfig) -> float:
    """
    (realized_return - risk_free_rate) / volatility.

    Raises rather than defaulting risk_free_rate to 0.0 (would silently
    overstate Sharpe) and rather than dividing by a non-positive
    volatility (undefined/meaningless ratio).
    """
    if config.risk_free_rate is None:
        raise ValueError(
            "compute_sharpe: SharpeConfig.risk_free_rate is None — it must be set "
            "explicitly as a manual constant until FRED is fixed upstream. "
            "Refusing to silently default to 0.0."
        )
    if config.volatility_source != "dcc_garch_current":
        raise ValueError(
            f"compute_sharpe: volatility_source='{config.volatility_source}' is not "
            "supported — 'dcc_garch_current' is the only implemented source "
            "(forecast_correlation() at any horizon is deliberately not an option)."
        )
    if volatility <= 0:
        raise ValueError(f"compute_sharpe: volatility must be > 0, got {volatility}")

    logger.info(
        f"compute_sharpe: lookback_days={config.lookback_days}, "
        f"volatility_source={config.volatility_source}, "
        f"risk_free_rate={config.risk_free_rate}, "
        f"realized_return={realized_return:.6f}, volatility={volatility:.6f}"
    )
    return (realized_return - config.risk_free_rate) / volatility


def render_mixed_period_disclosure() -> None:
    """
    Two-part progressive disclosure, not one dense string: a short badge
    ("Mixed-period estimate") that expands to the full caveat. Renders
    nothing else and computes nothing — purely a display component,
    isolated from ranking.py and the composite score same as the
    functions above.
    """
    import streamlit as st

    with st.expander("Mixed-period estimate", expanded=False):
        st.caption(
            "Return: trailing 12-month realized. Volatility: current model "
            "estimate, not the same period. Not a forward estimate."
        )


if __name__ == "__main__":
    def _smoke_test():
        from src.portfolio_builder.metrics import (
            SharpeConfig,
            compute_dcc_garch_volatility_current,
            compute_realized_return,
            compute_sharpe,
            render_mixed_period_disclosure,
        )

        # ── compute_realized_return: hand-computable constant-return series ──
        # 252 days of a constant 0.0005 daily return -> cumulative = 1.0005^252 - 1
        daily_r = 0.0005
        dates = pd.bdate_range("2025-01-01", periods=260)
        returns = pd.Series(daily_r, index=dates)
        realized = compute_realized_return(returns, lookback_days=252)
        expected = (1.0 + daily_r) ** 252 - 1.0
        assert abs(realized - expected) < 1e-12, (realized, expected)
        print("✓ compute_realized_return: matches hand-computed compounded return")

        # Too few observations -> raises
        try:
            compute_realized_return(returns.iloc[:100], lookback_days=252)
            raise AssertionError("expected ValueError for insufficient history")
        except ValueError:
            pass
        print("✓ compute_realized_return: insufficient history raises ValueError")

        # NaN in window -> raises
        gappy = returns.copy()
        gappy.iloc[-5] = float("nan")
        try:
            compute_realized_return(gappy, lookback_days=252)
            raise AssertionError("expected ValueError for NaN in window")
        except ValueError:
            pass
        print("✓ compute_realized_return: NaN in lookback window raises ValueError")

        # ── compute_dcc_garch_volatility_current: independent recomputation ──
        sectors = ["Tech", "Energy"]
        cond_vol = pd.DataFrame({"Tech": [0.010, 0.012], "Energy": [0.008, 0.009]})
        current_corr = pd.DataFrame([[1.0, 0.3], [0.3, 1.0]], index=sectors, columns=sectors)
        weights = pd.Series({"Tech": 0.6, "Energy": 0.4})

        class _FakeDCCResult:
            sector_names = sectors
            conditional_volatilities = cond_vol
            current_correlation = current_corr

        vol = compute_dcc_garch_volatility_current(_FakeDCCResult(), weights, annualization_days=252)

        # Independent recomputation (same formula, written fresh, not reusing the function's internals)
        sigma_last = cond_vol.iloc[-1][sectors].values
        D = np.diag(sigma_last)
        R = current_corr.loc[sectors, sectors].values
        sigma_daily_matrix = D @ R @ D
        w = weights.reindex(sectors).values
        expected_var_daily = w @ sigma_daily_matrix @ w
        expected_vol = np.sqrt(expected_var_daily) * np.sqrt(252)
        assert abs(vol - expected_vol) < 1e-12, (vol, expected_vol)
        print("✓ compute_dcc_garch_volatility_current: matches independent w'Sigma w recomputation")

        # Ticker-level (mismatched) weights -> raises rather than silently misaligning
        bad_weights = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
        try:
            compute_dcc_garch_volatility_current(_FakeDCCResult(), bad_weights)
            raise AssertionError("expected ValueError for sector/ticker weight mismatch")
        except ValueError:
            pass
        print("✓ compute_dcc_garch_volatility_current: mismatched weight index raises ValueError")

        # ── compute_sharpe ────────────────────────────────────────────────
        config = SharpeConfig(risk_free_rate=0.04)
        sharpe = compute_sharpe(0.12, 0.18, config)
        assert abs(sharpe - (0.12 - 0.04) / 0.18) < 1e-12
        print("✓ compute_sharpe: matches hand calc")

        # risk_free_rate=None -> raises, never silently defaults to 0.0
        try:
            compute_sharpe(0.12, 0.18, SharpeConfig(risk_free_rate=None))
            raise AssertionError("expected ValueError for risk_free_rate=None")
        except ValueError:
            pass
        print("✓ compute_sharpe: risk_free_rate=None raises ValueError, never defaults to 0.0")

        # non-positive volatility -> raises
        try:
            compute_sharpe(0.12, 0.0, SharpeConfig(risk_free_rate=0.04))
            raise AssertionError("expected ValueError for volatility <= 0")
        except ValueError:
            pass
        print("✓ compute_sharpe: non-positive volatility raises ValueError")

        # unsupported volatility_source -> raises
        bad_config = SharpeConfig(risk_free_rate=0.04)
        bad_config.volatility_source = "forecast_correlation_h21"
        try:
            compute_sharpe(0.12, 0.18, bad_config)
            raise AssertionError("expected ValueError for unsupported volatility_source")
        except ValueError:
            pass
        print("✓ compute_sharpe: non-'dcc_garch_current' volatility_source raises ValueError")

        # ── SharpeConfig validation ───────────────────────────────────────
        try:
            SharpeConfig(lookback_days=0)
            raise AssertionError("expected ValueError for lookback_days<=0")
        except ValueError:
            pass
        print("✓ SharpeConfig: lookback_days<=0 raises ValueError")

        # ── Rule 2: reused/new names resolve ──────────────────────────────
        assert callable(render_mixed_period_disclosure)
        print("✓ render_mixed_period_disclosure resolves")

        print("✓ metrics.py smoke test passed")

    _smoke_test()
