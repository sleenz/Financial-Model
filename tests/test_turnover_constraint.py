"""
Regression tests for the position reduction (turnover) constraint.

Covers PortfolioConstraints.compute_bounds() directly, and its consistent
application across every optimizer backend (PortfolioOptimizer AND
BlackLittermanModel). Black-Litterman previously ignored the turnover band
entirely — these tests guard against that regressing again.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.optimization.constraints import PortfolioConstraints
from src.optimization.optimizers import PortfolioOptimizer
from src.optimization.black_litterman import BlackLittermanModel


TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN"]


def _make_returns(n: int = 400, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, len(TICKERS))) * 0.01 + 0.0004
    return pd.DataFrame(data, columns=TICKERS)


# ---------------------------------------------------------------------------
# compute_bounds() — direct unit tests
# ---------------------------------------------------------------------------

def test_compute_bounds_standard_when_turnover_disabled():
    c = PortfolioConstraints(min_weight=0.02, max_weight=0.40, turnover_enabled=False)
    bounds = c.compute_bounds(TICKERS)
    assert bounds == [(0.02, 0.40) for _ in TICKERS]


def test_compute_bounds_falls_back_when_no_current_weights():
    c = PortfolioConstraints(min_weight=0.0, max_weight=0.40, turnover_enabled=True, current_weights=None)
    bounds = c.compute_bounds(TICKERS)
    assert bounds == [(0.0, 0.40) for _ in TICKERS]


def test_compute_bounds_turnover_band_around_current_weights():
    current = pd.Series([0.10, 0.10, 0.10, 0.70], index=TICKERS)
    c = PortfolioConstraints(
        min_weight=0.0, max_weight=0.90,
        turnover_enabled=True, reduction_pct=0.20, increase_pct=0.20,
        allow_full_exit=False, current_weights=current,
    )
    bounds = dict(zip(TICKERS, c.compute_bounds(TICKERS)))
    assert bounds["AAPL"] == pytest.approx((0.08, 0.12))
    assert bounds["AMZN"] == pytest.approx((0.56, 0.84))


def test_compute_bounds_allow_full_exit_zeroes_lower_bound():
    current = pd.Series([0.10, 0.10, 0.10, 0.70], index=TICKERS)
    c = PortfolioConstraints(
        min_weight=0.0, max_weight=0.90,
        turnover_enabled=True, reduction_pct=0.20, increase_pct=0.20,
        allow_full_exit=True, current_weights=current,
    )
    bounds = dict(zip(TICKERS, c.compute_bounds(TICKERS)))
    for lb, _ in bounds.values():
        assert lb == 0.0


def test_compute_bounds_new_position_gets_standard_bounds():
    # NVDA isn't in current_weights — it's a brand-new position.
    current = pd.Series([0.30, 0.30, 0.40], index=["AAPL", "MSFT", "GOOGL"])
    c = PortfolioConstraints(
        min_weight=0.01, max_weight=0.50,
        turnover_enabled=True, reduction_pct=0.20, increase_pct=0.20,
        current_weights=current,
    )
    bounds = dict(zip(TICKERS, c.compute_bounds(TICKERS)))
    assert bounds["AMZN"] == (0.01, 0.50)


def test_compute_bounds_infeasible_lower_raises_value_error():
    current = pd.Series([0.9, 0.9, 0.9, 0.9], index=TICKERS)
    c = PortfolioConstraints(
        min_weight=0.0, max_weight=1.0,
        turnover_enabled=True, reduction_pct=0.10, increase_pct=0.0,
        allow_full_exit=False, current_weights=current,
    )
    with pytest.raises(ValueError, match="Position reduction constraint infeasible"):
        c.compute_bounds(TICKERS)


def test_compute_bounds_infeasible_upper_raises_value_error():
    current = pd.Series([0.05, 0.05, 0.05, 0.05], index=TICKERS)
    c = PortfolioConstraints(
        min_weight=0.0, max_weight=1.0,
        turnover_enabled=True, reduction_pct=0.0, increase_pct=0.05,
        allow_full_exit=True, current_weights=current,
    )
    with pytest.raises(ValueError, match="Position increase constraint infeasible"):
        c.compute_bounds(TICKERS)


# ---------------------------------------------------------------------------
# End-to-end: PortfolioOptimizer respects the band (mainline scipy methods)
# ---------------------------------------------------------------------------

def test_optimizer_max_sharpe_respects_turnover_band():
    current = pd.Series([0.10, 0.10, 0.10, 0.70], index=TICKERS)
    c = PortfolioConstraints(
        min_weight=0.0, max_weight=0.90,
        turnover_enabled=True, reduction_pct=0.20, increase_pct=0.20,
        allow_full_exit=False, current_weights=current,
    )
    opt = PortfolioOptimizer(_make_returns(), risk_free_rate=0.02)
    result = opt.optimize("max_sharpe", c)
    w = result["weights"]
    for ticker in TICKERS:
        lb, ub = dict(zip(TICKERS, c.compute_bounds(TICKERS)))[ticker]
        assert lb - 1e-6 <= w[ticker] <= ub + 1e-6, f"{ticker} weight {w[ticker]} outside [{lb}, {ub}]"


def test_optimizer_infeasible_turnover_raises_value_error_not_wrapped():
    # Regression: PortfolioOptimizer.optimize() used to wrap every exception
    # (including this deliberate ValueError) into OptimizationError, which
    # callers catching `except ValueError` never actually caught.
    current = pd.Series([0.9, 0.9, 0.9, 0.9], index=TICKERS)
    c = PortfolioConstraints(
        min_weight=0.0, max_weight=1.0,
        turnover_enabled=True, reduction_pct=0.10, increase_pct=0.0,
        allow_full_exit=False, current_weights=current,
    )
    opt = PortfolioOptimizer(_make_returns(), risk_free_rate=0.02)
    with pytest.raises(ValueError, match="Position reduction constraint infeasible"):
        opt.optimize("max_sharpe", c)


# ---------------------------------------------------------------------------
# End-to-end: BlackLittermanModel respects the band (the actual bug)
# ---------------------------------------------------------------------------

def test_black_litterman_respects_turnover_band_when_constraints_given():
    current = pd.Series([0.10, 0.10, 0.10, 0.70], index=TICKERS)
    c = PortfolioConstraints(
        min_weight=0.0, max_weight=0.90,
        turnover_enabled=True, reduction_pct=0.20, increase_pct=0.20,
        allow_full_exit=False, current_weights=current,
    )
    model = BlackLittermanModel(returns=_make_returns(), risk_free_rate=0.02)
    out = model.optimize(constraints=c)
    w = out["weights"]
    bounds = dict(zip(TICKERS, c.compute_bounds(TICKERS)))
    for ticker in TICKERS:
        lb, ub = bounds[ticker]
        assert lb - 1e-6 <= w[ticker] <= ub + 1e-6, f"{ticker} weight {w[ticker]} outside [{lb}, {ub}]"


def test_black_litterman_ignores_turnover_without_constraints_arg():
    # Backward-compatible path: no `constraints` kwarg -> flat bounds only,
    # exactly like before this fix (existing callers/tests are unaffected).
    model = BlackLittermanModel(returns=_make_returns(), risk_free_rate=0.02)
    out = model.optimize(max_weight=0.5, min_weight=0.0)
    w = out["weights"]
    assert (w <= 0.5 + 1e-6).all()
    assert (w >= -1e-6).all()
