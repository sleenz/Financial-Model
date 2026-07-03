"""
Regression tests for PortfolioConstraints.project_to_bounds() and its use
as a universal post-processing step in PortfolioOptimizer.optimize().

Bug: HRP and Equal Weight compute weights purely from the covariance
structure (or a flat 1/n split) and never look at `constraints` at all.
Before this fix, changing Position Limits or the Position Reduction band
and re-running with one of those methods silently produced the exact same
weights every time — "run optimization once, tinker with constraints, the
value never changes."
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


TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN"]


def _make_returns(n: int = 400, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, len(TICKERS))) * 0.01 + 0.0004
    return pd.DataFrame(data, columns=TICKERS)


# ---------------------------------------------------------------------------
# project_to_bounds() — direct unit tests
# ---------------------------------------------------------------------------

def test_project_to_bounds_clips_and_renormalizes():
    c = PortfolioConstraints(min_weight=0.0, max_weight=0.30)
    w = np.array([0.05, 0.10, 0.60, 0.25])
    proj = c.project_to_bounds(w, TICKERS)
    assert np.all(proj <= 0.30 + 1e-9)
    assert proj.sum() == pytest.approx(1.0)


def test_project_to_bounds_respects_min_and_max():
    c = PortfolioConstraints(min_weight=0.10, max_weight=0.40)
    w = np.array([0.02, 0.02, 0.02, 0.94])
    proj = c.project_to_bounds(w, TICKERS)
    assert np.all(proj >= 0.10 - 1e-9)
    assert np.all(proj <= 0.40 + 1e-9)
    assert proj.sum() == pytest.approx(1.0)


def test_project_to_bounds_respects_turnover_band():
    current = pd.Series([0.10, 0.10, 0.10, 0.70], index=TICKERS)
    c = PortfolioConstraints(
        min_weight=0.0, max_weight=0.90,
        turnover_enabled=True, reduction_pct=0.20, increase_pct=0.20,
        allow_full_exit=False, current_weights=current,
    )
    w = np.array([0.219, 0.236, 0.315, 0.230])  # e.g. a raw HRP output
    proj = c.project_to_bounds(w, TICKERS)
    bounds = dict(zip(TICKERS, c.compute_bounds(TICKERS)))
    for ticker, val in zip(TICKERS, proj):
        lb, ub = bounds[ticker]
        assert lb - 1e-9 <= val <= ub + 1e-9
    assert proj.sum() == pytest.approx(1.0)


def test_project_to_bounds_is_near_noop_for_already_feasible_weights():
    c = PortfolioConstraints(min_weight=0.0, max_weight=0.40)
    w = np.array([0.25, 0.25, 0.25, 0.25])
    proj = c.project_to_bounds(w, TICKERS)
    assert proj == pytest.approx(w)


# ---------------------------------------------------------------------------
# compute_bounds() now validates feasibility for plain Position Limits too
# ---------------------------------------------------------------------------

def test_compute_bounds_infeasible_max_weight_raises_with_position_limits_message():
    # 4 assets * 15% cap = 60% max reachable — infeasible, no turnover involved.
    c = PortfolioConstraints(min_weight=0.0, max_weight=0.15)
    with pytest.raises(ValueError, match="Position limits infeasible"):
        c.compute_bounds(TICKERS)


def test_compute_bounds_infeasible_min_weight_raises_with_position_limits_message():
    # 4 assets * 30% floor = 120% minimum required — infeasible.
    c = PortfolioConstraints(min_weight=0.30, max_weight=1.0)
    with pytest.raises(ValueError, match="Position limits infeasible"):
        c.compute_bounds(TICKERS)


# ---------------------------------------------------------------------------
# End-to-end: HRP and Equal Weight now respect constraints, and change when
# constraints change (the actual reported bug).
# ---------------------------------------------------------------------------

def test_hrp_respects_position_limit_after_change():
    opt = PortfolioOptimizer(_make_returns(), risk_free_rate=0.02)

    loose = PortfolioConstraints(min_weight=0.0, max_weight=0.40)
    w1 = opt.optimize("hrp", loose)["weights"]

    tight = PortfolioConstraints(min_weight=0.0, max_weight=0.25)
    w2 = opt.optimize("hrp", tight)["weights"]

    assert (w2 <= 0.25 + 1e-6).all()
    assert w2.sum() == pytest.approx(1.0)
    assert not w1.equals(w2), (
        "HRP result did not change after tightening max_weight — "
        "the exact bug being regression-tested"
    )


def test_hrp_respects_turnover_band():
    current = pd.Series([0.05, 0.05, 0.05, 0.85], index=TICKERS)
    c = PortfolioConstraints(
        min_weight=0.0, max_weight=1.0,
        turnover_enabled=True, reduction_pct=0.5, increase_pct=0.2,
        allow_full_exit=True, current_weights=current,
    )
    opt = PortfolioOptimizer(_make_returns(), risk_free_rate=0.02)
    w = opt.optimize("hrp", c)["weights"]
    bounds = dict(zip(TICKERS, c.compute_bounds(TICKERS)))
    for ticker in TICKERS:
        lb, ub = bounds[ticker]
        assert lb - 1e-6 <= w[ticker] <= ub + 1e-6
    assert w.sum() == pytest.approx(1.0)


def test_equal_weight_respects_turnover_band():
    # Equal weight's unconstrained answer is a flat 25% each; a turnover band
    # around a concentrated current position forces a real, non-uniform split.
    current = pd.Series([0.05, 0.05, 0.05, 0.85], index=TICKERS)
    c = PortfolioConstraints(
        min_weight=0.0, max_weight=1.0,
        turnover_enabled=True, reduction_pct=0.5, increase_pct=0.2,
        allow_full_exit=True, current_weights=current,
    )
    opt = PortfolioOptimizer(_make_returns(), risk_free_rate=0.02)
    w = opt.optimize("equal_weight", c)["weights"]
    assert not np.allclose(w.values, 0.25), (
        "Equal Weight ignored the turnover band and stayed at a flat 25% each"
    )
    bounds = dict(zip(TICKERS, c.compute_bounds(TICKERS)))
    for ticker in TICKERS:
        lb, ub = bounds[ticker]
        assert lb - 1e-6 <= w[ticker] <= ub + 1e-6
    assert w.sum() == pytest.approx(1.0)


def test_max_sharpe_still_respects_bounds_after_projection_is_added():
    # Regression guard: adding a universal post-processing projection step
    # must not change results for methods that already solve within bounds.
    c = PortfolioConstraints(min_weight=0.0, max_weight=0.30)
    opt = PortfolioOptimizer(_make_returns(), risk_free_rate=0.02)
    w = opt.optimize("max_sharpe", c)["weights"]
    assert (w <= 0.30 + 1e-6).all()
    assert w.sum() == pytest.approx(1.0)
