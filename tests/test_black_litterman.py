"""
Smoke tests for src/optimization/black_litterman.py.

All tests use synthetic return data — no network access required.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.optimization.black_litterman import BlackLittermanModel, BlackLittermanError


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _make_returns(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
    # Mild positive drift, realistic daily vol
    data = rng.standard_normal((n, len(tickers))) * 0.01 + 0.0003
    return pd.DataFrame(data, columns=tickers)


# ---------------------------------------------------------------------------
# Test 1 — equilibrium returns have correct shape
# ---------------------------------------------------------------------------

def test_equilibrium_returns_shape():
    returns = _make_returns()
    model = BlackLittermanModel(returns)
    eq = model.equilibrium_returns
    assert isinstance(eq, pd.Series)
    assert list(eq.index) == list(returns.columns)
    assert len(eq) == len(returns.columns)


# ---------------------------------------------------------------------------
# Test 2 — no views: get_posterior_returns() equals equilibrium returns
# ---------------------------------------------------------------------------

def test_no_views_returns_equilibrium():
    returns = _make_returns()
    model = BlackLittermanModel(returns)
    posterior = model.get_posterior_returns()
    pd.testing.assert_series_equal(posterior, model.equilibrium_returns)


# ---------------------------------------------------------------------------
# Test 3 — absolute view shifts posterior toward the stated return
# ---------------------------------------------------------------------------

def test_absolute_view_shifts_posterior():
    returns = _make_returns()
    model = BlackLittermanModel(returns, tau=0.05)
    eq_aapl = float(model.equilibrium_returns["AAPL"])

    # State that AAPL will return 30% with high confidence
    model.add_absolute_view("AAPL", view_return=0.30, confidence=0.9)
    posterior = model.get_posterior_returns()

    # Posterior for AAPL should be pulled toward 30%, above equilibrium
    assert float(posterior["AAPL"]) > eq_aapl, (
        f"Posterior {posterior['AAPL']:.4f} should be above equilibrium {eq_aapl:.4f} "
        "when a strongly positive view is added."
    )


# ---------------------------------------------------------------------------
# Test 4 — relative view: long asset gets higher posterior than short asset
# ---------------------------------------------------------------------------

def test_relative_view_shifts_posterior():
    returns = _make_returns()
    model_no_view = BlackLittermanModel(returns, tau=0.05)
    eq_diff = float(
        model_no_view.equilibrium_returns["AAPL"]
        - model_no_view.equilibrium_returns["AMZN"]
    )

    model = BlackLittermanModel(returns, tau=0.05)
    # AAPL will outperform AMZN by 10%
    model.add_relative_view(["AAPL"], ["AMZN"], view_return=0.10, confidence=0.8)
    posterior = model.get_posterior_returns()

    post_diff = float(posterior["AAPL"] - posterior["AMZN"])
    assert post_diff > eq_diff, (
        f"Posterior diff ({post_diff:.4f}) should be larger than equilibrium diff "
        f"({eq_diff:.4f}) after a positive relative view."
    )


# ---------------------------------------------------------------------------
# Test 5 — optimize() returns valid weights (sum=1, all >= 0)
# ---------------------------------------------------------------------------

def test_optimize_valid_weights():
    returns = _make_returns()
    model = BlackLittermanModel(returns)
    out = model.optimize(max_weight=1.0, min_weight=0.0)

    weights = out["weights"]
    assert isinstance(weights, pd.Series)
    assert abs(weights.sum() - 1.0) < 1e-6, f"Weights sum to {weights.sum():.8f}, not 1."
    assert (weights >= -1e-8).all(), f"Negative weights found: {weights[weights < 0]}"
    assert (weights <= 1.0 + 1e-8).all(), f"Weights exceed max: {weights[weights > 1.0]}"


# ---------------------------------------------------------------------------
# Test 6 — optimize() with views produces different weights than without
# ---------------------------------------------------------------------------

def test_optimize_with_views_differs_from_no_views():
    """
    A very strong view on GOOG (50%, confidence=0.99) should push GOOG's
    posterior return far above the others and shift the max-weight-capped
    allocation materially toward GOOG.
    """
    returns = _make_returns()

    model_no_view = BlackLittermanModel(returns, tau=0.05)
    out_no_view = model_no_view.optimize(max_weight=0.5)

    model_with_view = BlackLittermanModel(returns, tau=0.05)
    model_with_view.add_absolute_view("GOOG", view_return=0.50, confidence=0.99)
    out_with_view = model_with_view.optimize(max_weight=0.5)

    # Verify the posterior return itself shifted
    assert (
        float(out_with_view["posterior_returns"]["GOOG"])
        > float(out_no_view["posterior_returns"]["GOOG"])
    ), "GOOG posterior return did not increase with a positive view."

    # The final weights should differ (GOOG should gain weight)
    diff = (out_with_view["weights"] - out_no_view["weights"]).abs().max()
    assert diff > 1e-4, (
        f"Max weight difference is only {diff:.6f} — views appear to have no effect on weights."
    )


# ---------------------------------------------------------------------------
# Test 7 — clear_views() resets model to no-view state
# ---------------------------------------------------------------------------

def test_clear_views_resets():
    returns = _make_returns()
    model = BlackLittermanModel(returns)
    model.add_absolute_view("MSFT", view_return=0.20, confidence=0.7)

    assert model.P is not None and len(model.Q) == 1

    model.clear_views()

    assert model.P is None
    assert model.Q is None
    assert model.omega is None

    # After clearing, posterior should equal equilibrium
    posterior = model.get_posterior_returns()
    pd.testing.assert_series_equal(posterior, model.equilibrium_returns)


# ---------------------------------------------------------------------------
# Test 8 — unknown asset raises BlackLittermanError
# ---------------------------------------------------------------------------

def test_unknown_asset_raises():
    returns = _make_returns()
    model = BlackLittermanModel(returns)
    with pytest.raises(BlackLittermanError):
        model.add_absolute_view("INVALID_TICKER", view_return=0.10)


# ---------------------------------------------------------------------------
# Test 9 — posterior covariance has correct shape and is positive semi-definite
# ---------------------------------------------------------------------------

def test_posterior_covariance_shape_and_psd():
    returns = _make_returns()
    model = BlackLittermanModel(returns)
    model.add_absolute_view("AAPL", 0.15, confidence=0.6)
    cov = model.get_posterior_covariance()

    n = len(returns.columns)
    assert cov.shape == (n, n), f"Expected ({n},{n}), got {cov.shape}"

    # Minimum eigenvalue should be >= 0 (PSD)
    min_eig = float(np.linalg.eigvalsh(cov.values).min())
    assert min_eig >= -1e-8, f"Posterior covariance is not PSD: min eigenvalue={min_eig:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
