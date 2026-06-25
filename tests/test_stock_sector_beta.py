"""
Smoke tests for src/risk/stock_sector_beta.py.

Tests use synthetic price data and mocked ETF price series so they run
offline without yfinance network access. Test 6 explicitly mocks
yf.download to verify fallback behaviour.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.risk.stock_sector_beta import (
    SECTOR_ETF_MAP,
    CIRCULARITY_THRESHOLD,
    IDX_MARKET_PROXY,
    IDX_TICKER_SUFFIX,
    StockBetaEntry,
    StockBetaResult,
    compute_etf_ex_stock_returns,
    compute_sector_relative_beta,
    compute_all_stock_betas,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_prices(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Synthetic daily close prices for several tickers + XLB + XLK."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n, freq="B")

    # XLB (Basic Materials) factor
    xlb_factor = rng.standard_normal(n).cumsum() * 0.008
    xlb_prices = 100 * np.exp(xlb_factor)

    # XLK (Technology) factor
    xlk_factor = rng.standard_normal(n).cumsum() * 0.010
    xlk_prices = 100 * np.exp(xlk_factor)

    # LIN — Basic Materials, non-dominant (weight << 10%)
    lin_factor = xlb_factor + rng.standard_normal(n) * 0.006
    lin_prices = 200 * np.exp(lin_factor)

    # NVDA — Technology, dominant (≈24% of XLK)
    nvda_factor = xlk_factor + rng.standard_normal(n) * 0.015
    nvda_prices = 400 * np.exp(nvda_factor)

    # AVGO, PANW, ADI — Technology, all non-dominant
    avgo_factor = xlk_factor + rng.standard_normal(n) * 0.008
    panw_factor = xlk_factor + rng.standard_normal(n) * 0.012
    adi_factor  = xlk_factor + rng.standard_normal(n) * 0.007

    return pd.DataFrame(
        {
            "LIN":  200 * np.exp(lin_factor),
            "NVDA": 400 * np.exp(nvda_factor),
            "AVGO": 600 * np.exp(avgo_factor),
            "PANW": 300 * np.exp(panw_factor),
            "ADI":  180 * np.exp(adi_factor),
            "XLB":  xlb_prices,
            "XLK":  xlk_prices,
        },
        index=dates,
    )


def _make_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def _etf_prices_from(prices: pd.DataFrame) -> dict[str, pd.Series]:
    """Build the etf_prices dict that compute_sector_relative_beta() expects."""
    etf_map = {}
    for col in ("XLB", "XLK"):
        if col in prices.columns:
            etf_map[col] = prices[col]
    return etf_map


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Non-dominant stock: no circularity correction applied
# ─────────────────────────────────────────────────────────────────────────────

def test_non_dominant_no_correction():
    """LIN (Basic Materials, weight in XLB << 10%) must have circularity_corrected=False."""
    prices = _make_prices()
    returns = _make_returns(prices)
    etf_prices = _etf_prices_from(prices)

    # Mock get_stock_weight_in_etf to return a small weight (non-dominant)
    with patch("src.risk.stock_sector_beta.get_stock_weight_in_etf", return_value=0.02):
        entry = compute_sector_relative_beta(
            ticker="LIN",
            sector="Basic Materials",
            stock_returns=returns["LIN"],
            etf_prices=etf_prices,
            min_observations=60,
        )

    assert entry.circularity_corrected is False, "Non-dominant stock must not be corrected"
    assert entry.source == "sector_etf"
    assert entry.etf_proxy == "XLB"
    assert -2.0 <= entry.beta <= 3.0, f"Beta out of plausible range: {entry.beta}"
    assert entry.r_squared is not None and entry.r_squared > 0.0
    assert entry.n_observations > 0


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Dominant stock: circularity correction applied and reduces beta
# ─────────────────────────────────────────────────────────────────────────────

def test_dominant_stock_corrected():
    """NVDA (Technology, weight ~24%) — corrected beta must be lower than naive beta."""
    prices = _make_prices()
    returns = _make_returns(prices)
    etf_prices = _etf_prices_from(prices)

    nvda_weight = 0.24  # above CIRCULARITY_THRESHOLD

    # Naive beta (no correction)
    with patch("src.risk.stock_sector_beta.get_stock_weight_in_etf", return_value=0.01):
        naive_entry = compute_sector_relative_beta(
            ticker="NVDA",
            sector="Technology",
            stock_returns=returns["NVDA"],
            etf_prices=etf_prices,
            min_observations=60,
        )

    # Corrected beta
    with patch("src.risk.stock_sector_beta.get_stock_weight_in_etf", return_value=nvda_weight):
        corrected_entry = compute_sector_relative_beta(
            ticker="NVDA",
            sector="Technology",
            stock_returns=returns["NVDA"],
            etf_prices=etf_prices,
            min_observations=60,
        )

    print(f"NVDA naive beta={naive_entry.beta:.4f}, corrected beta={corrected_entry.beta:.4f}")

    assert corrected_entry.circularity_corrected is True, "Dominant stock must be corrected"
    assert corrected_entry.source == "etf_ex_stock"
    # Circularity correction removes the stock's self-contribution → lower beta
    assert corrected_entry.beta < naive_entry.beta, (
        f"Corrected beta ({corrected_entry.beta:.4f}) should be lower than "
        f"naive beta ({naive_entry.beta:.4f})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Different stocks in same sector get different betas
# ─────────────────────────────────────────────────────────────────────────────

def test_same_sector_different_betas():
    """AVGO, PANW, ADI (all Technology) must have distinct beta values."""
    prices = _make_prices()
    returns = _make_returns(prices)
    etf_prices = _etf_prices_from(prices)

    # Non-dominant for all (weight << 10%)
    with patch("src.risk.stock_sector_beta.get_stock_weight_in_etf", return_value=0.01):
        avgo_entry = compute_sector_relative_beta(
            "AVGO", "Technology", returns["AVGO"], etf_prices, min_observations=60
        )
        panw_entry = compute_sector_relative_beta(
            "PANW", "Technology", returns["PANW"], etf_prices, min_observations=60
        )
        adi_entry = compute_sector_relative_beta(
            "ADI", "Technology", returns["ADI"], etf_prices, min_observations=60
        )

    betas = [avgo_entry.beta, panw_entry.beta, adi_entry.beta]
    print(f"AVGO={betas[0]:.4f}, PANW={betas[1]:.4f}, ADI={betas[2]:.4f}")

    assert len(set(round(b, 4) for b in betas)) == 3, (
        f"All three tech betas must be different — got {betas}. "
        "Uniform-shock bug may still be present."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — IDX ticker uses market proxy, not sector ETF
# ─────────────────────────────────────────────────────────────────────────────

def test_idx_ticker_market_proxy():
    """BBCA.JK must use ^JKSE as proxy, source='market_proxy', corrected=False."""
    rng = np.random.default_rng(7)
    n = 300
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    jkse_prices = pd.Series(1000 * np.exp(rng.standard_normal(n).cumsum() * 0.005), index=dates)
    factor = jkse_prices.pct_change().dropna()
    bbca_returns = factor + pd.Series(rng.standard_normal(len(factor)) * 0.004, index=factor.index)

    etf_prices = {IDX_MARKET_PROXY: jkse_prices}

    entry = compute_sector_relative_beta(
        ticker="BBCA.JK",
        sector="Financials",
        stock_returns=bbca_returns,
        etf_prices=etf_prices,
        min_observations=60,
    )

    assert entry.etf_proxy == IDX_MARKET_PROXY, f"Expected ^JKSE, got {entry.etf_proxy}"
    assert entry.source == "market_proxy"
    assert entry.circularity_corrected is False
    assert -2.0 <= entry.beta <= 3.0


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — Stress output differentiates implied returns within same sector
# ─────────────────────────────────────────────────────────────────────────────

def test_stress_output_differentiated():
    """AVGO, PANW, ADI in Tech Selloff must produce distinct beta_implied_returns."""
    from src.simulation.sector_stress import (
        SectorStressEngine, SectorStressConfig, SectorStressScenario,
    )
    from src.risk.sector_beta import SectorBetaConfig
    from src.risk.dcc_garch import DCCGARCHConfig
    from src.risk.copula import CopulaConfig
    from src.risk.regime_detection import RegimeConfig

    prices = _make_prices(n=600)
    returns = _make_returns(prices)
    stock_returns = returns[["AVGO", "PANW", "ADI"]]

    sector_map = {"AVGO": "Technology", "PANW": "Technology", "ADI": "Technology"}

    cfg = SectorStressConfig(
        beta_config=SectorBetaConfig(),
        dcc_config=DCCGARCHConfig(estimate_dcc_params=False),
        copula_config=CopulaConfig(n_simulation_paths=100),
        regime_config=RegimeConfig(n_states=2, n_init=2),
    )

    # Distinct betas — what compute_all_stock_betas() WOULD return with live ETF data
    injected = StockBetaResult(
        entries={
            "AVGO": StockBetaEntry("AVGO", "Technology", "XLK", 0.90, 0.65, False, 0.02, "sector_etf", 400, None),
            "PANW": StockBetaEntry("PANW", "Technology", "XLK", 1.25, 0.58, False, 0.01, "sector_etf", 400, None),
            "ADI":  StockBetaEntry("ADI",  "Technology", "XLK", 0.72, 0.61, False, 0.01, "sector_etf", 400, None),
        },
        computed_at="2024-01-01T00:00:00",
        data_start="2022-01-03",
        data_end="2024-04-01",
        n_fallbacks=0,
    )

    # Patch compute_all_stock_betas so fit() uses injected betas (bypasses ETF download)
    engine = SectorStressEngine(config=cfg)
    with patch("src.simulation.sector_stress.compute_all_stock_betas", return_value=injected):
        engine.fit(stock_returns, sector_map)

    scenario = SectorStressScenario(
        name="Tech Selloff",
        description="test",
        shocked_sectors={"Technology": -0.20},
        use_copula=False,
        use_dcc=False,
        use_regime=False,
    )
    holdings = {"AVGO": 1/3, "PANW": 1/3, "ADI": 1/3}
    result = engine.run_stress(scenario, holdings)

    implied = [h.beta_implied_return for h in result.holdings_results]
    print(f"AVGO={implied[0]:.4f}, PANW={implied[1]:.4f}, ADI={implied[2]:.4f}")

    # All three must be different
    assert len(set(round(v, 4) for v in implied)) == 3, (
        f"All implied returns must be distinct: {implied}. "
        "Uniform shock bug may still be present."
    )
    # All should be negative (shock is -20%)
    for v in implied:
        assert v < 0.0, f"Implied return should be negative with -20% shock: {v}"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — Fallback when ETF download fails
# ─────────────────────────────────────────────────────────────────────────────

def test_etf_download_failure_fallback():
    """When yf.download raises, all tech stocks must get source='fallback', beta=1.0."""
    prices = _make_prices()
    returns = _make_returns(prices)

    sector_map = {"AVGO": "Technology", "PANW": "Technology", "ADI": "Technology"}
    dates = returns.index
    start = str(dates[0].date())
    end = str(dates[-1].date())

    with patch("src.risk.stock_sector_beta.get_stock_weight_in_etf", return_value=None), \
         patch("yfinance.download", side_effect=Exception("TLS handshake failed")):
        result = compute_all_stock_betas(
            tickers=["AVGO", "PANW", "ADI"],
            sector_map=sector_map,
            stock_returns=returns[["AVGO", "PANW", "ADI"]],
            start_date=start,
            end_date=end,
            min_observations=60,
        )

    # Must not raise — result must be a valid StockBetaResult
    assert isinstance(result, StockBetaResult)

    for ticker in ["AVGO", "PANW", "ADI"]:
        entry = result.entries[ticker]
        assert entry.source == "fallback", f"{ticker} should have source=fallback, got {entry.source}"
        assert entry.beta == 1.0, f"{ticker} should have beta=1.0, got {entry.beta}"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7 — R² sanity: low R² tickers should have warning populated
# ─────────────────────────────────────────────────────────────────────────────

def test_r_squared_reasonable():
    """
    For stocks with source in (sector_etf, etf_ex_stock), R² should be > 0.10.
    Stocks failing this must have warning populated.
    """
    rng = np.random.default_rng(99)
    n = 400
    dates = pd.date_range("2022-01-03", periods=n, freq="B")

    xlk_prices = pd.Series(100 * np.exp(rng.standard_normal(n).cumsum() * 0.010), index=dates)
    # High-correlation ticker (good R²)
    correlated_returns = xlk_prices.pct_change().dropna() + pd.Series(rng.standard_normal(n - 1) * 0.003, index=dates[1:])
    # Low-correlation ticker (poor R²) — mostly noise
    noise_returns = pd.Series(rng.standard_normal(n - 1) * 0.015, index=dates[1:])

    etf_prices = {"XLK": xlk_prices}

    with patch("src.risk.stock_sector_beta.get_stock_weight_in_etf", return_value=0.01):
        good_entry = compute_sector_relative_beta(
            "GOOD", "Technology", correlated_returns, etf_prices, min_observations=60
        )
        bad_entry = compute_sector_relative_beta(
            "NOISY", "Technology", noise_returns, etf_prices, min_observations=60
        )

    # Good entry: R² should be high, no warning about low R²
    assert good_entry.r_squared is not None
    assert good_entry.r_squared > 0.10, (
        f"Correlated stock should have R² > 0.10, got {good_entry.r_squared:.3f}"
    )

    # Noisy entry: R² should be low, warning must be populated on the data object
    assert bad_entry.r_squared is not None
    if bad_entry.r_squared < 0.10:
        assert bad_entry.warning is not None, "Low R² ticker must have warning populated"
        assert "R²" in bad_entry.warning or "Low" in (bad_entry.warning or "")


# ─────────────────────────────────────────────────────────────────────────────
# Direct invocation
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
