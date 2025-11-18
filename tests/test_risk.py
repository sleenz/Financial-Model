"""Tests for risk analytics module (Phase 3)."""

import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.risk.metrics import RiskMetrics, calculate_metrics
from src.risk.var import VaRCalculator, PortfolioVaR, calculate_var, calculate_cvar
from src.risk.garch import GARCHModel, MultiAssetGARCH, ewma_volatility


def generate_sample_returns(n_assets=3, n_periods=500, seed=42):
    """Generate sample return data for testing."""
    np.random.seed(seed)

    tickers = ['AAPL', 'GOOGL', 'MSFT'][:n_assets]
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')

    # Generate returns with realistic properties
    returns_data = {}
    for i, ticker in enumerate(tickers):
        returns_data[ticker] = np.random.normal(0.0005, 0.02, n_periods)

    return pd.DataFrame(returns_data, index=dates)


class TestRiskMetrics:
    """Tests for RiskMetrics class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.returns = generate_sample_returns()
        self.rm = RiskMetrics(self.returns)

    def test_initialization(self):
        """Test RiskMetrics initialization."""
        assert self.rm.frequency == 252
        assert self.rm.risk_free_rate == 0.02

    def test_historical_volatility(self):
        """Test historical volatility calculation."""
        vol = self.rm.historical_volatility(annualize=True)

        assert len(vol) == 3
        # Volatility should be positive
        assert (vol > 0).all()
        # Annualized vol should be reasonable (not extreme)
        assert (vol < 1).all()

    def test_rolling_volatility(self):
        """Test rolling volatility calculation."""
        vol = self.rm.historical_volatility(window=30, annualize=True)

        # Should have NaN for first window-1 periods
        assert vol.isna().sum().sum() > 0
        # Non-NaN values should be positive
        assert (vol.dropna() > 0).all().all()

    def test_downside_deviation(self):
        """Test downside deviation calculation."""
        dd = self.rm.downside_deviation(annualize=True)

        assert len(dd) == 3
        assert (dd > 0).all()
        # Downside deviation should be less than or equal to total volatility
        vol = self.rm.historical_volatility(annualize=True)
        assert (dd <= vol * 1.1).all()  # Allow small numerical tolerance

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        sharpe = self.rm.sharpe_ratio()

        assert len(sharpe) == 3
        # Sharpe should be finite
        assert sharpe.isna().sum() == 0

    def test_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        sortino = self.rm.sortino_ratio()

        assert len(sortino) == 3
        # Sortino should be >= Sharpe for positive skew
        # (not always true, so just check it's finite)
        assert sortino.isna().sum() == 0

    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        prices = (1 + self.returns).cumprod()
        mdd = self.rm.max_drawdown(prices)

        assert len(mdd) == 3
        # Drawdowns are negative or zero
        assert (mdd <= 0).all()
        # Shouldn't lose more than 100%
        assert (mdd >= -1).all()

    def test_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        prices = (1 + self.returns).cumprod()
        calmar = self.rm.calmar_ratio(prices)

        assert len(calmar) == 3

    def test_omega_ratio(self):
        """Test Omega ratio calculation."""
        omega = self.rm.omega_ratio()

        assert len(omega) == 3
        # Omega should be positive
        assert (omega > 0).all()

    def test_skewness(self):
        """Test skewness calculation."""
        skew = self.rm.skewness()

        assert len(skew) == 3
        # Skewness should be reasonable
        assert (skew > -5).all()
        assert (skew < 5).all()

    def test_kurtosis(self):
        """Test kurtosis calculation."""
        kurt = self.rm.kurtosis()

        assert len(kurt) == 3
        # Excess kurtosis can be negative (platykurtic)
        assert (kurt > -3).all()

    def test_drawdown_series(self):
        """Test drawdown time series."""
        prices = (1 + self.returns).cumprod()
        dd_series = self.rm.drawdown_series(prices)

        assert dd_series.shape == prices.shape
        # All drawdowns should be <= 0
        assert (dd_series <= 0).all().all()

    def test_ulcer_index(self):
        """Test Ulcer Index calculation."""
        prices = (1 + self.returns).cumprod()
        ulcer = self.rm.ulcer_index(prices)

        assert len(ulcer) == 3
        assert (ulcer >= 0).all()

    def test_diversification_ratio(self):
        """Test diversification ratio calculation."""
        weights = np.array([0.4, 0.3, 0.3])
        div_ratio = self.rm.diversification_ratio(weights)

        # Diversification ratio >= 1 for diversified portfolio
        assert div_ratio >= 1.0

    def test_effective_number_of_bets(self):
        """Test ENB calculation."""
        weights = np.array([0.4, 0.3, 0.3])
        enb = self.rm.effective_number_of_bets(weights)

        # ENB should be between 1 and n_assets
        assert enb >= 1
        assert enb <= 3

    def test_herfindahl_index(self):
        """Test Herfindahl Index calculation."""
        weights = np.array([0.4, 0.3, 0.3])
        hhi = self.rm.herfindahl_index(weights)

        # HHI for 3 equal weights would be 0.333
        # For our weights: 0.16 + 0.09 + 0.09 = 0.34
        assert abs(hhi - 0.34) < 0.01

    def test_correlation_matrix(self):
        """Test correlation matrix calculation."""
        corr = self.rm.correlation_matrix()

        assert corr.shape == (3, 3)
        # Diagonal should be 1
        for i in range(3):
            assert abs(corr.iloc[i, i] - 1.0) < 1e-6

    def test_summary_table(self):
        """Test summary table generation."""
        prices = (1 + self.returns).cumprod()
        summary = self.rm.summary_table(prices)

        assert 'Annual Return' in summary.columns
        assert 'Sharpe Ratio' in summary.columns
        assert len(summary) == 3

    def test_calculate_all_metrics(self):
        """Test calculating all metrics."""
        weights = np.array([0.4, 0.3, 0.3])
        prices = (1 + self.returns).cumprod()
        metrics = self.rm.calculate_all_metrics(weights, prices)

        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'diversification_ratio' in metrics


class TestVaRCalculator:
    """Tests for VaRCalculator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.returns = generate_sample_returns()
        self.var_calc = VaRCalculator(self.returns)

    def test_historical_var(self):
        """Test Historical VaR calculation."""
        var = self.var_calc.historical_var(0.95)

        assert len(var) == 3
        # VaR should be negative (loss)
        assert (var < 0).all()

    def test_parametric_var(self):
        """Test Parametric VaR calculation."""
        var = self.var_calc.parametric_var(0.95)

        assert len(var) == 3
        assert (var < 0).all()

    def test_cornish_fisher_var(self):
        """Test Cornish-Fisher VaR calculation."""
        var = self.var_calc.cornish_fisher_var(0.95)

        assert len(var) == 3

    def test_historical_cvar(self):
        """Test Historical CVaR calculation."""
        cvar = self.var_calc.historical_cvar(0.95)
        var = self.var_calc.historical_var(0.95)

        assert len(cvar) == 3
        # CVaR should be more extreme than VaR
        assert (cvar <= var).all()

    def test_parametric_cvar(self):
        """Test Parametric CVaR calculation."""
        cvar = self.var_calc.parametric_cvar(0.95)
        var = self.var_calc.parametric_var(0.95)

        assert len(cvar) == 3
        # CVaR should be more extreme than VaR
        assert (cvar <= var).all()

    def test_var_with_portfolio_value(self):
        """Test VaR calculation with dollar value."""
        var = self.var_calc.historical_var(0.95, portfolio_value=10000)

        # Dollar VaR should be in hundreds range for $10k
        assert (abs(var) < 5000).all()

    def test_calculate_all(self):
        """Test calculating all VaR measures."""
        results = self.var_calc.calculate_all(0.95)

        assert 'Historical VaR' in results.columns
        assert 'Parametric VaR' in results.columns
        assert 'Historical CVaR' in results.columns

    def test_var_summary(self):
        """Test VaR summary generation."""
        summary = self.var_calc.var_summary()

        assert 'Confidence' in summary.columns
        assert len(summary) > 0

    def test_rolling_var(self):
        """Test rolling VaR calculation."""
        rolling = self.var_calc.rolling_var(window=100, confidence=0.95)

        # Should have NaN for first window periods
        assert rolling.isna().sum().sum() > 0

    def test_different_confidence_levels(self):
        """Test VaR at different confidence levels."""
        var_95 = self.var_calc.historical_var(0.95)
        var_99 = self.var_calc.historical_var(0.99)

        # 99% VaR should be more extreme than 95%
        assert (var_99 <= var_95).all()


class TestPortfolioVaR:
    """Tests for PortfolioVaR class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.returns = generate_sample_returns()
        self.weights = np.array([0.4, 0.3, 0.3])
        self.pvar = PortfolioVaR(self.returns, self.weights, 10000)

    def test_marginal_var(self):
        """Test Marginal VaR calculation."""
        marginal = self.pvar.marginal_var(0.95)

        assert len(marginal) == 3

    def test_component_var(self):
        """Test Component VaR calculation."""
        component = self.pvar.component_var(0.95)

        assert len(component) == 3
        # Components should sum to total VaR (approximately)

    def test_var_decomposition(self):
        """Test VaR decomposition."""
        decomp = self.pvar.var_decomposition(0.95)

        assert 'Weight' in decomp.columns
        assert 'Component VaR' in decomp.columns
        assert 'Contribution %' in decomp.columns


class TestGARCHModel:
    """Tests for GARCHModel class."""

    def setup_method(self):
        """Set up test fixtures."""
        returns = generate_sample_returns(n_assets=1)
        self.returns = returns.iloc[:, 0]
        self.garch = GARCHModel(self.returns)

    def test_fit(self):
        """Test GARCH model fitting."""
        result = self.garch.fit()

        assert 'params' in result
        assert 'omega' in result['params']
        assert 'alpha' in result['params']
        assert 'beta' in result['params']

    def test_forecast(self):
        """Test GARCH volatility forecasting."""
        self.garch.fit()
        forecast = self.garch.forecast(horizon=30)

        assert len(forecast) == 30
        assert 'volatility' in forecast.columns
        assert 'annualized_vol' in forecast.columns

    def test_conditional_volatility(self):
        """Test conditional volatility retrieval."""
        self.garch.fit()
        cond_vol = self.garch.get_conditional_volatility(annualize=True)

        assert len(cond_vol) == len(self.returns)
        assert (cond_vol > 0).all()

    def test_persistence(self):
        """Test volatility persistence calculation."""
        self.garch.fit()
        persistence = self.garch.persistence()

        # Persistence should be between 0 and 1 for stationary process
        assert persistence >= 0
        assert persistence < 1

    def test_half_life(self):
        """Test half-life calculation."""
        self.garch.fit()
        half_life = self.garch.half_life()

        assert half_life > 0

    def test_unconditional_volatility(self):
        """Test unconditional volatility calculation."""
        self.garch.fit()
        uncond_vol = self.garch.unconditional_volatility(annualize=True)

        assert uncond_vol > 0
        assert uncond_vol < 1  # Reasonable annualized vol

    def test_summary(self):
        """Test model summary."""
        self.garch.fit()
        summary = self.garch.summary()

        assert 'model_type' in summary
        assert 'parameters' in summary
        assert 'persistence' in summary


class TestMultiAssetGARCH:
    """Tests for MultiAssetGARCH class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.returns = generate_sample_returns(n_assets=3)
        self.multi_garch = MultiAssetGARCH(self.returns)

    def test_fit_all(self):
        """Test fitting GARCH for all assets."""
        summaries = self.multi_garch.fit_all()

        assert len(summaries) == 3
        for ticker in self.returns.columns:
            assert ticker in summaries

    def test_forecast_all(self):
        """Test forecasting for all assets."""
        self.multi_garch.fit_all()
        forecasts = self.multi_garch.forecast_all(horizon=30)

        assert forecasts.shape == (30, 3)

    def test_conditional_volatilities(self):
        """Test getting all conditional volatilities."""
        self.multi_garch.fit_all()
        vols = self.multi_garch.get_conditional_volatilities()

        assert vols.shape[1] == 3


class TestEWMAVolatility:
    """Tests for EWMA volatility function."""

    def test_ewma_volatility(self):
        """Test EWMA volatility calculation."""
        returns = generate_sample_returns()
        ewma = ewma_volatility(returns, decay=0.94, annualize=True)

        assert ewma.shape == returns.shape
        # Should be positive after initial NaN
        assert (ewma.dropna() > 0).all().all()


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.returns = generate_sample_returns()

    def test_calculate_var(self):
        """Test calculate_var convenience function."""
        var = calculate_var(self.returns, 0.95, 'historical')
        assert len(var) == 3

    def test_calculate_cvar(self):
        """Test calculate_cvar convenience function."""
        cvar = calculate_cvar(self.returns, 0.95, 'historical')
        assert len(cvar) == 3

    def test_calculate_metrics(self):
        """Test calculate_metrics convenience function."""
        metrics = calculate_metrics(self.returns)
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
