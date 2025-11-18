"""
Tests for Factor Analysis Module (Phase 8)
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from src.factors.fama_french import FamaFrenchAnalyzer, get_factor_data
from src.factors.attribution import (
    SectorAttribution, BrinsonAttribution,
    calculate_sector_attribution, run_brinson_attribution
)
from src.factors.style import (
    StyleFactorAnalyzer, calculate_momentum,
    calculate_value_score, calculate_quality_score,
    calculate_low_volatility_score
)
from src.factors.decomposition import (
    FactorRiskDecomposition, calculate_factor_risk,
    get_factor_stress_impact
)


@pytest.fixture
def sample_returns():
    """Create sample returns data."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='B')
    returns = pd.DataFrame(
        np.random.randn(252, 5) * 0.01,
        index=dates,
        columns=['AAPL', 'MSFT', 'GOOGL', 'JPM', 'XOM']
    )
    return returns


@pytest.fixture
def sample_prices(sample_returns):
    """Create sample price data."""
    return (1 + sample_returns).cumprod() * 100


@pytest.fixture
def sample_weights():
    """Create sample portfolio weights."""
    return pd.Series({
        'AAPL': 0.25,
        'MSFT': 0.25,
        'GOOGL': 0.20,
        'JPM': 0.15,
        'XOM': 0.15
    })


class TestFamaFrenchAnalyzer:
    """Tests for Fama-French factor analysis."""

    def test_initialization(self, sample_returns):
        """Test analyzer initialization."""
        analyzer = FamaFrenchAnalyzer(sample_returns)
        assert analyzer is not None
        assert len(analyzer.returns) > 0

    def test_analyze_asset(self, sample_returns):
        """Test single asset analysis."""
        analyzer = FamaFrenchAnalyzer(sample_returns)
        result = analyzer.analyze_asset('AAPL', model='5')

        assert 'alpha' in result
        assert 'betas' in result
        assert 'r_squared' in result
        assert 'Mkt-RF' in result['betas']

    def test_analyze_portfolio(self, sample_returns, sample_weights):
        """Test portfolio analysis."""
        analyzer = FamaFrenchAnalyzer(sample_returns)
        result = analyzer.analyze_portfolio(sample_weights)

        assert 'alpha_annualized' in result
        assert 'r_squared' in result
        assert 'betas' in result

    def test_3_factor_model(self, sample_returns, sample_weights):
        """Test 3-factor model."""
        analyzer = FamaFrenchAnalyzer(sample_returns)
        result = analyzer.analyze_portfolio(sample_weights, model='3')

        # Should have 3 factors
        assert 'Mkt-RF' in result['betas']
        assert 'SMB' in result['betas']
        assert 'HML' in result['betas']

    def test_5_factor_model(self, sample_returns, sample_weights):
        """Test 5-factor model."""
        analyzer = FamaFrenchAnalyzer(sample_returns)
        result = analyzer.analyze_portfolio(sample_weights, model='5')

        # Should have 5 factors
        assert 'Mkt-RF' in result['betas']
        assert 'RMW' in result['betas']
        assert 'CMA' in result['betas']

    def test_analyze_all_assets(self, sample_returns):
        """Test analyzing all assets."""
        analyzer = FamaFrenchAnalyzer(sample_returns)
        result = analyzer.analyze_all_assets()

        assert len(result) == len(sample_returns.columns)
        assert 'Alpha' in result.columns

    def test_factor_contribution(self, sample_returns, sample_weights):
        """Test factor contribution calculation."""
        analyzer = FamaFrenchAnalyzer(sample_returns)
        contrib = analyzer.factor_contribution(sample_weights)

        assert 'Mkt-RF' in contrib
        assert 'alpha' in contrib
        assert 'total_explained' in contrib

    def test_rolling_betas(self, sample_returns):
        """Test rolling beta calculation."""
        analyzer = FamaFrenchAnalyzer(sample_returns)
        rolling = analyzer.rolling_betas('AAPL', window=60)

        assert len(rolling) > 0
        assert 'Mkt-RF' in rolling.columns

    def test_summary(self, sample_returns, sample_weights):
        """Test summary generation."""
        analyzer = FamaFrenchAnalyzer(sample_returns)
        summary = analyzer.summary(sample_weights)

        assert isinstance(summary, str)
        assert 'Alpha' in summary
        assert 'Factor Exposures' in summary


class TestGetFactorData:
    """Tests for factor data retrieval."""

    def test_get_factor_data_5(self):
        """Test fetching 5-factor data."""
        data = get_factor_data('2023-01-01', '2023-12-31', model='5')

        assert 'Mkt-RF' in data.columns
        assert 'SMB' in data.columns
        assert 'HML' in data.columns

    def test_get_factor_data_3(self):
        """Test fetching 3-factor data."""
        data = get_factor_data('2023-01-01', '2023-12-31', model='3')

        assert 'Mkt-RF' in data.columns
        assert len(data) > 0


class TestSectorAttribution:
    """Tests for sector attribution."""

    def test_initialization(self, sample_returns, sample_weights):
        """Test sector attribution initialization."""
        attr = SectorAttribution(sample_returns, sample_weights)
        assert attr is not None

    def test_get_sector_weights(self, sample_returns, sample_weights):
        """Test sector weight calculation."""
        attr = SectorAttribution(sample_returns, sample_weights)
        weights = attr.get_sector_weights()

        assert isinstance(weights, pd.Series)
        assert abs(weights.sum() - 1.0) < 1e-10

    def test_get_sector_returns(self, sample_returns, sample_weights):
        """Test sector returns calculation."""
        attr = SectorAttribution(sample_returns, sample_weights)
        returns = attr.get_sector_returns()

        assert isinstance(returns, pd.DataFrame)
        assert len(returns) == len(sample_returns)

    def test_sector_contribution(self, sample_returns, sample_weights):
        """Test sector contribution calculation."""
        attr = SectorAttribution(sample_returns, sample_weights)
        contrib = attr.sector_contribution()

        assert 'Weight' in contrib.columns
        assert 'Return' in contrib.columns
        assert 'Contribution' in contrib.columns

    def test_sector_correlation(self, sample_returns, sample_weights):
        """Test sector correlation calculation."""
        attr = SectorAttribution(sample_returns, sample_weights)
        corr = attr.sector_correlation()

        assert isinstance(corr, pd.DataFrame)


class TestBrinsonAttribution:
    """Tests for Brinson attribution model."""

    def test_initialization(self, sample_returns, sample_weights):
        """Test Brinson attribution initialization."""
        brinson = BrinsonAttribution(sample_returns, sample_weights)
        assert brinson is not None

    def test_calculate_attribution(self, sample_returns, sample_weights):
        """Test attribution calculation."""
        brinson = BrinsonAttribution(sample_returns, sample_weights)
        result = brinson.calculate_attribution()

        assert 'detailed' in result
        assert 'summary' in result
        assert 'allocation_effect' in result['summary']
        assert 'selection_effect' in result['summary']
        assert 'interaction_effect' in result['summary']

    def test_rolling_attribution(self, sample_returns, sample_weights):
        """Test rolling attribution."""
        brinson = BrinsonAttribution(sample_returns, sample_weights)
        rolling = brinson.rolling_attribution(window=60)

        assert 'allocation' in rolling.columns
        assert 'selection' in rolling.columns

    def test_summary(self, sample_returns, sample_weights):
        """Test summary generation."""
        brinson = BrinsonAttribution(sample_returns, sample_weights)
        summary = brinson.summary()

        assert isinstance(summary, str)
        assert 'Allocation Effect' in summary


class TestStyleFactorAnalyzer:
    """Tests for style factor analysis."""

    def test_initialization(self, sample_prices):
        """Test style analyzer initialization."""
        analyzer = StyleFactorAnalyzer(sample_prices)
        assert analyzer is not None

    def test_calculate_all_factors(self, sample_prices):
        """Test calculating all style factors."""
        analyzer = StyleFactorAnalyzer(sample_prices)
        factors = analyzer.calculate_all_factors()

        assert 'Momentum' in factors.columns
        assert 'Low_Volatility' in factors.columns
        assert 'Quality' in factors.columns

    def test_portfolio_factor_exposure(self, sample_prices, sample_weights):
        """Test portfolio factor exposure."""
        analyzer = StyleFactorAnalyzer(sample_prices)
        exposures = analyzer.portfolio_factor_exposure(sample_weights)

        assert isinstance(exposures, dict)
        assert 'Momentum' in exposures

    def test_factor_tilt_analysis(self, sample_prices, sample_weights):
        """Test factor tilt analysis."""
        analyzer = StyleFactorAnalyzer(sample_prices)
        tilts = analyzer.factor_tilt_analysis(sample_weights)

        assert 'Portfolio' in tilts.columns
        assert 'Equal Weight' in tilts.columns
        assert 'Tilt' in tilts.columns

    def test_factor_return_attribution(self, sample_prices, sample_weights):
        """Test factor return attribution."""
        analyzer = StyleFactorAnalyzer(sample_prices)
        attribution = analyzer.factor_return_attribution(sample_weights)

        assert 'total' in attribution

    def test_top_factor_stocks(self, sample_prices):
        """Test getting top stocks by factor."""
        analyzer = StyleFactorAnalyzer(sample_prices)
        top = analyzer.top_factor_stocks('Momentum', n=3)

        assert len(top) == 3

    def test_factor_correlation(self, sample_prices):
        """Test factor correlation."""
        analyzer = StyleFactorAnalyzer(sample_prices)
        corr = analyzer.factor_correlation()

        assert isinstance(corr, pd.DataFrame)


class TestStyleFactorFunctions:
    """Tests for style factor calculation functions."""

    def test_calculate_momentum(self, sample_prices):
        """Test momentum calculation."""
        momentum = calculate_momentum(sample_prices)

        assert isinstance(momentum, pd.Series)
        assert len(momentum) == len(sample_prices.columns)

    def test_calculate_low_volatility_score(self, sample_returns):
        """Test low volatility score calculation."""
        score = calculate_low_volatility_score(sample_returns)

        assert isinstance(score, pd.Series)
        assert len(score) == len(sample_returns.columns)

    def test_calculate_value_score(self):
        """Test value score calculation."""
        pe = pd.Series({'A': 15, 'B': 25, 'C': 10})
        score = calculate_value_score(pe_ratios=pe)

        assert len(score) == 3
        # Lower P/E should have higher score
        assert score['C'] > score['B']

    def test_calculate_quality_score(self, sample_returns):
        """Test quality score calculation."""
        score = calculate_quality_score(returns=sample_returns)

        assert isinstance(score, pd.Series)


class TestFactorRiskDecomposition:
    """Tests for factor risk decomposition."""

    def test_initialization(self, sample_returns):
        """Test decomposition initialization."""
        decomp = FactorRiskDecomposition(sample_returns)
        assert decomp is not None

    def test_decompose_asset_variance(self, sample_returns):
        """Test single asset variance decomposition."""
        decomp = FactorRiskDecomposition(sample_returns)
        result = decomp.decompose_asset_variance('AAPL')

        assert 'total_variance' in result
        assert 'systematic_variance' in result
        assert 'specific_variance' in result
        assert 'systematic_pct' in result

    def test_decompose_portfolio_risk(self, sample_returns, sample_weights):
        """Test portfolio risk decomposition."""
        decomp = FactorRiskDecomposition(sample_returns)
        result = decomp.decompose_portfolio_risk(sample_weights)

        assert 'total_variance' in result
        assert 'systematic_variance' in result
        assert 'specific_variance' in result
        assert 'portfolio_betas' in result
        assert 'factor_contributions' in result

    def test_variance_decomposition_sums(self, sample_returns, sample_weights):
        """Test that variance components sum correctly."""
        decomp = FactorRiskDecomposition(sample_returns)
        result = decomp.decompose_portfolio_risk(sample_weights)

        total = result['systematic_variance'] + result['specific_variance']
        assert abs(total - result['total_variance']) < 1e-6

    def test_factor_correlation_matrix(self, sample_returns):
        """Test factor correlation matrix."""
        decomp = FactorRiskDecomposition(sample_returns)
        corr = decomp.factor_correlation_matrix()

        assert isinstance(corr, pd.DataFrame)

    def test_factor_stress_scenarios(self, sample_returns, sample_weights):
        """Test factor stress scenarios."""
        decomp = FactorRiskDecomposition(sample_returns)
        stress = decomp.factor_stress_scenarios(sample_weights)

        assert isinstance(stress, pd.DataFrame)
        assert 'Portfolio Impact' in stress.columns

    def test_tracking_error_decomposition(self, sample_returns, sample_weights):
        """Test tracking error decomposition."""
        decomp = FactorRiskDecomposition(sample_returns)

        # Create different benchmark weights
        bench_weights = pd.Series({
            'AAPL': 0.20,
            'MSFT': 0.20,
            'GOOGL': 0.20,
            'JPM': 0.20,
            'XOM': 0.20
        })

        result = decomp.tracking_error_decomposition(sample_weights, bench_weights)

        assert 'tracking_error' in result
        assert 'active_betas' in result

    def test_summary(self, sample_returns, sample_weights):
        """Test summary generation."""
        decomp = FactorRiskDecomposition(sample_returns)
        summary = decomp.summary(sample_weights)

        assert isinstance(summary, str)
        assert 'Risk Breakdown' in summary


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_calculate_sector_attribution(self, sample_returns, sample_weights):
        """Test sector attribution convenience function."""
        result = calculate_sector_attribution(sample_returns, sample_weights)

        assert isinstance(result, pd.DataFrame)

    def test_run_brinson_attribution(self, sample_returns, sample_weights):
        """Test Brinson attribution convenience function."""
        result = run_brinson_attribution(sample_returns, sample_weights)

        assert 'detailed' in result
        assert 'summary' in result

    def test_calculate_factor_risk(self, sample_returns, sample_weights):
        """Test factor risk convenience function."""
        result = calculate_factor_risk(sample_returns, sample_weights)

        assert 'total_variance' in result

    def test_get_factor_stress_impact(self, sample_returns, sample_weights):
        """Test stress impact convenience function."""
        scenario = {'Mkt-RF': -0.20, 'SMB': -0.10}
        impact = get_factor_stress_impact(sample_returns, sample_weights, scenario)

        assert isinstance(impact, float)
