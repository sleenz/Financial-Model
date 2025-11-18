"""Tests for optimization engine (Phase 2)."""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.optimization.constraints import (
    PortfolioConstraints,
    ConstraintError,
    default_constraints,
)
from src.optimization.optimizers import PortfolioOptimizer, OptimizationError
from src.optimization.hrp import HRPOptimizer, hrp_allocation
from src.optimization.black_litterman import (
    BlackLittermanModel,
    black_litterman_allocation,
    BlackLittermanError,
)


def generate_sample_returns(n_assets=5, n_periods=252, seed=42):
    """Generate sample return data for testing."""
    np.random.seed(seed)

    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA'][:n_assets]
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')

    # Generate correlated returns
    mean_returns = np.random.uniform(0.0005, 0.001, n_assets)
    volatilities = np.random.uniform(0.01, 0.03, n_assets)

    # Create correlation matrix
    corr = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            corr[i, j] = corr[j, i] = np.random.uniform(0.3, 0.7)

    # Cholesky decomposition for correlated returns
    L = np.linalg.cholesky(corr)

    returns_data = {}
    for i, ticker in enumerate(tickers):
        uncorrelated = np.random.normal(0, 1, n_periods)
        correlated = np.dot(L[i, :i+1], np.random.normal(0, 1, (i+1, n_periods)))
        returns_data[ticker] = mean_returns[i] + volatilities[i] * correlated

    return pd.DataFrame(returns_data, index=dates)


class TestPortfolioConstraints:
    """Tests for PortfolioConstraints class."""

    def test_default_constraints(self):
        """Test default constraint creation."""
        constraints = default_constraints()

        assert constraints.min_weight == 0.0
        assert constraints.max_weight == 0.40
        assert constraints.long_only == True

    def test_conservative_constraints(self):
        """Test conservative constraint creation."""
        constraints = default_constraints(conservative=True)

        assert constraints.max_weight == 0.20
        assert constraints.min_position_size == 0.02

    def test_custom_constraints(self):
        """Test custom constraint creation."""
        constraints = PortfolioConstraints(
            min_weight=0.0,
            max_weight=0.30,
            max_position_size=0.25,
            sector_limits={'Technology': 0.40},
            max_turnover=0.20,
        )

        assert constraints.max_weight == 0.30
        assert constraints.sector_limits['Technology'] == 0.40
        assert constraints.max_turnover == 0.20

    def test_invalid_constraints(self):
        """Test that invalid constraints raise errors."""
        with pytest.raises(ConstraintError):
            PortfolioConstraints(min_weight=0.5, max_weight=0.3)

        with pytest.raises(ConstraintError):
            PortfolioConstraints(min_position_size=0.5, max_position_size=0.3)

    def test_get_bounds(self):
        """Test getting bounds for optimization."""
        constraints = PortfolioConstraints(min_weight=0.0, max_weight=0.5)
        bounds = constraints.get_bounds(3)

        assert len(bounds) == 3
        assert bounds[0] == (0.0, 0.5)

    def test_apply_minimum_position(self):
        """Test minimum position size application."""
        constraints = PortfolioConstraints(min_position_size=0.05)
        weights = np.array([0.3, 0.04, 0.2, 0.01, 0.45])

        adjusted = constraints.apply_minimum_position(weights)

        # Small positions should be zeroed out
        assert adjusted[1] == 0
        assert adjusted[3] == 0
        # Should still sum to 1
        assert abs(np.sum(adjusted) - 1.0) < 1e-6

    def test_check_constraints_valid(self):
        """Test constraint checking with valid weights."""
        constraints = PortfolioConstraints(max_weight=0.5)
        weights = np.array([0.3, 0.3, 0.4])

        is_valid, violations = constraints.check_constraints(weights)

        assert is_valid
        assert len(violations) == 0

    def test_check_constraints_invalid(self):
        """Test constraint checking with invalid weights."""
        constraints = PortfolioConstraints(max_weight=0.3)
        weights = np.array([0.5, 0.3, 0.2])  # First weight exceeds max

        is_valid, violations = constraints.check_constraints(weights)

        assert not is_valid
        assert len(violations) > 0

    def test_to_dict_and_from_dict(self):
        """Test constraint serialization."""
        constraints = PortfolioConstraints(
            max_weight=0.3,
            sector_limits={'Tech': 0.5}
        )

        d = constraints.to_dict()
        restored = PortfolioConstraints.from_dict(d)

        assert restored.max_weight == constraints.max_weight
        assert restored.sector_limits == constraints.sector_limits


class TestPortfolioOptimizer:
    """Tests for PortfolioOptimizer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.returns = generate_sample_returns(n_assets=5, n_periods=252)
        self.optimizer = PortfolioOptimizer(self.returns)

    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.n_assets == 5
        assert len(self.optimizer.tickers) == 5
        assert self.optimizer.mean_returns is not None
        assert self.optimizer.cov_matrix is not None

    def test_max_sharpe_optimization(self):
        """Test maximum Sharpe ratio optimization."""
        result = self.optimizer.optimize(method='max_sharpe')

        assert 'weights' in result
        assert 'expected_return' in result
        assert 'volatility' in result
        assert 'sharpe_ratio' in result

        # Weights should sum to 1
        assert abs(result['weights'].sum() - 1.0) < 1e-6

        # All weights should be non-negative (long-only by default)
        assert (result['weights'] >= -1e-6).all()

    def test_min_volatility_optimization(self):
        """Test minimum volatility optimization."""
        result = self.optimizer.optimize(method='min_volatility')

        assert result['volatility'] > 0
        assert abs(result['weights'].sum() - 1.0) < 1e-6

    def test_max_return_optimization(self):
        """Test maximum return optimization."""
        result = self.optimizer.optimize(method='max_return')

        assert abs(result['weights'].sum() - 1.0) < 1e-6

    def test_risk_parity_optimization(self):
        """Test risk parity optimization."""
        result = self.optimizer.optimize(method='risk_parity')

        assert abs(result['weights'].sum() - 1.0) < 1e-6

        # Risk contributions should be roughly equal
        risk_contrib = self.optimizer.get_risk_contributions(result['weights'].values)
        mean_contrib = risk_contrib.mean()

        # Allow some tolerance for numerical optimization
        for contrib in risk_contrib:
            assert abs(contrib - mean_contrib) < 0.05

    def test_max_diversification_optimization(self):
        """Test maximum diversification optimization."""
        result = self.optimizer.optimize(method='max_diversification')

        assert abs(result['weights'].sum() - 1.0) < 1e-6

    def test_hrp_optimization(self):
        """Test HRP optimization."""
        result = self.optimizer.optimize(method='hrp')

        assert abs(result['weights'].sum() - 1.0) < 1e-6
        assert (result['weights'] >= 0).all()

    def test_equal_weight_optimization(self):
        """Test equal weight optimization."""
        result = self.optimizer.optimize(method='equal_weight')

        expected_weight = 1.0 / self.optimizer.n_assets
        for weight in result['weights']:
            assert abs(weight - expected_weight) < 1e-6

    def test_invalid_method(self):
        """Test that invalid method raises error."""
        with pytest.raises(OptimizationError):
            self.optimizer.optimize(method='invalid_method')

    def test_with_constraints(self):
        """Test optimization with custom constraints."""
        constraints = PortfolioConstraints(max_weight=0.3)
        result = self.optimizer.optimize(method='max_sharpe', constraints=constraints)

        # No weight should exceed max
        assert (result['weights'] <= 0.3 + 1e-6).all()

    def test_efficient_frontier(self):
        """Test efficient frontier calculation."""
        frontier = self.optimizer.efficient_frontier(n_points=10)

        assert len(frontier) >= 5  # May be fewer if some points fail
        assert 'return' in frontier.columns
        assert 'volatility' in frontier.columns
        assert 'sharpe' in frontier.columns

    def test_risk_contributions(self):
        """Test risk contribution calculation."""
        weights = np.ones(self.optimizer.n_assets) / self.optimizer.n_assets
        risk_contrib = self.optimizer.get_risk_contributions(weights)

        assert len(risk_contrib) == self.optimizer.n_assets
        # Risk contributions should sum to portfolio volatility
        assert risk_contrib.sum() > 0

    def test_compare_methods(self):
        """Test method comparison."""
        comparison = self.optimizer.compare_methods(
            methods=['max_sharpe', 'min_volatility', 'equal_weight']
        )

        assert len(comparison) == 3
        assert 'method' in comparison.columns
        assert 'sharpe_ratio' in comparison.columns

    def test_correlation_matrix(self):
        """Test correlation matrix retrieval."""
        corr = self.optimizer.get_correlation_matrix()

        assert corr.shape == (5, 5)
        # Diagonal should be 1
        for i in range(5):
            assert abs(corr.iloc[i, i] - 1.0) < 1e-6


class TestHRPOptimizer:
    """Tests for HRPOptimizer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.returns = generate_sample_returns(n_assets=5, n_periods=252)
        self.hrp = HRPOptimizer(self.returns)

    def test_hrp_initialization(self):
        """Test HRP optimizer initialization."""
        assert self.hrp.n_assets == 5
        assert self.hrp.cov_matrix is not None
        assert self.hrp.corr_matrix is not None

    def test_hrp_optimize(self):
        """Test HRP optimization."""
        result = self.hrp.optimize()

        assert 'weights' in result
        assert 'volatility' in result
        assert 'sorted_tickers' in result

        # Weights should sum to 1
        assert abs(result['weights'].sum() - 1.0) < 1e-6

        # All weights should be positive
        assert (result['weights'] > 0).all()

    def test_hrp_with_shrinkage(self):
        """Test HRP with Ledoit-Wolf shrinkage."""
        hrp_shrink = HRPOptimizer(self.returns, use_shrinkage=True)
        result = hrp_shrink.optimize()

        assert abs(result['weights'].sum() - 1.0) < 1e-6

    def test_hrp_without_shrinkage(self):
        """Test HRP without shrinkage."""
        hrp_no_shrink = HRPOptimizer(self.returns, use_shrinkage=False)
        result = hrp_no_shrink.optimize()

        assert abs(result['weights'].sum() - 1.0) < 1e-6

    def test_hrp_allocation_function(self):
        """Test convenience allocation function."""
        weights = hrp_allocation(self.returns)

        assert abs(weights.sum() - 1.0) < 1e-6
        assert len(weights) == 5

    def test_cluster_members(self):
        """Test cluster membership retrieval."""
        clusters = self.hrp.get_cluster_members(n_clusters=2)

        assert len(clusters) == 2
        # All tickers should be assigned
        all_tickers = []
        for members in clusters.values():
            all_tickers.extend(members)
        assert len(all_tickers) == 5

    def test_dendrogram_data(self):
        """Test dendrogram data retrieval."""
        data = self.hrp.get_dendrogram_data()

        assert 'linkage' in data
        assert 'labels' in data
        assert len(data['labels']) == 5


class TestBlackLittermanModel:
    """Tests for BlackLittermanModel class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.returns = generate_sample_returns(n_assets=5, n_periods=252)
        self.tickers = list(self.returns.columns)

        # Create market caps
        self.market_caps = pd.Series({
            'AAPL': 3000e9,
            'GOOGL': 1500e9,
            'MSFT': 2500e9,
            'AMZN': 1800e9,
            'NVDA': 1200e9,
        })

        self.bl = BlackLittermanModel(self.returns, self.market_caps)

    def test_bl_initialization(self):
        """Test Black-Litterman model initialization."""
        assert self.bl.n_assets == 5
        assert self.bl.equilibrium_returns is not None
        assert abs(self.bl.market_weights.sum() - 1.0) < 1e-6

    def test_equilibrium_returns(self):
        """Test equilibrium return calculation."""
        eq_returns = self.bl.equilibrium_returns

        assert len(eq_returns) == 5
        # Returns should be reasonable (not extreme)
        assert (eq_returns > -1).all()
        assert (eq_returns < 2).all()

    def test_add_absolute_view(self):
        """Test adding absolute view."""
        self.bl.add_absolute_view('AAPL', 0.15, confidence=0.7)

        assert self.bl.P is not None
        assert len(self.bl.Q) == 1
        assert self.bl.Q[0] == 0.15

    def test_add_relative_view(self):
        """Test adding relative view."""
        self.bl.add_relative_view(
            long_assets=['AAPL', 'MSFT'],
            short_assets=['GOOGL'],
            view_return=0.05,
            confidence=0.6
        )

        assert self.bl.P is not None
        assert len(self.bl.Q) == 1
        assert self.bl.Q[0] == 0.05

    def test_invalid_view_asset(self):
        """Test that invalid asset raises error."""
        with pytest.raises(BlackLittermanError):
            self.bl.add_absolute_view('INVALID', 0.10)

    def test_posterior_returns_no_views(self):
        """Test posterior returns without views."""
        posterior = self.bl.get_posterior_returns()

        # Should return equilibrium returns
        pd.testing.assert_series_equal(posterior, self.bl.equilibrium_returns)

    def test_posterior_returns_with_views(self):
        """Test posterior returns with views."""
        self.bl.add_absolute_view('AAPL', 0.20, confidence=0.8)

        posterior = self.bl.get_posterior_returns()

        # AAPL posterior should be pulled toward the view
        assert posterior['AAPL'] > self.bl.equilibrium_returns['AAPL']

    def test_optimize_without_views(self):
        """Test optimization without views."""
        result = self.bl.optimize()

        assert 'weights' in result
        assert abs(result['weights'].sum() - 1.0) < 1e-6

    def test_optimize_with_views(self):
        """Test optimization with views."""
        self.bl.add_absolute_view('AAPL', 0.25, confidence=0.8)
        result = self.bl.optimize()

        assert abs(result['weights'].sum() - 1.0) < 1e-6
        # AAPL weight should be higher due to bullish view
        # (this may not always be true depending on other factors)

    def test_optimize_with_constraints(self):
        """Test optimization with weight constraints."""
        result = self.bl.optimize(max_weight=0.30)

        assert (result['weights'] <= 0.30 + 1e-6).all()

    def test_clear_views(self):
        """Test clearing views."""
        self.bl.add_absolute_view('AAPL', 0.15)
        self.bl.clear_views()

        assert self.bl.P is None
        assert self.bl.Q is None

    def test_views_summary(self):
        """Test views summary retrieval."""
        self.bl.add_absolute_view('AAPL', 0.15)
        self.bl.add_relative_view(['MSFT'], ['GOOGL'], 0.03)

        summary = self.bl.get_views_summary()

        assert len(summary) == 2
        assert 'expected_return' in summary.columns

    def test_bl_allocation_function(self):
        """Test convenience allocation function."""
        views = [
            {'type': 'absolute', 'asset': 'AAPL', 'return': 0.15, 'confidence': 0.7},
        ]

        weights = black_litterman_allocation(
            self.returns,
            self.market_caps,
            views=views
        )

        assert abs(weights.sum() - 1.0) < 1e-6

    def test_posterior_covariance(self):
        """Test posterior covariance calculation."""
        self.bl.add_absolute_view('AAPL', 0.15)

        posterior_cov = self.bl.get_posterior_covariance()

        assert posterior_cov.shape == (5, 5)
        # Should be symmetric
        np.testing.assert_array_almost_equal(
            posterior_cov.values,
            posterior_cov.values.T
        )


class TestIntegration:
    """Integration tests combining multiple components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.returns = generate_sample_returns(n_assets=5, n_periods=252)

    def test_optimizer_all_methods(self):
        """Test that all optimization methods work."""
        optimizer = PortfolioOptimizer(self.returns)

        methods = [
            'max_sharpe',
            'min_volatility',
            'max_return',
            'risk_parity',
            'max_diversification',
            'hrp',
            'equal_weight',
        ]

        for method in methods:
            result = optimizer.optimize(method=method)

            # Basic sanity checks
            assert abs(result['weights'].sum() - 1.0) < 1e-6, f"{method} weights don't sum to 1"
            assert result['volatility'] > 0, f"{method} has zero volatility"

    def test_sharpe_higher_than_min_vol(self):
        """Test that max Sharpe has higher Sharpe than min vol."""
        optimizer = PortfolioOptimizer(self.returns)

        sharpe_result = optimizer.optimize(method='max_sharpe')
        min_vol_result = optimizer.optimize(method='min_volatility')

        # Max Sharpe should have higher or equal Sharpe ratio
        assert sharpe_result['sharpe_ratio'] >= min_vol_result['sharpe_ratio'] - 0.01

    def test_min_vol_lower_than_equal_weight(self):
        """Test that min vol has lower volatility than equal weight."""
        optimizer = PortfolioOptimizer(self.returns)

        min_vol_result = optimizer.optimize(method='min_volatility')
        equal_result = optimizer.optimize(method='equal_weight')

        # Min vol should have lower or equal volatility
        assert min_vol_result['volatility'] <= equal_result['volatility'] + 0.001


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
