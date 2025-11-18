"""Tests for portfolio module (Phase 5)."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.portfolio.calculator import (
    PositionCalculator,
    TaxLotOptimizer,
    calculate_positions,
    generate_allocation_report,
)
from src.portfolio.rebalancer import (
    PortfolioRebalancer,
    DCAScheduler,
    PerformanceAttributor,
    check_rebalancing,
    calculate_turnover,
)


class TestPositionCalculator:
    """Tests for PositionCalculator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.weights = pd.Series({
            'AAPL': 0.30,
            'GOOGL': 0.25,
            'MSFT': 0.25,
            'AMZN': 0.20,
        })
        self.prices = pd.Series({
            'AAPL': 150.00,
            'GOOGL': 140.00,
            'MSFT': 380.00,
            'AMZN': 175.00,
        })
        self.capital = 10000

    def test_initialization(self):
        """Test calculator initialization."""
        calc = PositionCalculator(self.capital, self.weights, self.prices)
        assert calc.total_capital == 10000
        assert len(calc.weights) == 4

    def test_weights_normalization(self):
        """Test that unnormalized weights get normalized."""
        bad_weights = pd.Series({
            'AAPL': 0.40,
            'GOOGL': 0.30,
            'MSFT': 0.20,
            'AMZN': 0.20,
        })  # Sums to 1.1

        calc = PositionCalculator(self.capital, bad_weights, self.prices)
        assert abs(calc.weights.sum() - 1.0) < 0.01

    def test_calculate_positions(self):
        """Test position calculation."""
        calc = PositionCalculator(self.capital, self.weights, self.prices)
        positions = calc.calculate_positions()

        assert len(positions) == 4
        assert 'Weight' in positions.columns
        assert 'Shares' in positions.columns
        assert 'Actual Amount' in positions.columns

    def test_whole_shares(self):
        """Test that shares are whole numbers when not fractional."""
        calc = PositionCalculator(
            self.capital, self.weights, self.prices, allow_fractional=False
        )
        positions = calc.calculate_positions()

        # All shares should be integers
        for shares in positions['Shares']:
            assert shares == int(shares)

    def test_fractional_shares(self):
        """Test fractional share calculation."""
        calc = PositionCalculator(
            self.capital, self.weights, self.prices, allow_fractional=True
        )
        positions = calc.calculate_positions()

        # AAPL: $3000 / $150 = 20 shares exactly
        assert abs(positions.loc['AAPL', 'Shares'] - 20.0) < 0.01

    def test_get_summary(self):
        """Test summary statistics."""
        calc = PositionCalculator(self.capital, self.weights, self.prices)
        summary = calc.get_summary()

        assert 'total_capital' in summary
        assert 'total_invested' in summary
        assert 'unallocated_cash' in summary
        assert summary['total_capital'] == 10000

    def test_unallocated_cash(self):
        """Test unallocated cash calculation."""
        calc = PositionCalculator(
            self.capital, self.weights, self.prices, allow_fractional=False
        )
        summary = calc.get_summary()

        # Unallocated should be positive (can't buy partial shares)
        assert summary['unallocated_cash'] >= 0
        assert summary['unallocated_cash'] < self.capital

    def test_format_report(self):
        """Test report formatting."""
        calc = PositionCalculator(self.capital, self.weights, self.prices)
        report = calc.format_report(
            expected_return=0.10,
            expected_volatility=0.20,
            sharpe_ratio=0.50
        )

        assert 'Portfolio Allocation Summary' in report
        assert '$10,000' in report
        assert 'AAPL' in report

    def test_optimize_for_minimum_remainder(self):
        """Test remainder optimization."""
        calc = PositionCalculator(
            self.capital, self.weights, self.prices, allow_fractional=False
        )

        original = calc.calculate_positions()
        optimized = calc.optimize_for_minimum_remainder()

        original_invested = original['Actual Amount'].sum()
        optimized_invested = optimized['Actual Amount'].sum()

        # Optimized should invest more (less remainder)
        assert optimized_invested >= original_invested

    def test_commission_calculation(self):
        """Test commission handling."""
        calc = PositionCalculator(
            self.capital,
            self.weights,
            self.prices,
            commission_per_trade=5.0
        )
        positions = calc.calculate_positions()

        # Each position with shares should have commission
        for ticker in positions.index:
            if positions.loc[ticker, 'Shares'] > 0:
                assert positions.loc[ticker, 'Commission'] == 5.0

    def test_missing_prices_error(self):
        """Test that missing prices raise error."""
        bad_prices = pd.Series({'AAPL': 150.00})  # Missing others

        with pytest.raises(ValueError):
            PositionCalculator(self.capital, self.weights, bad_prices)

    def test_zero_weight_handling(self):
        """Test handling of zero weights."""
        weights_with_zero = pd.Series({
            'AAPL': 0.50,
            'GOOGL': 0.50,
            'MSFT': 0.0,
            'AMZN': 0.0,
        })

        calc = PositionCalculator(self.capital, weights_with_zero, self.prices)
        positions = calc.calculate_positions()

        assert positions.loc['MSFT', 'Shares'] == 0
        assert positions.loc['AMZN', 'Shares'] == 0


class TestTaxLotOptimizer:
    """Tests for TaxLotOptimizer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.lots = pd.DataFrame([
            {'ticker': 'AAPL', 'purchase_date': '2020-01-01', 'shares': 10, 'cost_basis': 100},
            {'ticker': 'AAPL', 'purchase_date': '2021-01-01', 'shares': 15, 'cost_basis': 130},
            {'ticker': 'AAPL', 'purchase_date': '2022-01-01', 'shares': 20, 'cost_basis': 160},
        ])
        self.lots['purchase_date'] = pd.to_datetime(self.lots['purchase_date'])
        self.optimizer = TaxLotOptimizer(self.lots)

    def test_fifo(self):
        """Test FIFO lot selection."""
        selected = self.optimizer.fifo('AAPL', 15)

        # Should select oldest first
        assert len(selected) == 2
        assert selected.iloc[0]['cost_basis'] == 100

    def test_lifo(self):
        """Test LIFO lot selection."""
        selected = self.optimizer.lifo('AAPL', 15)

        # Should select newest first
        assert len(selected) == 1
        assert selected.iloc[0]['cost_basis'] == 160

    def test_highest_cost(self):
        """Test highest cost lot selection."""
        selected = self.optimizer.highest_cost('AAPL', 15)

        # Should select highest cost first
        assert selected.iloc[0]['cost_basis'] == 160

    def test_lowest_cost(self):
        """Test lowest cost lot selection."""
        selected = self.optimizer.lowest_cost('AAPL', 15)

        # Should select lowest cost first
        assert selected.iloc[0]['cost_basis'] == 100

    def test_tax_loss_harvest(self):
        """Test tax-loss harvesting selection."""
        current_price = 120  # Below some cost bases

        selected = self.optimizer.tax_loss_harvest('AAPL', current_price)

        # Should only select lots with losses
        for _, lot in selected.iterrows():
            assert lot['cost_basis'] > current_price


class TestPortfolioRebalancer:
    """Tests for PortfolioRebalancer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.target_weights = pd.Series({
            'AAPL': 0.30,
            'GOOGL': 0.30,
            'MSFT': 0.40,
        })
        self.current_weights = pd.Series({
            'AAPL': 0.35,  # Over by 5%
            'GOOGL': 0.25,  # Under by 5%
            'MSFT': 0.40,  # On target
        })
        self.prices = pd.Series({
            'AAPL': 150.00,
            'GOOGL': 140.00,
            'MSFT': 380.00,
        })

    def test_needs_rebalancing_true(self):
        """Test rebalancing detection when needed."""
        rebalancer = PortfolioRebalancer(
            self.target_weights,
            self.current_weights,
            self.prices,
            portfolio_value=10000,
            threshold_pct=0.03  # 3% threshold
        )

        needs_rebal, details = rebalancer.needs_rebalancing()

        assert needs_rebal
        assert abs(details['max_deviation'] - 0.05) < 1e-10

    def test_needs_rebalancing_false(self):
        """Test rebalancing detection when not needed."""
        rebalancer = PortfolioRebalancer(
            self.target_weights,
            self.current_weights,
            self.prices,
            portfolio_value=10000,
            threshold_pct=0.10  # 10% threshold
        )

        needs_rebal, details = rebalancer.needs_rebalancing()

        assert not needs_rebal

    def test_calculate_trades(self):
        """Test trade calculation."""
        rebalancer = PortfolioRebalancer(
            self.target_weights,
            self.current_weights,
            self.prices,
            portfolio_value=10000,
            threshold_pct=0.03
        )

        trades = rebalancer.calculate_trades()

        assert len(trades) == 3
        assert 'Action' in trades.columns
        assert 'Trade Value' in trades.columns

        # AAPL should be sold
        assert trades.loc['AAPL', 'Action'] == 'SELL'
        # GOOGL should be bought
        assert trades.loc['GOOGL', 'Action'] == 'BUY'

    def test_get_trade_summary(self):
        """Test trade summary generation."""
        rebalancer = PortfolioRebalancer(
            self.target_weights,
            self.current_weights,
            self.prices,
            portfolio_value=10000,
            threshold_pct=0.03
        )

        summary = rebalancer.get_trade_summary()

        assert 'n_buys' in summary
        assert 'n_sells' in summary
        assert 'total_turnover' in summary
        assert summary['n_buys'] == 1
        assert summary['n_sells'] == 1

    def test_format_recommendations(self):
        """Test recommendations formatting."""
        rebalancer = PortfolioRebalancer(
            self.target_weights,
            self.current_weights,
            self.prices,
            portfolio_value=10000,
            threshold_pct=0.03
        )

        recommendations = rebalancer.format_recommendations()

        assert 'Rebalancing Recommendations' in recommendations
        assert 'BUY' in recommendations or 'SELL' in recommendations


class TestDCAScheduler:
    """Tests for DCAScheduler class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.weights = pd.Series({
            'AAPL': 0.50,
            'GOOGL': 0.50,
        })

    def test_initialization(self):
        """Test scheduler initialization."""
        scheduler = DCAScheduler(12000, self.weights, 'monthly', 12)

        assert scheduler.amount_per_period == 1000

    def test_generate_schedule(self):
        """Test schedule generation."""
        scheduler = DCAScheduler(12000, self.weights, 'monthly', 12)
        schedule = scheduler.generate_schedule()

        assert len(schedule) == 12
        assert 'Period' in schedule.columns
        assert 'Total Amount' in schedule.columns
        assert 'AAPL' in schedule.columns

        # Each period should have $1000 total
        assert (schedule['Total Amount'] == 1000).all()

    def test_get_summary(self):
        """Test summary generation."""
        scheduler = DCAScheduler(12000, self.weights, 'monthly', 12)
        summary = scheduler.get_summary()

        assert summary['total_amount'] == 12000
        assert summary['n_periods'] == 12
        assert summary['frequency'] == 'monthly'


class TestPerformanceAttributor:
    """Tests for PerformanceAttributor class."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n_periods = 100

        self.returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, n_periods),
            'GOOGL': np.random.normal(0.0005, 0.025, n_periods),
            'MSFT': np.random.normal(0.0008, 0.018, n_periods),
        })

        self.weights = pd.Series({
            'AAPL': 0.40,
            'GOOGL': 0.30,
            'MSFT': 0.30,
        })

        self.portfolio_returns = (self.returns * self.weights).sum(axis=1)

    def test_contribution_analysis(self):
        """Test return contribution analysis."""
        attributor = PerformanceAttributor(
            self.portfolio_returns,
            self.returns,
            self.weights
        )

        contribution = attributor.contribution_analysis()

        assert len(contribution) == 3
        assert 'Weight' in contribution.columns
        assert 'Contribution' in contribution.columns

        # Contributions should sum to total return
        total_contrib = contribution['Contribution'].sum()
        total_return = self.returns.mean().dot(self.weights)
        assert abs(total_contrib - total_return) < 1e-10

    def test_risk_contribution(self):
        """Test risk contribution analysis."""
        attributor = PerformanceAttributor(
            self.portfolio_returns,
            self.returns,
            self.weights
        )

        risk_contrib = attributor.risk_contribution()

        assert len(risk_contrib) == 3
        assert 'Risk Contribution' in risk_contrib.columns
        assert 'Risk Contribution %' in risk_contrib.columns

        # Percentages should sum to 100%
        total_pct = risk_contrib['Risk Contribution %'].sum()
        assert abs(total_pct - 100) < 0.01


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_check_rebalancing(self):
        """Test check_rebalancing function."""
        target = pd.Series({'A': 0.5, 'B': 0.5})
        current = pd.Series({'A': 0.6, 'B': 0.4})

        needs_rebal, max_dev = check_rebalancing(target, current, 0.05)

        assert needs_rebal
        assert abs(max_dev - 0.1) < 1e-10

    def test_calculate_turnover(self):
        """Test turnover calculation."""
        old_weights = pd.Series({'A': 0.5, 'B': 0.5})
        new_weights = pd.Series({'A': 0.7, 'B': 0.3})

        turnover = calculate_turnover(old_weights, new_weights)

        # Turnover = |0.2 + 0.2| / 2 = 0.2
        assert abs(turnover - 0.2) < 1e-10

    def test_calculate_positions(self):
        """Test calculate_positions convenience function."""
        weights = pd.Series({'AAPL': 0.5, 'GOOGL': 0.5})
        prices = pd.Series({'AAPL': 150, 'GOOGL': 140})

        positions = calculate_positions(10000, weights, prices)

        assert len(positions) == 2

    def test_generate_allocation_report(self):
        """Test allocation report generation."""
        weights = pd.Series({'AAPL': 0.5, 'GOOGL': 0.5})
        prices = pd.Series({'AAPL': 150, 'GOOGL': 140})

        report = generate_allocation_report(10000, weights, prices)

        assert 'Portfolio Allocation' in report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
