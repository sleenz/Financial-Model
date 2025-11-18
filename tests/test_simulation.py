"""Tests for simulation module (Phase 4)."""

import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.simulation.monte_carlo import MonteCarloSimulator, run_monte_carlo
from src.simulation.scenarios import (
    StressTester,
    StressTestScenario,
    HISTORICAL_SCENARIOS,
    CUSTOM_SCENARIOS,
    get_scenario,
    list_scenarios,
)


def generate_sample_returns(n_assets=5, n_periods=252, seed=42):
    """Generate sample return data for testing."""
    np.random.seed(seed)

    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA'][:n_assets]
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')

    returns_data = {}
    for ticker in tickers:
        returns_data[ticker] = np.random.normal(0.0005, 0.02, n_periods)

    return pd.DataFrame(returns_data, index=dates)


class TestMonteCarloSimulator:
    """Tests for MonteCarloSimulator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.returns = generate_sample_returns()
        self.weights = np.array([0.3, 0.25, 0.20, 0.15, 0.10])
        self.simulator = MonteCarloSimulator(
            self.returns, self.weights, initial_value=10000
        )

    def test_initialization(self):
        """Test simulator initialization."""
        assert self.simulator.n_assets == 5
        assert self.simulator.initial_value == 10000
        assert self.simulator.portfolio_mean is not None
        assert self.simulator.portfolio_vol > 0

    def test_simulate_gbm(self):
        """Test GBM simulation."""
        sim_values = self.simulator.simulate_gbm(
            n_simulations=100,
            horizon_days=30,
            random_seed=42
        )

        assert sim_values.shape == (100, 30)
        # All values should be positive
        assert (sim_values > 0).all()

    def test_simulate_bootstrap(self):
        """Test bootstrap simulation."""
        sim_values = self.simulator.simulate_bootstrap(
            n_simulations=100,
            horizon_days=30,
            random_seed=42
        )

        assert sim_values.shape == (100, 30)
        assert (sim_values > 0).all()

    def test_simulate_student_t(self):
        """Test Student-t simulation."""
        sim_values = self.simulator.simulate_student_t(
            n_simulations=100,
            horizon_days=30,
            random_seed=42
        )

        assert sim_values.shape == (100, 30)
        assert (sim_values > 0).all()

    def test_simulate_jump_diffusion(self):
        """Test jump diffusion simulation."""
        sim_values = self.simulator.simulate_jump_diffusion(
            n_simulations=100,
            horizon_days=30,
            random_seed=42
        )

        assert sim_values.shape == (100, 30)
        assert (sim_values > 0).all()

    def test_simulate_method_selector(self):
        """Test simulate method with method selector."""
        methods = ['gbm', 'bootstrap', 'student_t', 'jump_diffusion']

        for method in methods:
            sim_values = self.simulator.simulate(
                n_simulations=50,
                horizon_days=20,
                method=method,
                random_seed=42
            )
            assert sim_values.shape == (50, 20)

    def test_invalid_method(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError):
            self.simulator.simulate(method='invalid')

    def test_analyze_results(self):
        """Test results analysis."""
        sim_values = self.simulator.simulate_gbm(100, 30, random_seed=42)
        analysis = self.simulator.analyze_results(sim_values)

        assert 'mean_final_value' in analysis
        assert 'median_final_value' in analysis
        assert 'prob_loss' in analysis
        assert 'percentiles' in analysis
        assert 'mean_max_drawdown' in analysis

        # Probability should be between 0 and 1
        assert 0 <= analysis['prob_loss'] <= 1

    def test_value_at_risk(self):
        """Test VaR calculation from simulation."""
        sim_values = self.simulator.simulate_gbm(1000, 30, random_seed=42)
        var = self.simulator.value_at_risk(sim_values, 0.95)

        # VaR should be positive (it's the loss amount)
        assert var >= 0

    def test_conditional_var(self):
        """Test CVaR calculation from simulation."""
        sim_values = self.simulator.simulate_gbm(1000, 30, random_seed=42)
        cvar = self.simulator.conditional_var(sim_values, 0.95)
        var = self.simulator.value_at_risk(sim_values, 0.95)

        # CVaR should be >= VaR
        assert cvar >= var

    def test_probability_of_ruin(self):
        """Test probability of ruin calculation."""
        sim_values = self.simulator.simulate_gbm(1000, 252, random_seed=42)
        prob_ruin = self.simulator.probability_of_ruin(sim_values, ruin_threshold=0.5)

        assert 0 <= prob_ruin <= 1

    def test_time_to_target(self):
        """Test time to target calculation."""
        sim_values = self.simulator.simulate_gbm(1000, 252, random_seed=42)
        result = self.simulator.time_to_target(sim_values, target_return=0.10)

        assert 'probability_reaching' in result
        assert 0 <= result['probability_reaching'] <= 1

    def test_distribution_plot_data(self):
        """Test distribution plot data generation."""
        sim_values = self.simulator.simulate_gbm(1000, 30, random_seed=42)
        plot_data = self.simulator.generate_distribution_plot_data(sim_values)

        assert 'final_values' in plot_data
        assert 'final_returns' in plot_data
        assert 'value_histogram' in plot_data

    def test_multi_horizon(self):
        """Test multi-horizon simulation."""
        results = self.simulator.run_multi_horizon(
            horizons=[21, 63],
            n_simulations=100,
            method='gbm'
        )

        assert len(results) == 2
        assert 'Horizon (days)' in results.columns
        assert 'Mean Return' in results.columns


class TestStressTester:
    """Tests for StressTester class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.returns = generate_sample_returns()
        self.weights = np.array([0.3, 0.25, 0.20, 0.15, 0.10])
        self.tester = StressTester(self.returns, self.weights, portfolio_value=10000)

    def test_initialization(self):
        """Test stress tester initialization."""
        assert self.tester.n_assets == 5
        assert self.tester.portfolio_value == 10000

    def test_historical_scenario(self):
        """Test historical scenario application."""
        result = self.tester.historical_scenario('2008_financial_crisis')

        assert 'scenario' in result
        assert 'portfolio_return' in result
        assert 'portfolio_loss' in result
        assert 'ending_value' in result

        # Should show a loss for crisis scenario
        assert result['portfolio_return'] < 0

    def test_all_historical_scenarios(self):
        """Test that all historical scenarios run."""
        for scenario_key in HISTORICAL_SCENARIOS.keys():
            result = self.tester.historical_scenario(scenario_key)
            assert 'portfolio_return' in result

    def test_custom_scenario(self):
        """Test custom scenario application."""
        scenario = StressTestScenario(
            name='Test Scenario',
            shocks={'equity': -0.20},
            description='Test'
        )

        result = self.tester.custom_scenario(scenario)

        assert abs(result['portfolio_return'] - (-0.20)) < 1e-10
        assert abs(result['portfolio_loss'] - (-2000)) < 1e-10

    def test_custom_scenario_with_mapping(self):
        """Test custom scenario with asset mapping."""
        scenario = StressTestScenario(
            name='Sector Shock',
            shocks={'tech': -0.30, 'finance': -0.10}
        )

        mapping = {
            'AAPL': 'tech',
            'MSFT': 'tech',
            'GOOGL': 'tech',
            'AMZN': 'tech',
            'NVDA': 'tech',
        }

        result = self.tester.custom_scenario(scenario, mapping)
        assert result['portfolio_return'] == -0.30

    def test_run_all_historical(self):
        """Test running all historical scenarios."""
        results = self.tester.run_all_historical()

        assert len(results) > 0
        assert 'Scenario' in results.columns
        assert 'Portfolio Return' in results.columns

    def test_parametric_stress(self):
        """Test parametric stress test."""
        result = self.tester.parametric_stress(
            equity_shock=-0.25,
            volatility_multiplier=2.0,
            correlation_adjustment=0.90
        )

        assert result['portfolio_return'] == -0.25
        assert result['stressed_volatility'] > 0

    def test_sensitivity_analysis(self):
        """Test sensitivity analysis."""
        sensitivity = self.tester.sensitivity_analysis()

        assert len(sensitivity) == 8  # Default 8 shock levels
        assert 'Shock' in sensitivity.columns
        assert 'Ending Value' in sensitivity.columns

    def test_reverse_stress_test(self):
        """Test reverse stress testing."""
        result = self.tester.reverse_stress_test(target_loss=-5000)

        assert 'required_market_shock' in result
        # Should need significant shock to lose $5000
        assert result['required_market_shock'] < 0


class TestStressTestScenario:
    """Tests for StressTestScenario class."""

    def test_scenario_creation(self):
        """Test scenario creation."""
        scenario = StressTestScenario(
            name='Test',
            shocks={'equity': -0.20},
            description='Test scenario'
        )

        assert scenario.name == 'Test'
        assert scenario.shocks['equity'] == -0.20

    def test_to_dict(self):
        """Test scenario serialization."""
        scenario = StressTestScenario(
            name='Test',
            shocks={'equity': -0.20}
        )

        d = scenario.to_dict()
        assert d['name'] == 'Test'
        assert d['shocks']['equity'] == -0.20


class TestScenarioHelpers:
    """Tests for scenario helper functions."""

    def test_get_scenario(self):
        """Test getting predefined scenario."""
        scenario = get_scenario('market_crash')
        assert scenario.name == 'Market Crash'

    def test_get_invalid_scenario(self):
        """Test getting invalid scenario raises error."""
        with pytest.raises(ValueError):
            get_scenario('invalid_scenario')

    def test_list_scenarios(self):
        """Test listing all scenarios."""
        scenarios = list_scenarios()

        assert len(scenarios) > 0
        # Should have both historical and custom
        assert any('[Historical]' in v for v in scenarios.values())
        assert any('[Custom]' in v for v in scenarios.values())

    def test_historical_scenarios_dict(self):
        """Test HISTORICAL_SCENARIOS structure."""
        for key, scenario in HISTORICAL_SCENARIOS.items():
            assert 'name' in scenario
            assert 'start_date' in scenario
            assert 'end_date' in scenario
            assert 'characteristics' in scenario

    def test_custom_scenarios_dict(self):
        """Test CUSTOM_SCENARIOS structure."""
        for key, scenario in CUSTOM_SCENARIOS.items():
            assert isinstance(scenario, StressTestScenario)
            assert scenario.name is not None
            assert scenario.shocks is not None


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_run_monte_carlo(self):
        """Test run_monte_carlo convenience function."""
        returns = generate_sample_returns()
        result = run_monte_carlo(
            returns,
            n_simulations=100,
            horizon_days=30,
            initial_value=10000
        )

        assert 'simulated_values' in result
        assert 'analysis' in result
        assert 'var_95' in result
        assert 'cvar_95' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
