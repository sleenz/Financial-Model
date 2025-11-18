"""Monte Carlo simulation engine for portfolio analysis."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cholesky

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for portfolio returns.

    Supports multiple distribution models:
    - Geometric Brownian Motion (GBM)
    - Historical Bootstrap
    - Student's t distribution
    - Jump Diffusion
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray = None,
        initial_value: float = 10000,
        frequency: int = 252,
    ):
        """
        Initialize Monte Carlo simulator.

        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights
            initial_value: Initial portfolio value
            frequency: Trading periods per year
        """
        self.returns = returns
        self.tickers = list(returns.columns)
        self.n_assets = len(self.tickers)
        self.weights = weights if weights is not None else np.ones(self.n_assets) / self.n_assets
        self.initial_value = initial_value
        self.frequency = frequency

        # Calculate statistics
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.corr_matrix = returns.corr()

        # Portfolio statistics
        self.portfolio_mean = np.dot(self.weights, self.mean_returns)
        self.portfolio_vol = np.sqrt(np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights)))

        logger.info(f"Monte Carlo simulator initialized: {self.n_assets} assets")

    def simulate_gbm(
        self,
        n_simulations: int = 10000,
        horizon_days: int = 252,
        random_seed: int = None,
    ) -> np.ndarray:
        """
        Simulate using Geometric Brownian Motion.

        Args:
            n_simulations: Number of simulation paths
            horizon_days: Simulation horizon in days
            random_seed: Random seed for reproducibility

        Returns:
            Array of simulated portfolio values (n_simulations x horizon_days)
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Daily drift and volatility
        dt = 1  # Daily
        drift = self.portfolio_mean - 0.5 * self.portfolio_vol ** 2

        # Generate random shocks
        shocks = np.random.normal(0, 1, (n_simulations, horizon_days))

        # Calculate daily returns
        daily_returns = drift * dt + self.portfolio_vol * np.sqrt(dt) * shocks

        # Cumulative returns
        cumulative_returns = np.cumsum(daily_returns, axis=1)

        # Portfolio values
        portfolio_values = self.initial_value * np.exp(cumulative_returns)

        return portfolio_values

    def simulate_bootstrap(
        self,
        n_simulations: int = 10000,
        horizon_days: int = 252,
        block_size: int = 5,
        random_seed: int = None,
    ) -> np.ndarray:
        """
        Simulate using historical bootstrap.

        Non-parametric method using actual historical returns.

        Args:
            n_simulations: Number of simulation paths
            horizon_days: Simulation horizon in days
            block_size: Block size for block bootstrap
            random_seed: Random seed

        Returns:
            Array of simulated portfolio values
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Calculate historical portfolio returns
        portfolio_returns = (self.returns * self.weights).sum(axis=1).values
        n_obs = len(portfolio_returns)

        # Block bootstrap
        n_blocks = int(np.ceil(horizon_days / block_size))

        simulated_values = np.zeros((n_simulations, horizon_days))

        for sim in range(n_simulations):
            # Sample random block starting points
            block_starts = np.random.randint(0, n_obs - block_size, n_blocks)

            # Build return sequence from blocks
            sim_returns = []
            for start in block_starts:
                sim_returns.extend(portfolio_returns[start:start + block_size])

            sim_returns = np.array(sim_returns[:horizon_days])

            # Calculate portfolio values
            cumulative = np.cumprod(1 + sim_returns)
            simulated_values[sim, :] = self.initial_value * cumulative

        return simulated_values

    def simulate_student_t(
        self,
        n_simulations: int = 10000,
        horizon_days: int = 252,
        degrees_of_freedom: int = None,
        random_seed: int = None,
    ) -> np.ndarray:
        """
        Simulate using Student's t distribution.

        Captures fat tails in return distribution.

        Args:
            n_simulations: Number of simulations
            horizon_days: Simulation horizon
            degrees_of_freedom: DoF for t-distribution (estimated if None)
            random_seed: Random seed

        Returns:
            Array of simulated portfolio values
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Estimate degrees of freedom from data if not provided
        if degrees_of_freedom is None:
            portfolio_returns = (self.returns * self.weights).sum(axis=1)
            # Use excess kurtosis to estimate DoF
            kurt = portfolio_returns.kurtosis()
            if kurt > 0:
                degrees_of_freedom = max(4, int(6 / kurt + 4))
            else:
                degrees_of_freedom = 30  # Approximate normal

        # Generate t-distributed shocks
        shocks = stats.t.rvs(degrees_of_freedom, size=(n_simulations, horizon_days))

        # Scale to match volatility
        scale = self.portfolio_vol * np.sqrt((degrees_of_freedom - 2) / degrees_of_freedom)

        # Daily returns
        daily_returns = self.portfolio_mean + scale * shocks

        # Cumulative returns
        cumulative_returns = np.cumprod(1 + daily_returns, axis=1)

        return self.initial_value * cumulative_returns

    def simulate_jump_diffusion(
        self,
        n_simulations: int = 10000,
        horizon_days: int = 252,
        jump_intensity: float = 0.1,
        jump_mean: float = -0.02,
        jump_std: float = 0.05,
        random_seed: int = None,
    ) -> np.ndarray:
        """
        Simulate using Merton's Jump Diffusion model.

        Captures rare extreme events.

        Args:
            n_simulations: Number of simulations
            horizon_days: Simulation horizon
            jump_intensity: Average jumps per day (lambda)
            jump_mean: Mean jump size
            jump_std: Jump size standard deviation
            random_seed: Random seed

        Returns:
            Array of simulated portfolio values
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Diffusion component
        drift = self.portfolio_mean - 0.5 * self.portfolio_vol ** 2
        diffusion = self.portfolio_vol * np.random.normal(0, 1, (n_simulations, horizon_days))

        # Jump component
        n_jumps = np.random.poisson(jump_intensity, (n_simulations, horizon_days))
        jump_sizes = np.random.normal(jump_mean, jump_std, (n_simulations, horizon_days))
        jumps = n_jumps * jump_sizes

        # Combined returns (in log space)
        log_returns = drift + diffusion + jumps

        # Portfolio values
        cumulative = np.cumsum(log_returns, axis=1)
        portfolio_values = self.initial_value * np.exp(cumulative)

        return portfolio_values

    def simulate(
        self,
        n_simulations: int = 10000,
        horizon_days: int = 252,
        method: str = 'gbm',
        random_seed: int = None,
        **kwargs
    ) -> np.ndarray:
        """
        Run Monte Carlo simulation with specified method.

        Args:
            n_simulations: Number of simulation paths
            horizon_days: Simulation horizon in days
            method: 'gbm', 'bootstrap', 'student_t', or 'jump_diffusion'
            random_seed: Random seed
            **kwargs: Additional method-specific arguments

        Returns:
            Array of simulated portfolio values
        """
        methods = {
            'gbm': self.simulate_gbm,
            'bootstrap': self.simulate_bootstrap,
            'student_t': self.simulate_student_t,
            'jump_diffusion': self.simulate_jump_diffusion,
        }

        if method not in methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(methods.keys())}")

        logger.info(f"Running {method} simulation: {n_simulations} paths, {horizon_days} days")

        return methods[method](
            n_simulations=n_simulations,
            horizon_days=horizon_days,
            random_seed=random_seed,
            **kwargs
        )

    def analyze_results(
        self,
        simulated_values: np.ndarray,
        percentiles: List[float] = None,
    ) -> Dict:
        """
        Analyze simulation results.

        Args:
            simulated_values: Simulated portfolio values
            percentiles: Percentiles to calculate

        Returns:
            Dictionary with analysis results
        """
        if percentiles is None:
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

        # Final values
        final_values = simulated_values[:, -1]

        # Returns
        final_returns = (final_values - self.initial_value) / self.initial_value

        # Percentile values
        percentile_values = {
            f'{p}th': np.percentile(final_values, p) for p in percentiles
        }

        # Probability calculations
        prob_loss = (final_returns < 0).mean()
        prob_10pct_loss = (final_returns < -0.10).mean()
        prob_20pct_loss = (final_returns < -0.20).mean()
        prob_10pct_gain = (final_returns > 0.10).mean()
        prob_20pct_gain = (final_returns > 0.20).mean()

        # Path statistics
        min_values = simulated_values.min(axis=1)
        max_drawdown = (simulated_values.max(axis=1) - min_values) / simulated_values.max(axis=1)

        return {
            'initial_value': self.initial_value,
            'mean_final_value': final_values.mean(),
            'median_final_value': np.median(final_values),
            'std_final_value': final_values.std(),
            'min_final_value': final_values.min(),
            'max_final_value': final_values.max(),
            'percentiles': percentile_values,
            'mean_return': final_returns.mean(),
            'median_return': np.median(final_returns),
            'prob_loss': prob_loss,
            'prob_10pct_loss': prob_10pct_loss,
            'prob_20pct_loss': prob_20pct_loss,
            'prob_10pct_gain': prob_10pct_gain,
            'prob_20pct_gain': prob_20pct_gain,
            'mean_max_drawdown': max_drawdown.mean(),
            'median_max_drawdown': np.median(max_drawdown),
            'worst_max_drawdown': max_drawdown.max(),
        }

    def value_at_risk(
        self,
        simulated_values: np.ndarray,
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate VaR from simulation.

        Args:
            simulated_values: Simulated portfolio values
            confidence: Confidence level

        Returns:
            VaR value (potential loss)
        """
        final_values = simulated_values[:, -1]
        var_value = np.percentile(final_values, (1 - confidence) * 100)
        return self.initial_value - var_value

    def conditional_var(
        self,
        simulated_values: np.ndarray,
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate CVaR (Expected Shortfall) from simulation.

        Args:
            simulated_values: Simulated portfolio values
            confidence: Confidence level

        Returns:
            CVaR value
        """
        final_values = simulated_values[:, -1]
        var_threshold = np.percentile(final_values, (1 - confidence) * 100)
        tail_values = final_values[final_values <= var_threshold]
        cvar_value = tail_values.mean()
        return self.initial_value - cvar_value

    def probability_of_ruin(
        self,
        simulated_values: np.ndarray,
        ruin_threshold: float = 0.5,
    ) -> float:
        """
        Calculate probability of ruin.

        Args:
            simulated_values: Simulated portfolio values
            ruin_threshold: Fraction of initial value considered ruin

        Returns:
            Probability of ruin
        """
        ruin_level = self.initial_value * ruin_threshold
        min_values = simulated_values.min(axis=1)
        return (min_values < ruin_level).mean()

    def time_to_target(
        self,
        simulated_values: np.ndarray,
        target_return: float = 0.10,
    ) -> Dict:
        """
        Calculate time to reach target return.

        Args:
            simulated_values: Simulated portfolio values
            target_return: Target return (e.g., 0.10 for 10%)

        Returns:
            Time statistics
        """
        target_value = self.initial_value * (1 + target_return)

        times_to_target = []
        for sim in range(simulated_values.shape[0]):
            path = simulated_values[sim, :]
            reached = np.where(path >= target_value)[0]
            if len(reached) > 0:
                times_to_target.append(reached[0])

        if len(times_to_target) == 0:
            return {
                'probability_reaching': 0,
                'mean_time': np.nan,
                'median_time': np.nan,
            }

        return {
            'probability_reaching': len(times_to_target) / simulated_values.shape[0],
            'mean_time': np.mean(times_to_target),
            'median_time': np.median(times_to_target),
            'min_time': np.min(times_to_target),
            'max_time': np.max(times_to_target),
        }

    def generate_distribution_plot_data(
        self,
        simulated_values: np.ndarray,
        n_bins: int = 50,
    ) -> Dict:
        """
        Generate data for distribution plots.

        Args:
            simulated_values: Simulated portfolio values
            n_bins: Number of histogram bins

        Returns:
            Dictionary with plot data
        """
        final_values = simulated_values[:, -1]
        final_returns = (final_values - self.initial_value) / self.initial_value

        # Histogram data
        hist_values, bin_edges = np.histogram(final_values, bins=n_bins)
        hist_returns, return_edges = np.histogram(final_returns, bins=n_bins)

        return {
            'final_values': final_values,
            'final_returns': final_returns,
            'value_histogram': {
                'counts': hist_values,
                'bins': bin_edges,
            },
            'return_histogram': {
                'counts': hist_returns,
                'bins': return_edges,
            },
        }

    def run_multi_horizon(
        self,
        horizons: List[int] = None,
        n_simulations: int = 10000,
        method: str = 'gbm',
    ) -> pd.DataFrame:
        """
        Run simulations for multiple horizons.

        Args:
            horizons: List of horizons in days
            n_simulations: Number of simulations
            method: Simulation method

        Returns:
            DataFrame with results for each horizon
        """
        if horizons is None:
            horizons = [21, 63, 126, 252, 756]  # 1m, 3m, 6m, 1y, 3y

        results = []

        for horizon in horizons:
            sim_values = self.simulate(n_simulations, horizon, method)
            analysis = self.analyze_results(sim_values)

            results.append({
                'Horizon (days)': horizon,
                'Horizon (months)': horizon / 21,
                'Mean Return': analysis['mean_return'],
                'Median Return': analysis['median_return'],
                'Prob Loss': analysis['prob_loss'],
                'Prob 10% Loss': analysis['prob_10pct_loss'],
                'Prob 20% Loss': analysis['prob_20pct_loss'],
                '5th Percentile': analysis['percentiles']['5th'],
                'Median Value': analysis['median_final_value'],
                '95th Percentile': analysis['percentiles']['95th'],
            })

        return pd.DataFrame(results)


def run_monte_carlo(
    returns: pd.DataFrame,
    weights: np.ndarray = None,
    n_simulations: int = 10000,
    horizon_days: int = 252,
    method: str = 'gbm',
    initial_value: float = 10000,
) -> Dict:
    """
    Convenience function to run Monte Carlo simulation.

    Args:
        returns: Return data
        weights: Portfolio weights
        n_simulations: Number of simulations
        horizon_days: Simulation horizon
        method: Simulation method
        initial_value: Initial portfolio value

    Returns:
        Dictionary with simulation results
    """
    simulator = MonteCarloSimulator(returns, weights, initial_value)
    simulated = simulator.simulate(n_simulations, horizon_days, method)
    analysis = simulator.analyze_results(simulated)

    return {
        'simulated_values': simulated,
        'analysis': analysis,
        'var_95': simulator.value_at_risk(simulated, 0.95),
        'cvar_95': simulator.conditional_var(simulated, 0.95),
    }
