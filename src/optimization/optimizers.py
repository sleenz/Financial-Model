"""Portfolio optimization algorithms."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from .constraints import PortfolioConstraints, default_constraints
from ..utils.logger import get_logger
from ..utils.helpers import annualize_returns, annualize_volatility

logger = get_logger(__name__)


class OptimizationError(Exception):
    """Custom exception for optimization errors."""
    pass


class PortfolioOptimizer:
    """
    Main portfolio optimization class.

    Supports multiple optimization methods:
    - Maximum Sharpe Ratio
    - Maximum Return
    - Minimum Volatility
    - Maximum Diversification
    - Risk Parity
    - Hierarchical Risk Parity (HRP)
    - Black-Litterman
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.02,
        frequency: int = 252,
    ):
        """
        Initialize the optimizer.

        Args:
            returns: DataFrame of asset returns (index=dates, columns=tickers)
            risk_free_rate: Annual risk-free rate (default: 2%)
            frequency: Number of periods per year (252 for daily)
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.frequency = frequency

        # Calculate basic statistics
        self.tickers = list(returns.columns)
        self.n_assets = len(self.tickers)
        self.mean_returns = returns.mean() * frequency
        self.cov_matrix = returns.cov() * frequency

        logger.info(f"Optimizer initialized with {self.n_assets} assets")

    def optimize(
        self,
        method: str = "max_sharpe",
        constraints: PortfolioConstraints = None,
        **kwargs
    ) -> Dict:
        """
        Run portfolio optimization.

        Args:
            method: Optimization method
            constraints: Portfolio constraints
            **kwargs: Additional method-specific arguments

        Returns:
            Dictionary with weights and metrics
        """
        if constraints is None:
            constraints = default_constraints()

        method_map = {
            'max_sharpe': self._optimize_sharpe,
            'max_return': self._optimize_return,
            'min_volatility': self._optimize_volatility,
            'max_diversification': self._optimize_diversification,
            'risk_parity': self._optimize_risk_parity,
            'hrp': self._optimize_hrp,
            'equal_weight': self._optimize_equal_weight,
        }

        if method not in method_map:
            raise OptimizationError(
                f"Unknown method: {method}. Available: {list(method_map.keys())}"
            )

        logger.info(f"Running {method} optimization")

        try:
            weights = method_map[method](constraints, **kwargs)

            # Enforce position limits / turnover bounds. Scipy-bounded methods
            # (max_sharpe, min_volatility, ...) already solved within these
            # bounds, so this is a no-op for them. HRP and equal-weight never
            # look at `constraints` while computing their raw weights, so
            # without this step those two methods would silently ignore
            # Position Limits and Position Reduction entirely.
            weights = constraints.project_to_bounds(weights, self.tickers)

            # Apply minimum position size
            weights = constraints.apply_minimum_position(weights)

            # Calculate portfolio metrics
            metrics = self._calculate_metrics(weights)

            return {
                'weights': pd.Series(weights, index=self.tickers),
                'expected_return': metrics['expected_return'],
                'volatility': metrics['volatility'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'method': method,
            }

        except ValueError:
            # Constraint validation errors (e.g. infeasible turnover bounds)
            # already carry an actionable, user-facing message — let them
            # propagate as-is instead of masking them behind OptimizationError,
            # which callers don't catch for this case.
            raise
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise OptimizationError(f"Optimization failed: {e}")

    def _optimize_sharpe(
        self,
        constraints: PortfolioConstraints,
        **kwargs
    ) -> np.ndarray:
        """Maximize Sharpe Ratio."""
        def neg_sharpe(weights):
            port_return = np.dot(weights, self.mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            if port_vol == 0:
                return 0
            return -(port_return - self.risk_free_rate) / port_vol

        return self._run_optimization(neg_sharpe, constraints)

    def _optimize_return(
        self,
        constraints: PortfolioConstraints,
        **kwargs
    ) -> np.ndarray:
        """Maximize expected return."""
        def neg_return(weights):
            return -np.dot(weights, self.mean_returns)

        return self._run_optimization(neg_return, constraints)

    def _optimize_volatility(
        self,
        constraints: PortfolioConstraints,
        **kwargs
    ) -> np.ndarray:
        """Minimize portfolio volatility."""
        def portfolio_vol(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

        return self._run_optimization(portfolio_vol, constraints)

    def _optimize_diversification(
        self,
        constraints: PortfolioConstraints,
        **kwargs
    ) -> np.ndarray:
        """Maximize diversification ratio."""
        asset_vols = np.sqrt(np.diag(self.cov_matrix))

        def neg_diversification(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            weighted_vol = np.dot(weights, asset_vols)
            if port_vol == 0:
                return 0
            return -weighted_vol / port_vol

        return self._run_optimization(neg_diversification, constraints)

    def _optimize_risk_parity(
        self,
        constraints: PortfolioConstraints,
        **kwargs
    ) -> np.ndarray:
        """Risk parity: equal risk contribution from each asset."""
        def risk_parity_objective(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

            if port_vol == 0:
                return 0

            # Marginal risk contributions
            marginal_contrib = np.dot(self.cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / port_vol

            # Target: equal risk contribution
            target_risk = port_vol / self.n_assets

            # Minimize squared differences from target
            return np.sum((risk_contrib - target_risk) ** 2)

        return self._run_optimization(risk_parity_objective, constraints)

    def _optimize_hrp(
        self,
        constraints: PortfolioConstraints,
        **kwargs
    ) -> np.ndarray:
        """Hierarchical Risk Parity optimization."""
        # Calculate correlation matrix
        corr = self.returns.corr()

        # Calculate distance matrix
        dist = np.sqrt(0.5 * (1 - corr))

        # Hierarchical clustering
        dist_condensed = squareform(dist.values, checks=False)
        link = linkage(dist_condensed, method='single')

        # Get quasi-diagonal order
        sort_ix = leaves_list(link)
        sorted_tickers = [self.tickers[i] for i in sort_ix]

        # Recursive bisection
        weights = self._hrp_recursive_bisection(
            sorted_tickers,
            self.cov_matrix
        )

        # Reorder to original ticker order
        return np.array([weights[ticker] for ticker in self.tickers])

    def _hrp_recursive_bisection(
        self,
        sorted_tickers: List[str],
        cov_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Recursive bisection for HRP allocation.

        Args:
            sorted_tickers: Tickers in quasi-diagonal order
            cov_matrix: Covariance matrix

        Returns:
            Dictionary of ticker weights
        """
        weights = {ticker: 1.0 for ticker in sorted_tickers}

        clusters = [sorted_tickers]

        while len(clusters) > 0:
            # Bisect each cluster
            new_clusters = []

            for cluster in clusters:
                if len(cluster) == 1:
                    continue

                # Split cluster in half
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                # Calculate cluster variances
                left_var = self._get_cluster_var(left, cov_matrix)
                right_var = self._get_cluster_var(right, cov_matrix)

                # Allocate based on inverse variance
                alloc_factor = 1 - left_var / (left_var + right_var)

                # Update weights
                for ticker in left:
                    weights[ticker] *= alloc_factor
                for ticker in right:
                    weights[ticker] *= (1 - alloc_factor)

                # Add sub-clusters for next iteration
                if len(left) > 1:
                    new_clusters.append(left)
                if len(right) > 1:
                    new_clusters.append(right)

            clusters = new_clusters

        return weights

    def _get_cluster_var(
        self,
        tickers: List[str],
        cov_matrix: pd.DataFrame
    ) -> float:
        """Calculate variance of an inverse-variance weighted cluster."""
        cov_slice = cov_matrix.loc[tickers, tickers]
        ivp = 1 / np.diag(cov_slice)
        ivp = ivp / ivp.sum()
        return np.dot(ivp, np.dot(cov_slice, ivp))

    def _optimize_equal_weight(
        self,
        constraints: PortfolioConstraints,
        **kwargs
    ) -> np.ndarray:
        """Equal weight allocation."""
        return np.ones(self.n_assets) / self.n_assets

    def _run_optimization(
        self,
        objective: callable,
        constraints: PortfolioConstraints
    ) -> np.ndarray:
        """
        Run scipy optimization with constraints.

        Args:
            objective: Objective function to minimize
            constraints: Portfolio constraints

        Returns:
            Optimal weights
        """
        # Initial guess: equal weights
        init_weights = np.ones(self.n_assets) / self.n_assets

        # Bounds — uses turnover trading band when turnover_enabled=True
        bounds = constraints.compute_bounds(self.tickers)

        # Constraints for scipy
        scipy_constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]

        # Run optimization
        result = minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=scipy_constraints,
            options={'ftol': 1e-10, 'maxiter': 1000}
        )

        if not result.success:
            logger.warning(f"Optimization may not have converged: {result.message}")

        return result.x

    def _calculate_metrics(self, weights: np.ndarray) -> Dict:
        """
        Calculate portfolio metrics for given weights.

        Args:
            weights: Portfolio weights

        Returns:
            Dictionary of metrics
        """
        expected_return = np.dot(weights, self.mean_returns)
        volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

        if volatility > 0:
            sharpe_ratio = (expected_return - self.risk_free_rate) / volatility
        else:
            sharpe_ratio = 0

        return {
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
        }

    def efficient_frontier(
        self,
        n_points: int = 50,
        constraints: PortfolioConstraints = None
    ) -> pd.DataFrame:
        """
        Calculate the efficient frontier.

        Args:
            n_points: Number of points on the frontier
            constraints: Portfolio constraints

        Returns:
            DataFrame with frontier points
        """
        if constraints is None:
            constraints = default_constraints()

        # Get return range
        min_ret_result = self.optimize('min_volatility', constraints)
        max_ret_result = self.optimize('max_return', constraints)

        min_ret = min_ret_result['expected_return']
        max_ret = max_ret_result['expected_return']

        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier_data = []

        for target in target_returns:
            try:
                # Minimize volatility for target return
                def portfolio_vol(weights):
                    return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

                init_weights = np.ones(self.n_assets) / self.n_assets
                bounds = constraints.get_bounds(self.n_assets)

                scipy_constraints = [
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                    {'type': 'eq', 'fun': lambda w, t=target: np.dot(w, self.mean_returns) - t}
                ]

                result = minimize(
                    portfolio_vol,
                    init_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=scipy_constraints,
                    # maxiter must match _run_optimization; the default of 100 is
                    # too low for frontier points which carry an extra equality
                    # constraint and cause most points to silently time out.
                    options={'ftol': 1e-9, 'maxiter': 1000},
                )

                # Accept the solution when SLSQP converged OR when it ran out of
                # iterations but still produced a feasible, full-investment portfolio.
                # Using result.success alone rejects many valid near-converged points.
                weights_feasible = abs(np.sum(result.x) - 1.0) < 1e-3
                if result.success or (result.fun > 0 and weights_feasible):
                    # Recompute vol directly from weights rather than trusting
                    # result.fun, which can carry small SLSQP floating-point artifacts.
                    vol = float(np.sqrt(
                        np.dot(result.x.T, np.dot(self.cov_matrix, result.x))
                    ))
                    ret = float(np.dot(result.x, self.mean_returns))
                    sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0.0

                    frontier_data.append({
                        'return': ret,
                        'volatility': vol,
                        'sharpe': sharpe,
                        'weights': result.x,
                    })

            except Exception as e:
                logger.debug(f"Failed to compute frontier point: {e}")
                continue

        if not frontier_data:
            return pd.DataFrame()

        # Sort by volatility so the plotted line runs left-to-right without gaps.
        df = pd.DataFrame(frontier_data)
        return df.sort_values('volatility').reset_index(drop=True)

    def get_risk_contributions(self, weights: np.ndarray = None) -> pd.Series:
        """
        Calculate risk contribution of each asset.

        Args:
            weights: Portfolio weights (default: equal weight)

        Returns:
            Series of risk contributions
        """
        if weights is None:
            weights = np.ones(self.n_assets) / self.n_assets

        port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

        # Marginal contributions
        marginal = np.dot(self.cov_matrix, weights)

        # Risk contributions
        risk_contrib = weights * marginal / port_vol

        return pd.Series(risk_contrib, index=self.tickers)

    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get asset correlation matrix."""
        return self.returns.corr()

    def get_covariance_matrix(self) -> pd.DataFrame:
        """Get annualized covariance matrix."""
        return self.cov_matrix

    def compare_methods(
        self,
        methods: List[str] = None,
        constraints: PortfolioConstraints = None
    ) -> pd.DataFrame:
        """
        Compare multiple optimization methods.

        Args:
            methods: List of methods to compare
            constraints: Portfolio constraints

        Returns:
            DataFrame comparing methods
        """
        if methods is None:
            methods = ['max_sharpe', 'min_volatility', 'risk_parity', 'hrp', 'equal_weight']

        results = []

        for method in methods:
            try:
                result = self.optimize(method, constraints)
                results.append({
                    'method': method,
                    'expected_return': result['expected_return'],
                    'volatility': result['volatility'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'max_weight': result['weights'].max(),
                    'min_weight': result['weights'][result['weights'] > 0].min(),
                    'n_positions': (result['weights'] > 0.001).sum(),
                })
            except Exception as e:
                logger.warning(f"Failed to optimize with {method}: {e}")

        return pd.DataFrame(results)
