"""Black-Litterman model implementation.

Based on: Black, F. & Litterman, R. (1992)
"Global Portfolio Optimization"
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .constraints import PortfolioConstraints
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BlackLittermanError(Exception):
    """Custom exception for Black-Litterman errors."""
    pass


class BlackLittermanModel:
    """
    Black-Litterman portfolio optimization model.

    Combines market equilibrium returns with investor views to produce
    more stable and intuitive portfolio allocations.

    Key components:
    - Market equilibrium (implied returns from market cap weights)
    - Investor views (absolute or relative)
    - Uncertainty in views
    - Posterior expected returns
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        market_caps: pd.Series = None,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        risk_free_rate: float = 0.02,
        frequency: int = 252,
    ):
        """
        Initialize Black-Litterman model.

        Args:
            returns: DataFrame of asset returns
            market_caps: Series of market capitalizations
            risk_aversion: Risk aversion coefficient (lambda)
            tau: Uncertainty in equilibrium (typically small, 0.01-0.05)
            risk_free_rate: Annual risk-free rate
            frequency: Number of periods per year
        """
        self.returns = returns
        self.tickers = list(returns.columns)
        self.n_assets = len(self.tickers)
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.risk_free_rate = risk_free_rate
        self.frequency = frequency

        # Annualized covariance
        self.cov_matrix = returns.cov() * frequency

        # Market cap weights (equilibrium weights)
        if market_caps is not None:
            self.market_caps = market_caps
            total_cap = market_caps.sum()
            self.market_weights = market_caps / total_cap
        else:
            # Use equal weights as fallback
            self.market_weights = pd.Series(
                np.ones(self.n_assets) / self.n_assets,
                index=self.tickers
            )
            self.market_caps = None

        # Calculate implied equilibrium returns
        self.equilibrium_returns = self._calculate_equilibrium_returns()

        # Initialize views
        self.P = None  # View matrix
        self.Q = None  # View returns
        self.omega = None  # View uncertainty

        logger.info(f"Black-Litterman model initialized with {self.n_assets} assets")

    def _calculate_equilibrium_returns(self) -> pd.Series:
        """
        Calculate implied equilibrium returns (pi).

        Using reverse optimization:
        pi = delta * Sigma * w_mkt

        Returns:
            Series of equilibrium returns
        """
        pi = self.risk_aversion * np.dot(
            self.cov_matrix,
            self.market_weights
        )
        return pd.Series(pi, index=self.tickers)

    def add_absolute_view(
        self,
        asset: str,
        view_return: float,
        confidence: float = 0.5,
    ):
        """
        Add an absolute view on an asset.

        Example: "AAPL will return 15%"

        Args:
            asset: Ticker symbol
            view_return: Expected return (annualized)
            confidence: Confidence level (0-1)
        """
        if asset not in self.tickers:
            raise BlackLittermanError(f"Unknown asset: {asset}")

        # Initialize matrices if needed
        if self.P is None:
            self.P = np.zeros((0, self.n_assets))
            self.Q = np.array([])
            self.omega = []

        # Create view vector (1 for the asset, 0 for others)
        p = np.zeros(self.n_assets)
        p[self.tickers.index(asset)] = 1

        # Add to matrices
        self.P = np.vstack([self.P, p])
        self.Q = np.append(self.Q, view_return)
        self.omega.append(self._calculate_view_uncertainty(p, confidence))

        logger.info(f"Added absolute view: {asset} = {view_return:.2%}")

    def add_relative_view(
        self,
        long_assets: List[str],
        short_assets: List[str],
        view_return: float,
        confidence: float = 0.5,
        equal_weight: bool = True,
    ):
        """
        Add a relative view between assets.

        Example: "Tech will outperform Finance by 3%"

        Args:
            long_assets: Assets expected to outperform
            short_assets: Assets expected to underperform
            view_return: Expected outperformance (annualized)
            confidence: Confidence level (0-1)
            equal_weight: Whether to equal-weight within groups
        """
        # Validate assets
        for asset in long_assets + short_assets:
            if asset not in self.tickers:
                raise BlackLittermanError(f"Unknown asset: {asset}")

        # Initialize matrices if needed
        if self.P is None:
            self.P = np.zeros((0, self.n_assets))
            self.Q = np.array([])
            self.omega = []

        # Create view vector
        p = np.zeros(self.n_assets)

        if equal_weight:
            long_weight = 1 / len(long_assets)
            short_weight = -1 / len(short_assets)
        else:
            long_weight = 1
            short_weight = -1

        for asset in long_assets:
            p[self.tickers.index(asset)] = long_weight
        for asset in short_assets:
            p[self.tickers.index(asset)] = short_weight

        # Add to matrices
        self.P = np.vstack([self.P, p])
        self.Q = np.append(self.Q, view_return)
        self.omega.append(self._calculate_view_uncertainty(p, confidence))

        logger.info(
            f"Added relative view: {long_assets} vs {short_assets} = {view_return:.2%}"
        )

    def _calculate_view_uncertainty(
        self,
        p: np.ndarray,
        confidence: float
    ) -> float:
        """
        Calculate uncertainty (variance) for a view.

        Uses the formula: omega = (1/c - 1) * p' * Sigma * p
        where c is confidence level.

        Args:
            p: View vector
            confidence: Confidence level (0-1)

        Returns:
            View variance
        """
        # Ensure confidence is in valid range
        confidence = np.clip(confidence, 0.01, 0.99)

        # Calculate view portfolio variance
        view_var = np.dot(p.T, np.dot(self.cov_matrix, p))

        # Scale by confidence
        # Higher confidence = lower uncertainty
        omega = view_var * (1 / confidence - 1)

        return omega

    def get_posterior_returns(self) -> pd.Series:
        """
        Calculate posterior expected returns.

        Combines equilibrium returns with views using Bayesian updating.

        Returns:
            Series of posterior returns
        """
        if self.P is None or len(self.Q) == 0:
            logger.warning("No views added, returning equilibrium returns")
            return self.equilibrium_returns

        # Convert to matrices
        Sigma = self.cov_matrix.values
        pi = self.equilibrium_returns.values
        P = self.P
        Q = self.Q
        Omega = np.diag(self.omega)

        # Black-Litterman formula
        # M = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1
        # posterior = M * [(tau*Sigma)^-1*pi + P'*Omega^-1*Q]

        tau_sigma = self.tau * Sigma
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(Omega)

        # Calculate M (posterior covariance of returns)
        M = np.linalg.inv(
            tau_sigma_inv + np.dot(P.T, np.dot(omega_inv, P))
        )

        # Calculate posterior returns
        posterior = np.dot(
            M,
            np.dot(tau_sigma_inv, pi) + np.dot(P.T, np.dot(omega_inv, Q))
        )

        return pd.Series(posterior, index=self.tickers)

    def get_posterior_covariance(self) -> pd.DataFrame:
        """
        Calculate posterior covariance matrix.

        Returns:
            DataFrame of posterior covariance
        """
        if self.P is None or len(self.Q) == 0:
            return self.cov_matrix * (1 + self.tau)

        Sigma = self.cov_matrix.values
        P = self.P
        Omega = np.diag(self.omega)

        tau_sigma = self.tau * Sigma
        omega_inv = np.linalg.inv(Omega)

        # Posterior covariance
        M = np.linalg.inv(
            np.linalg.inv(tau_sigma) + np.dot(P.T, np.dot(omega_inv, P))
        )

        posterior_cov = Sigma + M

        return pd.DataFrame(
            posterior_cov,
            index=self.tickers,
            columns=self.tickers
        )

    def optimize(
        self,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
        constraints: Optional[PortfolioConstraints] = None,
    ) -> Dict:
        """
        Optimize portfolio using Black-Litterman returns.

        Args:
            max_weight: Maximum weight per asset. Ignored if `constraints` is
                given (its own min_weight/max_weight take over).
            min_weight: Minimum weight per asset. Ignored if `constraints` is
                given (its own min_weight/max_weight take over).
            constraints: Optional PortfolioConstraints. When provided, bounds
                are computed via constraints.compute_bounds(), which honors
                the position reduction (turnover) trading band the same way
                every other optimization method does — without it, Black-
                Litterman would silently ignore that constraint.

        Returns:
            Dictionary with weights and metrics
        """
        posterior_returns = self.get_posterior_returns()
        posterior_cov = self.get_posterior_covariance()

        # Maximum Sharpe ratio optimization
        def neg_sharpe(weights):
            ret = np.dot(weights, posterior_returns)
            vol = np.sqrt(np.dot(weights.T, np.dot(posterior_cov, weights)))
            if vol == 0:
                return 0
            return -(ret - self.risk_free_rate) / vol

        # Initial guess
        init_weights = np.array(self.market_weights)

        # Bounds — turnover-aware when a PortfolioConstraints is supplied,
        # otherwise flat (min_weight, max_weight) for every asset.
        if constraints is not None:
            bounds = constraints.compute_bounds(self.tickers)
        else:
            bounds = [(min_weight, max_weight) for _ in range(self.n_assets)]

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        # Optimize
        result = minimize(
            neg_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-10}
        )

        weights = result.x

        # Calculate metrics
        expected_return = np.dot(weights, posterior_returns)
        volatility = np.sqrt(np.dot(weights.T, np.dot(posterior_cov, weights)))
        sharpe = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0

        return {
            'weights': pd.Series(weights, index=self.tickers),
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'posterior_returns': posterior_returns,
            'equilibrium_returns': self.equilibrium_returns,
        }

    def clear_views(self):
        """Remove all views."""
        self.P = None
        self.Q = None
        self.omega = None
        logger.info("All views cleared")

    def get_views_summary(self) -> pd.DataFrame:
        """
        Get summary of current views.

        Returns:
            DataFrame with view information
        """
        if self.P is None or len(self.Q) == 0:
            return pd.DataFrame()

        views = []
        for i in range(len(self.Q)):
            view_assets = []
            for j, ticker in enumerate(self.tickers):
                if abs(self.P[i, j]) > 1e-6:
                    view_assets.append(f"{ticker}:{self.P[i,j]:.2f}")

            views.append({
                'view_id': i,
                'assets': ', '.join(view_assets),
                'expected_return': self.Q[i],
                'uncertainty': self.omega[i],
            })

        return pd.DataFrame(views)


def black_litterman_allocation(
    returns: pd.DataFrame,
    market_caps: pd.Series = None,
    views: List[Dict] = None,
    **kwargs
) -> pd.Series:
    """
    Convenience function for Black-Litterman allocation.

    Args:
        returns: DataFrame of asset returns
        market_caps: Market capitalizations
        views: List of view dictionaries
        **kwargs: Additional arguments for BlackLittermanModel

    Returns:
        Series of portfolio weights
    """
    model = BlackLittermanModel(returns, market_caps, **kwargs)

    if views:
        for view in views:
            if view.get('type') == 'absolute':
                model.add_absolute_view(
                    view['asset'],
                    view['return'],
                    view.get('confidence', 0.5)
                )
            elif view.get('type') == 'relative':
                model.add_relative_view(
                    view['long'],
                    view['short'],
                    view['return'],
                    view.get('confidence', 0.5)
                )

    result = model.optimize()
    return result['weights']
