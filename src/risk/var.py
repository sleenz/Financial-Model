"""Value at Risk (VaR) and Conditional VaR (CVaR) calculations."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

from ..utils.logger import get_logger

logger = get_logger(__name__)


class VaRCalculator:
    """
    Calculate Value at Risk using multiple methods.

    Supports:
    - Historical VaR
    - Parametric VaR (normal distribution)
    - Cornish-Fisher VaR (adjusted for skewness/kurtosis)
    - Conditional VaR (Expected Shortfall)
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        confidence_levels: List[float] = None,
        frequency: int = 252,
    ):
        """
        Initialize VaR calculator.

        Args:
            returns: DataFrame of asset returns
            confidence_levels: List of confidence levels (default: [0.95, 0.99])
            frequency: Number of periods per year
        """
        self.returns = returns
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        self.frequency = frequency

        logger.debug(f"VaR calculator initialized with {len(returns)} periods")

    def historical_var(
        self,
        confidence: float = 0.95,
        portfolio_value: float = None
    ) -> pd.Series:
        """
        Calculate Historical VaR.

        Uses empirical distribution of returns.

        Args:
            confidence: Confidence level (e.g., 0.95 for 95%)
            portfolio_value: Portfolio value for dollar VaR

        Returns:
            VaR values (negative returns)
        """
        alpha = 1 - confidence
        var = self.returns.quantile(alpha)

        if portfolio_value:
            var = var * portfolio_value

        return var

    def parametric_var(
        self,
        confidence: float = 0.95,
        portfolio_value: float = None
    ) -> pd.Series:
        """
        Calculate Parametric VaR (variance-covariance method).

        Assumes normal distribution.

        Args:
            confidence: Confidence level
            portfolio_value: Portfolio value for dollar VaR

        Returns:
            VaR values
        """
        alpha = 1 - confidence
        z_score = stats.norm.ppf(alpha)

        mean = self.returns.mean()
        std = self.returns.std()

        var = mean + z_score * std

        if portfolio_value:
            var = var * portfolio_value

        return var

    def cornish_fisher_var(
        self,
        confidence: float = 0.95,
        portfolio_value: float = None
    ) -> pd.Series:
        """
        Calculate Cornish-Fisher VaR.

        Adjusts for skewness and kurtosis (non-normality).

        Args:
            confidence: Confidence level
            portfolio_value: Portfolio value for dollar VaR

        Returns:
            Adjusted VaR values
        """
        alpha = 1 - confidence
        z = stats.norm.ppf(alpha)

        mean = self.returns.mean()
        std = self.returns.std()
        skew = self.returns.skew()
        kurt = self.returns.kurtosis()

        # Cornish-Fisher expansion
        z_cf = (
            z +
            (z**2 - 1) * skew / 6 +
            (z**3 - 3*z) * kurt / 24 -
            (2*z**3 - 5*z) * (skew**2) / 36
        )

        var = mean + z_cf * std

        if portfolio_value:
            var = var * portfolio_value

        return var

    def historical_cvar(
        self,
        confidence: float = 0.95,
        portfolio_value: float = None
    ) -> pd.Series:
        """
        Calculate Historical CVaR (Expected Shortfall).

        Average loss beyond VaR threshold.

        Args:
            confidence: Confidence level
            portfolio_value: Portfolio value for dollar CVaR

        Returns:
            CVaR values
        """
        var = self.historical_var(confidence)

        results = {}
        for col in self.returns.columns:
            returns = self.returns[col]
            cvar = returns[returns <= var[col]].mean()
            results[col] = cvar

        cvar = pd.Series(results)

        if portfolio_value:
            cvar = cvar * portfolio_value

        return cvar

    def parametric_cvar(
        self,
        confidence: float = 0.95,
        portfolio_value: float = None
    ) -> pd.Series:
        """
        Calculate Parametric CVaR.

        Assumes normal distribution.

        Args:
            confidence: Confidence level
            portfolio_value: Portfolio value for dollar CVaR

        Returns:
            CVaR values
        """
        alpha = 1 - confidence

        mean = self.returns.mean()
        std = self.returns.std()

        # For normal distribution: CVaR = mean - std * phi(z) / alpha
        z = stats.norm.ppf(alpha)
        cvar = mean - std * stats.norm.pdf(z) / alpha

        if portfolio_value:
            cvar = cvar * portfolio_value

        return cvar

    def calculate_all(
        self,
        confidence: float = 0.95,
        portfolio_value: float = None
    ) -> pd.DataFrame:
        """
        Calculate all VaR and CVaR measures.

        Args:
            confidence: Confidence level
            portfolio_value: Portfolio value

        Returns:
            DataFrame with all VaR measures
        """
        results = {
            'Historical VaR': self.historical_var(confidence, portfolio_value),
            'Parametric VaR': self.parametric_var(confidence, portfolio_value),
            'Cornish-Fisher VaR': self.cornish_fisher_var(confidence, portfolio_value),
            'Historical CVaR': self.historical_cvar(confidence, portfolio_value),
            'Parametric CVaR': self.parametric_cvar(confidence, portfolio_value),
        }

        return pd.DataFrame(results)

    def var_summary(
        self,
        portfolio_value: float = None
    ) -> pd.DataFrame:
        """
        Generate VaR summary for multiple confidence levels.

        Args:
            portfolio_value: Portfolio value

        Returns:
            Summary DataFrame
        """
        summaries = []

        for conf in self.confidence_levels:
            for col in self.returns.columns:
                summaries.append({
                    'Asset': col,
                    'Confidence': f"{conf:.0%}",
                    'Historical VaR': self.historical_var(conf, portfolio_value)[col],
                    'Parametric VaR': self.parametric_var(conf, portfolio_value)[col],
                    'Historical CVaR': self.historical_cvar(conf, portfolio_value)[col],
                })

        return pd.DataFrame(summaries)

    def rolling_var(
        self,
        window: int = 252,
        confidence: float = 0.95,
        method: str = 'historical'
    ) -> pd.DataFrame:
        """
        Calculate rolling VaR.

        Args:
            window: Rolling window size
            confidence: Confidence level
            method: 'historical' or 'parametric'

        Returns:
            Rolling VaR values
        """
        alpha = 1 - confidence

        if method == 'historical':
            return self.returns.rolling(window=window).quantile(alpha)
        elif method == 'parametric':
            z = stats.norm.ppf(alpha)
            mean = self.returns.rolling(window=window).mean()
            std = self.returns.rolling(window=window).std()
            return mean + z * std
        else:
            raise ValueError(f"Unknown method: {method}")

    def var_backtest(
        self,
        confidence: float = 0.95,
        window: int = 252,
        method: str = 'historical'
    ) -> pd.DataFrame:
        """
        Backtest VaR model.

        Counts exceptions (actual losses > VaR).

        Args:
            confidence: Confidence level
            window: Rolling window for VaR calculation
            method: VaR method to backtest

        Returns:
            Backtest results
        """
        # Calculate rolling VaR
        var = self.rolling_var(window, confidence, method)

        # Shift VaR to align with next day's return
        var_shifted = var.shift(1)

        # Count exceptions
        results = []

        for col in self.returns.columns:
            returns = self.returns[col]
            var_vals = var_shifted[col]

            # Align data
            valid_idx = var_vals.dropna().index
            actual = returns.loc[valid_idx]
            predicted = var_vals.loc[valid_idx]

            exceptions = (actual < predicted).sum()
            n_obs = len(valid_idx)
            expected = n_obs * (1 - confidence)
            exception_rate = exceptions / n_obs

            # Kupiec POF test
            if exceptions > 0 and exceptions < n_obs:
                lr_stat = -2 * (
                    np.log((1 - confidence) ** exceptions * confidence ** (n_obs - exceptions)) -
                    np.log(exception_rate ** exceptions * (1 - exception_rate) ** (n_obs - exceptions))
                )
                p_value = 1 - stats.chi2.cdf(lr_stat, 1)
            else:
                lr_stat = np.nan
                p_value = np.nan

            results.append({
                'Asset': col,
                'Observations': n_obs,
                'Exceptions': exceptions,
                'Expected': expected,
                'Exception Rate': exception_rate,
                'LR Statistic': lr_stat,
                'P-Value': p_value,
                'Model Valid': p_value > 0.05 if not np.isnan(p_value) else None
            })

        return pd.DataFrame(results)


class PortfolioVaR:
    """
    Calculate VaR for a portfolio of assets.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        portfolio_value: float = 1.0,
    ):
        """
        Initialize portfolio VaR calculator.

        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights
            portfolio_value: Total portfolio value
        """
        self.returns = returns
        self.weights = weights
        self.portfolio_value = portfolio_value

        # Calculate portfolio returns
        self.portfolio_returns = (returns * weights).sum(axis=1)

    def marginal_var(
        self,
        confidence: float = 0.95
    ) -> pd.Series:
        """
        Calculate Marginal VaR for each asset.

        Sensitivity of portfolio VaR to position change.

        Args:
            confidence: Confidence level

        Returns:
            Marginal VaR for each asset
        """
        alpha = 1 - confidence
        z = stats.norm.ppf(alpha)

        cov_matrix = self.returns.cov()
        port_vol = np.sqrt(np.dot(self.weights.T, np.dot(cov_matrix, self.weights)))

        # Marginal contribution to volatility
        marginal_vol = np.dot(cov_matrix, self.weights) / port_vol

        return pd.Series(
            z * marginal_vol * self.portfolio_value,
            index=self.returns.columns
        )

    def component_var(
        self,
        confidence: float = 0.95
    ) -> pd.Series:
        """
        Calculate Component VaR for each asset.

        Contribution of each asset to total VaR.

        Args:
            confidence: Confidence level

        Returns:
            Component VaR for each asset
        """
        marginal = self.marginal_var(confidence)
        return marginal * self.weights

    def incremental_var(
        self,
        confidence: float = 0.95,
        increment: float = 0.01
    ) -> pd.Series:
        """
        Calculate Incremental VaR.

        Change in VaR from adding increment to position.

        Args:
            confidence: Confidence level
            increment: Position increment

        Returns:
            Incremental VaR for each asset
        """
        base_var = self._parametric_portfolio_var(confidence)

        results = {}
        for i, col in enumerate(self.returns.columns):
            new_weights = self.weights.copy()
            new_weights[i] += increment
            new_weights = new_weights / new_weights.sum()  # Renormalize

            new_var = self._parametric_portfolio_var(confidence, new_weights)
            results[col] = new_var - base_var

        return pd.Series(results)

    def _parametric_portfolio_var(
        self,
        confidence: float,
        weights: np.ndarray = None
    ) -> float:
        """
        Calculate parametric VaR for portfolio.

        Args:
            confidence: Confidence level
            weights: Portfolio weights

        Returns:
            Portfolio VaR
        """
        if weights is None:
            weights = self.weights

        alpha = 1 - confidence
        z = stats.norm.ppf(alpha)

        cov_matrix = self.returns.cov()
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        port_mean = np.dot(weights, self.returns.mean())

        return (port_mean + z * port_vol) * self.portfolio_value

    def var_decomposition(
        self,
        confidence: float = 0.95
    ) -> pd.DataFrame:
        """
        Decompose portfolio VaR by asset.

        Args:
            confidence: Confidence level

        Returns:
            VaR decomposition table
        """
        component = self.component_var(confidence)
        total_var = component.sum()

        return pd.DataFrame({
            'Weight': self.weights,
            'Component VaR': component,
            'Contribution %': component / total_var * 100
        }, index=self.returns.columns)


def calculate_var(
    returns: pd.DataFrame,
    confidence: float = 0.95,
    method: str = 'historical',
    portfolio_value: float = None
) -> pd.Series:
    """
    Convenience function to calculate VaR.

    Args:
        returns: Return data
        confidence: Confidence level
        method: 'historical', 'parametric', or 'cornish_fisher'
        portfolio_value: Portfolio value

    Returns:
        VaR values
    """
    calc = VaRCalculator(returns)

    if method == 'historical':
        return calc.historical_var(confidence, portfolio_value)
    elif method == 'parametric':
        return calc.parametric_var(confidence, portfolio_value)
    elif method == 'cornish_fisher':
        return calc.cornish_fisher_var(confidence, portfolio_value)
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_cvar(
    returns: pd.DataFrame,
    confidence: float = 0.95,
    method: str = 'historical',
    portfolio_value: float = None
) -> pd.Series:
    """
    Convenience function to calculate CVaR.

    Args:
        returns: Return data
        confidence: Confidence level
        method: 'historical' or 'parametric'
        portfolio_value: Portfolio value

    Returns:
        CVaR values
    """
    calc = VaRCalculator(returns)

    if method == 'historical':
        return calc.historical_cvar(confidence, portfolio_value)
    elif method == 'parametric':
        return calc.parametric_cvar(confidence, portfolio_value)
    else:
        raise ValueError(f"Unknown method: {method}")
