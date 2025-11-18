"""Comprehensive risk metrics calculations."""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats

from ..utils.logger import get_logger

logger = get_logger(__name__)


class RiskMetrics:
    """
    Calculate comprehensive risk metrics for portfolios and assets.

    Includes volatility, performance ratios, drawdown metrics,
    higher moments, and diversification measures.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.02,
        frequency: int = 252,
        benchmark_returns: pd.Series = None,
    ):
        """
        Initialize RiskMetrics calculator.

        Args:
            returns: DataFrame of asset/portfolio returns
            risk_free_rate: Annual risk-free rate
            frequency: Number of periods per year (252 for daily)
            benchmark_returns: Optional benchmark returns for relative metrics
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.frequency = frequency
        self.benchmark_returns = benchmark_returns

        # Periodic risk-free rate
        self.rf_periodic = (1 + risk_free_rate) ** (1 / frequency) - 1

        logger.debug(f"RiskMetrics initialized with {len(returns)} periods")

    # ==================== Volatility Metrics ====================

    def historical_volatility(
        self,
        window: int = None,
        annualize: bool = True
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate historical volatility.

        Args:
            window: Rolling window size (None for full period)
            annualize: Whether to annualize the result

        Returns:
            Volatility values
        """
        if window:
            vol = self.returns.rolling(window=window).std()
        else:
            vol = self.returns.std()

        if annualize:
            vol = vol * np.sqrt(self.frequency)

        return vol

    def parkinson_volatility(
        self,
        high: pd.DataFrame,
        low: pd.DataFrame,
        window: int = None,
        annualize: bool = True
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate Parkinson volatility using high-low prices.

        More efficient estimator than close-to-close volatility.

        Args:
            high: High prices
            low: Low prices
            window: Rolling window size
            annualize: Whether to annualize

        Returns:
            Parkinson volatility
        """
        log_hl = np.log(high / low) ** 2
        factor = 1 / (4 * np.log(2))

        if window:
            vol = np.sqrt(factor * log_hl.rolling(window=window).mean())
        else:
            vol = np.sqrt(factor * log_hl.mean())

        if annualize:
            vol = vol * np.sqrt(self.frequency)

        return vol

    def downside_deviation(
        self,
        threshold: float = 0,
        annualize: bool = True
    ) -> pd.Series:
        """
        Calculate downside deviation (semi-deviation).

        Only considers returns below threshold.

        Args:
            threshold: Return threshold (default: 0)
            annualize: Whether to annualize

        Returns:
            Downside deviation
        """
        downside = self.returns.copy()
        downside[downside > threshold] = 0
        downside_var = (downside ** 2).mean()
        dd = np.sqrt(downside_var)

        if annualize:
            dd = dd * np.sqrt(self.frequency)

        return dd

    # ==================== Performance Ratios ====================

    def sharpe_ratio(self) -> pd.Series:
        """
        Calculate Sharpe ratio.

        Returns:
            Sharpe ratio for each asset
        """
        excess_returns = self.returns - self.rf_periodic
        return (excess_returns.mean() * self.frequency) / (
            self.returns.std() * np.sqrt(self.frequency)
        )

    def sortino_ratio(self, threshold: float = 0) -> pd.Series:
        """
        Calculate Sortino ratio.

        Uses downside deviation instead of total volatility.

        Args:
            threshold: Minimum acceptable return

        Returns:
            Sortino ratio
        """
        excess_returns = self.returns.mean() * self.frequency - self.risk_free_rate
        downside_dev = self.downside_deviation(threshold, annualize=True)

        return excess_returns / downside_dev

    def calmar_ratio(self, prices: pd.DataFrame = None) -> pd.Series:
        """
        Calculate Calmar ratio.

        Annual return / Maximum drawdown.

        Args:
            prices: Price data (if None, calculated from returns)

        Returns:
            Calmar ratio
        """
        annual_return = self.returns.mean() * self.frequency

        if prices is None:
            prices = (1 + self.returns).cumprod()

        mdd = self.max_drawdown(prices)

        return annual_return / abs(mdd)

    def omega_ratio(self, threshold: float = 0) -> pd.Series:
        """
        Calculate Omega ratio.

        Probability-weighted ratio of gains to losses.

        Args:
            threshold: Return threshold

        Returns:
            Omega ratio
        """
        results = {}

        for col in self.returns.columns:
            returns = self.returns[col].dropna()
            gains = returns[returns > threshold] - threshold
            losses = threshold - returns[returns <= threshold]

            if losses.sum() == 0:
                results[col] = np.inf
            else:
                results[col] = gains.sum() / losses.sum()

        return pd.Series(results)

    def information_ratio(self) -> pd.Series:
        """
        Calculate Information ratio.

        Excess return over benchmark / tracking error.

        Returns:
            Information ratio
        """
        if self.benchmark_returns is None:
            logger.warning("No benchmark provided for Information Ratio")
            return pd.Series(index=self.returns.columns, data=np.nan)

        # Align returns with benchmark
        aligned = self.returns.sub(self.benchmark_returns, axis=0).dropna()
        active_returns = aligned.mean() * self.frequency
        tracking_error = aligned.std() * np.sqrt(self.frequency)

        return active_returns / tracking_error

    def treynor_ratio(self) -> pd.Series:
        """
        Calculate Treynor ratio.

        Excess return / Beta.

        Returns:
            Treynor ratio
        """
        if self.benchmark_returns is None:
            logger.warning("No benchmark provided for Treynor Ratio")
            return pd.Series(index=self.returns.columns, data=np.nan)

        excess_return = self.returns.mean() * self.frequency - self.risk_free_rate
        beta = self.calculate_beta()

        return excess_return / beta

    def calculate_beta(self) -> pd.Series:
        """
        Calculate beta relative to benchmark.

        Returns:
            Beta for each asset
        """
        if self.benchmark_returns is None:
            return pd.Series(index=self.returns.columns, data=1.0)

        results = {}
        benchmark = self.benchmark_returns.dropna()

        for col in self.returns.columns:
            asset = self.returns[col].dropna()
            # Align data
            common_idx = asset.index.intersection(benchmark.index)
            if len(common_idx) < 2:
                results[col] = np.nan
                continue

            cov = np.cov(asset[common_idx], benchmark[common_idx])[0, 1]
            var = np.var(benchmark[common_idx])
            results[col] = cov / var if var > 0 else 0

        return pd.Series(results)

    def m_squared(self) -> pd.Series:
        """
        Calculate M-squared (Modigliani-Modigliani).

        Risk-adjusted return comparable to benchmark.

        Returns:
            M-squared for each asset
        """
        if self.benchmark_returns is None:
            logger.warning("No benchmark provided for M-squared")
            return pd.Series(index=self.returns.columns, data=np.nan)

        sharpe = self.sharpe_ratio()
        benchmark_vol = self.benchmark_returns.std() * np.sqrt(self.frequency)

        return self.risk_free_rate + sharpe * benchmark_vol

    # ==================== Drawdown Metrics ====================

    def max_drawdown(self, prices: pd.DataFrame = None) -> pd.Series:
        """
        Calculate maximum drawdown.

        Args:
            prices: Price data (if None, calculated from returns)

        Returns:
            Maximum drawdown (negative value)
        """
        if prices is None:
            prices = (1 + self.returns).cumprod()

        rolling_max = prices.cummax()
        drawdown = (prices - rolling_max) / rolling_max

        return drawdown.min()

    def drawdown_series(self, prices: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate drawdown time series.

        Args:
            prices: Price data

        Returns:
            Drawdown series
        """
        if prices is None:
            prices = (1 + self.returns).cumprod()

        rolling_max = prices.cummax()
        return (prices - rolling_max) / rolling_max

    def ulcer_index(self, prices: pd.DataFrame = None) -> pd.Series:
        """
        Calculate Ulcer Index.

        Root mean square of drawdowns (emphasizes deep drawdowns).

        Args:
            prices: Price data

        Returns:
            Ulcer Index
        """
        drawdown = self.drawdown_series(prices)
        return np.sqrt((drawdown ** 2).mean())

    def pain_index(self, prices: pd.DataFrame = None) -> pd.Series:
        """
        Calculate Pain Index.

        Mean of absolute drawdowns.

        Args:
            prices: Price data

        Returns:
            Pain Index
        """
        drawdown = self.drawdown_series(prices)
        return abs(drawdown).mean()

    def pain_ratio(self, prices: pd.DataFrame = None) -> pd.Series:
        """
        Calculate Pain Ratio.

        Excess return / Pain Index.

        Args:
            prices: Price data

        Returns:
            Pain Ratio
        """
        excess_return = self.returns.mean() * self.frequency - self.risk_free_rate
        pain = self.pain_index(prices)

        return excess_return / pain

    def average_drawdown(self, prices: pd.DataFrame = None) -> pd.Series:
        """
        Calculate average drawdown.

        Args:
            prices: Price data

        Returns:
            Average drawdown
        """
        drawdown = self.drawdown_series(prices)
        return drawdown.mean()

    def drawdown_duration(self, prices: pd.DataFrame = None) -> pd.Series:
        """
        Calculate maximum drawdown duration (in periods).

        Args:
            prices: Price data

        Returns:
            Maximum drawdown duration
        """
        if prices is None:
            prices = (1 + self.returns).cumprod()

        results = {}

        for col in prices.columns:
            price = prices[col]
            rolling_max = price.cummax()

            # Find when we're in drawdown
            in_drawdown = price < rolling_max

            # Calculate durations
            duration = 0
            max_duration = 0

            for i in range(len(in_drawdown)):
                if in_drawdown.iloc[i]:
                    duration += 1
                    max_duration = max(max_duration, duration)
                else:
                    duration = 0

            results[col] = max_duration

        return pd.Series(results)

    # ==================== Higher Moments ====================

    def skewness(self) -> pd.Series:
        """
        Calculate skewness of returns.

        Positive = right-skewed (good), Negative = left-skewed (bad)

        Returns:
            Skewness
        """
        return self.returns.skew()

    def kurtosis(self) -> pd.Series:
        """
        Calculate excess kurtosis of returns.

        Higher = fatter tails (more extreme events)

        Returns:
            Excess kurtosis
        """
        return self.returns.kurtosis()

    def jarque_bera_test(self) -> pd.DataFrame:
        """
        Perform Jarque-Bera test for normality.

        Returns:
            DataFrame with test statistic and p-value
        """
        results = []

        for col in self.returns.columns:
            returns = self.returns[col].dropna()
            statistic, pvalue = stats.jarque_bera(returns)
            results.append({
                'asset': col,
                'statistic': statistic,
                'p_value': pvalue,
                'is_normal': pvalue > 0.05
            })

        return pd.DataFrame(results).set_index('asset')

    # ==================== Diversification Metrics ====================

    def diversification_ratio(
        self,
        weights: np.ndarray,
        cov_matrix: pd.DataFrame = None
    ) -> float:
        """
        Calculate diversification ratio.

        Weighted average volatility / Portfolio volatility.
        Higher = more diversified.

        Args:
            weights: Portfolio weights
            cov_matrix: Covariance matrix (if None, calculated from returns)

        Returns:
            Diversification ratio
        """
        if cov_matrix is None:
            cov_matrix = self.returns.cov() * self.frequency

        asset_vols = np.sqrt(np.diag(cov_matrix))
        weighted_vol = np.dot(weights, asset_vols)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        return weighted_vol / port_vol if port_vol > 0 else 0

    def effective_number_of_bets(
        self,
        weights: np.ndarray,
        cov_matrix: pd.DataFrame = None
    ) -> float:
        """
        Calculate Effective Number of Bets (ENB).

        Measures true diversification accounting for correlations.

        Args:
            weights: Portfolio weights
            cov_matrix: Covariance matrix

        Returns:
            ENB
        """
        if cov_matrix is None:
            cov_matrix = self.returns.cov() * self.frequency

        port_var = np.dot(weights.T, np.dot(cov_matrix, weights))

        if port_var == 0:
            return 0

        # Marginal contributions to risk
        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = weights * marginal_contrib / np.sqrt(port_var)

        # Normalize to get percentage contributions
        pct_contrib = risk_contrib / risk_contrib.sum()
        pct_contrib = pct_contrib[pct_contrib > 0]

        # ENB = exp(-sum(p * log(p)))
        return np.exp(-np.sum(pct_contrib * np.log(pct_contrib)))

    def herfindahl_index(self, weights: np.ndarray) -> float:
        """
        Calculate Herfindahl Index (concentration measure).

        Lower = more diversified. Range: 1/n to 1.

        Args:
            weights: Portfolio weights

        Returns:
            Herfindahl Index
        """
        return np.sum(weights ** 2)

    def correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix.

        Returns:
            Correlation matrix
        """
        return self.returns.corr()

    def average_correlation(self) -> float:
        """
        Calculate average pairwise correlation.

        Returns:
            Average correlation
        """
        corr = self.correlation_matrix()
        n = len(corr)

        if n < 2:
            return 0

        # Get upper triangle (excluding diagonal)
        upper = corr.values[np.triu_indices(n, k=1)]
        return np.mean(upper)

    # ==================== Summary Methods ====================

    def calculate_all_metrics(
        self,
        weights: np.ndarray = None,
        prices: pd.DataFrame = None
    ) -> Dict:
        """
        Calculate all available risk metrics.

        Args:
            weights: Portfolio weights (for diversification metrics)
            prices: Price data

        Returns:
            Dictionary of all metrics
        """
        metrics = {
            # Volatility
            'volatility': self.historical_volatility(annualize=True),
            'downside_deviation': self.downside_deviation(annualize=True),

            # Performance ratios
            'sharpe_ratio': self.sharpe_ratio(),
            'sortino_ratio': self.sortino_ratio(),
            'calmar_ratio': self.calmar_ratio(prices),
            'omega_ratio': self.omega_ratio(),

            # Drawdowns
            'max_drawdown': self.max_drawdown(prices),
            'ulcer_index': self.ulcer_index(prices),
            'pain_index': self.pain_index(prices),

            # Higher moments
            'skewness': self.skewness(),
            'kurtosis': self.kurtosis(),

            # Relative metrics (if benchmark provided)
            'information_ratio': self.information_ratio(),
            'treynor_ratio': self.treynor_ratio(),
            'beta': self.calculate_beta(),
        }

        # Diversification metrics (if weights provided)
        if weights is not None:
            metrics['diversification_ratio'] = self.diversification_ratio(weights)
            metrics['effective_number_of_bets'] = self.effective_number_of_bets(weights)
            metrics['herfindahl_index'] = self.herfindahl_index(weights)

        metrics['average_correlation'] = self.average_correlation()

        return metrics

    def summary_table(self, prices: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generate summary table of key metrics.

        Args:
            prices: Price data

        Returns:
            DataFrame with metrics summary
        """
        metrics = {
            'Annual Return': self.returns.mean() * self.frequency,
            'Annual Volatility': self.historical_volatility(annualize=True),
            'Sharpe Ratio': self.sharpe_ratio(),
            'Sortino Ratio': self.sortino_ratio(),
            'Max Drawdown': self.max_drawdown(prices),
            'Calmar Ratio': self.calmar_ratio(prices),
            'Skewness': self.skewness(),
            'Kurtosis': self.kurtosis(),
        }

        return pd.DataFrame(metrics)


def calculate_metrics(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.02,
    **kwargs
) -> Dict:
    """
    Convenience function to calculate risk metrics.

    Args:
        returns: Return data
        risk_free_rate: Risk-free rate
        **kwargs: Additional arguments

    Returns:
        Dictionary of metrics
    """
    rm = RiskMetrics(returns, risk_free_rate, **kwargs)
    return rm.calculate_all_metrics()
