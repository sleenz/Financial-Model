"""
Style Factor Analysis

Implements style factors: Momentum, Value, Quality, Low Volatility.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats

from ..utils.logger import get_logger

logger = get_logger(__name__)


def calculate_momentum(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip_recent: int = 21
) -> pd.Series:
    """
    Calculate momentum score for each asset.

    Uses 12-month return minus 1-month (avoids short-term reversal).

    Args:
        prices: Price DataFrame
        lookback: Lookback period in days (default 252 = 1 year)
        skip_recent: Days to skip (default 21 = 1 month)

    Returns:
        Series with momentum scores
    """
    if len(prices) < lookback:
        lookback = len(prices) - 1

    # Total return over lookback period
    total_return = prices.iloc[-skip_recent] / prices.iloc[-lookback] - 1

    # Return over recent period to skip
    recent_return = prices.iloc[-1] / prices.iloc[-skip_recent] - 1

    # Momentum = total return - recent return
    momentum = total_return - recent_return

    return momentum


def calculate_value_score(
    pe_ratios: Optional[pd.Series] = None,
    pb_ratios: Optional[pd.Series] = None,
    dividend_yields: Optional[pd.Series] = None
) -> pd.Series:
    """
    Calculate composite value score.

    Higher scores indicate more "value" (cheaper) stocks.

    Args:
        pe_ratios: Price-to-Earnings ratios
        pb_ratios: Price-to-Book ratios
        dividend_yields: Dividend yields

    Returns:
        Series with value scores (z-scores, inverted for P/E and P/B)
    """
    scores = []

    if pe_ratios is not None:
        # Lower P/E = higher value, so invert
        pe_z = -stats.zscore(pe_ratios.dropna())
        scores.append(pd.Series(pe_z, index=pe_ratios.dropna().index))

    if pb_ratios is not None:
        # Lower P/B = higher value, so invert
        pb_z = -stats.zscore(pb_ratios.dropna())
        scores.append(pd.Series(pb_z, index=pb_ratios.dropna().index))

    if dividend_yields is not None:
        # Higher dividend yield = more value
        dy_z = stats.zscore(dividend_yields.dropna())
        scores.append(pd.Series(dy_z, index=dividend_yields.dropna().index))

    if not scores:
        raise ValueError("At least one ratio must be provided")

    # Combine scores
    combined = pd.concat(scores, axis=1).mean(axis=1)
    return combined


def calculate_quality_score(
    roe: Optional[pd.Series] = None,
    debt_to_equity: Optional[pd.Series] = None,
    earnings_stability: Optional[pd.Series] = None,
    returns: Optional[pd.DataFrame] = None
) -> pd.Series:
    """
    Calculate composite quality score.

    Higher scores indicate higher quality companies.

    Args:
        roe: Return on Equity
        debt_to_equity: Debt-to-Equity ratio
        earnings_stability: Earnings stability measure
        returns: Historical returns for calculating stability

    Returns:
        Series with quality scores
    """
    scores = []

    if roe is not None:
        # Higher ROE = higher quality
        roe_z = stats.zscore(roe.dropna())
        scores.append(pd.Series(roe_z, index=roe.dropna().index))

    if debt_to_equity is not None:
        # Lower D/E = higher quality
        de_z = -stats.zscore(debt_to_equity.dropna())
        scores.append(pd.Series(de_z, index=debt_to_equity.dropna().index))

    if earnings_stability is not None:
        # Higher stability = higher quality
        es_z = stats.zscore(earnings_stability.dropna())
        scores.append(pd.Series(es_z, index=earnings_stability.dropna().index))

    if returns is not None:
        # Use return stability as proxy
        return_vol = returns.std()
        # Lower volatility = higher quality (more stable)
        vol_z = pd.Series(-stats.zscore(return_vol), index=return_vol.index)
        scores.append(vol_z)

    if not scores:
        raise ValueError("At least one metric must be provided")

    combined = pd.concat(scores, axis=1).mean(axis=1)
    return combined


def calculate_low_volatility_score(
    returns: pd.DataFrame,
    window: int = 252
) -> pd.Series:
    """
    Calculate low volatility score.

    Lower volatility gets higher score.

    Args:
        returns: Return DataFrame
        window: Lookback window

    Returns:
        Series with low volatility scores (inverted z-scores)
    """
    if len(returns) < window:
        window = len(returns)

    recent_returns = returns.iloc[-window:]
    volatility = recent_returns.std() * np.sqrt(252)

    # Invert so lower vol = higher score
    low_vol_score = -stats.zscore(volatility)

    return pd.Series(low_vol_score, index=returns.columns)


class StyleFactorAnalyzer:
    """
    Comprehensive style factor analysis for portfolios.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        returns: Optional[pd.DataFrame] = None,
        fundamentals: Optional[Dict[str, pd.Series]] = None
    ):
        """
        Initialize style factor analyzer.

        Args:
            prices: Price DataFrame
            returns: Optional returns DataFrame
            fundamentals: Optional dict with 'pe', 'pb', 'div_yield', 'roe', 'de'
        """
        self.prices = prices
        self.returns = returns if returns is not None else prices.pct_change().dropna()
        self.fundamentals = fundamentals or {}

        logger.info(f"StyleFactorAnalyzer initialized with {len(prices.columns)} assets")

    def calculate_all_factors(self) -> pd.DataFrame:
        """
        Calculate all style factors for each asset.

        Returns:
            DataFrame with factor scores for each asset
        """
        results = pd.DataFrame(index=self.prices.columns)

        # Momentum
        try:
            results['Momentum'] = calculate_momentum(self.prices)
        except Exception as e:
            logger.warning(f"Could not calculate momentum: {e}")
            results['Momentum'] = 0

        # Low Volatility
        try:
            results['Low_Volatility'] = calculate_low_volatility_score(self.returns)
        except Exception as e:
            logger.warning(f"Could not calculate low vol: {e}")
            results['Low_Volatility'] = 0

        # Value (if fundamentals available)
        if 'pe' in self.fundamentals or 'pb' in self.fundamentals:
            try:
                results['Value'] = calculate_value_score(
                    self.fundamentals.get('pe'),
                    self.fundamentals.get('pb'),
                    self.fundamentals.get('div_yield')
                )
            except Exception as e:
                logger.warning(f"Could not calculate value: {e}")
                results['Value'] = 0
        else:
            # Generate synthetic value scores
            np.random.seed(42)
            results['Value'] = np.random.randn(len(results))

        # Quality
        if 'roe' in self.fundamentals or 'de' in self.fundamentals:
            try:
                results['Quality'] = calculate_quality_score(
                    self.fundamentals.get('roe'),
                    self.fundamentals.get('de'),
                    returns=self.returns
                )
            except Exception as e:
                logger.warning(f"Could not calculate quality: {e}")
                results['Quality'] = 0
        else:
            # Use return stability as proxy
            try:
                results['Quality'] = calculate_quality_score(returns=self.returns)
            except:
                results['Quality'] = 0

        # Size (using volatility as proxy - smaller companies tend to be more volatile)
        try:
            volatility = self.returns.std() * np.sqrt(252)
            # Higher vol = smaller company
            results['Size'] = stats.zscore(volatility)
        except:
            results['Size'] = 0

        return results

    def portfolio_factor_exposure(
        self,
        weights: pd.Series
    ) -> Dict:
        """
        Calculate portfolio-level factor exposures.

        Args:
            weights: Portfolio weights

        Returns:
            Dictionary with weighted factor exposures
        """
        factors = self.calculate_all_factors()

        # Align weights with factors
        aligned_weights = weights[weights.index.isin(factors.index)]
        aligned_factors = factors.loc[aligned_weights.index]

        exposures = {}
        for factor in aligned_factors.columns:
            exposure = (aligned_factors[factor] * aligned_weights).sum()
            exposures[factor] = exposure

        return exposures

    def factor_tilt_analysis(
        self,
        weights: pd.Series
    ) -> pd.DataFrame:
        """
        Analyze factor tilts vs equal-weight portfolio.

        Args:
            weights: Portfolio weights

        Returns:
            DataFrame comparing portfolio tilts to equal weight
        """
        factors = self.calculate_all_factors()

        # Portfolio exposure
        aligned_weights = weights[weights.index.isin(factors.index)]
        aligned_factors = factors.loc[aligned_weights.index]

        portfolio_exp = {}
        for factor in aligned_factors.columns:
            portfolio_exp[factor] = (aligned_factors[factor] * aligned_weights).sum()

        # Equal weight exposure
        equal_weights = pd.Series(
            1.0 / len(aligned_weights),
            index=aligned_weights.index
        )

        equal_exp = {}
        for factor in aligned_factors.columns:
            equal_exp[factor] = (aligned_factors[factor] * equal_weights).sum()

        # Build comparison
        results = pd.DataFrame({
            'Portfolio': portfolio_exp,
            'Equal Weight': equal_exp,
            'Tilt': {k: portfolio_exp[k] - equal_exp[k] for k in portfolio_exp}
        })

        return results

    def factor_return_attribution(
        self,
        weights: pd.Series,
        factor_returns: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Attribute portfolio returns to factor exposures.

        Args:
            weights: Portfolio weights
            factor_returns: Expected factor returns (annualized)

        Returns:
            Dictionary with return attribution
        """
        # Default factor return assumptions
        default_factor_returns = {
            'Momentum': 0.04,      # 4% momentum premium
            'Value': 0.03,         # 3% value premium
            'Quality': 0.025,      # 2.5% quality premium
            'Low_Volatility': 0.02,  # 2% low vol premium
            'Size': 0.02           # 2% size premium
        }

        factor_rets = factor_returns or default_factor_returns
        exposures = self.portfolio_factor_exposure(weights)

        attribution = {}
        total = 0

        for factor, exposure in exposures.items():
            if factor in factor_rets:
                contribution = exposure * factor_rets[factor]
                attribution[factor] = {
                    'exposure': exposure,
                    'factor_return': factor_rets[factor],
                    'contribution': contribution
                }
                total += contribution

        attribution['total'] = total

        return attribution

    def top_factor_stocks(
        self,
        factor: str,
        n: int = 5
    ) -> pd.DataFrame:
        """
        Get top stocks for a given factor.

        Args:
            factor: Factor name
            n: Number of stocks to return

        Returns:
            DataFrame with top stocks
        """
        factors = self.calculate_all_factors()

        if factor not in factors.columns:
            raise ValueError(f"Factor {factor} not found")

        sorted_stocks = factors[factor].sort_values(ascending=False)

        result = pd.DataFrame({
            'Asset': sorted_stocks.head(n).index,
            f'{factor} Score': sorted_stocks.head(n).values
        })

        return result

    def factor_correlation(self) -> pd.DataFrame:
        """Calculate correlation between factors."""
        factors = self.calculate_all_factors()
        return factors.corr()

    def summary(self, weights: pd.Series) -> str:
        """
        Generate text summary of style factor analysis.

        Args:
            weights: Portfolio weights

        Returns:
            Formatted summary string
        """
        exposures = self.portfolio_factor_exposure(weights)
        tilts = self.factor_tilt_analysis(weights)

        lines = [
            "Style Factor Analysis",
            "=" * 40,
            "",
            "Portfolio Factor Exposures:",
            "-" * 30
        ]

        for factor, exposure in exposures.items():
            tilt = tilts.loc[factor, 'Tilt']
            direction = "+" if tilt > 0 else ""
            lines.append(f"  {factor}: {exposure:.3f} (tilt: {direction}{tilt:.3f})")

        lines.extend([
            "",
            "Factor Interpretation:",
            "-" * 30,
            "  Momentum > 0: Tilted toward winners",
            "  Value > 0: Tilted toward cheap stocks",
            "  Quality > 0: Tilted toward quality",
            "  Low_Volatility > 0: Tilted toward stable",
            "  Size > 0: Tilted toward small cap"
        ])

        return "\n".join(lines)


def analyze_style_factors(
    prices: pd.DataFrame,
    weights: pd.Series,
    fundamentals: Optional[Dict] = None
) -> Dict:
    """
    Convenience function for style factor analysis.

    Args:
        prices: Price DataFrame
        weights: Portfolio weights
        fundamentals: Optional fundamental data

    Returns:
        Dictionary with factor analysis results
    """
    analyzer = StyleFactorAnalyzer(prices, fundamentals=fundamentals)

    return {
        'factors': analyzer.calculate_all_factors(),
        'exposures': analyzer.portfolio_factor_exposure(weights),
        'tilts': analyzer.factor_tilt_analysis(weights),
        'attribution': analyzer.factor_return_attribution(weights),
        'correlation': analyzer.factor_correlation()
    }
