"""Current holdings tracking and portfolio diversity analysis."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger(__name__)


class HoldingsTracker:
    """
    Track current stock holdings and analyze portfolio diversity.
    """

    def __init__(self, holdings: Dict[str, float], current_prices: Optional[pd.Series] = None):
        """
        Initialize holdings tracker.

        Args:
            holdings: Dictionary mapping tickers to number of shares
                     Example: {'NVDA': 9, 'AAPL': 3, 'MSFT': 5}
            current_prices: Current prices for each holding (optional)
        """
        self.holdings = holdings
        self.current_prices = current_prices if current_prices is not None else pd.Series()

        logger.info(f"HoldingsTracker initialized with {len(holdings)} holdings")

    def get_holdings_dataframe(self) -> pd.DataFrame:
        """
        Get holdings as a DataFrame with calculated values.

        Returns:
            DataFrame with ticker, shares, price, value, and weight columns
        """
        data = []

        for ticker, shares in self.holdings.items():
            # Handle price lookup more robustly
            if len(self.current_prices) > 0:
                if ticker in self.current_prices.index:
                    _raw = self.current_prices[ticker]
                    # yfinance can return a single-element Series; extract scalar
                    price = float(_raw.iloc[0] if isinstance(_raw, pd.Series) else _raw)
                else:
                    logger.warning(f"No price found for {ticker}, using 0.0")
                    price = 0.0
            else:
                price = 0.0

            value = float(shares) * price

            data.append({
                'Ticker': ticker,
                'Shares': shares,
                'Price': price,
                'Value': value
            })

        df = pd.DataFrame(data).set_index('Ticker')

        # Calculate weights
        total_value = float(df['Value'].sum())
        if total_value > 0:
            df['Weight'] = df['Value'] / total_value
        else:
            logger.warning("Total portfolio value is 0, cannot calculate weights")
            df['Weight'] = 0.0

        return df

    def calculate_total_value(self) -> float:
        """
        Calculate total portfolio value.

        Returns:
            Total value in dollars
        """
        df = self.get_holdings_dataframe()
        return df['Value'].sum()

    def calculate_diversity_metrics(self) -> Dict:
        """
        Calculate comprehensive diversity metrics for the portfolio.

        Returns:
            Dictionary containing diversity metrics:
            - num_holdings: Number of unique stocks
            - herfindahl_index: Concentration measure (0-1, lower is more diverse)
            - effective_stocks: Effective number of stocks (inverse of HHI)
            - top_3_concentration: Percentage held in top 3 stocks
            - top_5_concentration: Percentage held in top 5 stocks
            - largest_position: Percentage in largest position
            - smallest_position: Percentage in smallest position
            - avg_position_size: Average position size as percentage
            - std_position_size: Standard deviation of position sizes
            - diversification_ratio: Measure of spread (higher is better)
            - gini_coefficient: Inequality measure (0-1, lower is more equal)
        """
        df = self.get_holdings_dataframe()

        if df.empty or df['Value'].sum() == 0:
            return {
                'num_holdings': 0,
                'herfindahl_index': 0.0,
                'effective_stocks': 0.0,
                'top_3_concentration': 0.0,
                'top_5_concentration': 0.0,
                'largest_position': 0.0,
                'smallest_position': 0.0,
                'avg_position_size': 0.0,
                'std_position_size': 0.0,
                'diversification_ratio': 0.0,
                'gini_coefficient': 0.0,
            }

        weights = df['Weight'].values
        weights_sorted = np.sort(weights)[::-1]  # Sort descending

        # Herfindahl-Hirschman Index (HHI)
        # Range: 1/N (perfectly diversified) to 1 (concentrated)
        hhi = np.sum(weights ** 2)

        # Effective number of stocks (1/HHI)
        effective_n = 1 / hhi if hhi > 0 else 0

        # Top N concentration
        top_3 = weights_sorted[:3].sum() if len(weights_sorted) >= 3 else weights_sorted.sum()
        top_5 = weights_sorted[:5].sum() if len(weights_sorted) >= 5 else weights_sorted.sum()

        # Position statistics
        largest = weights.max()
        smallest = weights[weights > 0].min() if (weights > 0).any() else 0
        avg_size = weights.mean()
        std_size = weights.std()

        # Diversification ratio: How spread out the positions are
        # Equal weight portfolio would have ratio = 1
        equal_weight = 1 / len(weights) if len(weights) > 0 else 0
        diversification_ratio = (1 - hhi) / (1 - equal_weight**2) if equal_weight > 0 else 0

        # Gini coefficient: Measure of inequality (0 = perfect equality, 1 = max inequality)
        gini = self._calculate_gini(weights_sorted)

        return {
            'num_holdings': len(self.holdings),
            'herfindahl_index': hhi,
            'effective_stocks': effective_n,
            'top_3_concentration': top_3,
            'top_5_concentration': top_5,
            'largest_position': largest,
            'smallest_position': smallest,
            'avg_position_size': avg_size,
            'std_position_size': std_size,
            'diversification_ratio': diversification_ratio,
            'gini_coefficient': gini,
        }

    def _calculate_gini(self, sorted_weights: np.ndarray) -> float:
        """
        Calculate Gini coefficient for portfolio concentration.

        Args:
            sorted_weights: Sorted array of portfolio weights

        Returns:
            Gini coefficient (0-1)
        """
        n = len(sorted_weights)
        if n == 0:
            return 0.0

        cumsum = np.cumsum(sorted_weights)
        sum_weights = cumsum[-1]

        if sum_weights == 0:
            return 0.0

        # Gini coefficient formula
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_weights)) / (n * sum_weights) - (n + 1) / n

        return gini

    def get_diversity_rating(self) -> str:
        """
        Get a qualitative rating of portfolio diversity.

        Returns:
            String rating: "Highly Concentrated", "Concentrated",
                          "Moderately Diversified", "Well Diversified", "Highly Diversified"
        """
        metrics = self.calculate_diversity_metrics()

        num_stocks = metrics['num_holdings']
        hhi = metrics['herfindahl_index']
        top_3 = metrics['top_3_concentration']

        # Rating based on multiple factors
        if num_stocks <= 2:
            return "Highly Concentrated"
        elif num_stocks <= 5 and (hhi > 0.3 or top_3 > 0.75):
            return "Concentrated"
        elif num_stocks <= 10 and (hhi > 0.2 or top_3 > 0.6):
            return "Moderately Diversified"
        elif num_stocks <= 20 and hhi <= 0.15:
            return "Well Diversified"
        elif num_stocks > 20 or (num_stocks > 10 and hhi <= 0.1):
            return "Highly Diversified"
        else:
            return "Moderately Diversified"

    def get_diversity_recommendations(self) -> List[str]:
        """
        Get recommendations for improving portfolio diversity.

        Returns:
            List of recommendation strings
        """
        metrics = self.calculate_diversity_metrics()
        recommendations = []

        num_stocks = metrics['num_holdings']
        hhi = metrics['herfindahl_index']
        top_3 = metrics['top_3_concentration']
        largest = metrics['largest_position']

        # Too few stocks
        if num_stocks < 5:
            recommendations.append(
                f"Consider adding more holdings. You have only {num_stocks} stocks. "
                "Most diversified portfolios have at least 10-20 stocks."
            )

        # Too concentrated in top positions
        if top_3 > 0.70:
            recommendations.append(
                f"Your top 3 holdings represent {top_3*100:.1f}% of your portfolio. "
                "Consider rebalancing to reduce concentration risk."
            )

        # Single position too large
        if largest > 0.40:
            recommendations.append(
                f"Your largest position is {largest*100:.1f}% of your portfolio. "
                "Consider reducing positions over 25-30% to limit single-stock risk."
            )

        # High HHI
        if hhi > 0.25 and num_stocks >= 5:
            recommendations.append(
                f"Your portfolio shows high concentration (HHI: {hhi:.3f}). "
                "Consider equalizing position sizes or adding more stocks."
            )

        # Good diversity
        if not recommendations:
            recommendations.append(
                "Your portfolio shows good diversification! "
                f"With {num_stocks} holdings and balanced weights, you have reasonable risk distribution."
            )

        return recommendations

    def format_holdings_report(self) -> str:
        """
        Generate a formatted report of holdings and diversity metrics.

        Returns:
            Formatted report string
        """
        df = self.get_holdings_dataframe()
        metrics = self.calculate_diversity_metrics()
        rating = self.get_diversity_rating()
        recommendations = self.get_diversity_recommendations()

        total_value = self.calculate_total_value()

        lines = [
            "=" * 70,
            "PORTFOLIO HOLDINGS REPORT",
            "=" * 70,
            f"Total Portfolio Value: ${total_value:,.2f}",
            f"Number of Holdings: {metrics['num_holdings']}",
            f"Diversity Rating: {rating}",
            "",
            "CURRENT HOLDINGS:",
            "-" * 70,
        ]

        # Sort by value descending
        df_sorted = df.sort_values('Value', ascending=False)

        for ticker in df_sorted.index:
            row = df_sorted.loc[ticker]
            lines.append(
                f"{ticker:8s} | {row['Shares']:>8.2f} shares @ ${row['Price']:>8.2f} | "
                f"${row['Value']:>12,.2f} ({row['Weight']*100:>5.1f}%)"
            )

        lines.extend([
            "",
            "DIVERSITY METRICS:",
            "-" * 70,
            f"Herfindahl Index (HHI):     {metrics['herfindahl_index']:.4f}  (lower = more diverse)",
            f"Effective Number of Stocks: {metrics['effective_stocks']:.2f}",
            f"Top 3 Concentration:        {metrics['top_3_concentration']*100:.1f}%",
            f"Top 5 Concentration:        {metrics['top_5_concentration']*100:.1f}%",
            f"Largest Position:           {metrics['largest_position']*100:.1f}%",
            f"Smallest Position:          {metrics['smallest_position']*100:.1f}%",
            f"Average Position Size:      {metrics['avg_position_size']*100:.1f}%",
            f"Gini Coefficient:           {metrics['gini_coefficient']:.4f}  (lower = more equal)",
            "",
            "RECOMMENDATIONS:",
            "-" * 70,
        ])

        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")

        lines.append("=" * 70)

        return "\n".join(lines)


def create_holdings_from_dict(holdings_dict: Dict[str, float]) -> HoldingsTracker:
    """
    Convenience function to create HoldingsTracker from dictionary.

    Args:
        holdings_dict: Dictionary of ticker -> shares

    Returns:
        HoldingsTracker instance
    """
    return HoldingsTracker(holdings_dict)


def analyze_portfolio_diversity(holdings: Dict[str, float], prices: pd.Series = None) -> Dict:
    """
    Convenience function to analyze portfolio diversity.

    Args:
        holdings: Dictionary of ticker -> shares
        prices: Current prices (optional)

    Returns:
        Dictionary of diversity metrics
    """
    tracker = HoldingsTracker(holdings, prices)
    return tracker.calculate_diversity_metrics()
