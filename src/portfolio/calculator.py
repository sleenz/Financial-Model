"""Position sizing and portfolio calculator."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime

from ..utils.logger import get_logger
from ..utils.helpers import format_currency, format_percentage

logger = get_logger(__name__)


class PositionCalculator:
    """
    Calculate position sizes, share counts, and allocation details.
    """

    def __init__(
        self,
        total_capital: float,
        weights: pd.Series,
        current_prices: pd.Series,
        allow_fractional: bool = False,
        commission_per_trade: float = 0.0,
        commission_per_share: float = 0.0,
        min_commission: float = 0.0,
    ):
        """
        Initialize position calculator.

        Args:
            total_capital: Total portfolio value in dollars
            weights: Target weights for each asset
            current_prices: Current prices for each asset
            allow_fractional: Whether to allow fractional shares
            commission_per_trade: Fixed commission per trade
            commission_per_share: Commission per share
            min_commission: Minimum commission per trade
        """
        self.total_capital = total_capital
        self.weights = weights
        self.current_prices = current_prices
        self.allow_fractional = allow_fractional
        self.commission_per_trade = commission_per_trade
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission

        # Validate inputs
        self._validate_inputs()

        logger.info(f"PositionCalculator initialized: ${total_capital:,.2f}, {len(weights)} assets")

    def _validate_inputs(self):
        """Validate input data."""
        # Check weights sum to approximately 1
        weight_sum = self.weights.sum()
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(f"Weights sum to {weight_sum:.4f}, normalizing to 1.0")
            self.weights = self.weights / weight_sum

        # Check for missing prices
        missing = set(self.weights.index) - set(self.current_prices.index)
        if missing:
            raise ValueError(f"Missing prices for: {missing}")

        # Check for zero or negative prices
        if (self.current_prices <= 0).any():
            raise ValueError("Prices must be positive")

    def calculate_positions(self) -> pd.DataFrame:
        """
        Calculate position sizes and share counts.

        Returns:
            DataFrame with position details
        """
        positions = []

        for ticker in self.weights.index:
            weight = self.weights[ticker]
            price = self.current_prices[ticker]

            # Target dollar amount
            target_amount = self.total_capital * weight

            # Calculate shares
            if self.allow_fractional:
                shares = target_amount / price
                actual_amount = target_amount
                remainder = 0
            else:
                shares = int(target_amount / price)
                actual_amount = shares * price
                remainder = target_amount - actual_amount

            # Calculate commission
            commission = self._calculate_commission(shares, actual_amount)

            positions.append({
                'Ticker': ticker,
                'Weight': weight,
                'Target Amount': target_amount,
                'Price': price,
                'Shares': shares,
                'Actual Amount': actual_amount,
                'Remainder': remainder,
                'Commission': commission,
                'Net Amount': actual_amount + commission,
            })

        df = pd.DataFrame(positions).set_index('Ticker')

        # Calculate actual weights after rounding
        total_invested = df['Actual Amount'].sum()
        df['Actual Weight'] = df['Actual Amount'] / total_invested if total_invested > 0 else 0

        return df

    def _calculate_commission(self, shares: float, amount: float) -> float:
        """Calculate trading commission."""
        if shares == 0:
            return 0

        commission = self.commission_per_trade + (shares * self.commission_per_share)
        return max(commission, self.min_commission)

    def get_summary(self) -> Dict:
        """
        Get portfolio allocation summary.

        Returns:
            Dictionary with summary statistics
        """
        positions = self.calculate_positions()

        total_invested = positions['Actual Amount'].sum()
        total_commission = positions['Commission'].sum()
        unallocated = self.total_capital - total_invested - total_commission

        return {
            'total_capital': self.total_capital,
            'total_invested': total_invested,
            'total_commission': total_commission,
            'unallocated_cash': unallocated,
            'unallocated_pct': unallocated / self.total_capital,
            'n_positions': (positions['Shares'] > 0).sum(),
            'avg_position_size': total_invested / (positions['Shares'] > 0).sum() if (positions['Shares'] > 0).any() else 0,
            'max_position': positions['Actual Amount'].max(),
            'min_position': positions[positions['Shares'] > 0]['Actual Amount'].min() if (positions['Shares'] > 0).any() else 0,
        }

    def format_report(
        self,
        expected_return: float = None,
        expected_volatility: float = None,
        sharpe_ratio: float = None,
    ) -> str:
        """
        Generate formatted allocation report.

        Args:
            expected_return: Expected annual return
            expected_volatility: Expected annual volatility
            sharpe_ratio: Portfolio Sharpe ratio

        Returns:
            Formatted report string
        """
        positions = self.calculate_positions()
        summary = self.get_summary()

        lines = [
            "=" * 60,
            "Portfolio Allocation Summary",
            "=" * 60,
            f"Total Capital: {format_currency(self.total_capital)}",
        ]

        if expected_return is not None:
            lines.append(f"Expected Annual Return: {format_percentage(expected_return)}")
        if expected_volatility is not None:
            lines.append(f"Expected Annual Volatility: {format_percentage(expected_volatility)}")
        if sharpe_ratio is not None:
            lines.append(f"Sharpe Ratio: {sharpe_ratio:.2f}")

        lines.extend([
            "",
            "Individual Positions:",
            "-" * 60,
        ])

        for ticker in positions.index:
            row = positions.loc[ticker]
            if row['Shares'] > 0:
                if self.allow_fractional:
                    shares_str = f"{row['Shares']:.4f}"
                else:
                    shares_str = f"{int(row['Shares'])}"

                lines.append(
                    f"{ticker:6s} - {row['Weight']*100:5.1f}% | "
                    f"{format_currency(row['Actual Amount']):>12s} | "
                    f"{shares_str:>8s} shares @ {format_currency(row['Price']):>10s} | "
                    f"Rem: {format_currency(row['Remainder'])}"
                )

        lines.extend([
            "",
            "-" * 60,
            f"Total Invested: {format_currency(summary['total_invested'])}",
            f"Total Commission: {format_currency(summary['total_commission'])}",
            f"Unallocated Cash: {format_currency(summary['unallocated_cash'])} "
            f"({format_percentage(summary['unallocated_pct'])})",
            "=" * 60,
        ])

        return "\n".join(lines)

    def optimize_for_minimum_remainder(self) -> pd.DataFrame:
        """
        Optimize allocation to minimize unallocated cash.

        Adjusts share counts to reduce remainder while staying within budget.

        Returns:
            Optimized positions DataFrame
        """
        positions = self.calculate_positions()

        if self.allow_fractional:
            return positions

        # Calculate current unallocated
        total_invested = positions['Actual Amount'].sum()
        unallocated = self.total_capital - total_invested

        # Try to buy additional shares with unallocated cash
        while unallocated > 0:
            # Find cheapest stock we can still buy
            affordable = positions[positions['Price'] <= unallocated]

            if affordable.empty:
                break

            # Buy one share of cheapest
            cheapest = affordable['Price'].idxmin()
            positions.loc[cheapest, 'Shares'] += 1
            positions.loc[cheapest, 'Actual Amount'] += positions.loc[cheapest, 'Price']
            positions.loc[cheapest, 'Remainder'] -= positions.loc[cheapest, 'Price']

            total_invested = positions['Actual Amount'].sum()
            unallocated = self.total_capital - total_invested

        # Recalculate actual weights
        positions['Actual Weight'] = positions['Actual Amount'] / total_invested

        return positions


class TaxLotOptimizer:
    """
    Optimize tax lots for selling positions.
    """

    def __init__(self, lots: pd.DataFrame):
        """
        Initialize tax lot optimizer.

        Args:
            lots: DataFrame with columns: ticker, purchase_date, shares, cost_basis
        """
        self.lots = lots

    def fifo(self, ticker: str, shares_to_sell: float) -> pd.DataFrame:
        """
        First In First Out selection.

        Args:
            ticker: Asset ticker
            shares_to_sell: Number of shares to sell

        Returns:
            Selected lots
        """
        ticker_lots = self.lots[self.lots['ticker'] == ticker].sort_values('purchase_date')
        return self._select_lots(ticker_lots, shares_to_sell)

    def lifo(self, ticker: str, shares_to_sell: float) -> pd.DataFrame:
        """
        Last In First Out selection.

        Args:
            ticker: Asset ticker
            shares_to_sell: Number of shares to sell

        Returns:
            Selected lots
        """
        ticker_lots = self.lots[self.lots['ticker'] == ticker].sort_values(
            'purchase_date', ascending=False
        )
        return self._select_lots(ticker_lots, shares_to_sell)

    def highest_cost(self, ticker: str, shares_to_sell: float) -> pd.DataFrame:
        """
        Highest cost basis first (minimize gains).

        Args:
            ticker: Asset ticker
            shares_to_sell: Number of shares to sell

        Returns:
            Selected lots
        """
        ticker_lots = self.lots[self.lots['ticker'] == ticker].sort_values(
            'cost_basis', ascending=False
        )
        return self._select_lots(ticker_lots, shares_to_sell)

    def lowest_cost(self, ticker: str, shares_to_sell: float) -> pd.DataFrame:
        """
        Lowest cost basis first (maximize gains for tax-loss harvesting).

        Args:
            ticker: Asset ticker
            shares_to_sell: Number of shares to sell

        Returns:
            Selected lots
        """
        ticker_lots = self.lots[self.lots['ticker'] == ticker].sort_values('cost_basis')
        return self._select_lots(ticker_lots, shares_to_sell)

    def tax_loss_harvest(
        self,
        ticker: str,
        current_price: float,
        shares_to_sell: float = None,
    ) -> pd.DataFrame:
        """
        Select lots with losses for tax-loss harvesting.

        Args:
            ticker: Asset ticker
            current_price: Current market price
            shares_to_sell: Number of shares (None = all losing lots)

        Returns:
            Selected lots with losses
        """
        ticker_lots = self.lots[self.lots['ticker'] == ticker].copy()
        ticker_lots['current_value'] = ticker_lots['shares'] * current_price
        ticker_lots['gain_loss'] = ticker_lots['current_value'] - (
            ticker_lots['shares'] * ticker_lots['cost_basis']
        )

        # Select only losing lots
        losing_lots = ticker_lots[ticker_lots['gain_loss'] < 0].sort_values('gain_loss')

        if shares_to_sell is None:
            return losing_lots

        return self._select_lots(losing_lots, shares_to_sell)

    def _select_lots(self, lots: pd.DataFrame, shares_to_sell: float) -> pd.DataFrame:
        """Select lots to meet share requirement."""
        selected = []
        remaining = shares_to_sell

        for _, lot in lots.iterrows():
            if remaining <= 0:
                break

            shares_from_lot = min(lot['shares'], remaining)
            selected.append({
                **lot.to_dict(),
                'shares_to_sell': shares_from_lot,
            })
            remaining -= shares_from_lot

        return pd.DataFrame(selected)


def calculate_positions(
    total_capital: float,
    weights: pd.Series,
    prices: pd.Series,
    allow_fractional: bool = False,
) -> pd.DataFrame:
    """
    Convenience function to calculate positions.

    Args:
        total_capital: Total portfolio value
        weights: Target weights
        prices: Current prices
        allow_fractional: Allow fractional shares

    Returns:
        Positions DataFrame
    """
    calc = PositionCalculator(total_capital, weights, prices, allow_fractional)
    return calc.calculate_positions()


def generate_allocation_report(
    total_capital: float,
    weights: pd.Series,
    prices: pd.Series,
    expected_return: float = None,
    expected_volatility: float = None,
    sharpe_ratio: float = None,
) -> str:
    """
    Generate allocation report.

    Args:
        total_capital: Total portfolio value
        weights: Target weights
        prices: Current prices
        expected_return: Expected return
        expected_volatility: Expected volatility
        sharpe_ratio: Sharpe ratio

    Returns:
        Formatted report string
    """
    calc = PositionCalculator(total_capital, weights, prices)
    return calc.format_report(expected_return, expected_volatility, sharpe_ratio)
