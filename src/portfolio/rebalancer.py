"""Portfolio rebalancing logic and recommendations."""

from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from ..utils.logger import get_logger
from ..utils.helpers import format_currency, format_percentage

logger = get_logger(__name__)


class PortfolioRebalancer:
    """
    Determine when and how to rebalance a portfolio.
    """

    def __init__(
        self,
        target_weights: pd.Series,
        current_weights: pd.Series,
        current_prices: pd.Series,
        portfolio_value: float,
        threshold_pct: float = 0.05,
        min_trade_value: float = 100,
        commission_per_trade: float = 0,
    ):
        """
        Initialize rebalancer.

        Args:
            target_weights: Target allocation weights
            current_weights: Current allocation weights
            current_prices: Current asset prices
            portfolio_value: Total portfolio value
            threshold_pct: Rebalancing threshold (e.g., 0.05 = 5%)
            min_trade_value: Minimum trade value to execute
            commission_per_trade: Commission per trade
        """
        self.target_weights = target_weights
        self.current_weights = current_weights
        self.current_prices = current_prices
        self.portfolio_value = portfolio_value
        self.threshold_pct = threshold_pct
        self.min_trade_value = min_trade_value
        self.commission_per_trade = commission_per_trade

        # Align indices
        self.tickers = list(set(target_weights.index) | set(current_weights.index))

        logger.debug(f"Rebalancer initialized with {len(self.tickers)} assets")

    def needs_rebalancing(self) -> Tuple[bool, Dict]:
        """
        Check if portfolio needs rebalancing.

        Returns:
            Tuple of (needs_rebalance, details)
        """
        deviations = {}
        max_deviation = 0

        for ticker in self.tickers:
            target = self.target_weights.get(ticker, 0)
            current = self.current_weights.get(ticker, 0)
            deviation = abs(current - target)

            deviations[ticker] = {
                'target': target,
                'current': current,
                'deviation': deviation,
                'exceeds_threshold': deviation > self.threshold_pct,
            }

            max_deviation = max(max_deviation, deviation)

        needs_rebalance = max_deviation > self.threshold_pct

        return needs_rebalance, {
            'max_deviation': max_deviation,
            'threshold': self.threshold_pct,
            'deviations': deviations,
        }

    def calculate_trades(self) -> pd.DataFrame:
        """
        Calculate trades needed to rebalance.

        Returns:
            DataFrame with trade recommendations
        """
        trades = []

        for ticker in self.tickers:
            target = self.target_weights.get(ticker, 0)
            current = self.current_weights.get(ticker, 0)

            target_value = self.portfolio_value * target
            current_value = self.portfolio_value * current
            trade_value = target_value - current_value

            # Get price
            price = self.current_prices.get(ticker, 0)
            if price > 0:
                shares = trade_value / price
            else:
                shares = 0

            # Determine action
            if trade_value > self.min_trade_value:
                action = 'BUY'
            elif trade_value < -self.min_trade_value:
                action = 'SELL'
            else:
                action = 'HOLD'

            trades.append({
                'Ticker': ticker,
                'Current Weight': current,
                'Target Weight': target,
                'Current Value': current_value,
                'Target Value': target_value,
                'Trade Value': trade_value,
                'Price': price,
                'Shares': shares,
                'Action': action,
            })

        df = pd.DataFrame(trades).set_index('Ticker')

        # Sort by trade value (sells first, then buys)
        df = df.sort_values('Trade Value')

        return df

    def get_trade_summary(self) -> Dict:
        """
        Get summary of rebalancing trades.

        Returns:
            Dictionary with trade summary
        """
        trades = self.calculate_trades()

        buys = trades[trades['Action'] == 'BUY']
        sells = trades[trades['Action'] == 'SELL']

        total_buy = buys['Trade Value'].sum()
        total_sell = abs(sells['Trade Value'].sum())
        n_trades = (trades['Action'] != 'HOLD').sum()
        total_commission = n_trades * self.commission_per_trade

        return {
            'n_buys': len(buys),
            'n_sells': len(sells),
            'n_holds': len(trades) - len(buys) - len(sells),
            'total_buy_value': total_buy,
            'total_sell_value': total_sell,
            'net_cash_flow': total_sell - total_buy,
            'total_turnover': (total_buy + total_sell) / 2,
            'turnover_pct': (total_buy + total_sell) / 2 / self.portfolio_value,
            'total_commission': total_commission,
        }

    def format_recommendations(self) -> str:
        """
        Generate formatted rebalancing recommendations.

        Returns:
            Formatted recommendation string
        """
        needs_rebal, details = self.needs_rebalancing()
        trades = self.calculate_trades()
        summary = self.get_trade_summary()

        lines = [
            "=" * 60,
            "Rebalancing Recommendations",
            "=" * 60,
            f"Portfolio Value: {format_currency(self.portfolio_value)}",
            f"Rebalancing Threshold: {format_percentage(self.threshold_pct)}",
            f"Maximum Deviation: {format_percentage(details['max_deviation'])}",
            f"Rebalancing Needed: {'YES' if needs_rebal else 'NO'}",
            "",
        ]

        if needs_rebal:
            lines.extend([
                "Recommended Trades:",
                "-" * 60,
            ])

            for ticker in trades.index:
                row = trades.loc[ticker]
                if row['Action'] != 'HOLD':
                    lines.append(
                        f"{row['Action']:4s} {ticker:6s} | "
                        f"{abs(row['Shares']):>8.2f} shares | "
                        f"{format_currency(abs(row['Trade Value'])):>12s} | "
                        f"{format_percentage(row['Current Weight'])} -> "
                        f"{format_percentage(row['Target Weight'])}"
                    )

            lines.extend([
                "",
                "-" * 60,
                f"Total Buys: {format_currency(summary['total_buy_value'])} ({summary['n_buys']} trades)",
                f"Total Sells: {format_currency(summary['total_sell_value'])} ({summary['n_sells']} trades)",
                f"Net Cash Flow: {format_currency(summary['net_cash_flow'])}",
                f"Total Turnover: {format_percentage(summary['turnover_pct'])}",
                f"Total Commission: {format_currency(summary['total_commission'])}",
            ])
        else:
            lines.append("Portfolio is within rebalancing thresholds. No trades needed.")

        lines.append("=" * 60)

        return "\n".join(lines)

    def optimize_trades(self, available_cash: float = 0) -> pd.DataFrame:
        """
        Optimize trades considering available cash.

        Prioritizes trades that can be funded by sells plus available cash.

        Args:
            available_cash: Additional cash available for buys

        Returns:
            Optimized trades DataFrame
        """
        trades = self.calculate_trades()

        # Calculate sell proceeds
        sells = trades[trades['Action'] == 'SELL']
        sell_proceeds = abs(sells['Trade Value'].sum())

        # Total available for buys
        total_available = sell_proceeds + available_cash

        # Prioritize buys by deviation from target
        buys = trades[trades['Action'] == 'BUY'].copy()
        buys['Deviation'] = abs(buys['Target Weight'] - buys['Current Weight'])
        buys = buys.sort_values('Deviation', ascending=False)

        # Allocate available funds to buys
        allocated = 0
        for ticker in buys.index:
            trade_value = buys.loc[ticker, 'Trade Value']
            if allocated + trade_value <= total_available:
                allocated += trade_value
            else:
                # Partial allocation
                remaining = total_available - allocated
                if remaining >= self.min_trade_value:
                    buys.loc[ticker, 'Trade Value'] = remaining
                    buys.loc[ticker, 'Shares'] = remaining / buys.loc[ticker, 'Price']
                else:
                    buys.loc[ticker, 'Action'] = 'HOLD'
                    buys.loc[ticker, 'Trade Value'] = 0
                    buys.loc[ticker, 'Shares'] = 0
                allocated = total_available

        # Combine sells and optimized buys
        result = pd.concat([sells, buys]).drop(columns=['Deviation'], errors='ignore')
        return result.sort_values('Trade Value')


class DCAScheduler:
    """
    Dollar-cost averaging scheduler.
    """

    def __init__(
        self,
        total_amount: float,
        target_weights: pd.Series,
        frequency: str = 'monthly',
        n_periods: int = 12,
    ):
        """
        Initialize DCA scheduler.

        Args:
            total_amount: Total amount to invest
            target_weights: Target allocation weights
            frequency: 'weekly', 'biweekly', 'monthly'
            n_periods: Number of periods to spread investment
        """
        self.total_amount = total_amount
        self.target_weights = target_weights
        self.frequency = frequency
        self.n_periods = n_periods

        self.amount_per_period = total_amount / n_periods

    def generate_schedule(self, start_date: datetime = None) -> pd.DataFrame:
        """
        Generate investment schedule.

        Args:
            start_date: Start date (default: today)

        Returns:
            DataFrame with investment schedule
        """
        if start_date is None:
            start_date = datetime.now()

        # Determine period delta
        freq_map = {
            'weekly': timedelta(weeks=1),
            'biweekly': timedelta(weeks=2),
            'monthly': timedelta(days=30),
        }
        delta = freq_map.get(self.frequency, timedelta(days=30))

        schedule = []
        for i in range(self.n_periods):
            date = start_date + (delta * i)

            period_data = {
                'Period': i + 1,
                'Date': date.date(),
                'Total Amount': self.amount_per_period,
            }

            # Add per-asset amounts
            for ticker in self.target_weights.index:
                period_data[ticker] = self.amount_per_period * self.target_weights[ticker]

            schedule.append(period_data)

        return pd.DataFrame(schedule)

    def get_summary(self) -> Dict:
        """
        Get DCA schedule summary.

        Returns:
            Dictionary with summary
        """
        return {
            'total_amount': self.total_amount,
            'n_periods': self.n_periods,
            'amount_per_period': self.amount_per_period,
            'frequency': self.frequency,
            'assets': list(self.target_weights.index),
        }


class PerformanceAttributor:
    """
    Analyze portfolio performance attribution.
    """

    def __init__(
        self,
        portfolio_returns: pd.Series,
        asset_returns: pd.DataFrame,
        weights: pd.Series,
        benchmark_returns: pd.Series = None,
    ):
        """
        Initialize performance attributor.

        Args:
            portfolio_returns: Portfolio return series
            asset_returns: Individual asset returns
            weights: Portfolio weights
            benchmark_returns: Benchmark return series
        """
        self.portfolio_returns = portfolio_returns
        self.asset_returns = asset_returns
        self.weights = weights
        self.benchmark_returns = benchmark_returns

    def allocation_effect(self) -> pd.Series:
        """
        Calculate allocation effect (weight decisions).

        Returns:
            Allocation effect by asset
        """
        if self.benchmark_returns is None:
            return pd.Series(index=self.asset_returns.columns, data=0)

        # Simplified: allocation effect = (portfolio_weight - benchmark_weight) * benchmark_return
        # Assuming equal weight benchmark
        n_assets = len(self.weights)
        benchmark_weight = 1 / n_assets

        allocation = (self.weights - benchmark_weight) * self.asset_returns.mean()
        return allocation

    def selection_effect(self) -> pd.Series:
        """
        Calculate selection effect (asset performance).

        Returns:
            Selection effect by asset
        """
        # Selection = weight * (asset_return - benchmark_return)
        if self.benchmark_returns is None:
            benchmark_return = self.asset_returns.mean().mean()
        else:
            benchmark_return = self.benchmark_returns.mean()

        selection = self.weights * (self.asset_returns.mean() - benchmark_return)
        return selection

    def contribution_analysis(self) -> pd.DataFrame:
        """
        Analyze return contribution by asset.

        Returns:
            DataFrame with contribution analysis
        """
        # Return contribution = weight * asset_return
        contribution = self.weights * self.asset_returns.mean()

        total_return = contribution.sum()

        return pd.DataFrame({
            'Weight': self.weights,
            'Return': self.asset_returns.mean(),
            'Contribution': contribution,
            'Contribution %': contribution / total_return * 100 if total_return != 0 else 0,
        })

    def risk_contribution(self) -> pd.DataFrame:
        """
        Analyze risk contribution by asset.

        Returns:
            DataFrame with risk contribution
        """
        cov_matrix = self.asset_returns.cov()
        portfolio_vol = np.sqrt(np.dot(self.weights.T, np.dot(cov_matrix, self.weights)))

        # Marginal contribution to risk
        marginal = np.dot(cov_matrix, self.weights) / portfolio_vol
        risk_contrib = self.weights * marginal

        return pd.DataFrame({
            'Weight': self.weights,
            'Volatility': self.asset_returns.std(),
            'Risk Contribution': risk_contrib,
            'Risk Contribution %': risk_contrib / risk_contrib.sum() * 100,
        })


def check_rebalancing(
    target_weights: pd.Series,
    current_weights: pd.Series,
    threshold: float = 0.05,
) -> Tuple[bool, float]:
    """
    Quick check if rebalancing is needed.

    Args:
        target_weights: Target weights
        current_weights: Current weights
        threshold: Rebalancing threshold

    Returns:
        Tuple of (needs_rebalancing, max_deviation)
    """
    deviations = abs(target_weights - current_weights)
    max_deviation = deviations.max()
    return max_deviation > threshold, max_deviation


def calculate_turnover(
    old_weights: pd.Series,
    new_weights: pd.Series,
) -> float:
    """
    Calculate portfolio turnover.

    Args:
        old_weights: Previous weights
        new_weights: New weights

    Returns:
        Turnover as fraction
    """
    return abs(new_weights - old_weights).sum() / 2
