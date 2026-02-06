"""Earnings analysis and insights."""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EarningsAnalyzer:
    """Analyze earnings history and generate insights."""

    def __init__(self, quarterly_earnings: pd.DataFrame):
        """
        Initialize earnings analyzer.

        Args:
            quarterly_earnings: DataFrame with quarterly earnings data
        """
        self.earnings = quarterly_earnings
        logger.info(f"EarningsAnalyzer initialized with {len(quarterly_earnings)} quarters")

    def get_last_n_quarters(self, n: int = 4) -> pd.DataFrame:
        """Get last N quarters of earnings data."""
        return self.earnings.head(n)

    def calculate_earnings_trends(self, n: int = 4) -> Dict:
        """
        Calculate earnings trends for last N quarters.

        Args:
            n: Number of quarters to analyze

        Returns:
            Dictionary with trend data
        """
        try:
            recent = self.get_last_n_quarters(n)

            if recent.empty:
                return {}

            trends = {
                'quarters': [],
                'eps_actual': [],
                'eps_estimate': [],
                'revenue': [],
                'qoq_eps_change': [],
                'qoq_revenue_change': [],
                'beat_or_miss': [],
            }

            for idx, row in recent.iterrows():
                # Get quarter info
                quarter_date = row.name if hasattr(row.name, 'strftime') else idx

                trends['quarters'].append(quarter_date)
                trends['eps_actual'].append(row.get('Earnings', None))
                trends['eps_estimate'].append(row.get('Estimate', None))
                trends['revenue'].append(row.get('Revenue', None))

                # Calculate beat/miss
                if pd.notna(row.get('Earnings')) and pd.notna(row.get('Estimate')):
                    actual = row['Earnings']
                    estimate = row['Estimate']
                    if actual > estimate:
                        trends['beat_or_miss'].append('beat')
                    elif actual < estimate:
                        trends['beat_or_miss'].append('miss')
                    else:
                        trends['beat_or_miss'].append('meet')
                else:
                    trends['beat_or_miss'].append(None)

            # Calculate QoQ changes
            eps_list = [e for e in trends['eps_actual'] if pd.notna(e)]
            revenue_list = [r for r in trends['revenue'] if pd.notna(r)]

            for i in range(len(eps_list)):
                if i > 0 and eps_list[i-1] != 0:
                    qoq_eps = (eps_list[i] - eps_list[i-1]) / abs(eps_list[i-1])
                    trends['qoq_eps_change'].append(qoq_eps)
                else:
                    trends['qoq_eps_change'].append(None)

            for i in range(len(revenue_list)):
                if i > 0 and revenue_list[i-1] != 0:
                    qoq_rev = (revenue_list[i] - revenue_list[i-1]) / abs(revenue_list[i-1])
                    trends['qoq_revenue_change'].append(qoq_rev)
                else:
                    trends['qoq_revenue_change'].append(None)

            return trends

        except Exception as e:
            logger.error(f"Error calculating earnings trends: {e}")
            return {}

    def generate_insights(self, n: int = 4) -> Dict:
        """
        Generate insights from earnings data.

        Args:
            n: Number of quarters to analyze

        Returns:
            Dictionary with insights
        """
        try:
            trends = self.calculate_earnings_trends(n)

            if not trends or not trends.get('beat_or_miss'):
                return {
                    'beat_streak': 0,
                    'momentum': 'unknown',
                    'revenue_trend': 'unknown',
                    'alerts': []
                }

            # Beat streak
            beat_or_miss = [b for b in trends['beat_or_miss'] if b is not None]
            beat_streak = 0
            for result in reversed(beat_or_miss):
                if result == 'beat':
                    beat_streak += 1
                else:
                    break

            # Momentum analysis
            qoq_changes = [c for c in trends['qoq_eps_change'] if c is not None]
            if len(qoq_changes) >= 2:
                recent_change = qoq_changes[0]
                prior_change = qoq_changes[1]

                if recent_change > prior_change:
                    momentum = 'accelerating'
                elif abs(recent_change - prior_change) < 0.02:  # Within 2%
                    momentum = 'stable'
                else:
                    momentum = 'decelerating'
            else:
                momentum = 'insufficient_data'

            # Revenue trend
            revenue_changes = [c for c in trends['qoq_revenue_change'] if c is not None]
            if revenue_changes:
                avg_revenue_change = np.mean(revenue_changes)
                if avg_revenue_change > 0.10:
                    revenue_trend = 'strong_growth'
                elif avg_revenue_change > 0.05:
                    revenue_trend = 'growing'
                elif avg_revenue_change > 0:
                    revenue_trend = 'slow_growth'
                else:
                    revenue_trend = 'declining'
            else:
                revenue_trend = 'insufficient_data'

            # Generate alerts
            alerts = []

            # Alert: First miss after streak
            if len(beat_or_miss) > 1 and beat_or_miss[0] == 'miss' and beat_or_miss[1] == 'beat':
                # Count consecutive beats before this miss
                prior_beats = 0
                for i in range(1, len(beat_or_miss)):
                    if beat_or_miss[i] == 'beat':
                        prior_beats += 1
                    else:
                        break

                if prior_beats >= 3:
                    alerts.append({
                        'type': 'warning',
                        'message': f'First earnings miss after {prior_beats} consecutive beats'
                    })

            # Alert: Momentum shift
            if momentum == 'decelerating' and len(qoq_changes) >= 2:
                alerts.append({
                    'type': 'info',
                    'message': 'Earnings momentum decelerating - watch next quarter'
                })

            # Alert: Revenue declining while earnings growing
            if revenue_trend == 'declining' and momentum == 'accelerating':
                alerts.append({
                    'type': 'warning',
                    'message': 'Earnings growing but revenue declining - check margin sustainability'
                })

            return {
                'beat_streak': beat_streak,
                'momentum': momentum,
                'revenue_trend': revenue_trend,
                'alerts': alerts,
                'quarters_analyzed': len(beat_or_miss)
            }

        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {
                'beat_streak': 0,
                'momentum': 'error',
                'revenue_trend': 'error',
                'alerts': []
            }

    def format_earnings_summary(self) -> str:
        """Generate formatted earnings summary."""
        trends = self.calculate_earnings_trends(4)
        insights = self.generate_insights(4)

        lines = [
            "=" * 60,
            "Earnings Summary (Last 4 Quarters)",
            "=" * 60,
        ]

        if trends.get('quarters'):
            for i in range(len(trends['quarters'])):
                quarter = trends['quarters'][i]
                eps = trends['eps_actual'][i]
                beat_miss = trends['beat_or_miss'][i]

                if eps is not None:
                    beat_symbol = "✓" if beat_miss == 'beat' else "✗" if beat_miss == 'miss' else "="
                    lines.append(f"{quarter}: EPS ${eps:.2f} {beat_symbol}")

        lines.extend([
            "",
            f"Beat Streak: {insights['beat_streak']} quarters",
            f"Momentum: {insights['momentum']}",
            f"Revenue Trend: {insights['revenue_trend']}",
            "",
        ])

        if insights['alerts']:
            lines.append("Alerts:")
            for alert in insights['alerts']:
                lines.append(f"  {alert['type'].upper()}: {alert['message']}")

        lines.append("=" * 60)

        return "\n".join(lines)
