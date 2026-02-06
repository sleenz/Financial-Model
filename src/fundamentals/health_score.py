"""Financial health score calculation."""

from typing import Dict
from ..utils.logger import get_logger

logger = get_logger(__name__)


def calculate_health_score(valuation: Dict, profitability: Dict,
                          growth: Dict, financial_health: Dict) -> Dict:
    """
    Calculate overall financial health score (0-10).

    This is a transparent, explainable score based on fundamental metrics.
    Users can see exactly how the score is calculated.

    Args:
        valuation: Valuation metrics
        profitability: Profitability metrics
        growth: Growth metrics
        financial_health: Financial health metrics

    Returns:
        Dictionary with score and breakdown
    """
    try:
        score = 0.0
        breakdown = {}

        # === VALUATION (2 points max) ===
        valuation_score = 0.0
        peg = valuation.get('peg_ratio')

        if peg is not None:
            if peg < 1.0:
                valuation_score = 2.0
                breakdown['valuation'] = {'score': 2.0, 'reason': 'PEG < 1.0 (undervalued relative to growth)'}
            elif peg < 1.5:
                valuation_score = 1.5
                breakdown['valuation'] = {'score': 1.5, 'reason': 'PEG 1.0-1.5 (fairly valued)'}
            elif peg < 2.0:
                valuation_score = 1.0
                breakdown['valuation'] = {'score': 1.0, 'reason': 'PEG 1.5-2.0 (slightly overvalued)'}
            else:
                valuation_score = 0.5
                breakdown['valuation'] = {'score': 0.5, 'reason': 'PEG > 2.0 (overvalued)'}
        else:
            # Fallback to P/E if PEG not available
            pe = valuation.get('pe_ratio_ttm')
            if pe is not None:
                if pe < 15:
                    valuation_score = 2.0
                    breakdown['valuation'] = {'score': 2.0, 'reason': 'P/E < 15 (undervalued)'}
                elif pe < 25:
                    valuation_score = 1.0
                    breakdown['valuation'] = {'score': 1.0, 'reason': 'P/E 15-25 (fair)'}
                else:
                    valuation_score = 0.5
                    breakdown['valuation'] = {'score': 0.5, 'reason': 'P/E > 25 (expensive)'}

        score += valuation_score

        # === PROFITABILITY (2 points max) ===
        profitability_score = 0.0
        net_margin = profitability.get('net_margin')
        roe = profitability.get('roe')

        if net_margin is not None:
            if net_margin > 0.15:
                profitability_score += 1.0
            elif net_margin > 0.10:
                profitability_score += 0.75
            elif net_margin > 0.05:
                profitability_score += 0.5

        if roe is not None:
            if roe > 0.15:
                profitability_score += 1.0
            elif roe > 0.10:
                profitability_score += 0.75
            elif roe > 0.05:
                profitability_score += 0.5

        breakdown['profitability'] = {
            'score': profitability_score,
            'reason': f'Net Margin: {net_margin*100:.1f}%, ROE: {roe*100:.1f}%' if net_margin and roe else 'Limited data'
        }
        score += profitability_score

        # === GROWTH (2 points max) ===
        growth_score = 0.0
        revenue_growth = growth.get('revenue_growth') or growth.get('revenue_growth_yoy')

        if revenue_growth is not None:
            if revenue_growth > 0.15:
                growth_score = 2.0
                breakdown['growth'] = {'score': 2.0, 'reason': 'Revenue growth > 15% (high growth)'}
            elif revenue_growth > 0.10:
                growth_score = 1.5
                breakdown['growth'] = {'score': 1.5, 'reason': 'Revenue growth 10-15% (solid growth)'}
            elif revenue_growth > 0.05:
                growth_score = 1.0
                breakdown['growth'] = {'score': 1.0, 'reason': 'Revenue growth 5-10% (moderate growth)'}
            else:
                growth_score = 0.5
                breakdown['growth'] = {'score': 0.5, 'reason': 'Revenue growth < 5% (slow/stagnant)'}
        else:
            breakdown['growth'] = {'score': 0.0, 'reason': 'No growth data available'}

        score += growth_score

        # === FINANCIAL HEALTH (2 points max) ===
        health_score = 0.0
        current_ratio = financial_health.get('current_ratio')
        debt_to_equity = financial_health.get('debt_to_equity')

        # Current ratio (1 point max)
        if current_ratio is not None:
            if current_ratio > 2.0:
                health_score += 1.0
            elif current_ratio > 1.5:
                health_score += 0.75
            elif current_ratio > 1.0:
                health_score += 0.5
            else:
                health_score += 0.25

        # Debt to equity (1 point max)
        if debt_to_equity is not None:
            if debt_to_equity < 0.3:
                health_score += 1.0
            elif debt_to_equity < 0.5:
                health_score += 0.75
            elif debt_to_equity < 1.0:
                health_score += 0.5
            else:
                health_score += 0.25

        breakdown['financial_health'] = {
            'score': health_score,
            'reason': f'Current Ratio: {current_ratio:.1f}, Debt/Equity: {debt_to_equity:.1f}' if current_ratio and debt_to_equity else 'Limited data'
        }
        score += health_score

        # === EARNINGS QUALITY (2 points max) ===
        # If we have FCF data, compare to earnings growth
        fcf = financial_health.get('free_cash_flow')
        ocf = financial_health.get('operating_cash_flow')
        earnings_growth = growth.get('earnings_growth')

        quality_score = 0.0
        if fcf is not None and fcf > 0:
            quality_score += 1.0
            if ocf is not None and ocf > fcf:
                quality_score += 1.0
                breakdown['earnings_quality'] = {'score': 2.0, 'reason': 'Strong FCF generation'}
            else:
                breakdown['earnings_quality'] = {'score': 1.0, 'reason': 'Positive FCF'}
        else:
            breakdown['earnings_quality'] = {'score': 0.0, 'reason': 'No FCF data or negative FCF'}

        score += quality_score

        # Normalize to 0-10 scale
        max_score = 10.0
        normalized_score = min(score, max_score)

        # Determine rating
        if normalized_score >= 8.0:
            rating = 'Excellent'
        elif normalized_score >= 6.5:
            rating = 'Strong'
        elif normalized_score >= 5.0:
            rating = 'Moderate'
        elif normalized_score >= 3.5:
            rating = 'Fair'
        elif normalized_score >= 2.0:
            rating = 'Weak'
        else:
            rating = 'Poor'

        return {
            'score': round(normalized_score, 1),
            'rating': rating,
            'breakdown': breakdown,
            'max_score': max_score
        }

    except Exception as e:
        logger.error(f"Error calculating health score: {e}")
        return {
            'score': 0.0,
            'rating': 'Error',
            'breakdown': {},
            'max_score': 10.0
        }


def explain_health_score() -> str:
    """Return explanation of how health score is calculated."""
    return """
Financial Health Score Calculation (0-10):

1. Valuation (2 points)
   - PEG < 1.0: 2.0 points (undervalued)
   - PEG 1.0-1.5: 1.5 points (fair)
   - PEG 1.5-2.0: 1.0 points (expensive)
   - PEG > 2.0: 0.5 points (overvalued)

2. Profitability (2 points)
   - Net Margin > 15%: 1.0 point
   - ROE > 15%: 1.0 point

3. Growth (2 points)
   - Revenue growth > 15%: 2.0 points
   - Revenue growth 10-15%: 1.5 points
   - Revenue growth 5-10%: 1.0 points
   - Revenue growth < 5%: 0.5 points

4. Financial Health (2 points)
   - Current Ratio > 2.0: 1.0 point
   - Debt/Equity < 0.3: 1.0 point

5. Earnings Quality (2 points)
   - Positive Free Cash Flow: 1.0 point
   - OCF > FCF: 1.0 point

Rating Scale:
- 8.0-10.0: Excellent
- 6.5-7.9: Strong
- 5.0-6.4: Moderate
- 3.5-4.9: Fair
- 2.0-3.4: Weak
- 0.0-1.9: Poor
"""
