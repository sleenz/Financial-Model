"""Fundamentals metrics calculations."""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from ..utils.logger import get_logger

logger = get_logger(__name__)


def calculate_valuation_metrics(info: Dict) -> Dict:
    """
    Calculate valuation metrics from stock info.

    Args:
        info: Stock info dictionary from yfinance

    Returns:
        Dictionary of valuation metrics
    """
    try:
        metrics = {
            'pe_ratio_ttm': info.get('trailingPE'),
            'pe_forward': info.get('forwardPE'),
            'peg_ratio': info.get('pegRatio'),
            'price_to_book': info.get('priceToBook'),
            'ev_to_ebitda': info.get('enterpriseToEbitda'),
            'price_to_sales': info.get('priceToSalesTrailing12Months'),
            'market_cap': info.get('marketCap'),
            'enterprise_value': info.get('enterpriseValue'),
        }

        # Calculate vs sector if available
        sector_pe = info.get('sectorPE')
        if sector_pe and metrics['pe_ratio_ttm']:
            metrics['vs_sector'] = metrics['pe_ratio_ttm'] / sector_pe
        else:
            metrics['vs_sector'] = None

        # Determine valuation signal
        pe = metrics['pe_ratio_ttm']
        peg = metrics['peg_ratio']

        if pe and pe < 15:
            metrics['valuation_signal'] = 'undervalued'
        elif pe and pe > 25:
            metrics['valuation_signal'] = 'overvalued'
        elif peg and peg < 1.0:
            metrics['valuation_signal'] = 'undervalued'
        elif peg and peg > 2.0:
            metrics['valuation_signal'] = 'overvalued'
        else:
            metrics['valuation_signal'] = 'fair'

        return metrics

    except Exception as e:
        logger.error(f"Error calculating valuation metrics: {e}")
        return {}


def calculate_profitability_metrics(info: Dict, financials: Optional[pd.DataFrame] = None) -> Dict:
    """
    Calculate profitability metrics.

    Args:
        info: Stock info dictionary
        financials: Financial statements DataFrame

    Returns:
        Dictionary of profitability metrics
    """
    try:
        metrics = {
            'gross_margin': info.get('grossMargins'),
            'operating_margin': info.get('operatingMargins'),
            'net_margin': info.get('profitMargins'),
            'roe': info.get('returnOnEquity'),
            'roa': info.get('returnOnAssets'),
            'roic': info.get('returnOnCapital'),
        }

        # Calculate trends if financials available
        if financials is not None and not financials.empty:
            try:
                # Get margins from financials for trend analysis
                if 'Gross Profit' in financials.index and 'Total Revenue' in financials.index:
                    revenue = financials.loc['Total Revenue']
                    gross_profit = financials.loc['Gross Profit']

                    # Calculate margin trend (most recent 2 periods)
                    if len(revenue) >= 2:
                        current_margin = gross_profit.iloc[0] / revenue.iloc[0]
                        prior_margin = gross_profit.iloc[1] / revenue.iloc[1]
                        metrics['gross_margin_trend'] = current_margin - prior_margin

            except Exception as e:
                logger.warning(f"Could not calculate margin trends: {e}")
                metrics['gross_margin_trend'] = None

        # Determine profitability signal
        net_margin = metrics['net_margin']
        roe = metrics['roe']

        if net_margin and net_margin > 0.15 and roe and roe > 0.15:
            metrics['profitability_signal'] = 'strong'
        elif net_margin and net_margin > 0.10:
            metrics['profitability_signal'] = 'moderate'
        elif net_margin and net_margin > 0.05:
            metrics['profitability_signal'] = 'weak'
        else:
            metrics['profitability_signal'] = 'poor'

        return metrics

    except Exception as e:
        logger.error(f"Error calculating profitability metrics: {e}")
        return {}


def calculate_growth_metrics(info: Dict, financials: Optional[pd.DataFrame] = None) -> Dict:
    """
    Calculate growth metrics.

    Args:
        info: Stock info dictionary
        financials: Financial statements DataFrame

    Returns:
        Dictionary of growth metrics
    """
    try:
        metrics = {
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth'),
        }

        # Calculate YoY and QoQ if financials available
        if financials is not None and not financials.empty:
            try:
                if 'Total Revenue' in financials.index:
                    revenue = financials.loc['Total Revenue']

                    # QoQ (most recent quarter vs previous)
                    if len(revenue) >= 2:
                        metrics['revenue_growth_qoq'] = (revenue.iloc[0] / revenue.iloc[1]) - 1

                    # YoY (most recent vs 4 quarters ago)
                    if len(revenue) >= 5:
                        metrics['revenue_growth_yoy'] = (revenue.iloc[0] / revenue.iloc[4]) - 1

                if 'Net Income' in financials.index:
                    net_income = financials.loc['Net Income']

                    if len(net_income) >= 2:
                        current = net_income.iloc[0]
                        prior = net_income.iloc[1]
                        if prior != 0:
                            metrics['earnings_growth_qoq'] = (current / prior) - 1

            except Exception as e:
                logger.warning(f"Could not calculate detailed growth metrics: {e}")

        # Determine growth signal
        revenue_growth = metrics.get('revenue_growth')

        if revenue_growth and revenue_growth > 0.15:
            metrics['growth_signal'] = 'high_growth'
        elif revenue_growth and revenue_growth > 0.10:
            metrics['growth_signal'] = 'growing'
        elif revenue_growth and revenue_growth > 0.05:
            metrics['growth_signal'] = 'slow_growth'
        else:
            metrics['growth_signal'] = 'stagnant'

        return metrics

    except Exception as e:
        logger.error(f"Error calculating growth metrics: {e}")
        return {}


def calculate_financial_health_metrics(balance_sheet: Optional[pd.DataFrame],
                                       cash_flow: Optional[pd.DataFrame],
                                       info: Dict) -> Dict:
    """
    Calculate financial health metrics.

    Args:
        balance_sheet: Balance sheet DataFrame
        cash_flow: Cash flow DataFrame
        info: Stock info dictionary

    Returns:
        Dictionary of financial health metrics
    """
    try:
        metrics = {
            'current_ratio': info.get('currentRatio'),
            'quick_ratio': info.get('quickRatio'),
            'debt_to_equity': info.get('debtToEquity'),
            'total_cash': info.get('totalCash'),
            'total_debt': info.get('totalDebt'),
            'free_cash_flow': info.get('freeCashflow'),
            'operating_cash_flow': info.get('operatingCashflow'),
        }

        # Calculate derived metrics
        if balance_sheet is not None and not balance_sheet.empty:
            try:
                # Cash to assets ratio
                if 'Total Assets' in balance_sheet.index and 'Cash' in balance_sheet.index:
                    total_assets = balance_sheet.loc['Total Assets'].iloc[0]
                    cash = balance_sheet.loc['Cash'].iloc[0]
                    metrics['cash_to_assets'] = cash / total_assets if total_assets != 0 else None

            except Exception as e:
                logger.warning(f"Could not calculate balance sheet ratios: {e}")

        # Interest coverage from cash flow and income
        if info.get('ebitda') and info.get('interestExpense'):
            ebitda = info['ebitda']
            interest = abs(info['interestExpense'])
            metrics['interest_coverage'] = ebitda / interest if interest != 0 else None
        else:
            metrics['interest_coverage'] = None

        # Operating cash flow to debt
        if metrics['operating_cash_flow'] and metrics['total_debt']:
            metrics['ocf_to_debt'] = metrics['operating_cash_flow'] / metrics['total_debt']
        else:
            metrics['ocf_to_debt'] = None

        # Determine health signal
        current_ratio = metrics['current_ratio']
        debt_to_equity = metrics['debt_to_equity']

        if current_ratio and current_ratio > 1.5 and debt_to_equity and debt_to_equity < 0.5:
            metrics['health_signal'] = 'strong'
        elif current_ratio and current_ratio > 1.0 and debt_to_equity and debt_to_equity < 1.0:
            metrics['health_signal'] = 'moderate'
        elif current_ratio and current_ratio > 1.0:
            metrics['health_signal'] = 'fair'
        else:
            metrics['health_signal'] = 'weak'

        return metrics

    except Exception as e:
        logger.error(f"Error calculating financial health metrics: {e}")
        return {}
