"""Main fundamentals analyzer class."""

from typing import Dict, Optional
from .metrics import (
    calculate_valuation_metrics,
    calculate_profitability_metrics,
    calculate_growth_metrics,
    calculate_financial_health_metrics
)
from .earnings import EarningsAnalyzer
from .health_score import calculate_health_score
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FundamentalsAnalyzer:
    """
    Main class for analyzing company fundamentals.

    Fetches and analyzes:
    - Valuation metrics
    - Profitability metrics
    - Growth metrics
    - Financial health
    - Earnings history
    - Overall health score
    """

    def __init__(self, ticker: str):
        """
        Initialize fundamentals analyzer.

        Args:
            ticker: Stock ticker symbol
        """
        import yfinance as yf

        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self.info = None
        self.financials = None
        self.balance_sheet = None
        self.cash_flow = None
        self.quarterly_earnings = None

        logger.info(f"FundamentalsAnalyzer initialized for {self.ticker}")

    def fetch_data(self) -> bool:
        """
        Fetch all fundamental data from yfinance.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Fetching fundamental data for {self.ticker}")

            # Get info
            self.info = self.stock.info

            # Get financial statements
            try:
                self.financials = self.stock.quarterly_financials
            except Exception as e:
                logger.warning(f"Could not fetch financials: {e}")
                self.financials = None

            try:
                self.balance_sheet = self.stock.quarterly_balance_sheet
            except Exception as e:
                logger.warning(f"Could not fetch balance sheet: {e}")
                self.balance_sheet = None

            try:
                self.cash_flow = self.stock.quarterly_cashflow
            except Exception as e:
                logger.warning(f"Could not fetch cash flow: {e}")
                self.cash_flow = None

            # Get earnings history
            try:
                self.quarterly_earnings = self.stock.quarterly_earnings
            except Exception as e:
                logger.warning(f"Could not fetch earnings: {e}")
                self.quarterly_earnings = None

            logger.info(f"Successfully fetched data for {self.ticker}")
            return True

        except Exception as e:
            logger.error(f"Error fetching fundamental data for {self.ticker}: {e}")
            return False

    def get_valuation_metrics(self) -> Dict:
        """Get valuation metrics."""
        if self.info is None:
            return {}
        return calculate_valuation_metrics(self.info)

    def get_profitability_metrics(self) -> Dict:
        """Get profitability metrics."""
        if self.info is None:
            return {}
        return calculate_profitability_metrics(self.info, self.financials)

    def get_growth_metrics(self) -> Dict:
        """Get growth metrics."""
        if self.info is None:
            return {}
        return calculate_growth_metrics(self.info, self.financials)

    def get_financial_health_metrics(self) -> Dict:
        """Get financial health metrics."""
        if self.info is None:
            return {}
        return calculate_financial_health_metrics(
            self.balance_sheet,
            self.cash_flow,
            self.info
        )

    def get_earnings_analysis(self) -> Optional[EarningsAnalyzer]:
        """Get earnings analyzer."""
        if self.quarterly_earnings is None or self.quarterly_earnings.empty:
            return None
        return EarningsAnalyzer(self.quarterly_earnings)

    def get_health_score(self) -> Dict:
        """Calculate overall health score."""
        valuation = self.get_valuation_metrics()
        profitability = self.get_profitability_metrics()
        growth = self.get_growth_metrics()
        financial_health = self.get_financial_health_metrics()

        return calculate_health_score(
            valuation,
            profitability,
            growth,
            financial_health
        )

    def get_company_info(self) -> Dict:
        """Get basic company information."""
        if self.info is None:
            return {}

        return {
            'name': self.info.get('longName', self.ticker),
            'sector': self.info.get('sector', 'Unknown'),
            'industry': self.info.get('industry', 'Unknown'),
            'market_cap': self.info.get('marketCap'),
            'employees': self.info.get('fullTimeEmployees'),
            'description': self.info.get('longBusinessSummary'),
            'website': self.info.get('website'),
            'current_price': self.info.get('currentPrice'),
        }

    def generate_summary(self) -> Dict:
        """
        Generate comprehensive fundamentals summary.

        Returns:
            Dictionary with all fundamental metrics and analysis
        """
        try:
            # Ensure data is fetched
            if self.info is None:
                self.fetch_data()

            summary = {
                'ticker': self.ticker,
                'company_info': self.get_company_info(),
                'valuation': self.get_valuation_metrics(),
                'profitability': self.get_profitability_metrics(),
                'growth': self.get_growth_metrics(),
                'financial_health': self.get_financial_health_metrics(),
                'health_score': self.get_health_score(),
            }

            # Add earnings if available
            earnings_analyzer = self.get_earnings_analysis()
            if earnings_analyzer:
                summary['earnings'] = {
                    'trends': earnings_analyzer.calculate_earnings_trends(),
                    'insights': earnings_analyzer.generate_insights(),
                }
            else:
                summary['earnings'] = None

            return summary

        except Exception as e:
            logger.error(f"Error generating summary for {self.ticker}: {e}")
            return {
                'ticker': self.ticker,
                'error': str(e)
            }

    def format_summary_report(self) -> str:
        """Generate formatted text report."""
        summary = self.generate_summary()

        if 'error' in summary:
            return f"Error analyzing {self.ticker}: {summary['error']}"

        lines = [
            "=" * 70,
            f"FUNDAMENTALS ANALYSIS: {self.ticker}",
            "=" * 70,
        ]

        # Company info
        info = summary['company_info']
        lines.extend([
            f"\nCompany: {info.get('name', 'N/A')}",
            f"Sector: {info.get('sector', 'N/A')}",
            f"Industry: {info.get('industry', 'N/A')}",
            "",
        ])

        # Health score
        health = summary['health_score']
        lines.extend([
            f"FINANCIAL HEALTH SCORE: {health['score']}/10 ({health['rating']})",
            "",
        ])

        # Valuation
        val = summary['valuation']
        if val:
            lines.extend([
                "VALUATION:",
                f"  P/E Ratio (TTM): {val.get('pe_ratio_ttm', 'N/A')}",
                f"  Forward P/E: {val.get('pe_forward', 'N/A')}",
                f"  PEG Ratio: {val.get('peg_ratio', 'N/A')}",
                f"  Signal: {val.get('valuation_signal', 'N/A').upper()}",
                "",
            ])

        # Profitability
        prof = summary['profitability']
        if prof:
            lines.extend([
                "PROFITABILITY:",
                f"  Gross Margin: {prof.get('gross_margin', 0)*100:.1f}%" if prof.get('gross_margin') else "  Gross Margin: N/A",
                f"  Net Margin: {prof.get('net_margin', 0)*100:.1f}%" if prof.get('net_margin') else "  Net Margin: N/A",
                f"  ROE: {prof.get('roe', 0)*100:.1f}%" if prof.get('roe') else "  ROE: N/A",
                "",
            ])

        # Growth
        growth = summary['growth']
        if growth:
            lines.extend([
                "GROWTH:",
                f"  Revenue Growth: {growth.get('revenue_growth', 0)*100:.1f}%" if growth.get('revenue_growth') else "  Revenue Growth: N/A",
                f"  Signal: {growth.get('growth_signal', 'N/A').upper()}",
                "",
            ])

        # Earnings
        if summary['earnings']:
            earnings = summary['earnings']
            insights = earnings['insights']
            lines.extend([
                "EARNINGS:",
                f"  Beat Streak: {insights['beat_streak']} quarters",
                f"  Momentum: {insights['momentum']}",
                "",
            ])

        lines.append("=" * 70)

        return "\n".join(lines)
