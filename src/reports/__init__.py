"""
Automated Reporting Module

Generate professional PDF reports for portfolio analysis.
"""

from .generator import ReportGenerator, generate_portfolio_report
from .templates import (
    PortfolioSummaryTemplate,
    PerformanceReviewTemplate,
    RiskDashboardTemplate
)
from .charts import ReportChartGenerator

__all__ = [
    'ReportGenerator',
    'generate_portfolio_report',
    'PortfolioSummaryTemplate',
    'PerformanceReviewTemplate',
    'RiskDashboardTemplate',
    'ReportChartGenerator'
]
