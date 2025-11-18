"""
Report Templates

Pre-built templates for common report types.
"""

import io
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


class PortfolioSummaryTemplate:
    """
    Template for portfolio summary reports.

    Includes:
    - Portfolio composition
    - Key metrics
    - Performance summary
    - Holdings table
    """

    def __init__(self, generator):
        """
        Initialize template.

        Args:
            generator: ReportGenerator instance
        """
        self.generator = generator

    def build(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        metrics: Dict,
        prices: Optional[pd.Series] = None,
        portfolio_value: float = 100000,
        chart_images: Optional[Dict[str, str]] = None
    ):
        """
        Build the portfolio summary report.

        Args:
            weights: Portfolio weights
            returns: Asset returns
            metrics: Portfolio metrics dict
            prices: Current asset prices
            portfolio_value: Total portfolio value
            chart_images: Dict of chart name -> image path
        """
        # Title page
        self.generator.add_title_page(
            subtitle="Portfolio Summary Report"
        )

        # Executive Summary
        self.generator.add_section_header("Executive Summary")

        # Key metrics row
        ann_return = metrics.get('annual_return', 0)
        ann_vol = metrics.get('annual_volatility', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = metrics.get('max_drawdown', 0)

        self.generator.add_metrics_row([
            ("Annual Return", f"{ann_return*100:.2f}%"),
            ("Volatility", f"{ann_vol*100:.2f}%"),
            ("Sharpe Ratio", f"{sharpe:.2f}"),
            ("Max Drawdown", f"{max_dd*100:.2f}%")
        ])

        self.generator.add_paragraph(
            f"This portfolio consists of {len(weights)} assets with a total value of "
            f"${portfolio_value:,.2f}. The optimization targets maximum risk-adjusted "
            f"returns while maintaining diversification across holdings."
        )

        # Holdings section
        self.generator.add_section_header("Portfolio Holdings")

        holdings_data = []
        for ticker, weight in weights.sort_values(ascending=False).items():
            value = portfolio_value * weight
            holdings_data.append([
                ticker,
                f"{weight*100:.1f}%",
                f"${value:,.0f}"
            ])

        self.generator.add_table(
            holdings_data,
            headers=["Ticker", "Weight", "Value"],
            col_widths=[2, 1.5, 2]
        )

        # Add pie chart if available
        if chart_images and 'allocation' in chart_images:
            self.generator.add_image(chart_images['allocation'], width=4, height=4)

        # Performance Statistics
        self.generator.add_section_header("Performance Statistics")

        # Calculate additional metrics
        portfolio_returns = (returns * weights).sum(axis=1)
        total_return = (1 + portfolio_returns).prod() - 1
        daily_vol = portfolio_returns.std()
        best_day = portfolio_returns.max()
        worst_day = portfolio_returns.min()

        stats_data = [
            ["Total Return", f"{total_return*100:.2f}%"],
            ["Annual Return", f"{ann_return*100:.2f}%"],
            ["Annual Volatility", f"{ann_vol*100:.2f}%"],
            ["Sharpe Ratio", f"{sharpe:.2f}"],
            ["Maximum Drawdown", f"{max_dd*100:.2f}%"],
            ["Best Day", f"{best_day*100:.2f}%"],
            ["Worst Day", f"{worst_day*100:.2f}%"],
            ["Daily Volatility", f"{daily_vol*100:.3f}%"]
        ]

        self.generator.add_table(
            stats_data,
            headers=["Metric", "Value"],
            col_widths=[3, 2]
        )

        # Add performance chart if available
        if chart_images and 'performance' in chart_images:
            self.generator.add_page_break()
            self.generator.add_section_header("Performance Chart")
            self.generator.add_image(chart_images['performance'], width=6, height=3.5)

        # Risk metrics
        if 'var_95' in metrics or 'cvar_95' in metrics:
            self.generator.add_section_header("Risk Metrics")

            risk_data = []
            if 'var_95' in metrics:
                risk_data.append(["VaR (95%)", f"{metrics['var_95']*100:.2f}%"])
            if 'cvar_95' in metrics:
                risk_data.append(["CVaR (95%)", f"{metrics['cvar_95']*100:.2f}%"])
            if 'sortino_ratio' in metrics:
                risk_data.append(["Sortino Ratio", f"{metrics['sortino_ratio']:.2f}"])
            if 'calmar_ratio' in metrics:
                risk_data.append(["Calmar Ratio", f"{metrics['calmar_ratio']:.2f}"])

            if risk_data:
                self.generator.add_table(
                    risk_data,
                    headers=["Risk Metric", "Value"],
                    col_widths=[3, 2]
                )

        logger.info("Portfolio summary template built")


class PerformanceReviewTemplate:
    """
    Template for periodic performance review reports.

    Includes:
    - Period returns vs benchmark
    - Attribution analysis
    - Risk evolution
    - Recommendations
    """

    def __init__(self, generator):
        self.generator = generator

    def build(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        period_start: datetime,
        period_end: datetime,
        benchmark_returns: Optional[pd.Series] = None,
        chart_images: Optional[Dict[str, str]] = None
    ):
        """
        Build performance review report.

        Args:
            weights: Portfolio weights
            returns: Asset returns
            period_start: Review period start
            period_end: Review period end
            benchmark_returns: Optional benchmark returns
            chart_images: Dict of chart name -> image path
        """
        # Title
        period_str = f"{period_start.strftime('%b %Y')} - {period_end.strftime('%b %Y')}"
        self.generator.add_title_page(
            subtitle=f"Performance Review: {period_str}"
        )

        # Filter returns to period
        mask = (returns.index >= period_start) & (returns.index <= period_end)
        period_returns = returns.loc[mask]

        if len(period_returns) == 0:
            self.generator.add_paragraph("No data available for selected period.")
            return

        # Calculate portfolio returns
        portfolio_returns = (period_returns * weights).sum(axis=1)

        # Period performance
        self.generator.add_section_header("Period Performance")

        total_return = (1 + portfolio_returns).prod() - 1
        ann_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe = ann_return / volatility if volatility > 0 else 0

        self.generator.add_metrics_row([
            ("Period Return", f"{total_return*100:.2f}%"),
            ("Ann. Return", f"{ann_return*100:.2f}%"),
            ("Volatility", f"{volatility*100:.2f}%"),
            ("Sharpe", f"{sharpe:.2f}")
        ])

        # Benchmark comparison
        if benchmark_returns is not None:
            bench_period = benchmark_returns.loc[mask]
            bench_return = (1 + bench_period).prod() - 1
            active_return = total_return - bench_return

            self.generator.add_spacer(0.3)
            self.generator.add_paragraph(
                f"<b>Benchmark Return:</b> {bench_return*100:.2f}%  |  "
                f"<b>Active Return:</b> {active_return*100:+.2f}%"
            )

        # Asset contribution
        self.generator.add_section_header("Asset Contribution")

        contrib_data = []
        for ticker in weights.index:
            if ticker in period_returns.columns:
                asset_return = (1 + period_returns[ticker]).prod() - 1
                contribution = weights[ticker] * asset_return
                contrib_data.append([
                    ticker,
                    f"{weights[ticker]*100:.1f}%",
                    f"{asset_return*100:.2f}%",
                    f"{contribution*100:.2f}%"
                ])

        contrib_data.sort(key=lambda x: float(x[3].replace('%', '')), reverse=True)

        self.generator.add_table(
            contrib_data,
            headers=["Asset", "Weight", "Return", "Contribution"],
            col_widths=[1.5, 1.2, 1.5, 1.5]
        )

        # Monthly returns
        self.generator.add_section_header("Monthly Returns")

        monthly = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

        monthly_data = []
        for date, ret in monthly.items():
            monthly_data.append([
                date.strftime('%b %Y'),
                f"{ret*100:.2f}%"
            ])

        if len(monthly_data) > 12:
            monthly_data = monthly_data[-12:]  # Last 12 months

        self.generator.add_table(
            monthly_data,
            headers=["Month", "Return"],
            col_widths=[2, 2]
        )

        # Add cumulative return chart
        if chart_images and 'cumulative' in chart_images:
            self.generator.add_page_break()
            self.generator.add_section_header("Cumulative Returns")
            self.generator.add_image(chart_images['cumulative'], width=6, height=3.5)

        # Drawdown analysis
        self.generator.add_section_header("Drawdown Analysis")

        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max

        max_dd = drawdowns.min()
        current_dd = drawdowns.iloc[-1]

        dd_data = [
            ["Maximum Drawdown", f"{max_dd*100:.2f}%"],
            ["Current Drawdown", f"{current_dd*100:.2f}%"],
            ["Time to Recovery", "N/A" if current_dd < 0 else "Recovered"]
        ]

        self.generator.add_table(
            dd_data,
            headers=["Metric", "Value"],
            col_widths=[3, 2]
        )

        logger.info("Performance review template built")


class RiskDashboardTemplate:
    """
    Template for risk dashboard reports.

    Includes:
    - VaR/CVaR summary
    - Correlation matrix
    - Stress test results
    - Risk decomposition
    """

    def __init__(self, generator):
        self.generator = generator

    def build(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        var_results: Optional[Dict] = None,
        stress_results: Optional[pd.DataFrame] = None,
        chart_images: Optional[Dict[str, str]] = None
    ):
        """
        Build risk dashboard report.

        Args:
            weights: Portfolio weights
            returns: Asset returns
            var_results: VaR calculation results
            stress_results: Stress test results
            chart_images: Dict of chart name -> image path
        """
        # Title
        self.generator.add_title_page(
            subtitle="Risk Dashboard Report"
        )

        # Portfolio risk overview
        self.generator.add_section_header("Risk Overview")

        portfolio_returns = (returns * weights).sum(axis=1)
        volatility = portfolio_returns.std() * np.sqrt(252)

        # Calculate VaR if not provided
        if var_results:
            var_95 = var_results.get('var_95', np.percentile(portfolio_returns, 5))
            cvar_95 = var_results.get('cvar_95', portfolio_returns[portfolio_returns <= var_95].mean())
        else:
            var_95 = np.percentile(portfolio_returns, 5)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

        self.generator.add_metrics_row([
            ("Volatility (Ann.)", f"{volatility*100:.2f}%"),
            ("VaR 95%", f"{abs(var_95)*100:.2f}%"),
            ("CVaR 95%", f"{abs(cvar_95)*100:.2f}%"),
            ("Worst Day", f"{portfolio_returns.min()*100:.2f}%")
        ])

        # VaR Analysis
        self.generator.add_section_header("Value at Risk Analysis")

        var_data = [
            ["VaR 90%", f"{abs(np.percentile(portfolio_returns, 10))*100:.2f}%"],
            ["VaR 95%", f"{abs(np.percentile(portfolio_returns, 5))*100:.2f}%"],
            ["VaR 99%", f"{abs(np.percentile(portfolio_returns, 1))*100:.2f}%"]
        ]

        self.generator.add_table(
            var_data,
            headers=["Confidence Level", "Daily VaR"],
            col_widths=[2.5, 2]
        )

        self.generator.add_paragraph(
            "VaR represents the maximum expected loss over a one-day period "
            "at the specified confidence level under normal market conditions."
        )

        # Correlation analysis
        self.generator.add_section_header("Correlation Analysis")

        corr_matrix = returns[weights.index].corr()

        # Find highest correlations
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    high_corr.append([
                        f"{corr_matrix.columns[i]} / {corr_matrix.columns[j]}",
                        f"{corr_val:.2f}"
                    ])

        if high_corr:
            high_corr.sort(key=lambda x: abs(float(x[1])), reverse=True)
            self.generator.add_paragraph("<b>High Correlations (>0.5):</b>")
            self.generator.add_table(
                high_corr[:5],
                headers=["Asset Pair", "Correlation"],
                col_widths=[3.5, 1.5]
            )
        else:
            self.generator.add_paragraph("No high correlations (>0.5) found between assets.")

        # Add correlation heatmap if available
        if chart_images and 'correlation' in chart_images:
            self.generator.add_image(chart_images['correlation'], width=5, height=4)

        # Stress test results
        if stress_results is not None and len(stress_results) > 0:
            self.generator.add_page_break()
            self.generator.add_section_header("Stress Test Results")

            stress_data = []
            for scenario in stress_results.index:
                impact = stress_results.loc[scenario, 'Portfolio Impact']
                stress_data.append([
                    scenario,
                    f"{impact*100:.2f}%"
                ])

            stress_data.sort(key=lambda x: float(x[1].replace('%', '')))

            self.generator.add_table(
                stress_data,
                headers=["Scenario", "Portfolio Impact"],
                col_widths=[3.5, 2]
            )

            self.generator.add_paragraph(
                "Stress tests simulate portfolio performance under extreme "
                "historical market conditions."
            )

        # Individual asset risk
        self.generator.add_section_header("Individual Asset Risk")

        asset_risk_data = []
        for ticker in weights.index:
            if ticker in returns.columns:
                asset_vol = returns[ticker].std() * np.sqrt(252)
                asset_var = np.percentile(returns[ticker], 5)
                risk_contrib = weights[ticker] * asset_vol / volatility

                asset_risk_data.append([
                    ticker,
                    f"{asset_vol*100:.2f}%",
                    f"{abs(asset_var)*100:.2f}%",
                    f"{risk_contrib*100:.1f}%"
                ])

        asset_risk_data.sort(key=lambda x: float(x[1].replace('%', '')), reverse=True)

        self.generator.add_table(
            asset_risk_data,
            headers=["Asset", "Volatility", "VaR 95%", "Risk Contrib."],
            col_widths=[1.5, 1.5, 1.5, 1.5]
        )

        # Risk recommendations
        self.generator.add_section_header("Risk Observations")

        observations = []

        # High volatility check
        if volatility > 0.20:
            observations.append(
                "• Portfolio volatility is elevated (>20%). Consider reducing "
                "exposure to high-volatility assets."
            )

        # Concentration check
        max_weight = weights.max()
        if max_weight > 0.30:
            top_asset = weights.idxmax()
            observations.append(
                f"• High concentration in {top_asset} ({max_weight*100:.0f}%). "
                "Consider diversifying to reduce single-asset risk."
            )

        # Correlation check
        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].mean()
        if avg_corr > 0.6:
            observations.append(
                f"• Average correlation is high ({avg_corr:.2f}). "
                "Portfolio may have limited diversification benefit."
            )

        if not observations:
            observations.append("• No significant risk concerns identified.")

        for obs in observations:
            self.generator.add_paragraph(obs)

        logger.info("Risk dashboard template built")
