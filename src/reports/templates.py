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


class ComprehensiveProfessionalTemplate:
    """
    Comprehensive professional financial analyst report template.

    Includes:
    - Executive Summary with key takeaways
    - Portfolio Overview with visualizations
    - Holdings Analysis with fundamentals for each stock
    - Risk Analysis with detailed metrics
    - Performance Analysis with charts
    - Optimization Recommendations
    - Appendix with methodology
    """

    def __init__(self, generator):
        self.generator = generator

    def build(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        prices: pd.DataFrame,
        metrics: Dict,
        portfolio_value: float = 100000,
        fundamentals_data: Optional[Dict] = None,
        chart_images: Optional[Dict[str, str]] = None,
        holdings_shares: Optional[Dict[str, float]] = None,
        optimization_result: Optional[Dict] = None,
        stress_results: Optional[pd.DataFrame] = None
    ):
        """
        Build comprehensive professional report.

        Args:
            weights: Portfolio weights
            returns: Asset returns
            prices: Price data
            metrics: Portfolio metrics
            portfolio_value: Total portfolio value
            fundamentals_data: Dict of ticker -> fundamentals summary
            chart_images: Dict of chart name -> image path
            holdings_shares: Dict of ticker -> number of shares
            optimization_result: Optimization results if available
            stress_results: Stress test results if available
        """
        # TITLE PAGE
        self.generator.add_title_page(
            subtitle="Comprehensive Portfolio Analysis"
        )

        # TABLE OF CONTENTS
        self._add_table_of_contents()

        # 1. EXECUTIVE SUMMARY
        self.generator.add_page_break()
        self._add_executive_summary(weights, returns, metrics, portfolio_value, fundamentals_data)

        # 2. PORTFOLIO OVERVIEW
        self.generator.add_page_break()
        self._add_portfolio_overview(weights, portfolio_value, returns, chart_images, holdings_shares)

        # 3. HOLDINGS ANALYSIS WITH FUNDAMENTALS
        self.generator.add_page_break()
        self._add_holdings_analysis(weights, portfolio_value, fundamentals_data, prices)

        # 4. PERFORMANCE ANALYSIS
        self.generator.add_page_break()
        self._add_performance_analysis(weights, returns, metrics, chart_images)

        # 5. RISK ANALYSIS
        self.generator.add_page_break()
        self._add_risk_analysis(weights, returns, metrics, stress_results, chart_images)

        # 6. OPTIMIZATION RECOMMENDATIONS
        if optimization_result:
            self.generator.add_page_break()
            self._add_optimization_recommendations(weights, optimization_result, portfolio_value)

        # 7. METHODOLOGY & DISCLAIMER
        self.generator.add_page_break()
        self._add_methodology_and_disclaimer()

        logger.info("Comprehensive professional report template built")

    def _add_table_of_contents(self):
        """Add table of contents."""
        self.generator.add_section_header("Table of Contents")

        toc_items = [
            "1. Executive Summary",
            "2. Portfolio Overview",
            "3. Holdings Analysis",
            "4. Performance Analysis",
            "5. Risk Analysis",
            "6. Optimization Recommendations",
            "7. Methodology & Disclaimer"
        ]

        for item in toc_items:
            self.generator.add_paragraph(item)

    def _add_executive_summary(self, weights, returns, metrics, portfolio_value, fundamentals_data):
        """Add executive summary section."""
        self.generator.add_section_header("1. Executive Summary")

        # Date stamp
        self.generator.add_paragraph(
            f"<b>Report Date:</b> {datetime.now().strftime('%B %d, %Y')}"
        )
        self.generator.add_spacer(0.2)

        # Key metrics overview
        self.generator.add_subsection_header("Portfolio Snapshot")

        ann_return = metrics.get('annual_return', 0)
        ann_vol = metrics.get('annual_volatility', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = metrics.get('max_drawdown', 0)

        self.generator.add_metrics_row([
            ("Portfolio Value", f"${portfolio_value:,.2f}"),
            ("Number of Holdings", f"{len(weights)}"),
            ("Annual Return", f"{ann_return*100:.2f}%"),
            ("Sharpe Ratio", f"{sharpe:.2f}")
        ])

        # Key takeaways
        self.generator.add_subsection_header("Key Findings")

        portfolio_returns = (returns * weights).sum(axis=1)
        total_return = (1 + portfolio_returns).prod() - 1

        # Calculate health scores if fundamentals available
        avg_health_score = None
        if fundamentals_data:
            health_scores = [
                data.get('health_score', {}).get('score', 0)
                for data in fundamentals_data.values()
                if data and 'health_score' in data
            ]
            if health_scores:
                avg_health_score = sum(health_scores) / len(health_scores)

        findings = []

        # Performance finding
        if ann_return > 0.15:
            findings.append("• <b>Strong Performance:</b> The portfolio has generated above-average returns with an annualized return of {:.1f}%.".format(ann_return*100))
        elif ann_return > 0.08:
            findings.append("• <b>Moderate Performance:</b> The portfolio has generated market-level returns of {:.1f}% annually.".format(ann_return*100))
        else:
            findings.append("• <b>Below-Average Performance:</b> The portfolio return of {:.1f}% is below typical market returns.".format(ann_return*100))

        # Risk finding
        if sharpe > 1.5:
            findings.append("• <b>Excellent Risk-Adjusted Returns:</b> Sharpe ratio of {:.2f} indicates strong returns relative to risk taken.".format(sharpe))
        elif sharpe > 1.0:
            findings.append("• <b>Good Risk-Adjusted Returns:</b> Sharpe ratio of {:.2f} demonstrates adequate compensation for risk.".format(sharpe))
        else:
            findings.append("• <b>Suboptimal Risk-Adjusted Returns:</b> Sharpe ratio of {:.2f} suggests returns may not justify the risk level.".format(sharpe))

        # Volatility finding
        if ann_vol > 0.25:
            findings.append("• <b>High Volatility:</b> Annual volatility of {:.1f}% indicates significant price fluctuations. Consider risk management strategies.".format(ann_vol*100))
        elif ann_vol > 0.15:
            findings.append("• <b>Moderate Volatility:</b> Annual volatility of {:.1f}% is within normal ranges for equity portfolios.".format(ann_vol*100))
        else:
            findings.append("• <b>Low Volatility:</b> Annual volatility of {:.1f}% suggests a conservative, stable portfolio.".format(ann_vol*100))

        # Concentration finding
        max_weight = weights.max()
        if max_weight > 0.30:
            top_ticker = weights.idxmax()
            findings.append("• <b>Concentration Risk:</b> {:.0f}% allocation to {} creates single-stock risk. Diversification recommended.".format(max_weight*100, top_ticker))

        # Fundamentals finding
        if avg_health_score is not None:
            if avg_health_score >= 7:
                findings.append("• <b>Strong Fundamentals:</b> Average health score of {:.1f}/10 indicates holdings with solid financial fundamentals.".format(avg_health_score))
            elif avg_health_score >= 5:
                findings.append("• <b>Moderate Fundamentals:</b> Average health score of {:.1f}/10 suggests mixed fundamental quality across holdings.".format(avg_health_score))
            else:
                findings.append("• <b>Weak Fundamentals:</b> Average health score of {:.1f}/10 indicates concerns about financial health of holdings.".format(avg_health_score))

        for finding in findings:
            self.generator.add_paragraph(finding)

        # Recommendations summary
        self.generator.add_subsection_header("Primary Recommendations")

        recommendations = []

        if sharpe < 1.0:
            recommendations.append("1. Consider rebalancing to improve risk-adjusted returns")

        if max_weight > 0.30:
            recommendations.append("2. Reduce concentration in top holdings to mitigate single-stock risk")

        if ann_vol > 0.25:
            recommendations.append("3. Implement hedging strategies to manage volatility")

        if avg_health_score and avg_health_score < 5:
            recommendations.append("4. Review holdings with weak fundamentals for potential replacement")

        if not recommendations:
            recommendations.append("1. Maintain current allocation and monitor performance")
            recommendations.append("2. Continue periodic rebalancing to target weights")

        for rec in recommendations[:4]:  # Max 4 recommendations
            self.generator.add_paragraph(rec)

    def _add_portfolio_overview(self, weights, portfolio_value, returns, chart_images, holdings_shares):
        """Add portfolio overview section."""
        self.generator.add_section_header("2. Portfolio Overview")

        self.generator.add_subsection_header("Current Allocation")

        # Holdings table with shares if available
        holdings_data = []
        for ticker, weight in weights.sort_values(ascending=False).items():
            value = portfolio_value * weight
            row = [
                ticker,
                f"{weight*100:.1f}%",
                f"${value:,.0f}"
            ]

            if holdings_shares and ticker in holdings_shares:
                row.append(f"{holdings_shares[ticker]:.0f}")

            holdings_data.append(row)

        headers = ["Ticker", "Weight", "Value"]
        col_widths = [1.5, 1.2, 1.5]

        if holdings_shares:
            headers.append("Shares")
            col_widths.append(1.2)

        self.generator.add_table(
            holdings_data,
            headers=headers,
            col_widths=col_widths
        )

        # Allocation pie chart
        if chart_images and 'allocation' in chart_images:
            self.generator.add_spacer(0.3)
            self.generator.add_image(chart_images['allocation'], width=5, height=4)

        # Diversification metrics
        self.generator.add_subsection_header("Diversification Analysis")

        # Calculate HHI and effective stocks
        hhi = (weights ** 2).sum()
        effective_stocks = 1 / hhi if hhi > 0 else 0

        # Top N concentration
        top3_concentration = weights.nlargest(3).sum()
        top5_concentration = weights.nlargest(5).sum() if len(weights) >= 5 else weights.sum()

        div_data = [
            ["Herfindahl-Hirschman Index", f"{hhi:.4f}"],
            ["Effective Number of Stocks", f"{effective_stocks:.2f}"],
            ["Top 3 Concentration", f"{top3_concentration*100:.1f}%"],
            ["Top 5 Concentration", f"{top5_concentration*100:.1f}%"]
        ]

        self.generator.add_table(
            div_data,
            headers=["Metric", "Value"],
            col_widths=[3, 2]
        )

        # Interpretation
        if hhi > 0.25:
            interpretation = "The portfolio exhibits high concentration. Consider adding more positions to improve diversification."
        elif hhi > 0.15:
            interpretation = "The portfolio shows moderate concentration with {:.1f} effective stocks.".format(effective_stocks)
        else:
            interpretation = "The portfolio is well-diversified with {:.1f} effective stocks.".format(effective_stocks)

        self.generator.add_paragraph(interpretation)

    def _add_holdings_analysis(self, weights, portfolio_value, fundamentals_data, prices):
        """Add holdings analysis with fundamentals."""
        self.generator.add_section_header("3. Holdings Analysis")

        self.generator.add_paragraph(
            "This section provides detailed fundamental analysis for each portfolio holding, "
            "including valuation metrics, profitability indicators, growth rates, and financial health scores."
        )

        # Analyze each holding
        for i, (ticker, weight) in enumerate(weights.sort_values(ascending=False).items()):
            if i > 0 and i % 2 == 0:  # Page break every 2 holdings
                self.generator.add_page_break()

            self.generator.add_subsection_header(f"{ticker} - {weight*100:.1f}% of Portfolio")

            position_value = portfolio_value * weight
            self.generator.add_paragraph(f"<b>Position Value:</b> ${position_value:,.2f}")

            # Get fundamentals if available
            if fundamentals_data and ticker in fundamentals_data:
                fund = fundamentals_data[ticker]

                if fund:
                    # Company info
                    company_info = fund.get('company_info', {})
                    if company_info:
                        self.generator.add_paragraph(
                            f"<b>Company:</b> {company_info.get('name', 'N/A')} | "
                            f"<b>Sector:</b> {company_info.get('sector', 'N/A')} | "
                            f"<b>Industry:</b> {company_info.get('industry', 'N/A')}"
                        )

                    # Health score
                    health_score = fund.get('health_score', {})
                    if health_score and 'score' in health_score:
                        score = health_score['score']
                        rating = health_score.get('rating', 'N/A')
                        self.generator.add_paragraph(
                            f"<b>Financial Health Score:</b> {score:.1f}/10 ({rating})"
                        )

                    # Key metrics table
                    metrics_data = []

                    # Valuation
                    valuation = fund.get('valuation', {})
                    if valuation:
                        pe = valuation.get('pe_ratio_ttm')
                        peg = valuation.get('peg_ratio')
                        pb = valuation.get('price_to_book')
                        signal = valuation.get('valuation_signal', 'N/A')

                        if pe: metrics_data.append(["P/E Ratio", f"{pe:.2f}"])
                        if peg: metrics_data.append(["PEG Ratio", f"{peg:.2f}"])
                        if pb: metrics_data.append(["P/B Ratio", f"{pb:.2f}"])
                        metrics_data.append(["Valuation", signal.title()])

                    # Profitability
                    profitability = fund.get('profitability', {})
                    if profitability:
                        net_margin = profitability.get('net_margin')
                        roe = profitability.get('return_on_equity')

                        if net_margin: metrics_data.append(["Net Margin", f"{net_margin*100:.1f}%"])
                        if roe: metrics_data.append(["ROE", f"{roe*100:.1f}%"])

                    # Growth
                    growth = fund.get('growth', {})
                    if growth:
                        rev_growth = growth.get('revenue_growth_yoy')
                        earn_growth = growth.get('earnings_growth_yoy')

                        if rev_growth: metrics_data.append(["Revenue Growth", f"{rev_growth*100:.1f}%"])
                        if earn_growth: metrics_data.append(["Earnings Growth", f"{earn_growth*100:.1f}%"])

                    # Financial health
                    fin_health = fund.get('financial_health', {})
                    if fin_health:
                        current_ratio = fin_health.get('current_ratio')
                        debt_equity = fin_health.get('debt_to_equity')

                        if current_ratio: metrics_data.append(["Current Ratio", f"{current_ratio:.2f}"])
                        if debt_equity: metrics_data.append(["Debt/Equity", f"{debt_equity:.2f}"])

                    if metrics_data:
                        self.generator.add_table(
                            metrics_data,
                            headers=["Metric", "Value"],
                            col_widths=[2.5, 2]
                        )

                    # Earnings insights
                    earnings = fund.get('earnings', {})
                    if earnings:
                        insights = earnings.get('insights', {})
                        if insights:
                            beat_streak = insights.get('beat_streak', 0)
                            momentum = insights.get('momentum', 'N/A')

                            self.generator.add_paragraph(
                                f"<b>Earnings:</b> {beat_streak} quarter beat streak | "
                                f"Momentum: {momentum.title()}"
                            )
                else:
                    self.generator.add_paragraph("<i>Fundamental data not available for this holding.</i>")
            else:
                self.generator.add_paragraph("<i>Fundamental analysis not included in this report.</i>")

            self.generator.add_spacer(0.2)

    def _add_performance_analysis(self, weights, returns, metrics, chart_images):
        """Add performance analysis section."""
        self.generator.add_section_header("4. Performance Analysis")

        portfolio_returns = (returns * weights).sum(axis=1)

        # Performance metrics
        self.generator.add_subsection_header("Performance Metrics")

        total_return = (1 + portfolio_returns).prod() - 1
        ann_return = metrics.get('annual_return', 0)
        ann_vol = metrics.get('annual_volatility', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = metrics.get('max_drawdown', 0)

        perf_data = [
            ["Total Return", f"{total_return*100:.2f}%"],
            ["Annualized Return", f"{ann_return*100:.2f}%"],
            ["Annualized Volatility", f"{ann_vol*100:.2f}%"],
            ["Sharpe Ratio", f"{sharpe:.2f}"],
            ["Maximum Drawdown", f"{max_dd*100:.2f}%"],
            ["Best Day", f"{portfolio_returns.max()*100:.2f}%"],
            ["Worst Day", f"{portfolio_returns.min()*100:.2f}%"]
        ]

        self.generator.add_table(
            perf_data,
            headers=["Metric", "Value"],
            col_widths=[3, 2]
        )

        # Performance chart
        if chart_images and 'performance' in chart_images:
            self.generator.add_spacer(0.3)
            self.generator.add_subsection_header("Cumulative Performance")
            self.generator.add_image(chart_images['performance'], width=6, height=3.5)

        # Drawdown chart
        if chart_images and 'drawdown' in chart_images:
            self.generator.add_subsection_header("Drawdown Analysis")
            self.generator.add_image(chart_images['drawdown'], width=6, height=3.5)

        # Monthly returns heatmap
        if chart_images and 'monthly' in chart_images:
            self.generator.add_page_break()
            self.generator.add_subsection_header("Monthly Returns Heatmap")
            self.generator.add_image(chart_images['monthly'], width=6, height=4)

    def _add_risk_analysis(self, weights, returns, metrics, stress_results, chart_images):
        """Add risk analysis section."""
        self.generator.add_section_header("5. Risk Analysis")

        portfolio_returns = (returns * weights).sum(axis=1)
        volatility = portfolio_returns.std() * np.sqrt(252)

        # Value at Risk
        self.generator.add_subsection_header("Value at Risk (VaR)")

        var_90 = np.percentile(portfolio_returns, 10)
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)

        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

        var_data = [
            ["VaR 90% (Daily)", f"-{abs(var_90)*100:.2f}%"],
            ["VaR 95% (Daily)", f"-{abs(var_95)*100:.2f}%"],
            ["VaR 99% (Daily)", f"-{abs(var_99)*100:.2f}%"],
            ["CVaR 95% (Expected Shortfall)", f"-{abs(cvar_95)*100:.2f}%"]
        ]

        self.generator.add_table(
            var_data,
            headers=["Risk Metric", "Value"],
            col_widths=[3.5, 2]
        )

        self.generator.add_paragraph(
            f"There is a 5% probability that the portfolio will lose more than "
            f"{abs(var_95)*100:.2f}% in a single day. If such a loss occurs, the "
            f"expected loss would be {abs(cvar_95)*100:.2f}% (CVaR)."
        )

        # Correlation heatmap
        if chart_images and 'correlation' in chart_images:
            self.generator.add_subsection_header("Asset Correlation Matrix")
            self.generator.add_image(chart_images['correlation'], width=5.5, height=4.5)

        # Stress tests
        if stress_results is not None and len(stress_results) > 0:
            self.generator.add_page_break()
            self.generator.add_subsection_header("Stress Test Results")

            stress_data = []
            for scenario in stress_results.index[:8]:  # Top 8 scenarios
                impact = stress_results.loc[scenario, 'Portfolio Impact']
                stress_data.append([
                    scenario,
                    f"{impact*100:.2f}%"
                ])

            stress_data.sort(key=lambda x: float(x[1].replace('%', '')))

            self.generator.add_table(
                stress_data,
                headers=["Historical Scenario", "Portfolio Impact"],
                col_widths=[3.5, 2]
            )

            worst_scenario = stress_data[0]
            self.generator.add_paragraph(
                f"Under the worst historical scenario ({worst_scenario[0]}), the portfolio "
                f"would have experienced a {worst_scenario[1]} decline."
            )

    def _add_optimization_recommendations(self, current_weights, optimization_result, portfolio_value):
        """Add optimization recommendations section."""
        self.generator.add_section_header("6. Optimization Recommendations")

        optimal_weights = optimization_result.get('weights')

        if optimal_weights is None:
            self.generator.add_paragraph("No optimization recommendations available.")
            return

        self.generator.add_paragraph(
            "Based on mean-variance optimization, the following adjustments are recommended "
            "to improve risk-adjusted returns:"
        )

        # Compare current vs optimal
        comparison_data = []
        for ticker in sorted(set(list(current_weights.index) + list(optimal_weights.index))):
            current = current_weights.get(ticker, 0)
            optimal = optimal_weights.get(ticker, 0)
            change = optimal - current

            comparison_data.append([
                ticker,
                f"{current*100:.1f}%",
                f"{optimal*100:.1f}%",
                f"{change*100:+.1f}%",
                f"${abs(change * portfolio_value):,.0f}"
            ])

        # Sort by absolute change
        comparison_data.sort(key=lambda x: abs(float(x[3].replace('%', '').replace('+', ''))), reverse=True)

        self.generator.add_table(
            comparison_data,
            headers=["Ticker", "Current", "Optimal", "Change", "$ Amount"],
            col_widths=[1.2, 1.2, 1.2, 1.2, 1.5]
        )

        # Expected improvement
        current_sharpe = (current_weights * optimization_result.get('expected_return', 0)) / optimization_result.get('volatility', 1)
        optimal_sharpe = optimization_result.get('sharpe_ratio', 0)

        if optimal_sharpe > current_sharpe:
            self.generator.add_spacer(0.3)
            self.generator.add_paragraph(
                f"<b>Expected Improvement:</b> Implementing these changes would improve the "
                f"portfolio Sharpe ratio from {current_sharpe:.2f} to {optimal_sharpe:.2f}."
            )

    def _add_methodology_and_disclaimer(self):
        """Add methodology and disclaimer section."""
        self.generator.add_section_header("7. Methodology & Disclaimer")

        self.generator.add_subsection_header("Analytical Methodology")

        methodology_points = [
            "• <b>Portfolio Optimization:</b> Mean-variance optimization (Markowitz, 1952) with SLSQP algorithm",
            "• <b>Risk Metrics:</b> Historical VaR/CVaR at 95% confidence level based on daily returns",
            "• <b>Performance:</b> Time-weighted returns annualized using 252 trading days",
            "• <b>Fundamentals:</b> Data sourced from Yahoo Finance including P/E, PEG, margins, growth rates",
            "• <b>Health Score:</b> Composite 0-10 scale based on valuation, profitability, growth, financial health, and earnings quality",
            "• <b>Correlations:</b> Pearson correlation coefficient calculated on daily returns"
        ]

        for point in methodology_points:
            self.generator.add_paragraph(point)

        self.generator.add_spacer(0.4)
        self.generator.add_subsection_header("Important Disclaimer")

        disclaimer = (
            "This report is for informational purposes only and does not constitute financial advice, "
            "investment recommendation, or an offer to buy or sell securities. Past performance is not "
            "indicative of future results. All investments carry risk, including potential loss of principal. "
            "The analysis is based on historical data and assumptions that may not hold in the future. "
            "Fundamental data is sourced from third-party providers and may contain errors or omissions. "
            "Consult with a qualified financial advisor before making investment decisions. "
            "This system is provided 'as-is' without warranties of any kind."
        )

        self.generator.add_paragraph(disclaimer)

        self.generator.add_spacer(0.3)
        self.generator.add_paragraph(
            f"<i>Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</i>"
        )
