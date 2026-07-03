"""
Reports Page

Generate and download professional PDF reports.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Reports", page_icon=None, layout="wide")

st.title(" Report Generation")
st.markdown("Generate professional PDF reports for your portfolio analysis")

# Check for required session state
if 'portfolio_data' not in st.session_state or st.session_state.portfolio_data is None:
    st.warning("Please load portfolio data first on the Portfolio Input page.")
    st.stop()

data = st.session_state.portfolio_data
returns = data['returns']
prices = data.get('prices')

# Get weights based on what's available
if 'optimization_result' in st.session_state and st.session_state.optimization_result:
    weights = st.session_state.optimization_result['weights']
    metrics = st.session_state.get('metrics', {})
    st.info("Generating report for **optimized portfolio**")
elif 'current_portfolio_weights' in st.session_state and st.session_state.current_portfolio_weights is not None:
    weights = st.session_state.current_portfolio_weights
    # Calculate metrics for current holdings
    portfolio_returns = (returns * weights).sum(axis=1)
    metrics = {
        'annual_return': portfolio_returns.mean() * 252,
        'annual_volatility': portfolio_returns.std() * np.sqrt(252),
        'sharpe_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))
    }
    st.info("Generating report for **your current holdings**")
else:
    weights = pd.Series(1/len(returns.columns), index=returns.columns)
    portfolio_returns = returns.mean(axis=1)
    metrics = {
        'annual_return': portfolio_returns.mean() * 252,
        'annual_volatility': portfolio_returns.std() * np.sqrt(252),
        'sharpe_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))
    }
    st.warning("Using equal weights. Enter holdings or run optimization for accurate report.")

# Check if reportlab is available
try:
    from src.reports.generator import ReportGenerator
    from src.reports.templates import (
        PortfolioSummaryTemplate,
        PerformanceReviewTemplate,
        RiskDashboardTemplate
    )
    from src.reports.charts import ReportChartGenerator
    REPORTLAB_AVAILABLE = True
except ImportError as e:
    REPORTLAB_AVAILABLE = False
    import_error = str(e)

if not REPORTLAB_AVAILABLE:
    st.error(f"""
    **ReportLab not installed**

    PDF report generation requires the reportlab library.

    Install it with:
    ```
    pip install reportlab
    ```

    Error: {import_error}
    """)
    st.stop()

# Report type selection
st.subheader("Select Report Type")

report_type = st.selectbox(
    "Report Type",
    [
        "Portfolio Summary",
        "Performance Review",
        "Risk Dashboard",
        "Complete Analysis"
    ],
    help="Choose the type of report to generate"
)

# Report settings
col1, col2 = st.columns(2)

with col1:
    report_title = st.text_input(
        "Report Title",
        value=f"Portfolio Analysis - {datetime.now().strftime('%B %Y')}"
    )

    author_name = st.text_input(
        "Author/Company Name",
        value="Portfolio Optimization System"
    )

with col2:
    portfolio_value = st.number_input(
        "Portfolio Value ($)",
        min_value=1000,
        value=int(st.session_state.get('portfolio_value', 100000)),
        step=1000
    )

    page_size = st.selectbox(
        "Page Size",
        ["Letter", "A4"]
    )

# Period selection for performance review
if report_type == "Performance Review":
    st.subheader("Review Period")
    col1, col2 = st.columns(2)

    with col1:
        period_start = st.date_input(
            "Start Date",
            value=returns.index[0].date()
        )

    with col2:
        period_end = st.date_input(
            "End Date",
            value=returns.index[-1].date()
        )

# Chart options
st.subheader("Chart Options")

include_charts = st.checkbox("Include charts in report", value=True)

if include_charts:
    chart_options = st.multiselect(
        "Select charts to include",
        [
            "Allocation Pie Chart",
            "Performance Chart",
            "Drawdown Chart",
            "Correlation Heatmap",
            "Monthly Returns",
            "Return Distribution"
        ],
        default=["Allocation Pie Chart", "Performance Chart", "Drawdown Chart"]
    )

# Generate report button
st.markdown("---")

if st.button("Generate Report", type="primary", width="stretch"):
    with st.spinner("Generating report..."):
        try:
            # Initialize generator
            generator = ReportGenerator(
                title=report_title,
                author=author_name,
                page_size=page_size.lower()
            )

            # Generate charts if requested
            chart_images = {}
            if include_charts:
                chart_gen = ReportChartGenerator()

                # Map selection to chart functions
                chart_mapping = {
                    "Allocation Pie Chart": ('allocation', lambda: chart_gen.allocation_pie_chart(weights)),
                    "Performance Chart": ('performance', lambda: chart_gen.performance_line_chart(
                        (returns[weights.index] * weights).sum(axis=1)
                    )),
                    "Drawdown Chart": ('drawdown', lambda: chart_gen.drawdown_chart(
                        (returns[weights.index] * weights).sum(axis=1)
                    )),
                    "Correlation Heatmap": ('correlation', lambda: chart_gen.correlation_heatmap(
                        returns[weights.index]
                    )),
                    "Monthly Returns": ('monthly', lambda: chart_gen.monthly_returns_heatmap(
                        (returns[weights.index] * weights).sum(axis=1)
                    )),
                    "Return Distribution": ('distribution', lambda: chart_gen.distribution_histogram(
                        (returns[weights.index] * weights).sum(axis=1)
                    ))
                }

                for chart_name in chart_options:
                    if chart_name in chart_mapping:
                        key, func = chart_mapping[chart_name]
                        chart_images[key] = func()

            # Build appropriate template
            if report_type == "Portfolio Summary":
                template = PortfolioSummaryTemplate(generator)
                template.build(
                    weights=weights,
                    returns=returns,
                    metrics=metrics,
                    portfolio_value=portfolio_value,
                    chart_images=chart_images
                )

            elif report_type == "Performance Review":
                template = PerformanceReviewTemplate(generator)
                template.build(
                    weights=weights,
                    returns=returns,
                    period_start=datetime.combine(period_start, datetime.min.time()),
                    period_end=datetime.combine(period_end, datetime.min.time()),
                    chart_images=chart_images
                )

            elif report_type == "Risk Dashboard":
                template = RiskDashboardTemplate(generator)

                # Get stress test results if available
                stress_results = st.session_state.get('stress_results', None)

                template.build(
                    weights=weights,
                    returns=returns,
                    var_results=metrics,
                    stress_results=stress_results,
                    chart_images=chart_images
                )

            elif report_type == "Complete Analysis":
                # Build all templates
                # Portfolio Summary
                template = PortfolioSummaryTemplate(generator)
                template.build(
                    weights=weights,
                    returns=returns,
                    metrics=metrics,
                    portfolio_value=portfolio_value,
                    chart_images=chart_images
                )

                generator.add_page_break()

                # Risk Dashboard
                risk_template = RiskDashboardTemplate(generator)
                risk_template.build(
                    weights=weights,
                    returns=returns,
                    var_results=metrics,
                    chart_images=chart_images
                )

            # Generate PDF bytes
            pdf_bytes = generator.generate_bytes()

            # Store in session for download
            st.session_state['generated_report'] = pdf_bytes
            st.session_state['report_filename'] = f"{report_title.replace(' ', '_')}.pdf"

            st.success("Report generated successfully!")

        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            raise e

# Download section
if 'generated_report' in st.session_state:
    st.markdown("---")
    st.subheader("Download Report")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.download_button(
            label=" Download PDF Report",
            data=st.session_state['generated_report'],
            file_name=st.session_state.get('report_filename', 'portfolio_report.pdf'),
            mime="application/pdf",
            width="stretch"
        )

    # Preview info
    st.info(f"""
    **Report Details:**
    - File: {st.session_state.get('report_filename', 'portfolio_report.pdf')}
    - Size: {len(st.session_state['generated_report']) / 1024:.1f} KB
    - Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    """)

# Report templates info
with st.expander("Report Templates Information"):
    st.markdown("""
    ### Available Report Types

    **Portfolio Summary**
    - Executive summary with key metrics
    - Holdings table with weights and values
    - Performance statistics
    - Risk metrics overview

    **Performance Review**
    - Period-specific performance analysis
    - Asset contribution breakdown
    - Monthly returns table
    - Drawdown analysis

    **Risk Dashboard**
    - VaR/CVaR analysis
    - Correlation analysis
    - Stress test results
    - Individual asset risk metrics
    - Risk observations and recommendations

    **Complete Analysis**
    - Combines all report sections
    - Comprehensive portfolio overview
    - Suitable for quarterly reviews
    """)

# Sidebar
with st.sidebar:
    st.header("Report Info")

    st.markdown("""
    ### Quick Guide

    1. Select report type
    2. Customize settings
    3. Choose charts to include
    4. Click Generate
    5. Download PDF

    ### Tips

    - Use **Portfolio Summary** for quick overviews
    - Use **Performance Review** for periodic reports
    - Use **Risk Dashboard** for risk-focused analysis
    - Use **Complete Analysis** for comprehensive reports
    """)

    st.markdown("---")

    if 'weights' in st.session_state:
        st.subheader("Current Portfolio")
        for ticker, weight in weights.head(5).items():
            st.write(f"{ticker}: {weight*100:.1f}%")
        if len(weights) > 5:
            st.write(f"... and {len(weights)-5} more")
