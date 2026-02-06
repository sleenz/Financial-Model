"""Fundamentals & Earnings Intelligence Page."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.fundamentals.analyzer import FundamentalsAnalyzer
from src.fundamentals.health_score import explain_health_score

st.set_page_config(page_title="Fundamentals", page_icon="💰", layout="wide")

st.title("💰 Fundamentals & Earnings Intelligence")

# Check if we have holdings or allow manual ticker entry
has_holdings = ('current_holdings' in st.session_state and
                st.session_state.current_holdings and
                len(st.session_state.current_holdings) > 0)

# Ticker selection
if has_holdings:
    st.info("💼 Analyze fundamentals for stocks in your portfolio")
    tickers_list = list(st.session_state.current_holdings.keys())

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_ticker = st.selectbox(
            "Select Stock",
            options=tickers_list,
            help="Choose from your current holdings"
        )
    with col2:
        custom_ticker = st.text_input("Or Enter Ticker", "").upper()
        if custom_ticker:
            selected_ticker = custom_ticker
else:
    st.info("Enter a ticker symbol to analyze company fundamentals")
    selected_ticker = st.text_input(
        "Stock Ticker",
        value="AAPL",
        help="Enter stock symbol (e.g., AAPL, MSFT, GOOGL)"
    ).upper()

# Analyze button
if st.button("📊 Analyze Fundamentals", type="primary", width="stretch"):
    if not selected_ticker:
        st.error("Please enter a ticker symbol")
    else:
        with st.spinner(f"Fetching fundamental data for {selected_ticker}..."):
            try:
                # Create analyzer
                analyzer = FundamentalsAnalyzer(selected_ticker)

                # Fetch data
                success = analyzer.fetch_data()

                if not success:
                    st.error(f"Could not fetch data for {selected_ticker}. Please check the ticker symbol.")
                else:
                    # Store in session state
                    st.session_state.fundamentals_analyzer = analyzer
                    st.session_state.fundamentals_ticker = selected_ticker
                    st.success(f"✓ Data fetched for {selected_ticker}")

            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())

# Display fundamentals if available
if 'fundamentals_analyzer' in st.session_state and st.session_state.fundamentals_ticker:
    analyzer = st.session_state.fundamentals_analyzer
    ticker = st.session_state.fundamentals_ticker

    st.markdown("---")

    # Get all metrics
    company_info = analyzer.get_company_info()
    valuation = analyzer.get_valuation_metrics()
    profitability = analyzer.get_profitability_metrics()
    growth = analyzer.get_growth_metrics()
    financial_health = analyzer.get_financial_health_metrics()
    health_score = analyzer.get_health_score()
    earnings_analyzer = analyzer.get_earnings_analysis()

    # Company header
    st.header(f"{company_info.get('name', ticker)}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sector", company_info.get('sector', 'N/A'))
    with col2:
        st.metric("Industry", company_info.get('industry', 'N/A'))
    with col3:
        market_cap = company_info.get('market_cap')
        if market_cap:
            st.metric("Market Cap", f"${market_cap/1e9:.1f}B")
        else:
            st.metric("Market Cap", "N/A")
    with col4:
        current_price = company_info.get('current_price')
        if current_price:
            st.metric("Price", f"${current_price:.2f}")
        else:
            st.metric("Price", "N/A")

    st.markdown("---")

    # ============================================================================
    # HEALTH SCORE (Prominent Display)
    # ============================================================================
    st.subheader("📊 Financial Health Score")

    score = health_score['score']
    rating = health_score['rating']

    # Visual gauge
    col1, col2 = st.columns([1, 2])

    with col1:
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': rating, 'font': {'size': 24}},
            gauge={
                'axis': {'range': [None, 10]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 2], 'color': "#ffcdd2"},
                    {'range': [2, 3.5], 'color': "#fff9c4"},
                    {'range': [3.5, 5], 'color': "#f0f4c3"},
                    {'range': [5, 6.5], 'color': "#c5e1a5"},
                    {'range': [6.5, 8], 'color': "#81c784"},
                    {'range': [8, 10], 'color': "#4caf50"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 5
                }
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"**Overall Score: {score}/10**")
        st.markdown(f"**Rating: {rating}**")
        st.write("")

        # Show breakdown
        st.markdown("**Score Breakdown:**")
        breakdown = health_score.get('breakdown', {})
        for category, details in breakdown.items():
            score_val = details['score']
            reason = details['reason']
            st.write(f"• **{category.replace('_', ' ').title()}**: {score_val:.1f} - {reason}")

        with st.expander("ℹ️ How is this calculated?"):
            st.code(explain_health_score())

    st.markdown("---")

    # ============================================================================
    # EARNINGS MOMENTUM (if available)
    # ============================================================================
    if earnings_analyzer and earnings_analyzer.earnings is not None:
        st.subheader("📈 Earnings Momentum")

        trends = earnings_analyzer.calculate_earnings_trends(4)
        insights = earnings_analyzer.generate_insights(4)

        if trends and trends.get('quarters'):
            # Earnings comparison
            col1, col2 = st.columns([2, 1])

            with col1:
                # Create earnings chart
                fig = go.Figure()

                quarters_display = [str(q)[:10] if hasattr(q, 'strftime') else str(q) for q in trends['quarters']]

                # EPS line
                eps_values = [e if e is not None else np.nan for e in trends['eps_actual']]
                fig.add_trace(go.Scatter(
                    x=quarters_display,
                    y=eps_values,
                    mode='lines+markers',
                    name='EPS Actual',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=10)
                ))

                # EPS estimate line (if available)
                if any(e is not None for e in trends['eps_estimate']):
                    estimate_values = [e if e is not None else np.nan for e in trends['eps_estimate']]
                    fig.add_trace(go.Scatter(
                        x=quarters_display,
                        y=estimate_values,
                        mode='lines+markers',
                        name='EPS Estimate',
                        line=dict(color='#ff7f0e', width=2, dash='dash'),
                        marker=dict(size=8)
                    ))

                fig.update_layout(
                    title="Quarterly EPS Trend",
                    xaxis_title="Quarter",
                    yaxis_title="EPS ($)",
                    height=300,
                    hovermode='x unified'
                )

                st.plotly_chart(fig, width="stretch")

            with col2:
                st.markdown("**Last 4 Quarters**")

                for i in range(min(4, len(trends['quarters']))):
                    quarter = str(trends['quarters'][i])[:10] if hasattr(trends['quarters'][i], 'strftime') else str(trends['quarters'][i])
                    eps = trends['eps_actual'][i]
                    beat_miss = trends['beat_or_miss'][i]
                    qoq_change = trends['qoq_eps_change'][i] if i < len(trends['qoq_eps_change']) else None

                    if eps is not None:
                        # Beat/miss indicator
                        if beat_miss == 'beat':
                            indicator = "✅"
                            color = "green"
                        elif beat_miss == 'miss':
                            indicator = "❌"
                            color = "red"
                        else:
                            indicator = "➖"
                            color = "gray"

                        # QoQ change
                        if qoq_change is not None:
                            change_str = f"{qoq_change*100:+.1f}%"
                            if qoq_change > 0:
                                arrow = "↗"
                            elif qoq_change < 0:
                                arrow = "↘"
                            else:
                                arrow = "→"
                        else:
                            change_str = ""
                            arrow = ""

                        st.markdown(f"{indicator} **{quarter}**: ${eps:.2f} {arrow} {change_str}")

            # Insights
            st.markdown("**💡 Insights:**")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Beat Streak", f"{insights['beat_streak']} quarters")
            with col2:
                momentum = insights['momentum']
                momentum_display = momentum.replace('_', ' ').title()
                st.metric("Momentum", momentum_display)
            with col3:
                revenue_trend = insights['revenue_trend']
                revenue_display = revenue_trend.replace('_', ' ').title()
                st.metric("Revenue Trend", revenue_display)

            # Alerts
            if insights['alerts']:
                st.warning("⚠️ **Alerts:**")
                for alert in insights['alerts']:
                    alert_type = alert['type']
                    message = alert['message']
                    if alert_type == 'warning':
                        st.warning(f"• {message}")
                    else:
                        st.info(f"• {message}")

        st.markdown("---")

    # ============================================================================
    # CORE METRICS CARDS
    # ============================================================================
    st.subheader("📊 Core Fundamentals")

    # Row 1: Valuation and Profitability
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💰 Valuation")

        if valuation:
            pe = valuation.get('pe_ratio_ttm')
            fwd_pe = valuation.get('pe_forward')
            peg = valuation.get('peg_ratio')
            pb = valuation.get('price_to_book')
            ev_ebitda = valuation.get('ev_to_ebitda')
            signal = valuation.get('valuation_signal', 'unknown')

            # Valuation signal indicator
            if signal == 'undervalued':
                st.success("✅ Signal: UNDERVALUED")
            elif signal == 'overvalued':
                st.warning("⚠️ Signal: OVERVALUED")
            else:
                st.info("ℹ️ Signal: FAIR VALUE")

            # Metrics
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                if pe:
                    st.metric("P/E (TTM)", f"{pe:.2f}")
                if peg:
                    st.metric("PEG Ratio", f"{peg:.2f}")
            with metric_col2:
                if fwd_pe:
                    change = ((fwd_pe - pe) / pe * 100) if pe and fwd_pe else None
                    st.metric("Forward P/E", f"{fwd_pe:.2f}",
                             f"{change:+.1f}%" if change else None)
                if pb:
                    st.metric("P/B Ratio", f"{pb:.2f}")

            if ev_ebitda:
                st.metric("EV/EBITDA", f"{ev_ebitda:.2f}")
        else:
            st.info("Valuation data not available")

    with col2:
        st.markdown("### 📈 Profitability")

        if profitability:
            gross_margin = profitability.get('gross_margin')
            operating_margin = profitability.get('operating_margin')
            net_margin = profitability.get('net_margin')
            roe = profitability.get('roe')
            signal = profitability.get('profitability_signal', 'unknown')

            # Profitability signal
            if signal == 'strong':
                st.success("✅ Signal: STRONG PROFITABILITY")
            elif signal == 'moderate':
                st.info("ℹ️ Signal: MODERATE PROFITABILITY")
            elif signal == 'weak':
                st.warning("⚠️ Signal: WEAK PROFITABILITY")
            else:
                st.error("❌ Signal: POOR PROFITABILITY")

            # Metrics
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                if gross_margin:
                    st.metric("Gross Margin", f"{gross_margin*100:.1f}%")
                if net_margin:
                    st.metric("Net Margin", f"{net_margin*100:.1f}%")
            with metric_col2:
                if operating_margin:
                    st.metric("Operating Margin", f"{operating_margin*100:.1f}%")
                if roe:
                    st.metric("ROE", f"{roe*100:.1f}%")
        else:
            st.info("Profitability data not available")

    st.markdown("---")

    # Row 2: Growth and Financial Health
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🚀 Growth")

        if growth:
            revenue_growth = growth.get('revenue_growth') or growth.get('revenue_growth_yoy')
            earnings_growth = growth.get('earnings_growth')
            signal = growth.get('growth_signal', 'unknown')

            # Growth signal
            if signal == 'high_growth':
                st.success("✅ Signal: HIGH GROWTH")
            elif signal == 'growing':
                st.info("ℹ️ Signal: GROWING")
            elif signal == 'slow_growth':
                st.warning("⚠️ Signal: SLOW GROWTH")
            else:
                st.error("❌ Signal: STAGNANT")

            # Metrics
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                if revenue_growth:
                    st.metric("Revenue Growth", f"{revenue_growth*100:.1f}%")
            with metric_col2:
                if earnings_growth:
                    st.metric("Earnings Growth", f"{earnings_growth*100:.1f}%")

            # QoQ metrics if available
            qoq_rev = growth.get('revenue_growth_qoq')
            qoq_eps = growth.get('earnings_growth_qoq')
            if qoq_rev or qoq_eps:
                st.markdown("**Quarter-over-Quarter:**")
                qoq_col1, qoq_col2 = st.columns(2)
                with qoq_col1:
                    if qoq_rev:
                        st.metric("Revenue QoQ", f"{qoq_rev*100:+.1f}%")
                with qoq_col2:
                    if qoq_eps:
                        st.metric("Earnings QoQ", f"{qoq_eps*100:+.1f}%")
        else:
            st.info("Growth data not available")

    with col2:
        st.markdown("### 🏥 Financial Health")

        if financial_health:
            current_ratio = financial_health.get('current_ratio')
            debt_to_equity = financial_health.get('debt_to_equity')
            interest_coverage = financial_health.get('interest_coverage')
            cash = financial_health.get('total_cash')
            signal = financial_health.get('health_signal', 'unknown')

            # Health signal
            if signal == 'strong':
                st.success("✅ Signal: STRONG BALANCE SHEET")
            elif signal == 'moderate':
                st.info("ℹ️ Signal: MODERATE HEALTH")
            elif signal == 'fair':
                st.warning("⚠️ Signal: FAIR HEALTH")
            else:
                st.error("❌ Signal: WEAK BALANCE SHEET")

            # Metrics
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                if current_ratio:
                    ratio_color = "🟢" if current_ratio > 1.5 else "🟡" if current_ratio > 1.0 else "🔴"
                    st.metric(f"{ratio_color} Current Ratio", f"{current_ratio:.2f}")
                if interest_coverage:
                    coverage_color = "🟢" if interest_coverage > 5 else "🟡" if interest_coverage > 2 else "🔴"
                    st.metric(f"{coverage_color} Interest Coverage", f"{interest_coverage:.1f}x")
            with metric_col2:
                if debt_to_equity:
                    # Normalize debt_to_equity (sometimes it's in percentage form)
                    de_normalized = debt_to_equity / 100 if debt_to_equity > 10 else debt_to_equity
                    de_color = "🟢" if de_normalized < 0.5 else "🟡" if de_normalized < 1.0 else "🔴"
                    st.metric(f"{de_color} Debt/Equity", f"{de_normalized:.2f}")
                if cash:
                    st.metric("Total Cash", f"${cash/1e9:.1f}B")
        else:
            st.info("Financial health data not available")

    st.markdown("---")

    # ============================================================================
    # EXPANDABLE SECTIONS
    # ============================================================================

    # Detailed Balance Sheet
    with st.expander("🏦 Balance Sheet Deep Dive"):
        if analyzer.balance_sheet is not None and not analyzer.balance_sheet.empty:
            st.dataframe(analyzer.balance_sheet, width="stretch")
        else:
            st.info("Balance sheet data not available")

    # Cash Flow Statement
    with st.expander("💵 Cash Flow Analysis"):
        if analyzer.cash_flow is not None and not analyzer.cash_flow.empty:
            st.dataframe(analyzer.cash_flow, width="stretch")

            # FCF analysis
            if 'Free Cash Flow' in analyzer.cash_flow.index:
                fcf = analyzer.cash_flow.loc['Free Cash Flow']
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[str(c)[:10] for c in fcf.index],
                    y=fcf.values,
                    marker_color='lightseagreen'
                ))
                fig.update_layout(
                    title="Free Cash Flow Trend",
                    xaxis_title="Quarter",
                    yaxis_title="FCF ($)",
                    height=300
                )
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("Cash flow data not available")

    # Company Description
    if company_info.get('description'):
        with st.expander("ℹ️ About the Company"):
            st.write(company_info['description'])
            if company_info.get('website'):
                st.write(f"**Website:** {company_info['website']}")
            if company_info.get('employees'):
                st.write(f"**Employees:** {company_info['employees']:,}")

else:
    # No data loaded yet - show introduction
    st.markdown("---")
    st.markdown("""
    ## Welcome to Fundamentals Analysis

    This dashboard provides comprehensive fundamental analysis including:

    **📊 Financial Health Score**: Overall assessment (0-10) based on valuation, profitability, growth, and balance sheet strength

    **📈 Earnings Momentum**: Quarterly earnings trends, beat/miss analysis, and forward momentum indicators

    **💰 Valuation Metrics**: P/E, PEG, P/B, EV/EBITDA with signals for under/over valuation

    **📊 Profitability**: Margin analysis (gross, operating, net) and ROE

    **🚀 Growth**: Revenue and earnings growth with QoQ and YoY comparisons

    **🏥 Financial Health**: Liquidity ratios, debt levels, cash position

    **💡 Actionable Insights**: Automated alerts and recommendations based on fundamental trends

    ---

    **Enter a ticker symbol above to begin analysis**
    """)
