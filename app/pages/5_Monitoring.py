"""Portfolio Monitoring Page - Rebalancing and performance tracking."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.portfolio.rebalancer import PortfolioRebalancer, PerformanceAttributor, DCAScheduler

st.set_page_config(page_title="Monitoring", page_icon=None, layout="wide")

st.title("Portfolio Monitoring")

# Check data
if 'portfolio_data' not in st.session_state or st.session_state.portfolio_data is None:
    st.warning("Please load portfolio data first.")
    st.stop()

data = st.session_state.portfolio_data
returns = data['returns']
prices = data['prices']

# Get target and current weights
if 'optimization_result' in st.session_state and st.session_state.optimization_result:
    target_weights = st.session_state.optimization_result['weights']
    st.info("Monitoring **optimized portfolio**")
else:
    target_weights = pd.Series(1/len(returns.columns), index=returns.columns)
    st.warning("Using equal weights as target (run optimization for custom allocation)")

# Check if we have actual current holdings
has_current_holdings = ('current_portfolio_weights' in st.session_state and
                        st.session_state.current_portfolio_weights is not None)

if has_current_holdings:
    st.success("You have current holdings entered - using actual portfolio for monitoring")

st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["Rebalancing", "Performance Attribution", "DCA Scheduler"])

with tab1:
    st.subheader("Rebalancing Analysis")

    # Simulate current weights (drifted from target)
    st.markdown("**Current vs Target Allocation**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("Enter current weights (or use simulated drift):")

        use_simulated = st.checkbox("Simulate drift from returns", value=True)

        if use_simulated:
            # Simulate drift using recent returns
            drift_period = st.slider("Drift period (days)", 30, 252, 90)
            recent_returns = returns.tail(drift_period)
            cumulative = (1 + recent_returns).prod()

            # Current weights after drift
            drifted_value = target_weights * cumulative
            current_weights = drifted_value / drifted_value.sum()
        else:
            # Manual input
            current_weights = target_weights.copy()
            st.markdown("Adjust weights manually:")
            for ticker in target_weights.index:
                current_weights[ticker] = st.number_input(
                    f"{ticker}",
                    0.0, 1.0,
                    float(target_weights[ticker]),
                    0.01
                )

    with col2:
        # Comparison chart
        comparison = pd.DataFrame({
            'Target': target_weights * 100,
            'Current': current_weights * 100
        })

        fig = go.Figure(data=[
            go.Bar(name='Target', x=comparison.index, y=comparison['Target']),
            go.Bar(name='Current', x=comparison.index, y=comparison['Current'])
        ])
        fig.update_layout(
            title="Weight Comparison",
            xaxis_title="Asset",
            yaxis_title="Weight (%)",
            barmode='group'
        )
        st.plotly_chart(fig, width="stretch")

    # Rebalancing analysis
    st.markdown("---")
    st.markdown("**Rebalancing Recommendations**")

    col1, col2 = st.columns(2)

    with col1:
        threshold = st.slider("Rebalancing Threshold (%)", 1, 20, 5) / 100
        portfolio_value = st.number_input(
            "Portfolio Value ($)",
            1000, 10000000,
            int(st.session_state.get('settings', {}).get('total_capital', 10000))
        )

    # Create rebalancer
    current_prices = prices.iloc[-1]
    rebalancer = PortfolioRebalancer(
        target_weights,
        current_weights,
        current_prices,
        portfolio_value,
        threshold
    )

    # Check if rebalancing needed
    needs_rebal, details = rebalancer.needs_rebalancing()

    with col2:
        if needs_rebal:
            st.error(f"Rebalancing NEEDED - Max deviation: {details['max_deviation']*100:.1f}%")
        else:
            st.success(f"Portfolio within threshold - Max deviation: {details['max_deviation']*100:.1f}%")

    # Trade recommendations
    if st.button("Generate Trade Recommendations"):
        trades = rebalancer.calculate_trades()
        summary = rebalancer.get_trade_summary()

        # Filter to actionable trades
        actionable = trades[trades['Action'] != 'HOLD'].copy()

        if len(actionable) > 0:
            # Display trades
            display_trades = actionable[['Action', 'Current Weight', 'Target Weight', 'Shares', 'Trade Value']].copy()
            display_trades['Current Weight'] = (display_trades['Current Weight'] * 100).round(2)
            display_trades['Target Weight'] = (display_trades['Target Weight'] * 100).round(2)
            display_trades['Shares'] = display_trades['Shares'].round(2)
            display_trades['Trade Value'] = display_trades['Trade Value'].apply(lambda x: f"${x:,.2f}")

            st.dataframe(display_trades, width="stretch")

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Buys", f"${summary['total_buy_value']:,.0f}")
            with col2:
                st.metric("Total Sells", f"${summary['total_sell_value']:,.0f}")
            with col3:
                st.metric("Turnover", f"{summary['turnover_pct']*100:.1f}%")
        else:
            st.info("No trades needed - portfolio is balanced.")

with tab2:
    st.subheader("Performance Attribution")

    # Calculate portfolio returns
    portfolio_returns = (returns * target_weights).sum(axis=1)

    # Create attributor
    attributor = PerformanceAttributor(
        portfolio_returns,
        returns,
        target_weights
    )

    col1, col2 = st.columns(2)

    with col1:
        # Return contribution
        st.markdown("**Return Contribution by Asset**")
        contribution = attributor.contribution_analysis()

        fig = go.Figure(data=[
            go.Bar(
                x=contribution.index,
                y=contribution['Contribution'] * 100,
                marker_color=['green' if x > 0 else 'red' for x in contribution['Contribution']]
            )
        ])
        fig.update_layout(
            title="Return Contribution (%)",
            xaxis_title="Asset",
            yaxis_title="Contribution (%)"
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        # Risk contribution
        st.markdown("**Risk Contribution by Asset**")
        risk_contrib = attributor.risk_contribution()

        fig = px.pie(
            values=risk_contrib['Risk Contribution %'],
            names=risk_contrib.index,
            title="Risk Contribution"
        )
        st.plotly_chart(fig, width="stretch")

    # Detailed tables
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Return Attribution**")
        display_contrib = contribution.copy()
        display_contrib['Weight'] = (display_contrib['Weight'] * 100).round(2)
        display_contrib['Return'] = (display_contrib['Return'] * 100).round(2)
        display_contrib['Contribution'] = (display_contrib['Contribution'] * 100).round(4)
        display_contrib['Contribution %'] = display_contrib['Contribution %'].round(2)
        st.dataframe(display_contrib, width="stretch")

    with col2:
        st.markdown("**Risk Attribution**")
        display_risk = risk_contrib.copy()
        display_risk['Weight'] = (display_risk['Weight'] * 100).round(2)
        display_risk['Volatility'] = (display_risk['Volatility'] * 100).round(2)
        display_risk['Risk Contribution'] = display_risk['Risk Contribution'].round(4)
        display_risk['Risk Contribution %'] = display_risk['Risk Contribution %'].round(2)
        st.dataframe(display_risk, width="stretch")

with tab3:
    st.subheader("Dollar-Cost Averaging Scheduler")

    st.markdown("Plan systematic investments over time.")

    col1, col2 = st.columns(2)

    with col1:
        total_investment = st.number_input("Total Amount to Invest ($)", 1000, 1000000, 10000)
        n_periods = st.number_input("Number of Periods", 1, 52, 12)

    with col2:
        frequency = st.selectbox("Frequency", ["weekly", "biweekly", "monthly"])

    # Create scheduler
    scheduler = DCAScheduler(
        total_investment,
        target_weights,
        frequency,
        n_periods
    )

    if st.button("Generate DCA Schedule"):
        schedule = scheduler.generate_schedule()
        summary = scheduler.get_summary()

        # Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Per Period", f"${summary['amount_per_period']:,.2f}")
        with col2:
            st.metric("Periods", summary['n_periods'])
        with col3:
            st.metric("Frequency", summary['frequency'].title())

        # Schedule table
        st.markdown("**Investment Schedule**")

        # Format for display
        display_schedule = schedule.copy()
        display_schedule['Total Amount'] = display_schedule['Total Amount'].apply(lambda x: f"${x:,.2f}")

        for ticker in target_weights.index:
            if ticker in display_schedule.columns:
                display_schedule[ticker] = display_schedule[ticker].apply(lambda x: f"${x:,.2f}")

        st.dataframe(display_schedule, width="stretch")

        # Download
        csv = schedule.to_csv(index=False)
        st.download_button(
            "Download Schedule (CSV)",
            csv,
            "dca_schedule.csv",
            "text/csv"
        )

# Portfolio performance summary
st.markdown("---")
st.subheader("Performance Summary")

# Calculate cumulative returns
portfolio_cumulative = (1 + (returns * target_weights).sum(axis=1)).cumprod()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=portfolio_cumulative.index,
    y=portfolio_cumulative.values,
    mode='lines',
    name='Portfolio',
    line=dict(color='blue', width=2)
))

fig.update_layout(
    title="Portfolio Cumulative Performance",
    xaxis_title="Date",
    yaxis_title="Growth of $1"
)
st.plotly_chart(fig, width="stretch")

# Period returns
st.markdown("**Period Returns**")
total_return = portfolio_cumulative.iloc[-1] - 1
annual_return = (1 + total_return) ** (252 / len(returns)) - 1

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Return", f"{total_return*100:.2f}%")
with col2:
    st.metric("Annualized Return", f"{annual_return*100:.2f}%")
with col3:
    vol = (returns * target_weights).sum(axis=1).std() * np.sqrt(252)
    st.metric("Volatility", f"{vol*100:.2f}%")
