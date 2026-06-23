"""Optimization Results Page - View optimized portfolio allocation."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.optimization.optimizers import PortfolioOptimizer
from src.optimization.constraints import PortfolioConstraints
from src.portfolio.calculator import PositionCalculator

st.set_page_config(page_title="Optimization", page_icon=None, layout="wide")

st.title("Portfolio Optimization")

# Check if data is loaded
if 'portfolio_data' not in st.session_state or st.session_state.portfolio_data is None:
    st.warning("Please load portfolio data first on the Portfolio Input page.")
    st.stop()

data = st.session_state.portfolio_data
settings = st.session_state.get('settings', {})
returns = data['returns']

# Method mapping
METHOD_MAP = {
    "Maximum Sharpe Ratio": "max_sharpe",
    "Minimum Volatility": "min_volatility",
    "Risk Parity": "risk_parity",
    "Hierarchical Risk Parity (HRP)": "hrp",
    "Maximum Diversification": "max_diversification",
    "Equal Weight": "equal_weight",
}

# Run optimization
st.subheader("Optimization Settings")
col1, col2, col3 = st.columns(3)

with col1:
    method_name = st.selectbox(
        "Method",
        list(METHOD_MAP.keys()),
        index=list(METHOD_MAP.keys()).index(settings.get('optimization_method', "Maximum Sharpe Ratio"))
    )
    method = METHOD_MAP[method_name]

with col2:
    capital = st.number_input(
        "Capital ($)",
        value=settings.get('total_capital', 10000),
        min_value=1000
    )

with col3:
    rf_rate = st.number_input(
        "Risk-Free Rate",
        value=settings.get('risk_free_rate', 0.02),
        format="%.3f"
    )

# Run button
if st.button("Run Optimization", type="primary"):
    with st.spinner("Optimizing portfolio..."):
        # Create constraints
        constraints = PortfolioConstraints(
            max_weight=settings.get('max_weight', 0.4),
            min_position_size=settings.get('min_weight', 0.0),
        )

        # Run optimizer
        optimizer = PortfolioOptimizer(returns, rf_rate)
        result = optimizer.optimize(method, constraints)

        # Store result and data for other pages
        st.session_state.optimization_result = result
        st.session_state.optimizer = optimizer
        st.session_state.weights = result['weights']
        st.session_state.returns = returns
        st.session_state.prices = data.get('prices')
        st.session_state.portfolio_value = capital

        # Store metrics for reports
        st.session_state.metrics = {
            'annual_return': result['expected_return'],
            'annual_volatility': result['volatility'],
            'sharpe_ratio': result['sharpe_ratio']
        }

        st.success("Optimization complete!")

# Display results if available
if 'optimization_result' in st.session_state and st.session_state.optimization_result is not None:
    result = st.session_state.optimization_result
    weights = result['weights']

    st.markdown("---")

    # Key metrics
    st.subheader("Portfolio Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Expected Return", f"{result['expected_return']*100:.2f}%")
    with col2:
        st.metric("Volatility", f"{result['volatility']*100:.2f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.3f}")
    with col4:
        st.metric("Positions", f"{(weights > 0.001).sum()}")

    st.markdown("---")

    # Rebalancing Section (if user has current holdings)
    if st.session_state.get('current_portfolio_weights') is not None:
        st.subheader(" Rebalancing Recommendations")

        current_weights = st.session_state.current_portfolio_weights
        optimal_weights = weights

        # Create comparison dataframe
        rebalance_data = []
        for ticker in set(current_weights.index) | set(optimal_weights.index):
            current = current_weights.get(ticker, 0)
            optimal = optimal_weights.get(ticker, 0)
            difference = optimal - current

            # Get current holdings
            holdings_dict = st.session_state.get('current_holdings', {})
            current_shares = holdings_dict.get(ticker, 0)

            # Calculate target shares
            current_price = data['prices'].iloc[-1].get(ticker, 0)
            target_value = capital * optimal
            target_shares = target_value / current_price if current_price > 0 else 0
            shares_change = target_shares - current_shares

            rebalance_data.append({
                'Ticker': ticker,
                'Current Weight': current,
                'Target Weight': optimal,
                'Change': difference,
                'Current Shares': current_shares,
                'Target Shares': target_shares,
                'Shares to Trade': shares_change
            })

        rebalance_df = pd.DataFrame(rebalance_data).set_index('Ticker')
        rebalance_df = rebalance_df.sort_values('Change', key=abs, ascending=False)

        # Visual comparison
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Current vs Target Allocation**")

            # Filter for display
            significant = rebalance_df[(rebalance_df['Current Weight'] > 0.001) |
                                       (rebalance_df['Target Weight'] > 0.001)]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Current',
                x=significant.index,
                y=significant['Current Weight'] * 100,
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                name='Target',
                x=significant.index,
                y=significant['Target Weight'] * 100,
                marker_color='steelblue'
            ))
            fig.update_layout(
                barmode='group',
                yaxis_title="Weight (%)",
                xaxis_title="Asset"
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.markdown("**Required Changes**")

            # Show changes table
            changes_display = rebalance_df[['Current Weight', 'Target Weight', 'Change', 'Shares to Trade']].copy()
            changes_display['Current Weight'] = (changes_display['Current Weight'] * 100).round(2).astype(str) + '%'
            changes_display['Target Weight'] = (changes_display['Target Weight'] * 100).round(2).astype(str) + '%'
            changes_display['Change'] = changes_display['Change'].apply(
                lambda x: f"+{x*100:.2f}%" if x > 0 else f"{x*100:.2f}%"
            )
            changes_display['Shares to Trade'] = changes_display['Shares to Trade'].round(2)

            st.dataframe(changes_display, width="stretch")

        # Action summary
        st.markdown("**Trading Actions:**")

        buy_positions = rebalance_df[rebalance_df['Shares to Trade'] > 0.5]
        sell_positions = rebalance_df[rebalance_df['Shares to Trade'] < -0.5]

        col1, col2 = st.columns(2)

        with col1:
            if not buy_positions.empty:
                st.markdown("** Buy:**")
                for ticker, row in buy_positions.iterrows():
                    st.write(f"- **{ticker}**: Buy {row['Shares to Trade']:.0f} shares")
            else:
                st.info("No buying needed")

        with col2:
            if not sell_positions.empty:
                st.markdown("** Sell:**")
                for ticker, row in sell_positions.iterrows():
                    st.write(f"- **{ticker}**: Sell {abs(row['Shares to Trade']):.0f} shares")
            else:
                st.info("No selling needed")

    st.markdown("---")

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Allocation")

        # Filter small positions for chart
        display_weights = weights[weights > 0.001]

        fig = px.pie(
            values=display_weights.values,
            names=display_weights.index,
            title="Portfolio Allocation"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Weight Comparison")

        fig = go.Figure(data=[
            go.Bar(
                x=weights.index,
                y=weights.values * 100,
                marker_color='steelblue'
            )
        ])
        fig.update_layout(
            title="Asset Weights (%)",
            xaxis_title="Asset",
            yaxis_title="Weight (%)"
        )
        st.plotly_chart(fig, width="stretch")

    # Position sizing table
    st.subheader("Position Sizing")

    # Get current prices (use last price from data)
    prices = data['prices'].iloc[-1]

    calc = PositionCalculator(
        capital,
        weights,
        prices,
        allow_fractional=settings.get('allow_fractional', False)
    )
    positions = calc.calculate_positions()
    summary = calc.get_summary()

    # Format for display
    display_df = positions[['Weight', 'Price', 'Shares', 'Actual Amount', 'Remainder']].copy()
    display_df['Weight'] = (display_df['Weight'] * 100).round(2).astype(str) + '%'
    display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:,.2f}")
    display_df['Actual Amount'] = display_df['Actual Amount'].apply(lambda x: f"${x:,.2f}")
    display_df['Remainder'] = display_df['Remainder'].apply(lambda x: f"${x:,.2f}")

    st.dataframe(display_df, width="stretch")

    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Invested", f"${summary['total_invested']:,.2f}")
    with col2:
        st.metric("Unallocated", f"${summary['unallocated_cash']:,.2f}")
    with col3:
        st.metric("Unallocated %", f"{summary['unallocated_pct']*100:.2f}%")

    # Efficient Frontier
    st.markdown("---")
    st.subheader("Efficient Frontier")

    if st.button("Calculate Efficient Frontier"):
        with st.spinner("Calculating frontier..."):
            optimizer = st.session_state.optimizer
            frontier = optimizer.efficient_frontier(n_points=30)

            if frontier.empty or 'volatility' not in frontier.columns:
                st.error("Could not calculate efficient frontier. Try a different optimization method or check your data.")
            else:
                fig = go.Figure()

                # Frontier line
                fig.add_trace(go.Scatter(
                    x=frontier['volatility'] * 100,
                    y=frontier['return'] * 100,
                    mode='lines',
                    name='Efficient Frontier',
                    line=dict(color='blue', width=2)
                ))

                # Current portfolio
                fig.add_trace(go.Scatter(
                    x=[result['volatility'] * 100],
                    y=[result['expected_return'] * 100],
                    mode='markers',
                    name='Optimal Portfolio',
                    marker=dict(size=15, color='red', symbol='star')
                ))

                # Individual assets
                asset_returns = returns.mean() * 252 * 100
                asset_vols = returns.std() * np.sqrt(252) * 100

                fig.add_trace(go.Scatter(
                    x=asset_vols,
                    y=asset_returns,
                    mode='markers+text',
                    name='Individual Assets',
                    text=returns.columns,
                    textposition='top center',
                    marker=dict(size=8, color='gray')
                ))

                fig.update_layout(
                    title="Efficient Frontier",
                    xaxis_title="Volatility (%)",
                    yaxis_title="Expected Return (%)",
                    showlegend=True
                )

                st.plotly_chart(fig, width="stretch")

    # Method comparison
    st.markdown("---")
    st.subheader("Method Comparison")

    if st.button("Compare All Methods"):
        with st.spinner("Comparing methods..."):
            optimizer = st.session_state.optimizer
            comparison = optimizer.compare_methods()

            # Format display
            display_comp = comparison.copy()
            display_comp['expected_return'] = (display_comp['expected_return'] * 100).round(2)
            display_comp['volatility'] = (display_comp['volatility'] * 100).round(2)
            display_comp['sharpe_ratio'] = display_comp['sharpe_ratio'].round(3)
            display_comp['max_weight'] = (display_comp['max_weight'] * 100).round(1)

            display_comp.columns = ['Method', 'Return (%)', 'Volatility (%)', 'Sharpe', 'Max Weight (%)', 'Min Weight', 'Positions']

            st.dataframe(display_comp, width="stretch")

    # Export
    st.markdown("---")
    st.subheader("Export Results")

    col1, col2 = st.columns(2)

    with col1:
        # CSV export
        csv = positions.to_csv()
        st.download_button(
            "Download Positions (CSV)",
            csv,
            "portfolio_positions.csv",
            "text/csv"
        )

    with col2:
        # Weights export
        weights_df = pd.DataFrame({'Ticker': weights.index, 'Weight': weights.values})
        csv_weights = weights_df.to_csv(index=False)
        st.download_button(
            "Download Weights (CSV)",
            csv_weights,
            "portfolio_weights.csv",
            "text/csv"
        )

else:
    st.info("Click 'Run Optimization' to generate results.")
