"""Risk Analytics Dashboard - Comprehensive risk metrics and visualizations."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.risk.metrics import RiskMetrics
from src.risk.var import VaRCalculator
from src.risk.garch import GARCHModel, ewma_volatility

st.set_page_config(page_title="Risk Analytics", page_icon="📊", layout="wide")

st.title("Risk Analytics Dashboard")

# Check data
if 'portfolio_data' not in st.session_state or st.session_state.portfolio_data is None:
    st.warning("Please load portfolio data first.")
    st.stop()

data = st.session_state.portfolio_data
returns = data['returns']
prices = data['prices']

# Calculate portfolio returns based on what's available
if 'optimization_result' in st.session_state and st.session_state.optimization_result:
    # Use optimized weights
    weights = st.session_state.optimization_result['weights']
    portfolio_returns = (returns * weights).sum(axis=1)
    st.info(f"📊 Analyzing **optimized portfolio** with {(weights > 0.001).sum()} positions")
elif 'current_portfolio_weights' in st.session_state and st.session_state.current_portfolio_weights is not None:
    # Use actual holdings weights
    weights = st.session_state.current_portfolio_weights
    portfolio_returns = (returns * weights).sum(axis=1)
    st.info(f"💼 Analyzing **your current holdings** with {len(weights)} positions")
else:
    # Equal weight fallback
    weights = pd.Series(1/len(returns.columns), index=returns.columns)
    portfolio_returns = returns.mean(axis=1)
    st.warning("⚠️ Using equal-weight portfolio. Enter holdings in Portfolio Input or run optimization for accurate analysis.")

# Initialize calculators
rm = RiskMetrics(returns)
var_calc = VaRCalculator(returns)

st.markdown("---")

# Key risk metrics
st.subheader("Key Risk Metrics")

col1, col2, col3, col4 = st.columns(4)

# Calculate metrics
annual_vol = returns.std() * np.sqrt(252)
sharpe = rm.sharpe_ratio()
max_dd = rm.max_drawdown(prices)
var_95 = var_calc.historical_var(0.95)

# Portfolio metrics
port_vol = portfolio_returns.std() * np.sqrt(252)
port_sharpe = (portfolio_returns.mean() * 252) / port_vol
port_prices = (1 + portfolio_returns).cumprod()
port_mdd = (port_prices / port_prices.cummax() - 1).min()

with col1:
    st.metric("Portfolio Volatility", f"{port_vol*100:.2f}%")
with col2:
    st.metric("Sharpe Ratio", f"{port_sharpe:.3f}")
with col3:
    st.metric("Max Drawdown", f"{port_mdd*100:.2f}%")
with col4:
    port_var = portfolio_returns.quantile(0.05)
    st.metric("VaR (95%)", f"{port_var*100:.2f}%")

st.markdown("---")

# Tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs(["VaR Analysis", "Drawdowns", "Correlations", "Volatility"])

with tab1:
    st.subheader("Value at Risk Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # VaR comparison
        confidence = st.slider("Confidence Level", 90, 99, 95) / 100

        var_results = var_calc.calculate_all(confidence)

        st.markdown("**VaR by Method (Daily)**")
        var_display = var_results.copy() * 100
        var_display = var_display.round(3)
        st.dataframe(var_display, use_container_width=True)

    with col2:
        # VaR distribution
        st.markdown("**Portfolio Return Distribution**")

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=portfolio_returns * 100,
            nbinsx=50,
            name='Returns'
        ))

        # Add VaR line
        var_val = portfolio_returns.quantile(1 - confidence) * 100
        fig.add_vline(x=var_val, line_dash="dash", line_color="red",
                      annotation_text=f"VaR ({confidence*100:.0f}%)")

        fig.update_layout(
            xaxis_title="Return (%)",
            yaxis_title="Frequency",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Drawdown Analysis")

    # Calculate drawdowns
    drawdown = rm.drawdown_series(prices)
    port_dd = (port_prices / port_prices.cummax() - 1)

    col1, col2 = st.columns(2)

    with col1:
        # Drawdown chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=port_dd.index,
            y=port_dd.values * 100,
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='red')
        ))
        fig.update_layout(
            title="Portfolio Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Drawdown metrics table
        st.markdown("**Drawdown Metrics**")

        dd_metrics = pd.DataFrame({
            'Max Drawdown': rm.max_drawdown(prices) * 100,
            'Avg Drawdown': rm.average_drawdown(prices) * 100,
            'Ulcer Index': rm.ulcer_index(prices) * 100,
        }).round(2)

        st.dataframe(dd_metrics, use_container_width=True)

        # Calmar ratio
        calmar = rm.calmar_ratio(prices)
        st.markdown("**Calmar Ratio (Return/MDD)**")
        calmar_df = pd.DataFrame({'Calmar': calmar}).round(3)
        st.dataframe(calmar_df, use_container_width=True)

with tab3:
    st.subheader("Correlation Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Correlation heatmap
        corr = returns.corr()

        fig = px.imshow(
            corr,
            text_auto='.2f',
            aspect='auto',
            color_continuous_scale='RdBu_r',
            title="Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Correlation stats
        st.markdown("**Correlation Statistics**")

        avg_corr = rm.average_correlation()
        st.metric("Average Correlation", f"{avg_corr:.3f}")

        # Most correlated pairs
        st.markdown("**Highest Correlations**")
        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                corr_pairs.append({
                    'Pair': f"{corr.columns[i]} - {corr.columns[j]}",
                    'Correlation': corr.iloc[i, j]
                })

        pairs_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)
        st.dataframe(pairs_df.head(5), use_container_width=True)

with tab4:
    st.subheader("Volatility Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Rolling volatility
        window = st.slider("Rolling Window (days)", 20, 120, 60)

        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)

        fig = go.Figure()
        for col in rolling_vol.columns[:5]:  # Limit to 5 for clarity
            fig.add_trace(go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol[col] * 100,
                name=col,
                mode='lines'
            ))

        fig.update_layout(
            title=f"Rolling {window}-Day Volatility",
            xaxis_title="Date",
            yaxis_title="Annualized Volatility (%)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # EWMA volatility
        st.markdown("**EWMA Volatility**")

        ewma_vol = ewma_volatility(returns, decay=0.94)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ewma_vol.index,
            y=(returns * weights).sum(axis=1).rolling(20).std() * np.sqrt(252) * 100,
            name='Portfolio EWMA Vol',
            line=dict(color='blue')
        ))

        fig.update_layout(
            title="Portfolio EWMA Volatility",
            xaxis_title="Date",
            yaxis_title="Volatility (%)"
        )
        st.plotly_chart(fig, use_container_width=True)

# Performance ratios summary
st.markdown("---")
st.subheader("Performance Ratios Summary")

summary = rm.summary_table(prices)
summary_display = summary.copy()
summary_display['Annual Return'] = (summary_display['Annual Return'] * 100).round(2)
summary_display['Annual Volatility'] = (summary_display['Annual Volatility'] * 100).round(2)
summary_display['Sharpe Ratio'] = summary_display['Sharpe Ratio'].round(3)
summary_display['Sortino Ratio'] = summary_display['Sortino Ratio'].round(3)
summary_display['Max Drawdown'] = (summary_display['Max Drawdown'] * 100).round(2)
summary_display['Calmar Ratio'] = summary_display['Calmar Ratio'].round(3)
summary_display['Skewness'] = summary_display['Skewness'].round(3)
summary_display['Kurtosis'] = summary_display['Kurtosis'].round(3)

st.dataframe(summary_display, use_container_width=True)

# Export
st.markdown("---")
csv = summary.to_csv()
st.download_button(
    "Download Risk Metrics (CSV)",
    csv,
    "risk_metrics.csv",
    "text/csv"
)
