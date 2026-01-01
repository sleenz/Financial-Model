"""Portfolio Input Page - Enter tickers, dates, and parameters."""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.data_manager import DataManager
from src.utils.helpers import validate_tickers
from src.portfolio.holdings import HoldingsTracker

st.set_page_config(page_title="Portfolio Input", page_icon="📝", layout="wide")

st.title("Portfolio Input")
st.markdown("Enter your portfolio parameters to begin optimization.")

# Initialize session state
if 'tickers' not in st.session_state:
    st.session_state.tickers = []
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None
if 'current_holdings' not in st.session_state:
    st.session_state.current_holdings = {}
if 'holdings_tracker' not in st.session_state:
    st.session_state.holdings_tracker = None

# Sidebar for quick settings
with st.sidebar:
    st.header("Quick Settings")
    preset = st.selectbox(
        "Load Preset Portfolio",
        ["Custom", "Tech Giants", "Diversified ETFs", "Blue Chips"]
    )

    if preset == "Tech Giants":
        default_tickers = "AAPL, MSFT, GOOGL, AMZN, NVDA"
    elif preset == "Diversified ETFs":
        default_tickers = "SPY, QQQ, IWM, EFA, AGG"
    elif preset == "Blue Chips":
        default_tickers = "JNJ, PG, KO, WMT, JPM"
    else:
        default_tickers = ""

# Main input form
col1, col2 = st.columns(2)

with col1:
    st.subheader("Asset Selection")

    # Ticker input
    ticker_input = st.text_area(
        "Enter Tickers (comma-separated)",
        value=default_tickers if preset != "Custom" else "",
        height=100,
        help="Enter stock symbols separated by commas (e.g., AAPL, MSFT, GOOGL)"
    )

    # File upload alternative
    uploaded_file = st.file_uploader(
        "Or upload CSV with tickers",
        type=['csv'],
        help="CSV should have a column named 'ticker' or 'symbol'"
    )

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if 'ticker' in df.columns:
                ticker_input = ', '.join(df['ticker'].tolist())
            elif 'symbol' in df.columns:
                ticker_input = ', '.join(df['symbol'].tolist())
            st.success(f"Loaded {len(df)} tickers from file")
        except Exception as e:
            st.error(f"Error reading file: {e}")

with col2:
    st.subheader("Time Period")

    # Date range
    col2a, col2b = st.columns(2)
    with col2a:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365*3),
            max_value=datetime.now()
        )
    with col2b:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )

    # Quick period selection
    period = st.selectbox(
        "Or Select Period",
        ["Custom", "1 Year", "2 Years", "3 Years", "5 Years"]
    )

    if period != "Custom":
        years = int(period.split()[0])
        start_date = datetime.now() - timedelta(days=365*years)
        end_date = datetime.now()

st.markdown("---")

# Current Holdings Input Section
st.subheader("📊 Current Holdings (Optional)")
st.markdown("Enter your current stock holdings to analyze portfolio diversity and concentration.")

with st.expander("Enter Current Holdings", expanded=False):
    st.markdown("**Add your current stock positions:**")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        holding_ticker = st.text_input(
            "Stock Ticker",
            key="holding_ticker",
            help="Enter stock symbol (e.g., NVDA, AAPL)"
        ).upper()

    with col2:
        holding_shares = st.number_input(
            "Number of Shares",
            min_value=0.0,
            value=0.0,
            step=1.0,
            key="holding_shares",
            help="Number of shares you own"
        )

    with col3:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("Add Holding", type="secondary"):
            if holding_ticker and holding_shares > 0:
                if 'current_holdings' not in st.session_state:
                    st.session_state.current_holdings = {}
                st.session_state.current_holdings[holding_ticker] = holding_shares
                st.success(f"Added {holding_shares} shares of {holding_ticker}")
                st.rerun()
            else:
                st.error("Please enter a valid ticker and number of shares")

    # Display current holdings
    if st.session_state.current_holdings:
        st.markdown("**Your Current Holdings:**")

        holdings_df = pd.DataFrame([
            {"Ticker": ticker, "Shares": shares}
            for ticker, shares in st.session_state.current_holdings.items()
        ])

        # Add delete buttons
        for idx, row in holdings_df.iterrows():
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{row['Ticker']}**")
            with col2:
                st.write(f"{row['Shares']:.2f} shares")
            with col3:
                if st.button(f"Remove", key=f"remove_{row['Ticker']}"):
                    del st.session_state.current_holdings[row['Ticker']]
                    st.rerun()

        # Clear all button
        if st.button("Clear All Holdings", type="secondary"):
            st.session_state.current_holdings = {}
            st.session_state.holdings_tracker = None
            st.rerun()

        # Analyze diversity button
        st.markdown("---")
        if st.button("Analyze Portfolio Diversity", type="primary"):
            # Fetch current prices for holdings
            try:
                dm = DataManager(show_progress=False)
                tickers_list = list(st.session_state.current_holdings.keys())

                with st.spinner("Fetching current prices..."):
                    prices = dm.get_price_data(
                        tickers_list,
                        datetime.now() - timedelta(days=5),
                        datetime.now()
                    )
                    current_prices = prices.iloc[-1]  # Get most recent prices

                    # Create holdings tracker
                    tracker = HoldingsTracker(
                        st.session_state.current_holdings,
                        current_prices
                    )
                    st.session_state.holdings_tracker = tracker

                    st.success("Diversity analysis complete!")

            except Exception as e:
                st.error(f"Error fetching prices: {e}")
                # Create tracker without prices
                tracker = HoldingsTracker(st.session_state.current_holdings)
                st.session_state.holdings_tracker = tracker

# Show diversity analysis if available
if st.session_state.holdings_tracker is not None:
    st.markdown("---")
    st.subheader("📈 Portfolio Diversity Analysis")

    tracker = st.session_state.holdings_tracker
    metrics = tracker.calculate_diversity_metrics()
    rating = tracker.get_diversity_rating()
    recommendations = tracker.get_diversity_recommendations()

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Number of Stocks", metrics['num_holdings'])
    with col2:
        st.metric("Diversity Rating", rating)
    with col3:
        st.metric("Effective Stocks", f"{metrics['effective_stocks']:.2f}")
    with col4:
        st.metric("Top 3 Concentration", f"{metrics['top_3_concentration']*100:.1f}%")

    # Holdings breakdown
    st.markdown("**Holdings Breakdown:**")
    holdings_df = tracker.get_holdings_dataframe().sort_values('Weight', ascending=False)

    # Format for display
    display_df = holdings_df.copy()
    display_df['Shares'] = display_df['Shares'].apply(lambda x: f"{x:.2f}")
    display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")
    display_df['Value'] = display_df['Value'].apply(lambda x: f"${x:,.2f}")
    display_df['Weight'] = display_df['Weight'].apply(lambda x: f"{x*100:.1f}%")

    st.dataframe(display_df, use_container_width=True)

    # Detailed metrics
    with st.expander("Detailed Diversity Metrics", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Concentration Metrics:**")
            st.write(f"- Herfindahl Index (HHI): {metrics['herfindahl_index']:.4f}")
            st.write(f"- Top 5 Concentration: {metrics['top_5_concentration']*100:.1f}%")
            st.write(f"- Largest Position: {metrics['largest_position']*100:.1f}%")
            st.write(f"- Smallest Position: {metrics['smallest_position']*100:.1f}%")

        with col2:
            st.markdown("**Distribution Metrics:**")
            st.write(f"- Average Position Size: {metrics['avg_position_size']*100:.1f}%")
            st.write(f"- Position Std Dev: {metrics['std_position_size']*100:.1f}%")
            st.write(f"- Gini Coefficient: {metrics['gini_coefficient']:.4f}")
            st.write(f"- Diversification Ratio: {metrics['diversification_ratio']:.4f}")

    # Recommendations
    st.markdown("**Recommendations:**")
    for i, rec in enumerate(recommendations, 1):
        if "good diversification" in rec.lower():
            st.success(f"{i}. {rec}")
        elif "consider" in rec.lower():
            st.warning(f"{i}. {rec}")
        else:
            st.info(f"{i}. {rec}")

st.markdown("---")

# Capital and optimization settings
col1, col2 = st.columns(2)

with col1:
    st.subheader("Investment Parameters")

    total_capital = st.number_input(
        "Total Capital ($)",
        min_value=1000,
        max_value=100000000,
        value=10000,
        step=1000,
        help="Total amount to invest"
    )

    optimization_method = st.selectbox(
        "Optimization Method",
        [
            "Maximum Sharpe Ratio",
            "Minimum Volatility",
            "Risk Parity",
            "Hierarchical Risk Parity (HRP)",
            "Maximum Diversification",
            "Equal Weight"
        ],
        help="Algorithm to use for portfolio optimization"
    )

    risk_free_rate = st.slider(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.1
    ) / 100

with col2:
    st.subheader("Constraints")

    with st.expander("Position Limits", expanded=True):
        max_weight = st.slider(
            "Maximum Position Size (%)",
            min_value=5,
            max_value=100,
            value=40,
            help="Maximum allocation to any single asset"
        ) / 100

        min_weight = st.slider(
            "Minimum Position Size (%)",
            min_value=0,
            max_value=20,
            value=0,
            help="Minimum allocation (positions below this become 0)"
        ) / 100

    with st.expander("Advanced Constraints"):
        allow_fractional = st.checkbox(
            "Allow Fractional Shares",
            value=False,
            help="Enable fractional share purchases"
        )

        target_volatility = st.number_input(
            "Target Volatility (%, 0 = no target)",
            min_value=0.0,
            max_value=100.0,
            value=0.0
        )

# Store settings in session state
st.session_state.settings = {
    'total_capital': total_capital,
    'optimization_method': optimization_method,
    'risk_free_rate': risk_free_rate,
    'max_weight': max_weight,
    'min_weight': min_weight,
    'allow_fractional': allow_fractional,
    'target_volatility': target_volatility / 100 if target_volatility > 0 else None,
}

st.markdown("---")

# Fetch data button
if st.button("Fetch Data & Continue", type="primary", use_container_width=True):
    if not ticker_input.strip():
        st.error("Please enter at least one ticker symbol.")
    else:
        try:
            # Validate and clean tickers
            tickers = validate_tickers(ticker_input)

            with st.spinner(f"Fetching data for {len(tickers)} tickers..."):
                # Fetch data
                dm = DataManager(show_progress=False)
                prices = dm.get_price_data(
                    tickers,
                    start_date,
                    end_date
                )

                # Calculate returns
                returns = prices.pct_change().dropna()

                # Store in session state
                st.session_state.tickers = list(prices.columns)
                st.session_state.portfolio_data = {
                    'prices': prices,
                    'returns': returns,
                    'start_date': start_date,
                    'end_date': end_date,
                }

                st.success(f"Successfully loaded data for {len(prices.columns)} tickers!")

                # Show summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tickers Loaded", len(prices.columns))
                with col2:
                    st.metric("Trading Days", len(prices))
                with col3:
                    st.metric("Date Range", f"{prices.index[0].date()} to {prices.index[-1].date()}")

                # Preview data
                st.subheader("Data Preview")
                st.dataframe(prices.tail(10), use_container_width=True)

                st.info("Go to **Optimization** page to run portfolio optimization.")

        except Exception as e:
            st.error(f"Error fetching data: {e}")

# Show current data if available
if st.session_state.portfolio_data is not None:
    st.markdown("---")
    st.subheader("Current Data Loaded")

    data = st.session_state.portfolio_data
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Tickers", len(st.session_state.tickers))
    with col2:
        st.metric("Trading Days", len(data['prices']))
    with col3:
        period = (data['prices'].index[-1] - data['prices'].index[0]).days
        st.metric("Period", f"{period} days")

    # Quick stats
    returns = data['returns']
    st.markdown("**Return Statistics (Annualized)**")

    stats = pd.DataFrame({
        'Mean Return': returns.mean() * 252,
        'Volatility': returns.std() * np.sqrt(252),
        'Sharpe': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
    }).round(4)

    st.dataframe(stats.T, use_container_width=True)
