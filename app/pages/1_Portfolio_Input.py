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
from src.utils.settings_manager import load_settings, save_settings
from src.utils.preset_manager import list_presets, load_preset, apply_preset_to_state

st.set_page_config(page_title="Portfolio Input", page_icon=None, layout="wide")

st.title("Portfolio Input")
st.markdown("Enter your current holdings or tickers to begin optimization.")

# Initialize session state
if 'tickers' not in st.session_state:
    st.session_state.tickers = []
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None
if 'current_holdings' not in st.session_state:
    st.session_state.current_holdings = {}
if 'holdings_tracker' not in st.session_state:
    st.session_state.holdings_tracker = None
if 'current_portfolio_weights' not in st.session_state:
    st.session_state.current_portfolio_weights = None
if 'settings' not in st.session_state:
    st.session_state.settings = {}

# Sidebar for quick settings
with st.sidebar:
    st.header("Quick Settings")

    # Saved presets (from the Portfolio Presets page) are appended after the
    # built-in starter baskets so they're one click away — picking one loads
    # its tickers/weights/value into session_state immediately.
    _saved_presets = list_presets()
    _name_counts: dict[str, int] = {}
    for _p in _saved_presets:
        _name_counts[_p["name"]] = _name_counts.get(_p["name"], 0) + 1

    _saved_id_by_label: dict[str, str] = {}
    _saved_labels = []
    for _p in _saved_presets:
        _label = f"⭐ {_p['name']}"
        if _name_counts[_p["name"]] > 1:
            _label += f" ({_p['preset_id'][:8]})"
        _saved_labels.append(_label)
        _saved_id_by_label[_label] = _p["preset_id"]

    _preset_options = ["Custom", "Tech Giants", "Diversified ETFs", "Blue Chips"] + _saved_labels

    def _on_quick_preset_change():
        selected = st.session_state.get("sidebar_preset_select")
        preset_id = _saved_id_by_label.get(selected)
        if preset_id is None:
            return
        data = load_preset(preset_id)
        if data is None:
            st.session_state["_quick_preset_load_error"] = True
        else:
            apply_preset_to_state(data, st.session_state)
            st.session_state["_quick_preset_loaded_name"] = data["name"]

    preset = st.selectbox(
        "Load Preset Portfolio",
        _preset_options,
        key="sidebar_preset_select",
        on_change=_on_quick_preset_change,
        help="Built-in starter baskets, or your own saved presets (⭐) — "
             "picking a saved preset instantly loads its tickers, weights, "
             "and value.",
    )

    if st.session_state.pop("_quick_preset_load_error", False):
        st.error("Failed to load that preset — it may be corrupt. Check logs for details.")
    _loaded_name = st.session_state.pop("_quick_preset_loaded_name", None)
    if _loaded_name:
        st.success(f"Loaded '{_loaded_name}'.")

    if preset in _saved_id_by_label:
        default_tickers = ", ".join(st.session_state.get("tickers", []))
    elif preset == "Tech Giants":
        default_tickers = "AAPL, MSFT, GOOGL, AMZN, NVDA"
    elif preset == "Diversified ETFs":
        default_tickers = "SPY, QQQ, IWM, EFA, AGG"
    elif preset == "Blue Chips":
        default_tickers = "JNJ, PG, KO, WMT, JPM"
    else:
        default_tickers = ""

# Time period selection (needed for both methods)
st.subheader("Time Period for Analysis")
col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=365*3),
        max_value=datetime.now()
    )
with col2:
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

# Two input methods with tabs
tab1, tab2 = st.tabs([" Option 1: My Current Holdings", " Option 2: Manual Ticker Entry"])

with tab1:
    st.markdown("""
    **Enter your current stock holdings** (Recommended if you already own stocks)

    Add the stocks you own and the number of shares. The system will:
    - Calculate your current portfolio value and allocation
    - Analyze your portfolio diversity
    - Use this as the starting point for optimization
    - Show you how to rebalance to improve your portfolio
    """)

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

    # Analyze and fetch data button
    st.markdown("---")
    if st.button(" Analyze My Portfolio & Fetch Data", type="primary", width="stretch"):
        # Fetch prices AND historical data for holdings
        if not st.session_state.current_holdings:
            st.error("Please add at least one holding first!")
        else:
            try:
                dm = DataManager(show_progress=False)
                tickers_list = list(st.session_state.current_holdings.keys())

                st.info(f"Fetching data for: {', '.join(tickers_list)}")

                with st.spinner("Fetching price data for your holdings..."):
                    # Fetch historical data for optimization
                    prices = dm.get_price_data(
                        tickers_list,
                        start_date,
                        end_date
                    )

                    # Debug: Show what we got
                    st.write(f" Fetched data for {len(prices.columns)} tickers: {list(prices.columns)}")
                    st.write(f" Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
                    st.write(f" Total rows: {len(prices)}")

                    # Calculate returns
                    returns = prices.pct_change().dropna()

                    # Get current prices (most recent)
                    current_prices = prices.iloc[-1]

                    # Debug: Show current prices
                    st.write("**Current Prices:**")
                    for ticker in current_prices.index:
                        st.write(f"- {ticker}: ${current_prices[ticker]:.2f}")

                    # Create holdings tracker
                    tracker = HoldingsTracker(
                        st.session_state.current_holdings,
                        current_prices
                    )
                    st.session_state.holdings_tracker = tracker

                    # Calculate current portfolio state
                    holdings_df = tracker.get_holdings_dataframe()
                    total_value = tracker.calculate_total_value()

                    # Debug: Show holdings values
                    st.write("**Portfolio Breakdown:**")
                    for ticker in holdings_df.index:
                        row = holdings_df.loc[ticker]
                        st.write(f"- {ticker}: {row['Shares']:.2f} shares × ${row['Price']:.2f} = ${row['Value']:.2f} ({row['Weight']*100:.1f}%)")

                    if total_value == 0:
                        st.error("Total portfolio value is $0. This means prices were not fetched correctly.")
                        st.warning("Possible issues: Invalid ticker symbols, no data available for date range, or API limits reached.")

                    current_weights = holdings_df['Weight']

                    # Store everything in session state
                    st.session_state.tickers = tickers_list
                    st.session_state.portfolio_data = {
                        'prices': prices,
                        'returns': returns,
                        'start_date': start_date,
                        'end_date': end_date,
                        'current_prices': current_prices,
                    }
                    st.session_state.current_portfolio_weights = current_weights
                    st.session_state.settings['total_capital'] = total_value

                    # Store for other analysis pages
                    st.session_state.optimizer = None  # Will be created in optimization
                    st.session_state.weights = current_weights  # Current weights as starting point
                    st.session_state.prices = prices
                    st.session_state.portfolio_value = total_value

                    st.success(f"Portfolio analyzed! Total value: ${total_value:,.2f}")
                    st.info("Your portfolio data is ready. You can now access:")
                    st.write("- **Optimization**: See rebalancing recommendations")
                    st.write("- **Risk Analytics**: Analyze VaR, CVaR, drawdowns")
                    st.write("- **Stress Testing**: Test portfolio under scenarios")
                    st.write("- **Monitoring**: Track portfolio performance")
                    st.write("- **Factor Analysis**: Analyze factor exposures")
                    st.write("- **Reports**: Generate comprehensive reports")

            except Exception as e:
                st.error(f"Error fetching data: {e}")
                st.warning("Common issues:")
                st.write("- Invalid ticker symbols (check spelling)")
                st.write("- Ticker not available in data sources")
                st.write("- Date range has no data")
                st.write("- API rate limits exceeded")
                import traceback
                with st.expander("Show Full Error Details"):
                    st.code(traceback.format_exc())

    # Show diversity analysis if available
    if st.session_state.holdings_tracker is not None:
        st.markdown("---")
        st.subheader(" Portfolio Diversity Analysis")

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

        st.dataframe(display_df, width="stretch")

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

with tab2:
    st.markdown("""
    **Manual ticker entry** (If you don't own stocks yet or want to explore other combinations)

    Enter stock symbols to analyze and optimize. The system will suggest optimal allocations.
    """)

    # Ticker input
    ticker_input = st.text_area(
        "Enter Tickers (comma-separated)",
        value=default_tickers if preset != "Custom" else "",
        height=100,
        help="Enter stock symbols separated by commas (e.g., AAPL, MSFT, GOOGL)"
    )

    # Fetch data button
    if st.button("Fetch Data & Continue", type="primary", width="stretch"):
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
                    st.session_state.current_portfolio_weights = None  # No current holdings

                    st.success(f"Successfully loaded data for {len(prices.columns)} tickers!")
                    st.info("Go to **Optimization** page to calculate optimal portfolio.")

            except Exception as e:
                st.error(f"Error fetching data: {e}")

st.markdown("---")

# Capital and optimization settings
col1, col2 = st.columns(2)

with col1:
    st.subheader("Investment Parameters")

    total_capital = st.number_input(
        "Total Capital ($)",
        min_value=1000,
        max_value=100000000,
        value=int(st.session_state.settings.get('total_capital', 100000)),
        step=1000,
        help="Total amount to invest"
    )

    _p1_method_opts = [
        "Maximum Sharpe Ratio",
        "Minimum Volatility",
        "Risk Parity",
        "Hierarchical Risk Parity (HRP)",
        "Maximum Diversification",
        "Equal Weight",
    ]
    _saved_method = st.session_state.settings.get('optimization_method', "Maximum Sharpe Ratio")
    _p1_method_idx = _p1_method_opts.index(_saved_method) if _saved_method in _p1_method_opts else 0
    optimization_method = st.selectbox(
        "Optimization Method",
        _p1_method_opts,
        index=_p1_method_idx,
        help="Algorithm to use for portfolio optimization"
    )

    risk_free_rate = st.slider(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=float(st.session_state.settings.get('risk_free_rate', 0.05) * 100),
        step=0.1
    ) / 100

with col2:
    st.subheader("Constraints")

    with st.expander("Position Limits", expanded=True):
        max_weight = st.slider(
            "Maximum Position Size (%)",
            min_value=5,
            max_value=100,
            value=int(st.session_state.settings.get('max_weight', 0.40) * 100),
            help="Maximum allocation to any single asset"
        ) / 100

        min_weight = st.slider(
            "Minimum Position Size (%)",
            min_value=0,
            max_value=20,
            value=int(st.session_state.settings.get('min_weight', 0.02) * 100),
            help="Minimum allocation (positions below this become 0)"
        ) / 100

    with st.expander("Advanced Constraints"):
        allow_fractional = st.checkbox(
            "Allow Fractional Shares",
            value=st.session_state.settings.get('allow_fractional', False),
            help="Enable fractional share purchases"
        )

        _tv_dec = st.session_state.settings.get('target_volatility', 0.0) or 0.0
        target_volatility = st.number_input(
            "Target Volatility (%, 0 = no target)",
            min_value=0.0,
            max_value=100.0,
            value=float(_tv_dec * 100)
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

if st.button("Save Settings", key="save_settings_p1"):
    current = load_settings()
    current["portfolio"]["total_capital"] = st.session_state.settings.get(
        "total_capital", current["portfolio"]["total_capital"]
    )
    current["portfolio"]["tickers"] = st.session_state.get(
        "tickers", current["portfolio"]["tickers"]
    )
    if save_settings(current):
        st.success("Settings saved — portfolio value and tickers will be "
                   "restored next session.")
    else:
        st.error("Failed to save settings. Check write permissions on data/.")

st.markdown("---")

# Show current data status
if st.session_state.portfolio_data is not None:
    st.subheader(" Portfolio Data Ready")

    data = st.session_state.portfolio_data
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Assets", len(st.session_state.tickers))
    with col2:
        st.metric("Trading Days", len(data['prices']))
    with col3:
        period = (data['prices'].index[-1] - data['prices'].index[0]).days
        st.metric("Period", f"{period} days")
    with col4:
        if st.session_state.current_portfolio_weights is not None:
            total_value = st.session_state.settings.get('total_capital', 0)
            st.metric("Portfolio Value", f"${total_value:,.0f}")
        else:
            st.metric("Mode", "New Portfolio")

    # Show mode
    if st.session_state.current_portfolio_weights is not None:
        st.info("**Mode: Rebalancing from Current Holdings** - The optimizer will show you how to adjust your existing positions.")
    else:
        st.info("**Mode: New Portfolio** - The optimizer will suggest an optimal allocation from scratch.")

    # Quick stats
    with st.expander("View Price Statistics", expanded=False):
        returns = data['returns']
        st.markdown("**Return Statistics (Annualized)**")

        stats = pd.DataFrame({
            'Mean Return': returns.mean() * 252,
            'Volatility': returns.std() * np.sqrt(252),
            'Sharpe': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
        }).round(4)

        st.dataframe(stats.T, width="stretch")
