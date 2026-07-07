"""Portfolio Input Page - Enter tickers, dates, and parameters.

Merged with the former standalone Portfolio Presets page (per explicit
instruction: "merge the portfolio preset with the portfolio input... since
they are correlated") — presets are just named snapshots of a portfolio,
and this page is where a portfolio is built/edited, so they now live
together as a third tab instead of two separate pages in the sidebar nav.

Bug fix (per explicit report): loading a preset used to only populate
st.session_state.tickers/weights, which pre-filled Option 2's Manual
Ticker Entry text area, not Option 1's My Current Holdings (the
ticker->shares dict a user can actually add to / edit / remove from).
_load_preset_and_populate_holdings() below now ALSO derives a share count
per ticker (shares = weight * portfolio_value / current_price, via a live
price fetch) and writes it into current_holdings, so a loaded preset shows
up as editable holdings — the whole point of loading one to adjust it.
"""

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
from src.utils.preset_manager import (
    list_presets,
    preset_name_exists,
    load_preset,
    save_preset,
    update_preset,
    rename_preset,
    delete_preset,
    apply_preset_to_state,
)

st.set_page_config(page_title="Portfolio Input", page_icon=None, layout="wide")

st.title("Portfolio Input")
st.markdown("Enter your current holdings or tickers to begin optimization.")

# Initialize session state
if 'tickers' not in st.session_state:
    st.session_state.tickers = []
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None
if 'current_holdings' not in st.session_state:
    # Restore previously saved holdings (ticker -> shares) so Option 1 is
    # pre-filled automatically instead of asking the user to re-enter them.
    _saved_holdings = load_settings()["portfolio"].get("holdings", {})
    st.session_state.current_holdings = dict(_saved_holdings)
    if _saved_holdings:
        st.session_state._holdings_restored = True
if 'holdings_tracker' not in st.session_state:
    st.session_state.holdings_tracker = None
if 'current_portfolio_weights' not in st.session_state:
    st.session_state.current_portfolio_weights = None
if 'settings' not in st.session_state:
    st.session_state.settings = {}
if 'loaded_preset_id' not in st.session_state:
    st.session_state.loaded_preset_id = None
if 'preset_pending_confirm' not in st.session_state:
    st.session_state.preset_pending_confirm = None


def _load_preset_and_populate_holdings(preset: dict) -> list:
    """
    Load a preset's saved (tickers, weights, value) via the existing
    apply_preset_to_state() — unchanged, since other pages (Optimization,
    Risk Analytics, etc.) read st.session_state.weights/portfolio_value
    directly and must keep seeing them populated the same way — AND
    additionally derive a per-ticker SHARE count into current_holdings
    (shares = weight * portfolio_value / current_price), so the preset
    shows up in the My Current Holdings tab as editable positions instead
    of only pre-filling Manual Ticker Entry's raw ticker list.

    Returns the list of tickers that couldn't get a current price (left
    out of current_holdings, not silently zeroed) — callers surface this
    as a warning.
    """
    apply_preset_to_state(preset, st.session_state)

    tickers = list(preset.get("tickers", []))
    weights = list(preset.get("weights", []))
    value = float(preset.get("portfolio_value", 0.0))

    # A stale tracker built from whatever holdings existed BEFORE this
    # load would otherwise keep showing diversity metrics for a different
    # portfolio until the user re-clicks Analyze.
    st.session_state.holdings_tracker = None

    if not tickers or value <= 0:
        st.session_state.current_holdings = {}
        return []

    try:
        dm = DataManager(show_progress=False)
        prices = dm.get_current_prices(tickers)
    except Exception:
        prices = pd.Series(dtype=float)

    holdings: dict = {}
    failed_tickers = []
    for ticker, weight in zip(tickers, weights):
        price = prices.get(ticker)
        if price is None or price <= 0:
            failed_tickers.append(ticker)
            continue
        holdings[ticker] = round((weight * value) / price, 2)

    st.session_state.current_holdings = holdings
    return failed_tickers


# Sidebar for quick settings
with st.sidebar:
    st.header("Quick Settings")

    # Saved presets (from the Presets tab below) are appended after the
    # built-in starter baskets so they're one click away — picking one loads
    # its tickers/weights/value into My Current Holdings immediately.
    _saved_presets = list_presets()
    _name_counts: dict = {}
    for _p in _saved_presets:
        _name_counts[_p["name"]] = _name_counts.get(_p["name"], 0) + 1

    _saved_id_by_label: dict = {}
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
            failed = _load_preset_and_populate_holdings(data)
            st.session_state["_quick_preset_loaded_name"] = data["name"]
            st.session_state["_quick_preset_price_failures"] = failed

    preset = st.selectbox(
        "Load Preset Portfolio",
        _preset_options,
        key="sidebar_preset_select",
        on_change=_on_quick_preset_change,
        help="Built-in starter baskets, or your own saved presets — "
             "picking a saved preset instantly loads its tickers, weights, "
             "and value into My Current Holdings (as editable share counts).",
    )

    if st.session_state.pop("_quick_preset_load_error", False):
        st.error("Failed to load that preset — it may be corrupt. Check logs for details.")
    _loaded_name = st.session_state.pop("_quick_preset_loaded_name", None)
    if _loaded_name:
        st.success(f"Loaded '{_loaded_name}' into My Current Holdings.")
    _price_failures = st.session_state.pop("_quick_preset_price_failures", None)
    if _price_failures:
        st.warning(
            f"Could not fetch a current price for: {', '.join(_price_failures)} — "
            "left out of My Current Holdings; add them manually if needed."
        )

    if preset == "Tech Giants":
        default_tickers = "AAPL, MSFT, GOOGL, AMZN, NVDA"
    elif preset == "Diversified ETFs":
        default_tickers = "SPY, QQQ, IWM, EFA, AGG"
    elif preset == "Blue Chips":
        default_tickers = "JNJ, PG, KO, WMT, JPM"
    else:
        # Includes the saved-preset case: those now load into My Current
        # Holdings (see _load_preset_and_populate_holdings), not here.
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

# Three input methods with tabs — Presets merged in as its own tab (was a
# separate page) since it's just another way to populate the same
# My Current Holdings / Manual Ticker Entry state below.
tab1, tab2, tab3 = st.tabs([
    " Option 1: My Current Holdings",
    " Option 2: Manual Ticker Entry",
    " Presets",
])

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
        if st.session_state.pop("_holdings_restored", False):
            st.info(
                f"Restored {len(st.session_state.current_holdings)} holding"
                f"{'s' if len(st.session_state.current_holdings) != 1 else ''} "
                "from your last saved session."
            )

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

with tab3:
    # Actions below call st.rerun() right after st.success()/st.error()/st.warning() —
    # those calls never render since the rerun starts a fresh script run before
    # Streamlit can paint them. Queue the message in session_state instead and
    # flush it here on the run that follows the rerun.
    for _flash_kind, _flash_msg in st.session_state.pop("_preset_flash", []):
        getattr(st, _flash_kind)(_flash_msg)

    st.markdown(
        "Save named snapshots of your portfolio (tickers, weights, value) so you can "
        "switch between them without re-entering data. This is separate from the "
        "automatic last-session save — presets are saved and loaded explicitly. "
        "Loading a preset here populates **My Current Holdings** (Option 1 tab above) "
        "with editable share counts, derived from the preset's saved weights/value at "
        "current market prices."
    )

    def _fmt_ts(iso_str: str) -> str:
        if not iso_str:
            return "unknown"
        try:
            dt = datetime.fromisoformat(iso_str)
            return dt.strftime("%Y-%m-%d %H:%M UTC")
        except ValueError:
            return iso_str

    def _queue_preset_flash(kind: str, message: str) -> None:
        st.session_state.setdefault("_preset_flash", []).append((kind, message))

    def _current_portfolio_state():
        """Read the live portfolio input (tickers, weights, value) from the same
        session_state keys the My Current Holdings / Manual Ticker Entry /
        Optimization widgets already use, for saving AS a preset."""
        weights_obj = st.session_state.get('weights')
        if weights_obj is not None:
            tickers = [str(t) for t in weights_obj.index]
            weights = [float(w) for w in weights_obj.values]
        else:
            tickers = [str(t) for t in st.session_state.get('tickers', [])]
            n = len(tickers)
            weights = [1.0 / n] * n if n else []
        portfolio_value = float(st.session_state.get('portfolio_value', 0.0) or 0.0)
        return tickers, weights, portfolio_value

    st.markdown("---")

    # --- Current portfolio summary + Save As New ---------------------------
    st.subheader("Current Portfolio")

    _cur_tickers, _cur_weights, _cur_value = _current_portfolio_state()

    if _cur_tickers:
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            st.metric("Tickers", len(_cur_tickers))
        with pcol2:
            st.metric("Portfolio Value", f"${_cur_value:,.2f}")
        with pcol3:
            _loaded = st.session_state.loaded_preset_id
            _loaded_preset_name = None
            if _loaded:
                for _p in list_presets():
                    if _p["preset_id"] == _loaded:
                        _loaded_preset_name = _p["name"]
                        break
            st.metric("Loaded Preset", _loaded_preset_name or "None")

        with st.expander("View current tickers / weights", expanded=False):
            st.dataframe(
                pd.DataFrame({"Ticker": _cur_tickers, "Weight": [f"{w*100:.2f}%" for w in _cur_weights]}),
                width="stretch",
            )
    else:
        st.info(
            "No portfolio loaded yet. Enter tickers in **My Current Holdings** or "
            "**Manual Ticker Entry** above first (and click Analyze / Fetch Data)."
        )

    save_new_name = st.text_input("Preset name", key="preset_save_new_name", placeholder="e.g. Portfolio 1")

    if st.button("Save As New Preset", type="primary", disabled=not _cur_tickers):
        name = save_new_name.strip()
        if not name:
            st.error("Please enter a name for the preset.")
        else:
            conflict_id = preset_name_exists(name)
            if conflict_id:
                st.session_state.preset_pending_confirm = {
                    "action": "save_new_overwrite",
                    "conflict_id": conflict_id,
                    "name": name,
                }
            else:
                new_id = save_preset(name, _cur_tickers, _cur_weights, _cur_value)
                st.session_state.loaded_preset_id = new_id
                _queue_preset_flash("success", f"Saved new preset '{name}'.")
            st.rerun()

    st.markdown("---")

    # --- Saved presets list / actions ---------------------------------------
    st.subheader("Saved Presets")

    presets = list_presets()

    if not presets:
        st.info("No presets saved yet. Use **Save As New Preset** above to create one.")
    else:
        listing_df = pd.DataFrame(
            [{"Name": p["name"], "Last Updated": _fmt_ts(p["updated_at"])} for p in presets]
        )
        st.dataframe(listing_df, width="stretch", hide_index=True)

        options = [p["preset_id"] for p in presets]
        labels = {p["preset_id"]: f"{p['name']}  —  {_fmt_ts(p['updated_at'])}" for p in presets}

        selected_id = st.selectbox(
            "Select a preset",
            options,
            format_func=lambda pid: labels.get(pid, pid),
            key="preset_selected_id",
        )

        col_load, col_update, col_rename, col_delete = st.columns(4)

        with col_load:
            if st.button("Load", width="stretch"):
                preset_data = load_preset(selected_id)
                if preset_data is None:
                    st.error("Failed to load this preset — the file may be corrupt. Check logs for details.")
                else:
                    failed = _load_preset_and_populate_holdings(preset_data)
                    _queue_preset_flash("success", f"Loaded preset '{preset_data['name']}' into My Current Holdings.")
                    if failed:
                        _queue_preset_flash(
                            "warning",
                            f"Could not fetch a current price for: {', '.join(failed)} — "
                            "left out of My Current Holdings; add them manually if needed.",
                        )
                    st.rerun()

        with col_update:
            _can_update = st.session_state.loaded_preset_id is not None
            if st.button("Update Current", width="stretch", disabled=not _can_update):
                target_id = st.session_state.loaded_preset_id
                tickers, weights, value = _current_portfolio_state()
                if update_preset(target_id, tickers, weights, value):
                    _queue_preset_flash("success", "Updated the loaded preset in place.")
                else:
                    _queue_preset_flash("error", "Failed to update — the preset file may have been deleted. Check logs.")
                st.rerun()
            if not _can_update:
                st.caption("Load a preset first to enable Update.")

        with col_rename:
            if st.button("Rename", width="stretch"):
                st.session_state.preset_pending_confirm = {
                    "action": "rename_form",
                    "target_id": selected_id,
                }
                st.rerun()

        with col_delete:
            if st.button("Delete", width="stretch"):
                st.session_state.preset_pending_confirm = {
                    "action": "delete_confirm",
                    "target_id": selected_id,
                }
                st.rerun()

    # --- Pending confirmation / follow-up UI ---------------------------------
    pending = st.session_state.preset_pending_confirm

    if pending is not None:
        st.markdown("---")

        if pending["action"] == "save_new_overwrite":
            conflict = load_preset(pending["conflict_id"])
            conflict_name = conflict["name"] if conflict else pending["name"]
            st.warning(
                f"A preset named **'{pending['name']}'** already exists. "
                "Saving with this name will overwrite that preset's data "
                "(its underlying file stays the same, only its contents change)."
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button(f"Overwrite '{conflict_name}'", type="primary"):
                    tickers, weights, value = _current_portfolio_state()
                    update_preset(pending["conflict_id"], tickers, weights, value)
                    st.session_state.loaded_preset_id = pending["conflict_id"]
                    st.session_state.preset_pending_confirm = None
                    _queue_preset_flash("success", f"Overwrote preset '{conflict_name}'.")
                    st.rerun()
            with c2:
                if st.button("Cancel"):
                    st.session_state.preset_pending_confirm = None
                    st.rerun()

        elif pending["action"] == "rename_form":
            target = load_preset(pending["target_id"])
            if target is None:
                st.error("This preset no longer exists.")
                st.session_state.preset_pending_confirm = None
            else:
                st.markdown(f"**Rename '{target['name']}'**")
                new_name = st.text_input("New name", value=target["name"], key="preset_rename_input")
                if st.button("Confirm Rename", type="primary"):
                    stripped = new_name.strip()
                    if not stripped:
                        st.error("Please enter a non-empty name.")
                    elif stripped == target["name"]:
                        st.session_state.preset_pending_confirm = None
                        st.rerun()
                    else:
                        conflict_id = preset_name_exists(stripped, exclude_id=pending["target_id"])
                        if conflict_id:
                            st.session_state.preset_pending_confirm = {
                                "action": "rename_overwrite_confirm",
                                "target_id": pending["target_id"],
                                "new_name": stripped,
                                "conflict_id": conflict_id,
                            }
                        else:
                            rename_preset(pending["target_id"], stripped)
                            st.session_state.preset_pending_confirm = None
                            _queue_preset_flash("success", f"Renamed to '{stripped}'.")
                        st.rerun()
                if st.button("Cancel", key="cancel_rename"):
                    st.session_state.preset_pending_confirm = None
                    st.rerun()

        elif pending["action"] == "rename_overwrite_confirm":
            conflict = load_preset(pending["conflict_id"])
            conflict_name = conflict["name"] if conflict else pending["new_name"]
            st.warning(
                f"Another preset is already named **'{pending['new_name']}'**. "
                "Two presets with the same display name would be ambiguous in the "
                "dropdown. Rename anyway?"
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Rename Anyway", type="primary"):
                    rename_preset(pending["target_id"], pending["new_name"])
                    st.session_state.preset_pending_confirm = None
                    _queue_preset_flash("success", f"Renamed to '{pending['new_name']}'.")
                    st.rerun()
            with c2:
                if st.button("Cancel", key="cancel_rename_overwrite"):
                    st.session_state.preset_pending_confirm = None
                    st.rerun()

        elif pending["action"] == "delete_confirm":
            target = load_preset(pending["target_id"])
            target_name = target["name"] if target else pending["target_id"]
            st.warning(f"Delete preset **'{target_name}'**? This cannot be undone.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Confirm Delete", type="primary"):
                    delete_preset(pending["target_id"])
                    if st.session_state.loaded_preset_id == pending["target_id"]:
                        st.session_state.loaded_preset_id = None
                    st.session_state.preset_pending_confirm = None
                    _queue_preset_flash("success", f"Deleted preset '{target_name}'.")
                    st.rerun()
            with c2:
                if st.button("Cancel", key="cancel_delete"):
                    st.session_state.preset_pending_confirm = None
                    st.rerun()

st.markdown("---")

st.caption(
    "Investment parameters (capital, method, risk-free rate) and constraints "
    "now live on the **Optimization** page, alongside the position reduction "
    "constraint — everything the optimizer needs is in one place there."
)

if st.button("Save Holdings", key="save_settings_p1"):
    current = load_settings()
    current["portfolio"]["tickers"] = st.session_state.get(
        "tickers", current["portfolio"]["tickers"]
    )
    current["portfolio"]["holdings"] = dict(st.session_state.get("current_holdings", {}))
    if save_settings(current):
        st.success(
            "Holdings saved — your tickers and share counts will be "
            "restored automatically next session."
        )
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
