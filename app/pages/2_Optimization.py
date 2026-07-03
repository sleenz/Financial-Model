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
from src.utils.settings_manager import load_settings, save_settings

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
    "Black-Litterman": "black_litterman",
    "Custom / Current Holdings": "use_current",
}

# Optimization Settings — capital, method, risk-free rate. Combined here
# with Constraints below so all optimizer inputs live on one page.
st.subheader("Optimization Settings")
col1, col2, col3 = st.columns(3)

with col1:
    _method_options = list(METHOD_MAP.keys())
    _default_method = settings.get('optimization_method', "Maximum Sharpe Ratio")
    if _default_method not in _method_options:
        _default_method = "Maximum Sharpe Ratio"
    method_name = st.selectbox(
        "Method",
        _method_options,
        index=_method_options.index(_default_method)
    )
    method = METHOD_MAP[method_name]

with col2:
    capital = st.number_input(
        "Capital ($)",
        value=int(settings.get('total_capital', 100000)),
        min_value=1000
    )

with col3:
    rf_rate = st.number_input(
        "Risk-Free Rate",
        value=settings.get('risk_free_rate', 0.05),
        format="%.3f"
    )

# Info banner for Custom / Current Holdings mode
if method == "use_current":
    _cw = st.session_state.get('current_portfolio_weights')
    if _cw is not None:
        _n_pos = int((_cw > 0).sum())
        st.info(
            f"**Custom / Current Holdings mode** — your existing allocation "
            f"({_n_pos} position{'s' if _n_pos != 1 else ''}) will be used as-is. "
            "No optimization will be run."
        )
    else:
        st.warning(
            "No current holdings found. "
            "Go to **Portfolio Input → Option 1: My Current Holdings**, "
            "add your positions, and click **Analyze My Portfolio** first."
        )

st.markdown("---")

# Constraints — position limits, advanced constraints, and the position
# reduction (turnover) band all grouped together here.
st.subheader("Constraints")

_saved_constraints = load_settings()
_CONSTRAINT_MODE_OPTIONS = ["Both", "Position Limits only", "Position Reduction only"]
_default_constraint_mode = _saved_constraints["constraints"].get("constraint_mode", "Both")
if _default_constraint_mode not in _CONSTRAINT_MODE_OPTIONS:
    _default_constraint_mode = "Both"

constraint_mode = st.radio(
    "Apply which constraints?",
    _CONSTRAINT_MODE_OPTIONS,
    index=_CONSTRAINT_MODE_OPTIONS.index(_default_constraint_mode),
    horizontal=True,
    help=(
        "**Both** — Position Limits always cap every trade; the Position "
        "Reduction band below (if enabled) further restricts moves away from "
        "your current holdings. **Position Limits only** — ignores the "
        "Position Reduction band even if it's enabled below. **Position "
        "Reduction only** — ignores the Maximum/Minimum Position Size sliders "
        "and bounds trades purely by the Position Reduction band around your "
        "current holdings."
    ),
)

col1, col2 = st.columns(2)

with col1:
    with st.expander("Position Limits", expanded=(constraint_mode != "Position Reduction only")):
        if constraint_mode == "Position Reduction only":
            st.caption("Not applied — constraint mode above is 'Position Reduction only'.")
        max_weight = st.slider(
            "Maximum Position Size (%)",
            min_value=5,
            max_value=100,
            value=int(settings.get('max_weight', 0.40) * 100),
            help="Maximum allocation to any single asset"
        ) / 100

        min_weight = st.slider(
            "Minimum Position Size (%)",
            min_value=0,
            max_value=20,
            value=int(settings.get('min_weight', 0.02) * 100),
            help="Minimum allocation (positions below this become 0)"
        ) / 100

with col2:
    with st.expander("Advanced Constraints"):
        allow_fractional = st.checkbox(
            "Allow Fractional Shares",
            value=settings.get('allow_fractional', False),
            help="Enable fractional share purchases"
        )

        _tv_dec = settings.get('target_volatility', 0.0) or 0.0
        target_volatility = st.number_input(
            "Target Volatility (%, 0 = no target)",
            min_value=0.0,
            max_value=100.0,
            value=float(_tv_dec * 100)
        )

# Turnover / position reduction constraint expander
with st.expander("Position Reduction Constraint", expanded=(constraint_mode == "Position Reduction only")):
    st.caption(
        "Limits how much each position can change from its current size. "
        "Requires current holdings to be entered in Portfolio Input first."
    )
    if constraint_mode == "Position Limits only":
        st.caption("Not applied — constraint mode above is 'Position Limits only'.")

    turnover_enabled = st.toggle(
        "Enable position reduction constraint",
        value=st.session_state.get(
            "turnover_enabled",
            _saved_constraints["constraints"]["turnover_enabled"]
        ),
        help="When enabled, the optimizer cannot move any position "
             "beyond the defined trading band."
    )
    st.session_state.turnover_enabled = turnover_enabled

    if turnover_enabled:
        col1, col2 = st.columns(2)

        with col1:
            reduction_pct = st.slider(
                "Max reduction from current position (%)",
                min_value=0,
                max_value=100,
                value=int(st.session_state.get(
                    "reduction_pct",
                    _saved_constraints["constraints"]["reduction_pct"]
                ) * 100),
                step=5,
                help="50% means a 100-share position can drop to minimum 50 shares."
            ) / 100.0
            st.session_state.reduction_pct = reduction_pct

        with col2:
            increase_pct = st.slider(
                "Max increase from current position (%)",
                min_value=0,
                max_value=200,
                value=int(st.session_state.get(
                    "increase_pct",
                    _saved_constraints["constraints"]["increase_pct"]
                ) * 100),
                step=5,
                help="30% means a 20% position can grow to maximum 26%."
            ) / 100.0
            st.session_state.increase_pct = increase_pct

        allow_full_exit = st.checkbox(
            "Allow full exit (sell 100% of any position)",
            value=st.session_state.get(
                "allow_full_exit",
                _saved_constraints["constraints"]["allow_full_exit"]
            ),
            help="When checked, positions can be sold entirely regardless "
                 "of the reduction constraint."
        )
        st.session_state.allow_full_exit = allow_full_exit

        if (
            "current_portfolio_weights" in st.session_state
            and st.session_state.current_portfolio_weights is not None
        ):
            current_w = st.session_state.current_portfolio_weights
            preview_rows = []
            for ticker, w in current_w.items():
                lb = 0.0 if allow_full_exit else max(0.0, w * (1 - reduction_pct))
                ub = min(max_weight, w * (1 + increase_pct))
                preview_rows.append({
                    "Ticker":      ticker,
                    "Current (%)": f"{w:.1%}",
                    "Min (%)":     f"{lb:.1%}",
                    "Max (%)":     f"{ub:.1%}",
                    "Band":        f"[{lb:.1%} – {ub:.1%}]",
                })
            st.dataframe(
                pd.DataFrame(preview_rows).set_index("Ticker"),
                width='stretch'
            )
            st.caption(
                "Min = lowest weight optimizer can assign. "
                "Max = highest weight optimizer can assign. "
                "Stocks not in current portfolio use standard min/max bounds."
            )

            lbs = [
                0.0 if allow_full_exit
                else max(0.0, w * (1 - reduction_pct))
                for w in current_w
            ]
            ubs = [
                min(max_weight, w * (1 + increase_pct))
                for w in current_w
            ]
            sum_lower = sum(lbs)
            sum_upper = sum(ubs)
            if sum_lower > 1.0:
                st.error(
                    f"Infeasible: minimum weights sum to {sum_lower:.1%} > 100%. "
                    f"Increase the reduction percentage or enable 'Allow full exit'."
                )
            elif sum_upper < 1.0:
                st.error(
                    f"Infeasible: maximum weights sum to {sum_upper:.1%} < 100%. "
                    f"Increase the increase percentage."
                )
            else:
                st.success(
                    f"✓ Constraints feasible — "
                    f"weights will be bounded between "
                    f"{sum_lower:.1%} and {sum_upper:.1%} total."
                )
        else:
            st.info(
                "Enter your current holdings in Portfolio Input first "
                "to see the trading band preview."
            )

st.session_state.constraint_mode = constraint_mode

# Resolve the effective bounds/turnover flag actually passed to the optimizer,
# after applying the constraint_mode choice above.
if constraint_mode == "Position Reduction only":
    _effective_min_weight, _effective_max_weight = 0.0, 1.0
else:
    _effective_min_weight, _effective_max_weight = min_weight, max_weight

_effective_turnover_enabled = turnover_enabled and constraint_mode != "Position Limits only"

if constraint_mode == "Position Reduction only":
    if not turnover_enabled:
        st.warning(
            "Constraint mode is 'Position Reduction only' but the Position "
            "Reduction Constraint toggle above is off — enable it, otherwise "
            "**no constraints will be applied** to the optimizer."
        )
    elif st.session_state.get("current_portfolio_weights") is None:
        st.warning(
            "Constraint mode is 'Position Reduction only' but no current "
            "holdings are loaded — there's nothing to band around, so "
            "**no constraints will be applied**. Enter holdings on the "
            "Portfolio Input page first."
        )

st.markdown("---")

# Keep session_state.settings in sync with the widgets above so other pages
# (Portfolio Input status, Presets) see the live values, not stale ones.
st.session_state.settings = {
    'total_capital': capital,
    'optimization_method': method_name,
    'risk_free_rate': rf_rate,
    'max_weight': max_weight,
    'min_weight': min_weight,
    'allow_fractional': allow_fractional,
    'target_volatility': target_volatility / 100 if target_volatility > 0 else None,
}

# Black-Litterman configuration — built before the run button so _bl_views_data
# is in scope when the button handler executes.
_bl_tau: float = 0.05
_bl_risk_aversion: float = 2.5
_bl_views_data: list = []

if method == "black_litterman":
    _tickers_list = list(returns.columns)
    with st.expander("Black-Litterman Configuration", expanded=True):
        _blc1, _blc2 = st.columns(2)
        with _blc1:
            _bl_tau = st.slider(
                "tau — prior uncertainty",
                0.005, 0.100, 0.050, step=0.005, format="%.3f",
                key="bl_tau",
                help=(
                    "Scalar that scales the uncertainty of the equilibrium prior. "
                    "Lower tau = trust the market equilibrium more; "
                    "higher tau = give more weight to investor views."
                ),
            )
            _bl_risk_aversion = st.number_input(
                "Risk aversion delta",
                min_value=0.5, max_value=10.0, value=2.5, step=0.1,
                format="%.1f", key="bl_risk_aversion",
                help=(
                    "Market-wide risk aversion coefficient used to back out implied "
                    "equilibrium returns from market-cap weights. Typical range: 2–4."
                ),
            )
        with _blc2:
            st.caption(
                "tau controls how strongly investor views pull the posterior returns "
                "away from the market equilibrium. delta reverse-engineers the "
                "equilibrium return vector: pi = delta * Sigma * w_market."
            )

        st.markdown("**Investor Views** (optional)")
        st.caption(
            "Absolute view: 'AAPL will return 15% per year.'  "
            "Relative view: 'MSFT will outperform GOOG by 5% per year.'"
        )

        _n_views = int(st.number_input(
            "Number of views", min_value=0, max_value=8, value=0, step=1,
            key="bl_n_views",
        ))

        for _vi in range(_n_views):
            st.markdown(f"---\n**View {_vi + 1}**")
            _vtype = st.selectbox(
                "Type", ["Absolute", "Relative"], key=f"bl_v{_vi}_type"
            )
            _vc1, _vc2, _vc3 = st.columns(3)
            with _vc1:
                if _vtype == "Absolute":
                    _vasset = st.selectbox(
                        "Asset", _tickers_list, key=f"bl_v{_vi}_asset"
                    )
                else:
                    _vlong = st.multiselect(
                        "Long assets", _tickers_list,
                        default=[_tickers_list[0]] if _tickers_list else [],
                        key=f"bl_v{_vi}_long",
                    )
                    _vshort = st.multiselect(
                        "Short assets", _tickers_list,
                        key=f"bl_v{_vi}_short",
                    )
            with _vc2:
                _vlabel = "Expected return (%)" if _vtype == "Absolute" else "Outperformance (%)"
                _vreturn = st.number_input(
                    _vlabel, -50.0, 100.0, 10.0, step=0.5,
                    key=f"bl_v{_vi}_return",
                ) / 100.0
            with _vc3:
                _vconf = st.slider(
                    "Confidence", 0.10, 0.90, 0.50, key=f"bl_v{_vi}_conf"
                )

            if _vtype == "Absolute":
                _bl_views_data.append({
                    "type": "absolute",
                    "asset": _vasset,
                    "return": _vreturn,
                    "confidence": _vconf,
                })
            elif _vtype == "Relative":
                _vlong_sel = locals().get("_vlong", [])
                _vshort_sel = locals().get("_vshort", [])
                if _vlong_sel and _vshort_sel:
                    _bl_views_data.append({
                        "type": "relative",
                        "long": _vlong_sel,
                        "short": _vshort_sel,
                        "return": _vreturn,
                        "confidence": _vconf,
                    })
                else:
                    st.warning(f"View {_vi + 1}: select at least one long and one short asset.")

# Save optimization settings button
if st.button("Save Optimization Settings", key="save_settings_p2"):
    current = load_settings()
    current["optimization"].update({
        "method":            method_name,
        "risk_free_rate":    rf_rate,
        "max_weight":        max_weight,
        "min_weight":        min_weight,
        "target_volatility": target_volatility / 100 if target_volatility > 0 else None,
        "allow_fractional":  allow_fractional,
    })
    current["constraints"].update({
        "turnover_enabled": st.session_state.get("turnover_enabled", False),
        "reduction_pct":    st.session_state.get("reduction_pct", 0.50),
        "increase_pct":     st.session_state.get("increase_pct", 0.30),
        "allow_full_exit":  st.session_state.get("allow_full_exit", True),
        "constraint_mode":  constraint_mode,
    })
    if save_settings(current):
        st.success("Optimization settings saved.")
    else:
        st.error("Failed to save settings.")

# Run button
_btn_label = "Analyze Portfolio" if method == "use_current" else "Run Optimization"
if st.button(_btn_label, type="primary"):
    with st.spinner("Analyzing portfolio..." if method == "use_current" else "Optimizing portfolio..."):

        if method == "use_current":
            _cw = st.session_state.get('current_portfolio_weights')
            if _cw is None:
                st.error("No current holdings found. Please enter holdings on the Portfolio Input page first.")
                st.stop()

            # Align with tickers that have return data; normalize
            _available = list(returns.columns)
            _missing = [t for t in _cw.index if t not in _available]
            if _missing:
                st.warning(f"No price data for: {', '.join(_missing)}. Their weight will be redistributed.")
            _wa = _cw.reindex(_available, fill_value=0.0)
            _total = _wa.sum()
            if _total <= 0:
                st.error("Could not match any holdings to the loaded tickers. Check spelling.")
                st.stop()
            _wa = _wa / _total

            _weights_arr = _wa.values
            _mean_ret = returns.mean() * 252
            _cov_mat = returns.cov() * 252
            _exp_ret = float(np.dot(_weights_arr, _mean_ret.values))
            _vol = float(np.sqrt(_weights_arr @ _cov_mat.values @ _weights_arr))
            _sharpe = (_exp_ret - rf_rate) / _vol if _vol > 0 else 0.0

            result = {
                'weights': _wa,
                'expected_return': _exp_ret,
                'volatility': _vol,
                'sharpe_ratio': _sharpe,
                'method': 'use_current',
            }
            optimizer = None

        elif method == "black_litterman":
            from src.optimization.black_litterman import BlackLittermanModel
            _bl_model = BlackLittermanModel(
                returns=returns,
                risk_aversion=float(st.session_state.get("bl_risk_aversion", 2.5)),
                tau=float(st.session_state.get("bl_tau", 0.05)),
                risk_free_rate=rf_rate,
            )
            for _view in _bl_views_data:
                if _view["type"] == "absolute":
                    _bl_model.add_absolute_view(
                        _view["asset"], _view["return"], _view["confidence"]
                    )
                elif _view["type"] == "relative":
                    _bl_model.add_relative_view(
                        _view["long"], _view["short"],
                        _view["return"], _view["confidence"],
                    )
            _bl_constraints = PortfolioConstraints(
                max_weight=_effective_max_weight,
                min_weight=_effective_min_weight,
                turnover_enabled=_effective_turnover_enabled,
                reduction_pct=st.session_state.get("reduction_pct", 0.50),
                increase_pct=st.session_state.get("increase_pct", 0.30),
                allow_full_exit=st.session_state.get("allow_full_exit", True),
                current_weights=st.session_state.get("current_portfolio_weights", None),
            )
            try:
                _bl_out = _bl_model.optimize(constraints=_bl_constraints)
            except ValueError as e:
                st.error(f"Optimization failed — constraint error: {e}")
                st.stop()
            # Recompute metrics using historical data so they match Risk Analytics.
            # BL.optimize() uses posterior_cov = Sigma + M for the optimisation
            # objective, but posterior_cov > Sigma inflates volatility and
            # depresses Sharpe vs what Risk Analytics would show for the same weights.
            _bl_w = _bl_out["weights"].values
            _hist_ret_ann = returns.mean() * 252
            _hist_cov_ann = returns.cov() * 252
            _bl_exp_ret = float(np.dot(_bl_w, _hist_ret_ann.values))
            _bl_vol = float(np.sqrt(_bl_w @ _hist_cov_ann.values @ _bl_w))
            _bl_sharpe = (_bl_exp_ret - rf_rate) / _bl_vol if _bl_vol > 0 else 0.0
            result = {
                "weights": _bl_out["weights"],
                "expected_return": _bl_exp_ret,
                "volatility": _bl_vol,
                "sharpe_ratio": _bl_sharpe,
                "method": "black_litterman",
                "posterior_returns": _bl_out["posterior_returns"],
                "equilibrium_returns": _bl_out["equilibrium_returns"],
            }
            optimizer = None

        else:
            # Create constraints — reflects the "Apply which constraints?"
            # mode chosen above (Both / Position Limits only / Position
            # Reduction only).
            constraints = PortfolioConstraints(
                max_weight=_effective_max_weight,
                min_weight=_effective_min_weight,
                min_position_size=_effective_min_weight,
                turnover_enabled=_effective_turnover_enabled,
                reduction_pct=st.session_state.get("reduction_pct", 0.50),
                increase_pct=st.session_state.get("increase_pct", 0.30),
                allow_full_exit=st.session_state.get("allow_full_exit", True),
                current_weights=st.session_state.get("current_portfolio_weights", None),
            )

            # Run optimizer
            optimizer = PortfolioOptimizer(returns, rf_rate)
            try:
                result = optimizer.optimize(method, constraints)
            except ValueError as e:
                st.error(f"Optimization failed — constraint error: {e}")
                st.stop()

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

        st.success("Analysis complete!" if method == "use_current" else "Optimization complete!")

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

    # Black-Litterman: posterior vs equilibrium returns
    if result.get("method") == "black_litterman" and "posterior_returns" in result:
        st.markdown("---")
        st.subheader("Black-Litterman: Posterior vs Equilibrium Returns")
        _bl_cmp_df = pd.DataFrame({
            "Equilibrium (prior)": result["equilibrium_returns"] * 100,
            "Posterior (views applied)": result["posterior_returns"] * 100,
        })
        _fig_bl = go.Figure()
        _fig_bl.add_trace(go.Bar(
            name="Equilibrium", x=_bl_cmp_df.index,
            y=_bl_cmp_df["Equilibrium (prior)"], marker_color="steelblue",
        ))
        _fig_bl.add_trace(go.Bar(
            name="Posterior", x=_bl_cmp_df.index,
            y=_bl_cmp_df["Posterior (views applied)"], marker_color="coral",
        ))
        _fig_bl.update_layout(
            barmode="group", yaxis_title="Expected Annual Return (%)",
            xaxis_title="Asset", height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(_fig_bl, width="stretch")
        st.caption(
            "Equilibrium returns are derived from market-cap weights via reverse optimization "
            "(pi = delta * Sigma * w_market). Posterior returns incorporate your investor views "
            "via Bayesian updating. The optimizer then maximizes Sharpe using the posterior estimates."
        )

    st.markdown("---")

    # Rebalancing Section (if user has current holdings AND optimization was actually run)
    if result.get('method') != 'use_current' and st.session_state.get('current_portfolio_weights') is not None:
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
        allow_fractional=allow_fractional
    )
    positions = calc.calculate_positions()
    summary = calc.get_summary()

    # Filter out zero-weight positions (excluded by min position size or not held)
    active_positions = positions[positions['Weight'] > 1e-4]
    n_excluded = len(positions) - len(active_positions)
    if n_excluded > 0:
        _min_pct = min_weight * 100
        if _min_pct > 0:
            st.info(
                f"{n_excluded} position{'s' if n_excluded != 1 else ''} excluded: "
                f"weight fell below the {_min_pct:.0f}% minimum position size threshold."
            )

    # Format for display
    display_df = active_positions[['Weight', 'Price', 'Shares', 'Actual Amount', 'Remainder']].copy()
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

    # Efficient Frontier (only when an optimizer was run, not for custom holdings)
    st.markdown("---")
    st.subheader("Efficient Frontier")

    if result.get('method') in ('use_current', 'black_litterman'):
        st.info("Efficient frontier is not available in this mode.")
    elif st.button("Calculate Efficient Frontier"):
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

    # Method comparison (only when an optimizer is available)
    st.markdown("---")
    st.subheader("Method Comparison")

    if result.get('method') in ('use_current', 'black_litterman'):
        st.info("Method comparison is not available in this mode.")
    elif st.button("Compare All Methods"):
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
