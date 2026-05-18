"""Stress Testing Page - Historical scenarios and Monte Carlo simulation."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.simulation.scenarios import StressTester, HISTORICAL_SCENARIOS, list_scenarios
from src.simulation.monte_carlo import MonteCarloSimulator

st.set_page_config(page_title="Stress Testing", page_icon="⚠️", layout="wide")

st.title("Stress Testing & Scenario Analysis")

# Check data
if 'portfolio_data' not in st.session_state or st.session_state.portfolio_data is None:
    st.warning("Please load portfolio data first.")
    st.stop()

data = st.session_state.portfolio_data
returns = data['returns']

# Get weights based on what's available
if 'optimization_result' in st.session_state and st.session_state.optimization_result:
    weights = st.session_state.optimization_result['weights'].values
    st.info("📊 Stress testing **optimized portfolio**")
elif 'current_portfolio_weights' in st.session_state and st.session_state.current_portfolio_weights is not None:
    weights = st.session_state.current_portfolio_weights.values
    st.info("💼 Stress testing **your current holdings**")
else:
    weights = np.ones(len(returns.columns)) / len(returns.columns)
    st.warning("⚠️ Using equal weights. Enter holdings or run optimization for accurate results.")

# Portfolio value
portfolio_value = st.session_state.get('settings', {}).get('total_capital', 10000)

# Initialize stress tester
stress_tester = StressTester(returns, weights, portfolio_value)

st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["Historical Scenarios", "Monte Carlo", "Custom Stress"])

with tab1:
    st.subheader("Historical Stress Scenarios")

    # Run all scenarios
    if st.button("Run All Historical Scenarios", type="primary"):
        with st.spinner("Running scenarios..."):
            results = stress_tester.run_all_historical()

            # Display results
            st.dataframe(results.round(2), use_container_width=True)

            # Chart
            fig = go.Figure(data=[
                go.Bar(
                    x=results['Scenario'],
                    y=results['Portfolio Return'] * 100,
                    marker_color=['red' if x < 0 else 'green' for x in results['Portfolio Return']]
                )
            ])
            fig.update_layout(
                title="Portfolio Impact by Scenario",
                xaxis_title="Scenario",
                yaxis_title="Return (%)",
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)

    # Individual scenario details
    st.markdown("---")
    st.markdown("**Scenario Details**")

    scenario_key = st.selectbox(
        "Select Scenario",
        list(HISTORICAL_SCENARIOS.keys()),
        format_func=lambda x: HISTORICAL_SCENARIOS[x]['name']
    )

    if scenario_key:
        scenario = HISTORICAL_SCENARIOS[scenario_key]
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**{scenario['name']}**")
            st.markdown(f"Period: {scenario['start_date']} to {scenario['end_date']}")
            st.markdown(f"Description: {scenario['description']}")

        with col2:
            chars = scenario['characteristics']
            st.metric("Equity Drop", f"{chars['equity_drop']*100:.0f}%")
            st.metric("Volatility Spike", f"{chars['volatility_spike']:.1f}x")

with tab2:
    st.subheader("Monte Carlo Simulation")

    col1, col2 = st.columns(2)

    with col1:
        n_sims = st.number_input("Number of Simulations", 1000, 50000, 10000, 1000)
        horizon = st.selectbox("Horizon", [21, 63, 126, 252, 504], index=3,
                               format_func=lambda x: f"{x} days ({x//21} months)")

    with col2:
        method = st.selectbox("Method", ["gbm", "bootstrap", "student_t", "jump_diffusion"])
        initial_value = st.number_input("Initial Value ($)", 1000, 1000000, int(portfolio_value))

    if st.button("Run Monte Carlo", type="primary"):
        with st.spinner(f"Running {n_sims} simulations..."):
            simulator = MonteCarloSimulator(returns, weights, initial_value)
            sim_values = simulator.simulate(n_sims, horizon, method, random_seed=42)
            analysis = simulator.analyze_results(sim_values)

            # Store for later
            st.session_state.mc_results = {
                'values': sim_values,
                'analysis': analysis
            }

            # Display results
            st.markdown("---")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Final Value", f"${analysis['mean_final_value']:,.0f}")
            with col2:
                st.metric("Prob of Loss", f"{analysis['prob_loss']*100:.1f}%")
            with col3:
                st.metric("VaR (95%)", f"${initial_value - analysis['percentiles']['5th']:,.0f}")
            with col4:
                st.metric("Mean Max DD", f"{analysis['mean_max_drawdown']*100:.1f}%")

            # Distribution chart
            final_values = sim_values[:, -1]

            fig = go.Figure()
            fig.add_trace(go.Histogram(x=final_values, nbinsx=50, name='Final Values'))
            fig.add_vline(x=initial_value, line_dash="dash", line_color="red",
                          annotation_text="Initial")
            fig.add_vline(x=analysis['percentiles']['5th'], line_dash="dash",
                          line_color="orange", annotation_text="5th %ile")

            fig.update_layout(
                title="Distribution of Final Portfolio Values",
                xaxis_title="Portfolio Value ($)",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Percentile table
            st.markdown("**Percentile Distribution**")
            pct_df = pd.DataFrame({
                'Percentile': analysis['percentiles'].keys(),
                'Value ($)': [f"${v:,.0f}" for v in analysis['percentiles'].values()]
            })
            st.dataframe(pct_df.T, use_container_width=True)

            # Sample paths
            st.markdown("**Sample Simulation Paths**")
            fig = go.Figure()
            for i in range(min(100, n_sims)):
                fig.add_trace(go.Scatter(
                    y=sim_values[i],
                    mode='lines',
                    line=dict(width=0.5, color='gray'),
                    showlegend=False,
                    opacity=0.3
                ))

            # Add percentile lines
            p5 = np.percentile(sim_values, 5, axis=0)
            p50 = np.percentile(sim_values, 50, axis=0)
            p95 = np.percentile(sim_values, 95, axis=0)

            fig.add_trace(go.Scatter(y=p5, name='5th %ile', line=dict(color='red')))
            fig.add_trace(go.Scatter(y=p50, name='Median', line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(y=p95, name='95th %ile', line=dict(color='green')))

            fig.update_layout(
                title="Simulation Paths",
                xaxis_title="Day",
                yaxis_title="Portfolio Value ($)"
            )
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Custom Stress Test")

    st.markdown("Define a custom shock scenario:")

    col1, col2 = st.columns(2)

    with col1:
        equity_shock = st.slider("Equity Shock (%)", -50, 0, -20) / 100
        vol_mult = st.slider("Volatility Multiplier", 1.0, 5.0, 2.0)

    with col2:
        corr_adj = st.slider("Correlation Adjustment", 0.5, 1.0, 0.9)

    if st.button("Apply Custom Stress"):
        result = stress_tester.parametric_stress(equity_shock, vol_mult, corr_adj)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Portfolio Return", f"{result['portfolio_return']*100:.1f}%")
        with col2:
            st.metric("Portfolio Loss", f"${abs(result['portfolio_loss']):,.0f}")
        with col3:
            st.metric("Ending Value", f"${result['ending_value']:,.0f}")

        st.markdown(f"**Stressed Volatility:** {result['stressed_volatility']*100:.1f}%")

    # Sensitivity analysis
    st.markdown("---")
    st.markdown("**Sensitivity Analysis**")

    if st.button("Run Sensitivity Analysis"):
        sensitivity = stress_tester.sensitivity_analysis()

        fig = go.Figure(data=[
            go.Scatter(
                x=sensitivity['Shock'] * 100,
                y=sensitivity['Ending Value'],
                mode='lines+markers',
                line=dict(color='blue')
            )
        ])
        fig.add_hline(y=portfolio_value, line_dash="dash", line_color="red",
                      annotation_text="Initial")

        fig.update_layout(
            title="Portfolio Value vs Market Shock",
            xaxis_title="Market Shock (%)",
            yaxis_title="Portfolio Value ($)"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(sensitivity.round(2), use_container_width=True)

# ── Hedging Effectiveness During Stress Events ────────────────────────────────
st.markdown("---")
st.subheader("Hedging Effectiveness During Stress Events")

if st.button("Analyse Hedging Effectiveness", key="hedge_stress_btn"):
    with st.spinner("Analysing stress-period hedging..."):
        # Portfolio return series aligned with returns index
        _port_ret_stress = pd.Series(returns.values @ weights, index=returns.index)

        # Full-period betas to classify assets as equity-like vs hedge
        _pv_stress = float(_port_ret_stress.var())
        if _pv_stress > 0:
            _betas_st = {col: float(np.cov(returns[col].values,
                                            _port_ret_stress.values)[0, 1] / _pv_stress)
                         for col in returns.columns}
        else:
            _betas_st = {col: 1.0 for col in returns.columns}

        _beta_threshold_st = 0.5
        _equity_assets = [c for c, b in _betas_st.items() if b >= _beta_threshold_st]
        _hedge_assets = [c for c, b in _betas_st.items() if b < _beta_threshold_st]

        col_info1, col_info2 = st.columns(2)
        col_info1.markdown(f"**Equity-like (β ≥ {_beta_threshold_st}):** "
                           f"{', '.join(_equity_assets) if _equity_assets else 'None'}")
        col_info2.markdown(f"**Hedge / Diversifier (β < {_beta_threshold_st}):** "
                           f"{', '.join(_hedge_assets) if _hedge_assets else 'None'}")

        # Gather per-scenario data
        _scenario_data = []
        for _sk, _sv in HISTORICAL_SCENARIOS.items():
            try:
                _start = pd.Timestamp(_sv["start_date"])
                _end = pd.Timestamp(_sv["end_date"])
                _window = returns.loc[_start:_end]
                if _window.empty:
                    continue
                _cum = (_window + 1).prod() - 1
                _port_cum = float((_port_ret_stress.loc[_start:_end] + 1).prod() - 1)

                # Identify best hedge (most negative return among hedge assets)
                _best_hedge = None
                _best_ret = float("inf")
                for _hc in _hedge_assets:
                    if _hc in _cum.index and float(_cum[_hc]) < _best_ret:
                        _best_ret = float(_cum[_hc])
                        _best_hedge = _hc

                _scenario_data.append({
                    "key": _sk, "name": _sv["name"],
                    "cum": _cum, "port_cum": _port_cum,
                    "best_hedge": _best_hedge,
                })
            except Exception:
                continue

        if not _scenario_data:
            st.warning("No historical scenario windows overlap with your return data.")
        else:
            for _sd in _scenario_data:
                st.markdown(f"### {_sd['name']}")
                col_left, col_right = st.columns(2)

                with col_left:
                    _cum_s = _sd["cum"]
                    _bar_colors = ["crimson" if float(_cum_s[c]) < 0 else "steelblue"
                                   for c in _cum_s.index]
                    _fig_bar = go.Figure(go.Bar(
                        x=list(_cum_s.index),
                        y=(_cum_s * 100).round(2).tolist(),
                        marker_color=_bar_colors, name="Asset Return"
                    ))
                    if _sd["best_hedge"] and _sd["best_hedge"] in _cum_s.index:
                        _bh = _sd["best_hedge"]
                        _fig_bar.add_annotation(
                            x=_bh, y=float(_cum_s[_bh]) * 100,
                            text="★ Best Hedge", showarrow=True, arrowhead=2,
                            font=dict(color="gold", size=13)
                        )
                    _fig_bar.update_layout(
                        title=f"Asset Returns — {_sd['name']}",
                        xaxis_title="Asset", yaxis_title="Cumulative Return (%)"
                    )
                    st.plotly_chart(_fig_bar, use_container_width=True)

                with col_right:
                    _eff_scores = {}
                    _ticker_list = list(returns.columns)
                    if _sd["port_cum"] < 0:
                        for _hc in _hedge_assets:
                            if _hc in _cum_s.index:
                                _hc_ret = float(_cum_s[_hc])
                                _hc_idx = _ticker_list.index(_hc)
                                _hc_wt = float(weights[_hc_idx])
                                _offset = -_hc_ret * _hc_wt
                                _eff_scores[_hc] = round(
                                    _offset / abs(_sd["port_cum"]) * 100, 2
                                )

                    if _eff_scores:
                        _eff_series = pd.Series(_eff_scores,
                                                name="Hedge Effectiveness (% offset)")
                        _fig_eff = px.bar(
                            _eff_series.reset_index(),
                            x="index", y="Hedge Effectiveness (% offset)",
                            color="Hedge Effectiveness (% offset)",
                            color_continuous_scale="Greens",
                            title="Hedge Effectiveness Score"
                        )
                        _fig_eff.update_layout(xaxis_title="Hedge Asset")
                        st.plotly_chart(_fig_eff, use_container_width=True)

                        _avg_eff = float(np.mean(list(_eff_scores.values())))
                        _verdict = ("Strong" if _avg_eff > 30 else
                                    "Moderate" if _avg_eff > 10 else "Weak")
                        st.metric("Avg Hedge Effectiveness",
                                  f"{_avg_eff:.1f}%", delta=_verdict)
                    elif _sd["port_cum"] >= 0:
                        st.info("Portfolio was profitable during this period — no hedging needed.")
                    else:
                        st.info("No hedge assets identified for this scenario.")

                st.markdown("---")
