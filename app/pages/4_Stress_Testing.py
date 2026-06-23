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
from src.simulation.historical_scenarios import (
    HistoricalStressor,
    HistoricalStressorConfig,
    HISTORICAL_SCENARIOS as NEW_HISTORICAL_SCENARIOS,
)
# st.session_state.historical_actual_results  dict[str, HistoricalScenarioResult]
from src.simulation.sector_stress import (
    DEFAULT_SCENARIOS,
    SectorStressConfig,
    SectorStressEngine,
    SectorStressScenario,
)
from src.risk.dcc_garch import DCCGARCHConfig
from src.risk.copula import CopulaConfig
from src.risk.regime_detection import RegimeConfig
from src.risk.sector_beta import SectorBetaConfig
from src.data.data_manager import DataManager

st.set_page_config(page_title="Stress Testing", page_icon=None, layout="wide")

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
    st.info("Stress testing **optimized portfolio**")
elif 'current_portfolio_weights' in st.session_state and st.session_state.current_portfolio_weights is not None:
    weights = st.session_state.current_portfolio_weights.values
    st.info("Stress testing **your current holdings**")
else:
    weights = np.ones(len(returns.columns)) / len(returns.columns)
    st.warning("Using equal weights. Enter holdings or run optimization for accurate results.")

# Portfolio value
portfolio_value = st.session_state.get('settings', {}).get('total_capital', 10000)

# Initialize stress tester
stress_tester = StressTester(returns, weights, portfolio_value)

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Historical Scenarios", "Monte Carlo", "Custom Stress", "🔬 Sector Shock"
])

with tab1:
    st.subheader("Historical Stress Scenarios")

    use_actual_returns = st.toggle(
        "Use actual per-stock returns (recommended)",
        value=True,
        help=(
            "ON: pulls real price data for each stock during the crisis window. "
            "Stocks that didn't exist yet use beta-scaled index returns. "
            "OFF: legacy mode — applies uniform hardcoded equity drop to all holdings."
        ),
    )

    if use_actual_returns:
        # ── New path — actual per-stock returns ───────────────────────────────
        if st.button("Run All Historical Scenarios", type="primary", key="run_hist_actual"):
            _tickers = list(returns.columns)
            _weights_series = pd.Series(
                {t: float(w) for t, w in zip(_tickers, weights)}
            )
            _pv = st.session_state.get("settings", {}).get("total_capital", portfolio_value)

            with st.spinner("Fetching actual crisis returns from yfinance…"):
                _actual_results = stress_tester.run_historical_actual(
                    tickers=_tickers,
                    weights=_weights_series,
                    portfolio_value=_pv,
                )
                st.session_state.historical_actual_results = _actual_results

        if "historical_actual_results" in st.session_state:
            _actual_results = st.session_state.historical_actual_results
            _stressor = HistoricalStressor()
            _summary_df = _stressor.to_comparison_dataframe(_actual_results)

            st.dataframe(
                _summary_df.style.format({
                    "Index Return": "{:.1%}",
                    "Portfolio Return": "{:.1%}",
                    "Portfolio P&L": "${:,.0f}",
                }).background_gradient(subset=["Portfolio Return"], cmap="RdYlGn"),
                use_container_width=True,
            )

            # Per-scenario drill-down
            st.markdown("---")
            st.markdown("**Drill into scenario**")
            _sel_scenario = st.selectbox(
                "Select scenario",
                options=list(_actual_results.keys()),
                key="hist_actual_sel",
            )

            if _sel_scenario:
                _res = _actual_results[_sel_scenario]
                _bd = _stressor.to_stock_breakdown(_res)

                st.caption(
                    f"Index ({_res.scenario.market_index}): "
                    f"{_res.index_return:.1%} | "
                    f"Portfolio: {_res.portfolio_return:.1%} | "
                    f"Actual data: {_res.n_actual}/{_res.n_actual + _res.n_beta_scaled} stocks"
                )

                def _highlight_source(row):
                    if row["Source"] == "beta_scaled":
                        return ["background-color: #fff3cd"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    _bd.style
                        .apply(_highlight_source, axis=1)
                        .format({
                            "Realized Return": "{:.1%}",
                            "Beta Used": "{:.2f}",
                            "P&L ($)": "${:,.0f}",
                        }),
                    use_container_width=True,
                )

                _warnings = [
                    f"⚠️ {row['Ticker']}: {row['Warning']}"
                    for _, row in _bd.iterrows()
                    if row["Warning"]
                ]
                for _w in _warnings:
                    st.warning(_w)

                st.caption(
                    "🟡 Yellow rows = beta-scaled (stock did not exist during this crisis). "
                    "White rows = actual historical returns."
                )

    else:
        # ── Legacy path — uniform hardcoded shocks (unchanged) ────────────────
        if st.button("Run All Historical Scenarios", type="primary", key="run_hist_legacy"):
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

    # Individual scenario details (legacy reference — always shown)
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

with tab4:
    st.subheader("Sector Shock Stress Test")
    st.markdown(
        "Propagates sector-level shocks through a portfolio using "
        "**DCC-GARCH** dynamic correlations, **Student-t Copula** tail dependence, "
        "and **HMM** regime-conditioned correlation selection."
    )

    # ── Model configuration ──────────────────────────────────────────────────
    with st.expander("Model Configuration", expanded=False):
        cfg_col1, cfg_col2, cfg_col3 = st.columns(3)

        with cfg_col1:
            st.markdown("**DCC-GARCH**")
            _dcc_alpha = st.number_input(
                "α init (news impact)", 0.01, 0.20, 0.05, step=0.01,
                key="ss_dcc_alpha",
            )
            _dcc_beta = st.number_input(
                "β init (correlation persistence)", 0.50, 0.99, 0.90, step=0.01,
                key="ss_dcc_beta",
            )
            _estimate_dcc = st.checkbox("Estimate DCC params (MLE)", True, key="ss_estimate_dcc")

        with cfg_col2:
            st.markdown("**Student-t Copula**")
            _copula_type = st.selectbox("Type", ["t", "gaussian"], key="ss_copula_type")
            _n_sim = st.number_input(
                "Simulation paths", 1000, 50000, 10000, step=1000, key="ss_n_sim"
            )
            _estimate_df = st.checkbox(
                "Estimate degrees of freedom (MLE)", True, key="ss_estimate_df"
            )

        with cfg_col3:
            st.markdown("**HMM Regime Detector**")
            _n_states = st.selectbox("Number of states", [2, 3, 4], index=1, key="ss_n_states")
            _n_init_hmm = st.number_input(
                "HMM initialisations", 3, 20, 10, step=1, key="ss_n_init"
            )

        st.markdown("---")
        _class_level = st.selectbox(
            "TRBC classification level",
            ["economic", "business", "industry"],
            help=(
                "economic = broadest (8-10 sectors) | "
                "business = mid (25-30) | "
                "industry = finest (70+)"
            ),
            key="ss_class_level",
        )
        _pv_sector = st.number_input(
            "Portfolio value ($)",
            min_value=1_000,
            max_value=1_000_000_000,
            value=int(st.session_state.get("settings", {}).get("total_capital", 1_000_000)),
            step=10_000,
            key="ss_portfolio_value",
        )

    # ── Fetch sectors & fit models ───────────────────────────────────────────
    st.markdown("---")
    _fit_clicked = st.button(
        "Fetch Sectors & Fit Models", type="primary", key="ss_fit_btn"
    )

    if _fit_clicked:
        _tickers = list(returns.columns)

        _stress_cfg = SectorStressConfig(
            beta_config=SectorBetaConfig(),
            dcc_config=DCCGARCHConfig(
                dcc_alpha_init=float(st.session_state.ss_dcc_alpha),
                dcc_beta_init=float(st.session_state.ss_dcc_beta),
                estimate_dcc_params=bool(st.session_state.ss_estimate_dcc),
            ),
            copula_config=CopulaConfig(
                copula_type=str(st.session_state.ss_copula_type),
                n_simulation_paths=int(st.session_state.ss_n_sim),
                estimate_df=bool(st.session_state.ss_estimate_df),
            ),
            regime_config=RegimeConfig(
                n_states=int(st.session_state.ss_n_states),
                n_init=int(st.session_state.ss_n_init),
            ),
            portfolio_value=float(st.session_state.ss_portfolio_value),
        )

        with st.spinner("Fetching TRBC sector classifications (yfinance fallback)…"):
            try:
                _dm = DataManager(show_progress=False)
                _sector_map = _dm.get_sector_classifications(
                    _tickers,
                    level=str(st.session_state.ss_class_level),
                )
                st.session_state.ss_sector_map = _sector_map
                st.session_state.ss_class_level_used = st.session_state.ss_class_level
                unique_s = sorted(set(_sector_map.values()) - {"Unknown"})
                st.success(
                    f"Sectors fetched: {len(unique_s)} unique — {', '.join(unique_s)}"
                )
            except Exception as _e:
                st.error(f"Sector fetch failed: {_e}")
                st.stop()

        with st.spinner(
            "Fitting DCC-GARCH → Student-t Copula → HMM Regime Detector…"
        ):
            try:
                _engine = SectorStressEngine(config=_stress_cfg)
                _engine.fit(returns, st.session_state.ss_sector_map)
                st.session_state.ss_engine = _engine
                st.session_state.ss_stress_cfg = _stress_cfg
            except Exception as _e:
                st.error(f"Model fitting failed: {_e}")
                st.stop()

        st.session_state.pop("ss_result", None)
        st.session_state.pop("ss_all_results", None)
        st.rerun()

    # ── Regime badge + sub-model status ─────────────────────────────────────
    if "ss_engine" in st.session_state:
        _engine = st.session_state.ss_engine
        _summary = _engine.get_fit_summary()
        _regime = _summary["current_regime"]
        _regime_prob = _summary["regime_probability"]

        _REGIME_ICON = {
            "calm": "🟢", "elevated": "🟡",
            "mild_stress": "🟠", "crisis": "🔴", "unknown": "⚪",
        }
        _icon = _REGIME_ICON.get(_regime, "⚪")

        col_regime, col_models = st.columns([1, 2])
        with col_regime:
            st.markdown("**Current Market Regime**")
            st.markdown(f"## {_icon} {_regime.replace('_', ' ').upper()}")
            st.markdown(f"Confidence: **{_regime_prob:.1%}**")

        with col_models:
            st.markdown("**Sub-model Status**")
            _sc1, _sc2, _sc3, _sc4 = st.columns(4)
            _sc1.metric("Beta", "OK" if _summary["beta"] else "BAD")
            _sc2.metric("DCC-GARCH", "OK" if _summary["dcc"] else "WARNING")
            _sc3.metric("Copula", "OK" if _summary["copula"] else "WARNING")
            _sc4.metric("HMM Regime", "OK" if _summary["regime"] else "WARNING")

        if _summary["warnings"]:
            with st.expander(
                f"{len(_summary['warnings'])} fitting warning(s)", expanded=False
            ):
                for _w in _summary["warnings"]:
                    st.warning(_w)

        st.markdown("---")

        # ── Matrix expanders ─────────────────────────────────────────────────
        _mx1, _mx2 = st.columns(2)

        with _mx1:
            with st.expander("Cross-Sector Beta Matrix", expanded=False):
                if _summary["beta"] and _engine._beta_result is not None:
                    _beta_df = _engine._beta_result.beta_matrix_average
                    _fig_beta = px.imshow(
                        _beta_df.round(3),
                        color_continuous_scale="RdBu_r",
                        color_continuous_midpoint=0,
                        text_auto=".2f",
                        title="Beta Matrix (1Y/3Y average)",
                    )
                    _fig_beta.update_layout(height=380)
                    st.plotly_chart(_fig_beta, use_container_width=True)
                    _n_un = _engine._beta_result.n_unstable_pairs
                    if _n_un > 0:
                        st.warning(f"{_n_un} unstable sector pair(s) — beta estimates may drift.")
                else:
                    st.info("Beta model not fitted.")

        with _mx2:
            with st.expander("DCC Correlation Matrix", expanded=False):
                if _summary["dcc"] and _engine._dcc_result is not None:
                    _corr_choice = st.radio(
                        "Snapshot",
                        ["Current", "Stress (95th %ile)", "Calm (5th %ile)"],
                        horizontal=True,
                        key="ss_corr_choice",
                    )
                    if _corr_choice == "Current":
                        _corr_df = _engine._dcc_result.current_correlation
                    elif "Stress" in _corr_choice:
                        _corr_df = _engine._dcc_result.stress_correlation
                    else:
                        _corr_df = _engine._dcc_result.calm_correlation

                    _fig_corr = px.imshow(
                        _corr_df.round(3),
                        color_continuous_scale="RdBu_r",
                        color_continuous_midpoint=0,
                        zmin=-1, zmax=1,
                        text_auto=".2f",
                        title="DCC Correlation",
                    )
                    _fig_corr.update_layout(height=380)
                    st.plotly_chart(_fig_corr, use_container_width=True)
                else:
                    st.info("DCC-GARCH model not fitted.")

        st.markdown("---")

        # ── Scenario selector ────────────────────────────────────────────────
        st.subheader("Scenario")

        _scenario_names = [s.name for s in DEFAULT_SCENARIOS] + ["Custom…"]
        _sel_name = st.selectbox(
            "Select a scenario", _scenario_names, key="ss_scenario_name"
        )

        if _sel_name == "Custom…":
            st.markdown("**Define custom shock:**")
            _sm = st.session_state.get("ss_sector_map", {})
            _uniq_secs = sorted(set(_sm.values()) - {"Unknown"})
            _shock_secs = st.multiselect(
                "Sectors to shock", _uniq_secs, key="ss_custom_secs"
            )
            _custom_shocks: dict = {}
            for _s in _shock_secs:
                _sv = (
                    st.slider(f"{_s} shock (%)", -50, 50, -20, key=f"ss_sl_{_s}") / 100.0
                )
                _custom_shocks[_s] = _sv
            _cop_q = st.slider(
                "Copula quantile (0.05 = 5th %-ile loss tail)",
                0.01, 0.99, 0.05, key="ss_custom_cop_q",
            )
            _active_scenario = SectorStressScenario(
                name="Custom Scenario",
                description="User-defined sector shock",
                shocked_sectors=_custom_shocks,
                copula_shock_quantile=_cop_q,
            )
        else:
            _active_scenario = next(
                s for s in DEFAULT_SCENARIOS if s.name == _sel_name
            )
            _dc1, _dc2 = st.columns([2, 1])
            with _dc1:
                st.markdown(f"**{_active_scenario.name}**")
                st.caption(_active_scenario.description)
            with _dc2:
                st.markdown("**Sector shocks:**")
                for _sec, _shk in _active_scenario.shocked_sectors.items():
                    _clr = "red" if _shk < 0 else "green"
                    st.markdown(f"- {_sec}: :{_clr}[{_shk:+.0%}]")

        # ── Run buttons ───────────────────────────────────────────────────────
        _rb1, _rb2 = st.columns(2)
        with _rb1:
            _run_single = st.button(
                f"Run: {_active_scenario.name}", type="primary", key="ss_run_single"
            )
        with _rb2:
            _run_all = st.button(
                "Run All 7 Default Scenarios", key="ss_run_all"
            )

        _tickers = list(returns.columns)
        _holdings = {t: float(w) for t, w in zip(_tickers, weights)}

        if _run_single:
            with st.spinner(f"Running '{_active_scenario.name}'…"):
                _result = _engine.run_stress(_active_scenario, _holdings)
                st.session_state.ss_result = _result

        if _run_all:
            with st.spinner("Running all 7 scenarios…"):
                st.session_state.ss_all_results = _engine.run_all_scenarios(_holdings)

        # ── Single scenario results ───────────────────────────────────────────
        if "ss_result" in st.session_state:
            _res = st.session_state.ss_result
            _pv = st.session_state.ss_stress_cfg.portfolio_value

            st.markdown("---")
            st.subheader(f"Results — {_res.scenario.name}")

            _rm1, _rm2, _rm3, _rm4 = st.columns(4)
            _rm1.metric("Beta P&L", f"${_res.total_beta_pnl:,.0f}")
            _rm2.metric("Copula VaR P&L", f"${_res.total_copula_pnl:,.0f}")
            _rm3.metric(
                "Regime",
                f"{_res.regime_at_shock.upper()} ({_res.regime_probability:.0%})",
            )
            _rm4.metric("Holdings", str(len(_res.holdings_results)))

            _df_res = _res.to_dataframe()

            if not _df_res.empty:
                # Waterfall chart — top 20 by |beta PnL|
                st.markdown("#### Beta P&L Contribution (Waterfall)")
                _wf_df = _df_res.head(20)
                _wf_measures = ["relative"] * len(_wf_df) + ["total"]
                _wf_x = list(_wf_df["ticker"]) + ["TOTAL"]
                _wf_y = list(_wf_df["pnl_contribution_beta"]) + [_res.total_beta_pnl]
                _wf_text = [
                    (f"+${v:,.0f}" if v >= 0 else f"-${abs(v):,.0f}") for v in _wf_y
                ]

                _fig_wf = go.Figure(go.Waterfall(
                    orientation="v",
                    measure=_wf_measures,
                    x=_wf_x,
                    y=_wf_y,
                    text=_wf_text,
                    textposition="outside",
                    connector={"line": {"color": "rgba(80,80,80,0.4)", "width": 1}},
                    increasing={"marker": {"color": "#3b82f6"}},
                    decreasing={"marker": {"color": "#ef4444"}},
                    totals={"marker": {"color": "#374151"}},
                ))
                _fig_wf.update_layout(
                    yaxis_title="P&L ($)",
                    xaxis_title="Holding",
                    height=460,
                    showlegend=False,
                    margin=dict(t=30),
                )
                st.plotly_chart(_fig_wf, use_container_width=True)

            # Holdings table
            st.markdown("#### Holdings Detail")
            st.dataframe(
                _df_res.style.format({
                    "weight": "{:.2%}",
                    "beta_implied_return": "{:+.2%}",
                    "copula_median_return": "{:+.2%}",
                    "copula_var_return": "{:+.2%}",
                    "pnl_contribution_beta": "${:,.0f}",
                    "pnl_contribution_copula": "${:,.0f}",
                }),
                use_container_width=True,
                height=340,
            )

            # Most exposed / natural hedges
            _exp_col, _hdg_col = st.columns(2)
            with _exp_col:
                st.markdown("**Most Exposed (Largest Loss)**")
                _exposed = _engine.get_most_exposed(_res, top_n=5)
                if not _exposed.empty:
                    st.dataframe(
                        _exposed.style.format(
                            {"weight": "{:.2%}", "pnl": "${:,.0f}"}
                        ),
                        use_container_width=True,
                    )
                else:
                    st.info("No holdings with negative P&L.")

            with _hdg_col:
                st.markdown("**Natural Hedges (Positive P&L)**")
                _hedges = _engine.get_hedge_candidates(_res, top_n=5)
                if not _hedges.empty:
                    st.dataframe(
                        _hedges.style.format({
                            "weight": "{:.2%}",
                            "pnl_contribution_beta": "${:,.0f}",
                        }),
                        use_container_width=True,
                    )
                else:
                    st.info("No holdings gain under this scenario.")

            # Beta stability warnings
            _unstable = _df_res[_df_res["beta_stability"] == "unstable"]
            if not _unstable.empty:
                with st.expander(
                    f"⚠️ Beta Stability Warnings ({len(_unstable)} holdings)", expanded=True
                ):
                    st.warning(
                        "These holdings sit in sectors where the 1Y and 3Y beta "
                        "estimates diverge by more than the stability threshold. "
                        "Beta-implied P&L figures may be unreliable."
                    )
                    st.dataframe(
                        _unstable[
                            ["ticker", "sector", "beta_implied_return", "pnl_contribution_beta"]
                        ].style.format({
                            "beta_implied_return": "{:+.2%}",
                            "pnl_contribution_beta": "${:,.0f}",
                        }),
                        use_container_width=True,
                    )

            # Copula correlation used
            if not _res.correlation_used.empty:
                with st.expander("Correlation Matrix Used in This Run", expanded=False):
                    _fig_cu = px.imshow(
                        _res.correlation_used.round(3),
                        color_continuous_scale="RdBu_r",
                        color_continuous_midpoint=0,
                        zmin=-1, zmax=1,
                        text_auto=".2f",
                    )
                    _fig_cu.update_layout(height=380)
                    st.plotly_chart(_fig_cu, use_container_width=True)

            # Run notes
            if _res.warnings:
                with st.expander(
                    f"ℹ️ {len(_res.warnings)} scenario note(s)", expanded=False
                ):
                    for _w in _res.warnings:
                        st.info(_w)

        # ── Scenario comparison ──────────────────────────────────────────────
        if "ss_all_results" in st.session_state:
            _all = st.session_state.ss_all_results
            _pv_cmp = (
                st.session_state.ss_stress_cfg.portfolio_value
                if "ss_stress_cfg" in st.session_state
                else 1_000_000.0
            )

            st.markdown("---")
            st.subheader("Scenario Comparison")

            _cmp_rows = []
            for _r in _all:
                _cmp_rows.append({
                    "Scenario": _r.scenario.name,
                    "Beta P&L ($)": _r.total_beta_pnl,
                    "Beta P&L (%)": _r.total_beta_pnl / _pv_cmp * 100,
                    "Copula P&L ($)": _r.total_copula_pnl,
                    "Copula P&L (%)": _r.total_copula_pnl / _pv_cmp * 100,
                    "Regime": _r.regime_at_shock.upper(),
                    "Notes": len(_r.warnings),
                })

            _cmp_df = pd.DataFrame(_cmp_rows)
            st.dataframe(
                _cmp_df.style.format({
                    "Beta P&L ($)": "${:,.0f}",
                    "Beta P&L (%)": "{:+.2f}%",
                    "Copula P&L ($)": "${:,.0f}",
                    "Copula P&L (%)": "{:+.2f}%",
                }).background_gradient(subset=["Beta P&L ($)"], cmap="RdYlGn"),
                use_container_width=True,
            )

            # Grouped bar chart
            _fig_cmp = go.Figure()
            _fig_cmp.add_trace(go.Bar(
                name="Beta P&L",
                x=_cmp_df["Scenario"],
                y=_cmp_df["Beta P&L ($)"],
                marker_color=[
                    "#ef4444" if v < 0 else "#3b82f6"
                    for v in _cmp_df["Beta P&L ($)"]
                ],
            ))
            _fig_cmp.add_trace(go.Bar(
                name="Copula VaR P&L",
                x=_cmp_df["Scenario"],
                y=_cmp_df["Copula P&L ($)"],
                marker_color=[
                    "#f97316" if v < 0 else "#22c55e"
                    for v in _cmp_df["Copula P&L ($)"]
                ],
            ))
            _fig_cmp.update_layout(
                barmode="group",
                xaxis_tickangle=-30,
                yaxis_title="P&L ($)",
                height=430,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(_fig_cmp, use_container_width=True)


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

                # Identify best hedge (highest return = best offset of portfolio loss)
                _best_hedge = None
                _best_ret = float("-inf")
                for _hc in _hedge_assets:
                    if _hc in _cum.index and float(_cum[_hc]) > _best_ret:
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
                            text=" Best Hedge", showarrow=True, arrowhead=2,
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
                                # positive _hc_ret means the hedge gained → offset loss
                                # negative _hc_ret means the hedge also lost → made it worse
                                _offset = _hc_ret * _hc_wt
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
