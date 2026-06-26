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

st.set_page_config(page_title="Risk Analytics", page_icon=None, layout="wide")

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
    st.info(f"Analyzing **optimized portfolio** with {(weights > 0.001).sum()} positions")
elif 'current_portfolio_weights' in st.session_state and st.session_state.current_portfolio_weights is not None:
    # Use actual holdings weights
    weights = st.session_state.current_portfolio_weights
    portfolio_returns = (returns * weights).sum(axis=1)
    st.info(f"Analyzing **your current holdings** with {len(weights)} positions")
else:
    # Equal weight fallback
    weights = pd.Series(1/len(returns.columns), index=returns.columns)
    portfolio_returns = returns.mean(axis=1)
    st.warning("Using equal-weight portfolio. Enter holdings in Portfolio Input or run optimization for accurate analysis.")

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
_rf = st.session_state.get('settings', {}).get('risk_free_rate', 0.02)
port_vol = portfolio_returns.std() * np.sqrt(252)
port_sharpe = (portfolio_returns.mean() * 252 - _rf) / port_vol if port_vol > 0 else 0.0
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
        st.dataframe(var_display, width="stretch")

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
        st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig, width="stretch")

    with col2:
        # Drawdown metrics table
        st.markdown("**Drawdown Metrics**")

        dd_metrics = pd.DataFrame({
            'Max Drawdown': rm.max_drawdown(prices) * 100,
            'Avg Drawdown': rm.average_drawdown(prices) * 100,
            'Ulcer Index': rm.ulcer_index(prices) * 100,
        }).round(2)

        st.dataframe(dd_metrics, width="stretch")

        # Calmar ratio
        calmar = rm.calmar_ratio(prices)
        st.markdown("**Calmar Ratio (Return/MDD)**")
        calmar_df = pd.DataFrame({'Calmar': calmar}).round(3)
        st.dataframe(calmar_df, width="stretch")

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
        st.plotly_chart(fig, width="stretch")

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
        st.dataframe(pairs_df.head(5), width="stretch")

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
        st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig, width="stretch")

# ── Module 2 & 3: Deep Risk Analysis + Hedging Effectiveness ──────────────────
try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    from sklearn.decomposition import PCA as _PCA
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# Normalise weights to a numpy array aligned with returns.columns
if isinstance(weights, (pd.Series, dict)):
    _w = np.array([weights[col] if col in weights else 0.0
                   for col in returns.columns], dtype=float)
else:
    _w = np.array(weights, dtype=float)
_w = _w / _w.sum() if _w.sum() > 0 else _w

# Shared precomputations used by both modules
_cov = returns.cov().values * 252
_n = len(_w)
_port_variance = float(_w @ _cov @ _w)
_port_vol_scalar = float(np.sqrt(_port_variance)) if _port_variance > 0 else 1e-8
_MRC = _cov @ _w                                         # marginal risk contribution
_RC = _w * _MRC                                          # risk contribution (variance units)
_RC_pct = _RC / _port_variance if _port_variance > 0 else _RC
_asset_vols = np.sqrt(np.diag(_cov))
_weighted_vol = float(_w @ _asset_vols)                  # weighted-avg individual vol
_DR = _weighted_vol / _port_vol_scalar                   # diversification ratio
_betas = _MRC / _port_variance if _port_variance > 0 else np.ones(_n)
_beta_series = pd.Series(_betas, index=returns.columns)


@st.cache_data(show_spinner=False)
def _mc_sim(ret_arr: np.ndarray, w_tuple: tuple, n: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    w = np.array(w_tuple)
    port_rets = ret_arr @ w
    port_rets = port_rets[~np.isnan(port_rets)]   # drop NaN rows
    mu = float(np.mean(port_rets)) if len(port_rets) > 0 else 0.0
    sigma = float(np.std(port_rets, ddof=1)) if len(port_rets) > 1 else 1e-8
    return rng.normal(mu, sigma, n)


# ── Module 2: Deep Risk Analysis ──────────────────────────────────────────────
st.markdown("---")
st.subheader("Deep Risk Analysis")

_tab_var2, _tab_tail, _tab_mc = st.tabs(["Enhanced VaR", "Tail Risk", "Monte Carlo"])

with _tab_var2:
    st.markdown("#### Enhanced VaR Comparison")

    _conf_levels = [0.90, 0.95, 0.99]
    _var_rows = {}
    for _cl in _conf_levels:
        _q = float(portfolio_returns.quantile(1 - _cl))
        _sorted_r = np.sort(portfolio_returns.values)
        _tail_idx = max(int(np.floor((1 - _cl) * len(_sorted_r))), 1)
        _cvar_val = float(_sorted_r[:_tail_idx].mean())
        _row = {"Historical VaR (%)": round(_q * 100, 3),
                "CVaR / ES (%)": round(_cvar_val * 100, 3)}
        if _HAS_SCIPY:
            _norm_q = float(portfolio_returns.mean()
                            + _scipy_stats.norm.ppf(1 - _cl) * portfolio_returns.std())
            _row["Normal VaR (%)"] = round(_norm_q * 100, 3)
        _var_rows[f"{int(_cl * 100)}%"] = _row

    _var_df = pd.DataFrame(_var_rows).T

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(_var_df, width="stretch")
        if _HAS_SCIPY and "Normal VaR (%)" in _var_df.columns:
            _divergence = abs(_var_rows["95%"]["Historical VaR (%)"]
                              - _var_rows["95%"]["Normal VaR (%)"])
            st.metric("Historical vs Normal VaR Divergence (95%)", f"{_divergence:.3f}%",
                      help="Large divergence signals fat tails")
    with col2:
        _fig_var2 = go.Figure()
        _fig_var2.add_trace(go.Bar(x=list(_var_rows.keys()),
                                    y=[v["Historical VaR (%)"] for v in _var_rows.values()],
                                    name="Historical VaR", marker_color="steelblue"))
        _fig_var2.add_trace(go.Bar(x=list(_var_rows.keys()),
                                    y=[v["CVaR / ES (%)"] for v in _var_rows.values()],
                                    name="CVaR", marker_color="crimson"))
        _fig_var2.update_layout(barmode="group",
                                 xaxis_title="Confidence Level", yaxis_title="Return (%)")
        st.plotly_chart(_fig_var2, width="stretch")

    st.markdown("#### Portfolio Return Distribution with VaR Overlay")
    _fig_dist = go.Figure()
    _fig_dist.add_trace(go.Histogram(x=portfolio_returns * 100, nbinsx=60,
                                      marker_color="steelblue", opacity=0.7, name="Returns"))
    _fig_dist.add_vline(x=_var_rows["95%"]["Historical VaR (%)"], line_dash="dash",
                         line_color="orange", annotation_text="VaR 95%")
    _fig_dist.add_vline(x=_var_rows["99%"]["Historical VaR (%)"], line_dash="dash",
                         line_color="red", annotation_text="VaR 99%")
    _fig_dist.update_layout(xaxis_title="Return (%)", yaxis_title="Frequency")
    st.plotly_chart(_fig_dist, width="stretch")

with _tab_tail:
    st.markdown("#### Tail Risk Metrics")

    _skew = float(portfolio_returns.skew())
    _kurt = float(portfolio_returns.kurt())
    _downside_vol = portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)
    _port_ann_ret = portfolio_returns.mean() * 252
    _sortino_port = _port_ann_ret / _downside_vol if _downside_vol > 0 else 0.0
    _gains = portfolio_returns[portfolio_returns > 0]
    _losses_neg = portfolio_returns[portfolio_returns <= 0]
    _omega = float(_gains.sum() / abs(_losses_neg.sum())) if len(_losses_neg) > 0 else float("inf")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Skewness", f"{_skew:.3f}")
    col2.metric("Excess Kurtosis", f"{_kurt:.3f}")
    col3.metric("Sortino Ratio", f"{_sortino_port:.3f}")
    col4.metric("Omega Ratio", f"{_omega:.3f}")

    if _HAS_SCIPY:
        _jb_stat, _jb_p = _scipy_stats.jarque_bera(portfolio_returns.dropna())
        _normality = "Non-normal" if _jb_p < 0.05 else "Normal"
        st.markdown(f"**Jarque–Bera test:** statistic = {_jb_stat:.2f}, "
                    f"p-value = {_jb_p:.4f} → **{_normality}** at 5% level")

    # Sortino by asset
    st.markdown("#### Sortino Ratio by Asset")
    _sortino_vals = []
    for _col in returns.columns:
        _ds = returns[_col][returns[_col] < 0].std() * np.sqrt(252)
        _ar = returns[_col].mean() * 252
        _sortino_vals.append({"Ticker": _col,
                               "Sortino": round(_ar / _ds, 3) if _ds > 0 else 0.0})
    _fig_srt = px.bar(pd.DataFrame(_sortino_vals), x="Ticker", y="Sortino",
                       color="Sortino", color_continuous_scale="RdYlGn")
    _fig_srt.update_layout(xaxis_title="Asset", yaxis_title="Sortino Ratio")
    st.plotly_chart(_fig_srt, width="stretch")

    if _HAS_SCIPY:
        st.markdown("#### QQ Plot (Portfolio Returns vs Normal)")
        _rets_clean = portfolio_returns.dropna().values
        _qq = _scipy_stats.probplot(_rets_clean)
        _fig_qq = go.Figure()
        _fig_qq.add_trace(go.Scatter(x=_qq[0][0], y=_qq[0][1], mode="markers",
                                      marker=dict(size=4, color="steelblue"), name="Empirical"))
        _qq_x = np.array([_qq[0][0].min(), _qq[0][0].max()])
        _qq_y = _qq[1][1] + _qq[1][0] * _qq_x
        _fig_qq.add_trace(go.Scatter(x=_qq_x, y=_qq_y, mode="lines",
                                      line=dict(color="red"), name="Normal"))
        _fig_qq.update_layout(xaxis_title="Theoretical Quantiles",
                               yaxis_title="Sample Quantiles")
        st.plotly_chart(_fig_qq, width="stretch")

    st.markdown("#### Tail Event Summary")
    _tail_rows = []
    for _pct in [1, 2.5, 5, 10]:
        _cut = float(np.percentile(portfolio_returns, _pct))
        _tevents = portfolio_returns[portfolio_returns <= _cut]
        _tail_rows.append({"Percentile": f"{_pct}th",
                            "Threshold (%)": round(_cut * 100, 3),
                            "Avg Loss (%)": round(_tevents.mean() * 100, 3),
                            "# Days": len(_tevents)})
    st.dataframe(pd.DataFrame(_tail_rows), width="stretch")

with _tab_mc:
    st.markdown("#### Monte Carlo Risk Simulation")

    _mc_n = st.slider("Simulations", 1000, 20000, 5000, 1000, key="mc_deep_n")
    _mc_rets = _mc_sim(returns.values, tuple(_w.tolist()), _mc_n, seed=42)

    _var_mc = {cl: round(float(np.percentile(_mc_rets, 100 - int(cl.rstrip("%")))) * 100, 3)
               for cl in ["90%", "95%", "99%"]}
    _cvar_mc = {}
    for _cl_str, _var_val in _var_mc.items():
        _mask = _mc_rets <= _var_val / 100
        _cvar_mc[_cl_str] = round(float(_mc_rets[_mask].mean()) * 100, 3) if _mask.any() else _var_val

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Simulated VaR & CVaR**")
        _mc_tbl = pd.DataFrame({"MC VaR (%)": _var_mc, "MC CVaR (%)": _cvar_mc})
        st.dataframe(_mc_tbl, width="stretch")
    with col2:
        _fig_mc_hist = go.Figure()
        _fig_mc_hist.add_trace(go.Histogram(x=_mc_rets * 100, nbinsx=60,
                                             marker_color="steelblue", opacity=0.7))
        _fig_mc_hist.add_vline(x=_var_mc["95%"], line_dash="dash", line_color="orange",
                                annotation_text="VaR 95%")
        _fig_mc_hist.add_vline(x=_var_mc["99%"], line_dash="dash", line_color="red",
                                annotation_text="VaR 99%")
        _fig_mc_hist.update_layout(xaxis_title="Simulated Return (%)", yaxis_title="Frequency")
        st.plotly_chart(_fig_mc_hist, width="stretch")

    st.markdown("#### Simulated Portfolio Paths (100 days)")
    _pv0 = st.session_state.get("portfolio_value", 10000)
    _rng_mc = np.random.default_rng(42)
    _paths = _rng_mc.choice(_mc_rets, size=(100, 100), replace=True)
    _cum_paths = _pv0 * np.cumprod(1 + _paths, axis=1)
    _fig_paths = go.Figure()
    for _pi in range(min(30, _cum_paths.shape[0])):
        _fig_paths.add_trace(go.Scatter(y=_cum_paths[_pi], mode="lines",
                                         line=dict(width=0.5, color="gray"),
                                         opacity=0.3, showlegend=False))
    for _pct_label, _pct_val, _color in [("5th %ile", 5, "red"),
                                           ("Median", 50, "blue"),
                                           ("95th %ile", 95, "green")]:
        _pct_line = np.percentile(_cum_paths, _pct_val, axis=0)
        _lw = 2 if _pct_label == "Median" else 1
        _fig_paths.add_trace(go.Scatter(y=_pct_line, name=_pct_label,
                                         line=dict(color=_color, width=_lw)))
    _fig_paths.update_layout(xaxis_title="Day", yaxis_title="Portfolio Value ($)")
    st.plotly_chart(_fig_paths, width="stretch")


# ── Module 3: Hedging Effectiveness ──────────────────────────────────────────
st.markdown("---")
st.subheader("Hedging Effectiveness")

_tab_h1, _tab_h2, _tab_h3, _tab_h4 = st.tabs([
    "Risk Contribution", "Hedge Classification",
    "Diversification Benefit", "Effective Bets"
])

with _tab_h1:
    st.markdown("#### Weight vs Risk Contribution")

    _rc_df = pd.DataFrame({
        "Weight (%)": (_w * 100).round(2),
        "Risk Contribution (%)": (_RC_pct * 100).round(2),
    }, index=returns.columns)

    col1, col2 = st.columns(2)
    with col1:
        _fig_rc = go.Figure()
        _fig_rc.add_trace(go.Bar(x=_rc_df.index, y=_rc_df["Weight (%)"],
                                  name="Weight", marker_color="steelblue"))
        _fig_rc.add_trace(go.Bar(x=_rc_df.index, y=_rc_df["Risk Contribution (%)"],
                                  name="Risk Contribution", marker_color="crimson"))
        _fig_rc.update_layout(barmode="group", xaxis_title="Asset", yaxis_title="(%)",
                               title="Weight vs Risk Contribution")
        st.plotly_chart(_fig_rc, width="stretch")

    with col2:
        st.dataframe(_rc_df, width="stretch")
        st.metric("Diversification Ratio", f"{_DR:.3f}",
                  help="Weighted-avg vol / portfolio vol. >1 means diversification benefit.")
        _hhi_rc = float(np.sum(_RC_pct ** 2))
        _hhi_w = float(np.sum(_w ** 2))
        st.metric("Risk HHI (concentration)", f"{_hhi_rc:.4f}")
        st.metric("Weight HHI", f"{_hhi_w:.4f}")

    st.markdown("#### Risk Budget")
    _fig_pie = px.pie(values=np.abs(_RC_pct), names=returns.columns,
                       title="Risk Contribution Share")
    _fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(_fig_pie, width="stretch")

with _tab_h2:
    st.markdown("#### Hedge Classification by Portfolio Beta")

    _beta_thresh = st.slider("Equity beta threshold", 0.0, 1.0, 0.5, 0.05, key="beta_thresh")
    _hedge_df = pd.DataFrame({
        "Beta to Portfolio": _beta_series.round(3),
        "Role": ["Equity" if b >= _beta_thresh else "Hedge / Diversifier"
                 for b in _beta_series],
        "Weight (%)": (_w * 100).round(2),
        "Risk Contribution (%)": (_RC_pct * 100).round(2),
    })

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(_hedge_df, width="stretch")
    with col2:
        _color_map = {"Equity": "steelblue", "Hedge / Diversifier": "seagreen"}
        _hedge_plot = _hedge_df.reset_index().rename(columns={_hedge_df.index.name or "index": "Asset"})
        _fig_beta = px.bar(_hedge_plot, x="Asset", y="Beta to Portfolio",
                            color="Role", color_discrete_map=_color_map,
                            title="Asset Beta to Portfolio")
        _fig_beta.update_layout(xaxis_title="Asset")
        st.plotly_chart(_fig_beta, width="stretch")

    st.markdown("#### Component VaR (95%, Normal)")
    _z95 = 1.645
    _comp_var = _z95 * _RC / _port_vol_scalar
    _comp_var_df = pd.DataFrame({
        "Component VaR (daily %)": (_comp_var * 100).round(4),
        "% of Total VaR": ((_comp_var / _comp_var.sum()) * 100).round(2),
    }, index=returns.columns)
    _comp_var_df.index.name = "Ticker"
    _fig_cvar2 = px.bar(_comp_var_df.reset_index(), x="Ticker",
                         y="Component VaR (daily %)",
                         color="Component VaR (daily %)",
                         color_continuous_scale="Reds",
                         title="Component VaR by Asset")
    _fig_cvar2.update_layout(xaxis_title="Asset")
    st.plotly_chart(_fig_cvar2, width="stretch")
    st.dataframe(_comp_var_df, width="stretch")

with _tab_h3:
    st.markdown("#### Diversification Benefit")

    _undiv_var = _weighted_vol ** 2   # variance if all correlations = 1
    _div_benefit = max(0.0, _undiv_var - _port_variance)
    _div_benefit_pct = _div_benefit / _undiv_var * 100 if _undiv_var > 0 else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Undiversified Variance (×10⁴)", f"{_undiv_var * 1e4:.4f}")
    col2.metric("Portfolio Variance (×10⁴)", f"{_port_variance * 1e4:.4f}")
    col3.metric("Diversification Benefit", f"{_div_benefit_pct:.2f}%")

    _fig_wf = go.Figure(go.Waterfall(
        x=["Undiversified", "Diversification Benefit", "Portfolio"],
        y=[_undiv_var * 1e4, -_div_benefit * 1e4, _port_variance * 1e4],
        measure=["absolute", "relative", "total"],
        connector={"line": {"color": "gray"}},
        decreasing={"marker": {"color": "seagreen"}},
        increasing={"marker": {"color": "crimson"}},
        totals={"marker": {"color": "steelblue"}},
    ))
    _fig_wf.update_layout(title="Variance Waterfall (×10⁴)", yaxis_title="Variance (×10⁴)")
    st.plotly_chart(_fig_wf, width="stretch")

    st.markdown("#### Pairwise Variance Contribution")
    _cov_df_disp = returns.cov() * 252
    _var_contrib = pd.DataFrame(
        np.outer(_w, _w) * _cov_df_disp.values * 1e4,
        index=returns.columns, columns=returns.columns
    )
    _fig_hm = px.imshow(_var_contrib.round(4), text_auto=".3f",
                         color_continuous_scale="RdBu_r", aspect="auto",
                         title="Pairwise Variance Contribution (×10⁴)")
    st.plotly_chart(_fig_hm, width="stretch")

with _tab_h4:
    st.markdown("#### Effective Number of Bets (ENB via PCA)")

    if not _HAS_SKLEARN:
        st.info("scikit-learn is required. Install with: `pip install scikit-learn`")
    elif _n < 2:
        st.info("Need at least 2 assets for PCA.")
    else:
        try:
            _pca = _PCA().fit(returns.dropna())
            _ev = _pca.explained_variance_ratio_
            _enb = float(np.exp(-np.sum(_ev * np.log(_ev + 1e-12))))

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Effective Number of Bets", f"{_enb:.2f}",
                          help="Exp(Shannon entropy of PCA eigenvalue distribution). "
                               "Higher = more diversified.")
                st.metric("Total Assets", str(_n))
                st.metric("ENB Efficiency (ENB / N)", f"{_enb / _n:.1%}")

                _scree_df = pd.DataFrame({
                    "Component": [f"PC{i+1}" for i in range(len(_ev))],
                    "Variance Explained (%)": (_ev * 100).round(2),
                    "Cumulative (%)": (np.cumsum(_ev) * 100).round(2),
                })
                st.dataframe(_scree_df, width="stretch")

            with col2:
                _fig_scree = go.Figure()
                _fig_scree.add_trace(go.Bar(x=_scree_df["Component"],
                                             y=_scree_df["Variance Explained (%)"],
                                             name="Individual", marker_color="steelblue"))
                _fig_scree.add_trace(go.Scatter(x=_scree_df["Component"],
                                                 y=_scree_df["Cumulative (%)"],
                                                 name="Cumulative", mode="lines+markers",
                                                 line=dict(color="red"), yaxis="y2"))
                _fig_scree.update_layout(
                    title="PCA Scree Plot",
                    yaxis=dict(title="Variance Explained (%)"),
                    yaxis2=dict(title="Cumulative (%)", overlaying="y", side="right"),
                )
                st.plotly_chart(_fig_scree, width="stretch")

            st.markdown("#### Factor Loadings (Top 5 PCs)")
            _n_pcs = min(5, len(_ev))
            _loadings = pd.DataFrame(
                _pca.components_[:_n_pcs].T,
                index=returns.columns,
                columns=[f"PC{i+1}" for i in range(_n_pcs)]
            )
            _fig_load = px.imshow(_loadings.round(3), text_auto=".2f",
                                   color_continuous_scale="RdBu_r", aspect="auto",
                                   title="PCA Factor Loadings")
            st.plotly_chart(_fig_load, width="stretch")

        except Exception as _pca_err:
            st.warning(f"PCA computation failed: {_pca_err}")


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

st.dataframe(summary_display, width="stretch")

# Export
st.markdown("---")
csv = summary.to_csv()
st.download_button(
    "Download Risk Metrics (CSV)",
    csv,
    "risk_metrics.csv",
    "text/csv"
)
