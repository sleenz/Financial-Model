"""
Portfolio Builder — assembles Phases 1-4 of src/portfolio_builder into one
Streamlit page: ticker chips, ranked list (heat-colored score + editable
share count + computed % weight), correlation network (sector overview
with drill-into-sector detail), and a metrics panel (HHI/diversification
+ period-matched Sharpe estimate).

Architecture note (Phase 5 CHECK: "no backend call fires on a
share-count edit"): every network/data-fetch/DCC-GARCH-fit call is
cached in st.session_state, keyed by the current TICKER SET only. Share
count edits change % weight (and the Sharpe/volatility numbers, which
are cheap re-aggregations of already-fetched data) via plain arithmetic
on rerun — they never re-key the cache or trigger a new fetch/fit. A
visible call counter at the bottom of the page makes this observable,
not just asserted.
"""

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.data_manager import DataManager
from src.portfolio_builder.cache import CacheConfig, UniverseCache
from src.portfolio_builder.fetch import FetchConfig, OnDemandFetcher, PortfolioDataLayer
from src.portfolio_builder.metrics import (
    DiversificationConfig,
    SharpeConfig,
    compute_dcc_garch_volatility_trailing,
    compute_diversification_rating,
    compute_realized_return,
    compute_sector_exposure,
    compute_sharpe,
    render_sharpe_methodology_disclosure,
)
from src.portfolio_builder.network import (
    CorrelationNetworkConfig,
    NetworkStyleConfig,
    build_semantic_zoom_network,
    correlation_from_distance,
    edge_color_for_correlation,
    filter_edges_by_threshold,
    get_sector_subgraph,
    node_color_for_percentile,
)
from src.risk.dcc_garch import DCCGARCHConfig, DCCGARCHModel

st.set_page_config(page_title="Portfolio Builder", layout="wide")
st.title("Portfolio Builder")

# ── Manual risk-free constant (SharpeConfig.risk_free_rate must be set
# explicitly — see metrics.py docstring: FRED is broken upstream, and
# compute_sharpe() refuses to silently default this to 0.0). ──────────────
_MANUAL_RISK_FREE_RATE = 0.045  # update periodically until FRED is fixed

# ── Correlation network edge-filter defaults — threshold is only the
# slider's INITIAL value below; the widget's own state drives every
# subsequent rerun (see the Correlation Network section). ────────────────
_CORRELATION_NETWORK_CONFIG = CorrelationNetworkConfig()

# ── Correlation network / DCC-GARCH lookback — DELIBERATELY DECOUPLED from
# SharpeConfig().lookback_days. Both features share the same fetched
# hist_prices DataFrame, but each slices its OWN trailing window from it
# below (see _compute_backend_data) — otherwise extending the Sharpe
# window (e.g. to 3 years, per explicit instruction) would silently also
# extend the correlation network's and DCC-GARCH's effective sample window,
# an unrelated feature nobody asked to change. 252 preserves their
# original, already-reviewed 1-year effective window exactly. ────────────
_CORRELATION_LOOKBACK_DAYS = 252


# ── Cached resources (survive reruns within a session — not reopened per
# rerun the way a plain UniverseCache()/DataManager() call would be) ──────
@st.cache_resource
def _get_universe_cache() -> UniverseCache:
    return UniverseCache(CacheConfig())


@st.cache_resource
def _get_data_layer() -> PortfolioDataLayer:
    return PortfolioDataLayer(FetchConfig())


@st.cache_resource
def _get_on_demand_fetcher() -> OnDemandFetcher:
    return OnDemandFetcher(_get_universe_cache(), _get_data_layer())


# ── Session state ─────────────────────────────────────────────────────────
if "pb_tickers" not in st.session_state:
    st.session_state.pb_tickers = []
if "pb_shares" not in st.session_state:
    st.session_state.pb_shares = {}
if "pb_shares_baseline" not in st.session_state:
    # Separate from pb_shares on purpose (real bug found + fixed in this
    # session): feeding pb_shares — which the data_editor's own output
    # updates every rerun via the sync-back loop below — back into that
    # SAME editor's `data=` argument creates a moving-baseline feedback
    # loop. Streamlit's data_editor only correctly accumulates edits
    # across reruns (via its `key`) when the `data=` it's given stays
    # stable; once the baseline itself starts reflecting the previous
    # edit, only the FIRST edit in a session ever registers — every
    # subsequent edit to any row is silently dropped (confirmed via
    # st.session_state["pb_ranked_editor"]["edited_rows"] staying `{}`
    # after the second edit, regardless of interaction method: mouse,
    # keyboard, or a different row each time). pb_shares_baseline is set
    # ONCE per ticker (on add, seeded from any remembered value) and never
    # touched by the sync-back loop, so the editor's `data=` argument
    # never moves out from under it and every edit registers correctly.
    st.session_state.pb_shares_baseline = {}
if "pb_backend_cache" not in st.session_state:
    st.session_state.pb_backend_cache = {}
if "pb_backend_call_count" not in st.session_state:
    st.session_state.pb_backend_call_count = 0


def _compute_backend_data(tickers: list) -> dict:
    """Everything that needs a fetch, a network build, or a model fit.
    Called at most once per distinct ticker SET — see _get_backend_data's
    cache check. Every failure is caught, logged, and surfaced via
    result['errors'] rather than silently omitted or crashing the page."""
    st.session_state.pb_backend_call_count += 1
    result: dict = {"errors": []}

    fetcher = _get_on_demand_fetcher()
    cache = _get_universe_cache()
    data_layer = _get_data_layer()

    entries = {}
    for t in tickers:
        try:
            entries[t] = fetcher.get_or_fetch(t)
        except Exception as exc:
            result["errors"].append(f"{t}: ranking data unavailable ({exc})")
    result["entries"] = entries
    result["sector_map"] = {t: e.sector for t, e in entries.items()}

    if entries:
        try:
            dm = DataManager(show_progress=False)
            result["prices"] = dm.get_current_prices(list(entries.keys()))
        except Exception as exc:
            result["errors"].append(f"Current prices unavailable: {exc}")

        try:
            end = datetime.now()
            start = end - timedelta(days=SharpeConfig().lookback_days * 2)
            dm = DataManager(show_progress=False)
            hist_prices = dm.get_price_data(list(entries.keys()), start, end)
            result["hist_prices"] = hist_prices
        except Exception as exc:
            result["errors"].append(f"Historical prices unavailable: {exc}")

    if len(entries) >= 2:
        # Prefer a correlation matrix computed live from the historical
        # prices just fetched above over UniverseCache's correlation_row.
        # OnDemandFetcher.get_or_fetch() always leaves correlation_row
        # empty for a freshly-fetched ticker (deferred to the next
        # run_nightly_refresh() — see fetch.py) and no such job is
        # actually scheduled in this deployment, so every first-time
        # ticker's row is permanently empty and the cache-only path
        # (build_correlation_matrix) would exclude every ticker and raise
        # "no tickers had usable correlation data" for any new user.
        # Falls back to the cache-only path if fresh prices aren't
        # available, so a ticker set that DOES have cached correlation
        # data (e.g. after a nightly refresh eventually runs) still works
        # even if a live price fetch fails.
        live_correlation = None
        hist_prices = result.get("hist_prices")
        if hist_prices is not None and not hist_prices.empty:
            # Sliced to _CORRELATION_LOOKBACK_DAYS, NOT the full (now
            # possibly 3-year) hist_prices fetch — see that constant's
            # comment for why this stays decoupled from Sharpe's window.
            live_returns = hist_prices.iloc[-_CORRELATION_LOOKBACK_DAYS:].pct_change().dropna(how="all")
            if len(live_returns) >= data_layer.config.min_correlation_overlap_days:
                live_correlation = live_returns.corr()
        try:
            result["zoom"] = build_semantic_zoom_network(
                cache, result["sector_map"], tickers=list(entries.keys()),
                correlation=live_correlation,
            )
        except Exception as exc:
            result["errors"].append(f"Correlation network unavailable: {exc}")

    n_sectors = len(set(result["sector_map"].values()))
    if "hist_prices" in result and result["hist_prices"] is not None and n_sectors >= 2:
        try:
            returns = result["hist_prices"].pct_change().dropna(how="all")
            sector_returns = {}
            for sector in set(result["sector_map"].values()):
                members = [t for t, s in result["sector_map"].items() if s == sector and t in returns.columns]
                if members:
                    sector_returns[sector] = returns[members].mean(axis=1)
            sector_returns_df = pd.DataFrame(sector_returns).dropna()
            dcc_config = DCCGARCHConfig(estimate_dcc_params=False, min_observations=60)
            result["dcc_result"] = DCCGARCHModel(dcc_config).fit(sector_returns_df)
        except Exception as exc:
            result["errors"].append(f"DCC-GARCH fit unavailable: {exc}")

    return result


def _get_backend_data(ticker_key: tuple):
    """Cache lookup keyed ONLY by the ticker set — a share-count edit
    never changes this key, so it never re-triggers _compute_backend_data."""
    if not ticker_key:
        return None
    if ticker_key not in st.session_state.pb_backend_cache:
        st.session_state.pb_backend_cache[ticker_key] = _compute_backend_data(list(ticker_key))
    return st.session_state.pb_backend_cache[ticker_key]


def _add_ticker_callback() -> None:
    t = st.session_state.pb_new_ticker_input.strip().upper()
    if t and t not in st.session_state.pb_tickers:
        st.session_state.pb_tickers.append(t)
        st.session_state.pb_shares.setdefault(t, 0)
        # Seed the editor's stable baseline from any remembered share count
        # (e.g. this ticker was removed and is now being re-added) — set
        # once here, never touched again while the ticker stays in the set.
        st.session_state.pb_shares_baseline[t] = st.session_state.pb_shares[t]
    st.session_state.pb_new_ticker_input = ""


# ── Ticker chips ───────────────────────────────────────────────────────────
st.subheader("Universe")
add_col, btn_col = st.columns([4, 1])
with add_col:
    st.text_input(
        "Add ticker", key="pb_new_ticker_input", label_visibility="collapsed",
        placeholder="Add a ticker, e.g. AAPL",
    )
with btn_col:
    # Clearing pb_new_ticker_input must happen in an on_click callback, not
    # inline below — Streamlit raises StreamlitAPIException if a widget's
    # session_state key is written after that widget has already been
    # instantiated in the same script run. Callbacks run before the rerun
    # (and before the widget is recreated), so this is the one place
    # writing to it is actually valid.
    st.button("Add", use_container_width=True, on_click=_add_ticker_callback)

if st.session_state.pb_tickers:
    chip_cols = st.columns(min(len(st.session_state.pb_tickers), 8) or 1)
    for i, t in enumerate(st.session_state.pb_tickers):
        with chip_cols[i % len(chip_cols)]:
            if st.button(f"{t}  ✕", key=f"pb_remove_{t}"):
                st.session_state.pb_tickers.remove(t)
                st.session_state.pb_shares.pop(t, None)
                st.session_state.pb_shares_baseline.pop(t, None)
                st.rerun()
else:
    st.caption("No tickers yet — add some above.")

ticker_key = tuple(sorted(st.session_state.pb_tickers))
backend = _get_backend_data(ticker_key)

for err in (backend or {}).get("errors", []):
    st.warning(err)

# ── Ranked list: ticker, sector, heat-colored score, shares, % weight ─────
st.subheader("Ranked List")

if not backend or not backend.get("entries"):
    st.caption("Add tickers above to see the ranked list.")
else:
    entries = backend["entries"]
    scores = pd.Series({t: e.composite_score for t, e in entries.items()})
    percentile = scores.rank(pct=True) if len(scores) > 1 else pd.Series(1.0, index=scores.index)

    def _heat_label(t: str) -> str:
        p = percentile[t]
        emoji = "🟢" if p >= 0.67 else ("🟡" if p >= 0.33 else "🔴")
        return f"{emoji} {scores[t]:.1f}"

    # Streamlit garbage-collects a keyed widget's internal state
    # (st.session_state["pb_ranked_editor"]["edited_rows"]) whenever that
    # widget isn't rendered on a script run — which happens every time the
    # user navigates to a different page, since this page's script (and
    # therefore this data_editor call) doesn't run at all while another
    # page is showing. Coming back re-creates the widget with an empty
    # diff, and since ranked_df's baseline is deliberately frozen in
    # pb_shares_baseline (see above — required so the earlier multi-edit
    # bug fix holds), the freshly-recreated widget showed 0 shares,
    # discarding everything the user had entered (reported bug). Only
    # refresh the baseline from the authoritative pb_shares when there is
    # no live in-widget diff to protect — this is exactly the case where
    # the widget has just been (re)created with nothing pending, whether
    # that's a genuinely first-ever render or a post-navigation one. While
    # edits are actively accumulating within a single visit, edited_rows
    # stays non-empty, so this never fires and the original fix's
    # protection against a moving baseline still holds.
    if not st.session_state.get("pb_ranked_editor", {}).get("edited_rows"):
        for t in entries:
            st.session_state.pb_shares_baseline[t] = st.session_state.pb_shares.get(t, 0)

    prices = backend.get("prices", pd.Series(dtype=float))
    rows = []
    for t, e in entries.items():
        rows.append({
            "Ticker": t,
            "Sector": e.sector,
            "Score": _heat_label(t),
            # pb_shares_baseline, NOT pb_shares — see the session-state
            # init comment above for why feeding the constantly-updated
            # pb_shares back in here breaks the editor after one edit.
            "Shares": int(st.session_state.pb_shares_baseline.get(t, 0)),
        })
    ranked_df = pd.DataFrame(rows).set_index("Ticker")

    edited_df = st.data_editor(
        ranked_df,
        column_config={
            "Sector": st.column_config.TextColumn(disabled=True),
            "Score": st.column_config.TextColumn(disabled=True, help="Heat-colored by percentile within this list: 🟢 top third, 🟡 middle, 🔴 bottom"),
            "Shares": st.column_config.NumberColumn(min_value=0, step=1),
        },
        use_container_width=True,
        key="pb_ranked_editor",
    )

    # Sync edited share counts back into session_state (for % weight and
    # the metrics panel below) — pure arithmetic from here on, no backend
    # call is triggered by this. Deliberately NOT written back into
    # ranked_df/pb_shares_baseline above — see the session-state init
    # comment for why that would break the editor after the first edit.
    for t in edited_df.index:
        st.session_state.pb_shares[t] = int(edited_df.loc[t, "Shares"])

    values = pd.Series({
        t: st.session_state.pb_shares.get(t, 0) * float(prices.get(t, 0.0))
        for t in entries
    })
    total_value = values.sum()
    if total_value > 0:
        weight_pct = (values / total_value * 100).round(2)
        st.dataframe(
            pd.DataFrame({"% Weight": weight_pct}),
            use_container_width=True,
        )
    else:
        st.caption("Enter share counts above to see % weight (needs current prices too).")

# ── Correlation network ───────────────────────────────────────────────────
st.subheader("Correlation Network")

if not backend or "zoom" not in backend:
    st.caption("Add at least 2 tickers to see the correlation network.")
else:
    zoom = backend["zoom"]
    sector_options = (
        ["(sector overview)", "(all assets)"] + sorted(zoom.sector_network.sector_members.keys())
    )
    selected = st.selectbox("View", sector_options, key="pb_network_zoom")

    pos_col, hedge_col = st.columns(2)
    with pos_col:
        # Purely a rendering filter over the already-cached zoom object below
        # (keyed by ticker set — see _get_backend_data) — moving this slider
        # never re-fetches data, re-runs HRP's .corr(), or rebuilds the MST.
        positive_threshold = st.slider(
            "Correlation threshold (show correlated pairs at/above this)",
            min_value=0.0, max_value=1.0,
            value=_CORRELATION_NETWORK_CONFIG.positive_threshold, step=0.05,
            key="pb_corr_positive_threshold",
        )
    with hedge_col:
        hedge_threshold = st.slider(
            "Hedge threshold (show anti-correlated/hedge pairs at/below this)",
            min_value=-1.0, max_value=0.0,
            value=_CORRELATION_NETWORK_CONFIG.hedge_threshold, step=0.05,
            key="pb_corr_hedge_threshold",
        )

    edge_filter_config = CorrelationNetworkConfig(
        always_include_mst=_CORRELATION_NETWORK_CONFIG.always_include_mst,
        positive_threshold=positive_threshold,
        hedge_threshold=hedge_threshold,
    )

    if selected == "(sector overview)":
        mst_source = zoom.sector_network.mst
        distance_source = zoom.sector_network.distance_matrix
        title = "Sector overview (default zoom)"
        node_basis = "sector"
    elif selected == "(all assets)":
        mst_source = zoom.ticker_network.mst
        distance_source = zoom.ticker_network.distance_matrix
        title = "All assets (every ticker, ticker-level)"
        node_basis = "ticker"
    else:
        members = zoom.sector_network.sector_members.get(selected, [])
        mst_source = get_sector_subgraph(zoom.ticker_network, zoom.sector_network.sector_members, selected)
        distance_source = zoom.ticker_network.distance_matrix.loc[members, members]
        title = f"{selected} — ticker detail (zoomed in)"
        node_basis = "ticker"

    graph = filter_edges_by_threshold(distance_source, mst_source, edge_filter_config)

    if graph.number_of_nodes() == 0:
        st.info("No nodes to display for this view.")
    else:
        import networkx as nx

        style_config = NetworkStyleConfig()

        # Node color by rank tier — same composite_score percentile basis as
        # the Ranked List's heat emoji above, so a ticker's node color and
        # its ranked-list emoji always agree. In sector-overview mode there's
        # no per-sector composite_score, so sectors are ranked against each
        # other by their members' mean score instead.
        composite_scores = pd.Series({t: e.composite_score for t, e in backend["entries"].items()})
        if node_basis == "ticker":
            node_scores = composite_scores
        else:
            node_scores = composite_scores.groupby(pd.Series(backend["sector_map"])).mean()
        node_percentile = (
            node_scores.rank(pct=True) if len(node_scores) > 1 else pd.Series(1.0, index=node_scores.index)
        )

        pos = nx.spring_layout(graph, seed=42)

        edge_traces = []
        for u, v, data in graph.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            corr = correlation_from_distance(data["weight"])
            color = edge_color_for_correlation(corr, style_config)
            # A midpoint (not just the two endpoints) gives Plotly a closer
            # point to snap hover to along the whole length of the line, not
            # just right at a node — otherwise hovering mid-edge often misses.
            xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            hover_text = f"{u} – {v}<br>Correlation: {corr:.3f}<br>Distance: {data['weight']:.3f}"
            edge_traces.append(go.Scatter(
                x=[x0, xm, x1], y=[y0, ym, y1], mode="lines",
                line=dict(width=3, color=color), opacity=style_config.edge_opacity,
                hoverinfo="text", text=[hover_text, hover_text, hover_text],
                showlegend=False,
            ))

        node_colors = [
            node_color_for_percentile(node_percentile.get(n, 1.0), style_config)
            for n in graph.nodes()
        ]
        node_x = [pos[n][0] for n in graph.nodes()]
        node_y = [pos[n][1] for n in graph.nodes()]
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers+text", text=list(graph.nodes()),
            textposition="top center", marker=dict(size=20, color=node_colors),
            hoverinfo="text", showlegend=False,
        )
        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title=title, showlegend=False, margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
        )
        st.plotly_chart(fig, use_container_width=True)

        legend_col1, legend_col2 = st.columns(2)
        with legend_col1:
            st.markdown(
                "**Node color — rank tier**<br>"
                f"<span style='color:{style_config.node_color_top}'>●</span> Top third&nbsp;&nbsp;"
                f"<span style='color:{style_config.node_color_mid}'>●</span> Middle third&nbsp;&nbsp;"
                f"<span style='color:{style_config.node_color_bottom}'>●</span> Bottom third",
                unsafe_allow_html=True,
            )
        with legend_col2:
            gradient_css = (
                f"background: linear-gradient(to right, {style_config.edge_color_negative}, "
                f"{style_config.edge_color_neutral}, {style_config.edge_color_positive}); "
                "height: 14px; border-radius: 3px; margin-top: 2px;"
            )
            st.markdown(
                "**Edge color — correlation strength (gradient)**<br>"
                f"<div style='{gradient_css}'></div>"
                "<div style='display:flex; justify-content:space-between; "
                "font-size:0.75em; color:gray;'>"
                "<span>-1.0 (strong hedge)</span><span>0.0</span>"
                "<span>+1.0 (strong positive)</span></div>",
                unsafe_allow_html=True,
            )
        st.caption(
            "The network is a minimum spanning tree built on the Mantegna "
            "distance transform of ticker-level correlation — shorter edges "
            "connect more correlated tickers, and edge COLOR (not opacity) "
            "shows correlation strength as a gradient toward the hedge or "
            "positive end. Hover an edge for its exact correlation and "
            "distance. MST edges are always shown regardless of the sliders "
            "above; the correlation threshold adds strongly-correlated "
            "pairs, the hedge threshold adds strongly anti-correlated "
            "(hedge-like) pairs — pick \"(all assets)\" above to see every "
            "ticker in one view instead of one sector at a time."
        )

    if zoom.ticker_network.excluded_tickers:
        st.caption(f"Excluded from the network (incomplete cached correlation data): {zoom.ticker_network.excluded_tickers}")

# ── Metrics panel ──────────────────────────────────────────────────────────
st.subheader("Metrics")

if not backend or not backend.get("entries"):
    st.caption("Add tickers above to see portfolio metrics.")
else:
    prices = backend.get("prices", pd.Series(dtype=float))
    values = pd.Series({
        t: st.session_state.pb_shares.get(t, 0) * float(prices.get(t, 0.0))
        for t in backend["entries"]
    })
    total_value = values.sum()

    if total_value <= 0:
        st.caption("Enter share counts in the ranked list above to see portfolio metrics.")
    else:
        weights = values / total_value
        diversification_config = DiversificationConfig()
        sector_exposure = compute_sector_exposure(weights, backend["sector_map"], diversification_config)

        st.markdown("**Sector Exposure**")
        sector_weight_pct = (
            pd.Series(sector_exposure.sector_weights).sort_values(ascending=False) * 100
        ).rename("Weight (%)")
        st.bar_chart(sector_weight_pct)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sector HHI", f"{sector_exposure.hhi:.3f}")
            if sector_exposure.is_concentrated:
                st.warning(
                    f"Sector concentration warning — HHI {sector_exposure.hhi:.3f} is "
                    f"at/above the {diversification_config.hhi_warning_threshold:.0%} threshold."
                )
        with col2:
            if "zoom" in backend:
                rating = compute_diversification_rating(sector_exposure, backend["zoom"].ticker_network.mst)
                st.metric("Diversification Rating", f"{rating:.1f} / 100")
            else:
                st.caption("Diversification rating needs at least 2 tickers.")

        st.markdown("---")

        dcc_result = backend.get("dcc_result")
        hist_prices = backend.get("hist_prices")
        if dcc_result is None or hist_prices is None:
            st.caption(
                "Sharpe estimate needs tickers spanning at least 2 "
                "sectors and available historical price data."
            )
        else:
            try:
                returns = hist_prices.pct_change().dropna(how="all")
                aligned_weights = weights.reindex(returns.columns).fillna(0.0)
                portfolio_returns = (returns[aligned_weights.index] * aligned_weights).sum(axis=1)

                sector_weights = weights.groupby(pd.Series(backend["sector_map"])).sum()

                sharpe_config = SharpeConfig(risk_free_rate=_MANUAL_RISK_FREE_RATE)
                realized = compute_realized_return(portfolio_returns, sharpe_config.lookback_days)
                # Same lookback_days window as the realized-return leg above —
                # see metrics.py's module docstring for why this replaced the
                # old current-day-only volatility (it silently distorted
                # Sharpe whenever "right now" wasn't representative of the
                # trailing year).
                vol = compute_dcc_garch_volatility_trailing(
                    dcc_result, sector_weights, lookback_days=sharpe_config.lookback_days,
                )
                sharpe = compute_sharpe(realized, vol, sharpe_config)

                # Derived from sharpe_config.lookback_days, not a hardcoded
                # "12mo"/"3yr" string — stays correct if that default ever
                # changes again without a matching UI-label edit.
                lookback_years = sharpe_config.lookback_days / 252
                if lookback_years == int(lookback_years):
                    period_label = f"{int(lookback_years)}yr avg"
                else:
                    period_label = f"{sharpe_config.lookback_days}d avg"

                m1, m2, m3 = st.columns(3)
                m1.metric(f"Realized Return ({period_label})", f"{realized * 100:.2f}%")
                m2.metric(f"Volatility ({period_label}, annualized)", f"{vol * 100:.2f}%")
                m3.metric("Sharpe", f"{sharpe:.2f}")
                render_sharpe_methodology_disclosure()
            except ValueError as exc:
                st.warning(f"Sharpe estimate unavailable: {exc}")

st.caption(f"(backend computed {st.session_state.pb_backend_call_count} time(s) this session — "
           "unaffected by share-count edits above)")
