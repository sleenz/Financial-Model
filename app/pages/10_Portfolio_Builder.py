"""
Portfolio Builder — assembles Phases 1-4 of src/portfolio_builder into one
Streamlit page: ticker chips, ranked list (heat-colored score + editable
share count + computed % weight), correlation network (sector overview
with drill-into-sector detail), and a metrics panel (HHI/diversification
+ mixed-period Sharpe estimate).

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
    SharpeConfig,
    compute_dcc_garch_volatility_current,
    compute_diversification_rating,
    compute_realized_return,
    compute_sector_exposure,
    compute_sharpe,
    render_mixed_period_disclosure,
)
from src.portfolio_builder.network import build_semantic_zoom_network, get_sector_subgraph
from src.risk.dcc_garch import DCCGARCHConfig, DCCGARCHModel

st.set_page_config(page_title="Portfolio Builder", layout="wide")
st.title("Portfolio Builder")

# ── Manual risk-free constant (SharpeConfig.risk_free_rate must be set
# explicitly — see metrics.py docstring: FRED is broken upstream, and
# compute_sharpe() refuses to silently default this to 0.0). ──────────────
_MANUAL_RISK_FREE_RATE = 0.045  # update periodically until FRED is fixed


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
            live_returns = hist_prices.pct_change().dropna(how="all")
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
    sector_options = ["(sector overview)"] + sorted(zoom.sector_network.sector_members.keys())
    selected = st.selectbox("Zoom into a sector", sector_options, key="pb_network_zoom")

    if selected == "(sector overview)":
        graph = zoom.sector_network.mst
        title = "Sector overview (default zoom)"
    else:
        graph = get_sector_subgraph(zoom.ticker_network, zoom.sector_network.sector_members, selected)
        title = f"{selected} — ticker detail (zoomed in)"

    if graph.number_of_nodes() == 0:
        st.info("No nodes to display for this view.")
    else:
        import networkx as nx

        pos = nx.spring_layout(graph, seed=42)
        edge_x, edge_y = [], []
        for u, v in graph.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#999"), hoverinfo="none")
        node_x = [pos[n][0] for n in graph.nodes()]
        node_y = [pos[n][1] for n in graph.nodes()]
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers+text", text=list(graph.nodes()),
            textposition="top center", marker=dict(size=20, color="#4C78A8"), hoverinfo="text",
        )
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=title, showlegend=False, margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
        )
        st.plotly_chart(fig, use_container_width=True)

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
        sector_exposure = compute_sector_exposure(weights, backend["sector_map"])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sector HHI", f"{sector_exposure.hhi:.3f}")
            if sector_exposure.is_concentrated:
                st.warning(f"Sector concentration warning — HHI {sector_exposure.hhi:.3f} is at/above the 0.40 threshold.")
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
                "Mixed-period Sharpe estimate needs tickers spanning at least 2 "
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
                vol = compute_dcc_garch_volatility_current(dcc_result, sector_weights)
                sharpe = compute_sharpe(realized, vol, sharpe_config)

                m1, m2, m3 = st.columns(3)
                m1.metric("Realized Return (12mo)", f"{realized * 100:.2f}%")
                m2.metric("Volatility (current, annualized)", f"{vol * 100:.2f}%")
                m3.metric("Sharpe", f"{sharpe:.2f}")
                render_mixed_period_disclosure()
            except ValueError as exc:
                st.warning(f"Mixed-period Sharpe estimate unavailable: {exc}")

st.caption(f"(backend computed {st.session_state.pb_backend_call_count} time(s) this session — "
           "unaffected by share-count edits above)")
