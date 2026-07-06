"""
Stock Valuation Page - 3-stage institutional valuation pipeline.

Stages
------
1. Greenblatt Magic Formula Screen  — rank on EBIT/EV and ROIC, filter junk
2. Multi-Factor Composite Score     — graduated 0-100 score across 5 factor groups
3. Reverse DCF Validation           — implied vs. historical FCF growth check
"""

import math
import sys
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.valuation.stock_valuer import (
    magic_formula_screen,
    multi_factor_score,
    reverse_dcf,
    _rating_label,
    SECTOR_EV_EBITDA_MEDIANS,
    EXCLUDED_SECTORS,
    MIN_MARKET_CAP_USD,
)

st.set_page_config(page_title="Stock Valuation", page_icon=None, layout="wide")

st.title("Stock Valuation")
st.caption(
    "3-stage institutional pipeline: Greenblatt Magic Formula screen "
    "→ Multi-factor composite scoring → Reverse DCF validation"
)

st.markdown("---")

# ── Ticker input ──────────────────────────────────────────────────────────────

st.subheader("1. Select Tickers")

_session_tickers: list[str] = st.session_state.get("tickers", [])

col_l, col_r = st.columns([2, 1])

with col_l:
    _default_str = ", ".join(_session_tickers) if _session_tickers else ""
    _ticker_input = st.text_area(
        "Tickers to analyse (comma- or space-separated)",
        value=_default_str,
        height=80,
        help=(
            "Pre-populated from Portfolio Input when available. "
            "Add or remove tickers freely."
        ),
    )

with col_r:
    if _session_tickers:
        st.info(
            f"Loaded **{len(_session_tickers)}** ticker(s) from your portfolio.\n\n"
            "Edit the box on the left to add or remove tickers."
        )
    else:
        st.info(
            "No portfolio loaded. Enter tickers manually in the text box."
        )
    st.caption(
        "Filters applied automatically:\n"
        f"- Market cap >= ${MIN_MARKET_CAP_USD/1e6:.0f} M\n"
        f"- Excluded sectors: {', '.join(sorted(EXCLUDED_SECTORS))}\n"
        "- Positive EBIT required\n"
        "- Total Debt <= 3x EBITDA"
    )

# Parse tickers
_raw = _ticker_input.replace(",", " ").split()
tickers = [t.strip().upper() for t in _raw if t.strip()]

if not tickers:
    st.warning("Enter at least one ticker above to run the analysis.")
    st.stop()

st.markdown("---")

# ── Settings ──────────────────────────────────────────────────────────────────

st.subheader("2. Settings")

_col1, _col2, _col3 = st.columns(3)

with _col1:
    wacc = st.slider(
        "WACC (Discount Rate)",
        min_value=0.05,
        max_value=0.20,
        value=0.10,
        step=0.005,
        format="%.1f%%",
        help="Weighted Average Cost of Capital used in the Reverse DCF model.",
    )
    # st.slider with format="%.1f%%" returns a fraction, so the value is already 0.10
    # but it displays as "10.0%". Actually no — it returns the raw float (0.05 to 0.20).

with _col2:
    dcf_score_threshold = st.slider(
        "Min score for DCF (Stage 3)",
        min_value=40,
        max_value=90,
        value=65,
        step=5,
        help=(
            "Only stocks with a total score >= this threshold proceed "
            "to the Reverse DCF stage."
        ),
    )

with _col3:
    top_pct = st.slider(
        "Stage 1 survivor % (top N %)",
        min_value=10,
        max_value=100,
        value=30,
        step=10,
        help=(
            "After the Magic Formula screen, keep only the top X% "
            "by combined rank (minimum 3 stocks always kept)."
        ),
    )

st.markdown("---")

# ── Run button ────────────────────────────────────────────────────────────────

st.subheader("3. Run Analysis")

_run = st.button("Run 3-Stage Valuation", type="primary", width="stretch")

if not _run:
    st.info(
        f"Ready to analyse **{len(tickers)}** ticker(s): "
        + ", ".join(f"`{t}`" for t in tickers)
    )
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

all_warnings: list[str] = []

# ── Stage 1 ──────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Stage 1 — Greenblatt Magic Formula Screen")

with st.status(
    f"Screening {len(tickers)} ticker(s) on EBIT/EV and ROIC...",
    expanded=True,
) as _status1:
    st.write("Fetching fundamental data from Yahoo Finance...")
    screen_df = magic_formula_screen(tickers)
    if screen_df.empty:
        _status1.update(label="Stage 1 complete — no survivors", state="error")
    else:
        _status1.update(
            label=f"Stage 1 complete — {len(screen_df)} ticker(s) passed",
            state="complete",
        )

if screen_df.empty:
    st.error(
        "No tickers passed the Magic Formula screen. "
        "Check that your tickers have sufficient market cap, positive EBIT, "
        "and are not in excluded sectors (Financials / Utilities)."
    )
    st.stop()

# Display screen results
_n_survivors = len(screen_df)
_n_keep = max(3, math.ceil(_n_survivors * top_pct / 100))
top_df = screen_df.head(_n_keep).copy()
top_tickers = top_df["ticker"].tolist()

col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown(
        f"**{_n_survivors}** ticker(s) passed.  "
        f"Keeping top **{top_pct}%** = **{_n_keep}** for Stage 2."
    )

    _disp = screen_df.copy()
    _disp["ebit_ev"] = (_disp["ebit_ev"] * 100).round(2)
    _disp["roic"] = (_disp["roic"] * 100).round(2)
    _disp = _disp.rename(
        columns={"ebit_ev": "EBIT/EV (%)", "roic": "ROIC (%)"}
    )

    def _style_rank(df_inner):
        styles = pd.DataFrame("", index=df_inner.index, columns=df_inner.columns)
        if "combined_rank" in df_inner.columns:
            # Highlight the kept rows
            kept_idx = df_inner.index[:_n_keep]
            for idx in kept_idx:
                styles.loc[idx, "combined_rank"] = "background-color: #d4edda; font-weight: bold"
        return styles

    st.dataframe(
        _disp.style.apply(_style_rank, axis=None),
        width="stretch",
        height=min(400, 36 + 35 * len(_disp)),
    )

with col_right:
    # Bar chart: EBIT/EV and ROIC side by side
    _kept = screen_df["ticker"].isin(top_tickers)
    _fig1 = go.Figure()
    _fig1.add_trace(go.Bar(
        x=screen_df["ticker"],
        y=(screen_df["ebit_ev"] * 100).round(2),
        name="EBIT/EV (%)",
        marker_color=["#2196F3" if k else "#90CAF9" for k in _kept],
    ))
    _fig1.add_trace(go.Bar(
        x=screen_df["ticker"],
        y=(screen_df["roic"] * 100).round(2),
        name="ROIC (%)",
        marker_color=["#4CAF50" if k else "#A5D6A7" for k in _kept],
    ))
    _fig1.update_layout(
        title="Magic Formula Metrics (darker = kept for Stage 2)",
        barmode="group",
        xaxis_title="Ticker",
        yaxis_title="(%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=350,
    )
    st.plotly_chart(_fig1, width="stretch")

st.markdown("---")

# ── Stage 2 ──────────────────────────────────────────────────────────────────

st.subheader("Stage 2 — Multi-Factor Composite Scoring")

# Pre-compute sector EV/EBITDA medians
_sector_medians: dict[str, float] = {}
for _, _row in top_df.iterrows():
    _sec = _row.get("sector", "Unknown")
    if _sec not in _sector_medians:
        _sector_medians[_sec] = SECTOR_EV_EBITDA_MEDIANS.get(
            _sec, SECTOR_EV_EBITDA_MEDIANS["default"]
        )

_score_rows: list[dict] = []
_progress_bar = st.progress(0, text="Starting multi-factor scoring...")

with st.status(
    f"Scoring {len(top_tickers)} ticker(s) across 5 factor groups...",
    expanded=True,
) as _status2:
    for _i, _ticker in enumerate(top_tickers, 1):
        st.write(f"Scoring **{_ticker}** ({_i}/{len(top_tickers)})...")
        _progress_bar.progress(
            _i / len(top_tickers),
            text=f"Scoring {_ticker} ({_i}/{len(top_tickers)})",
        )
        _sec_arr = top_df.loc[top_df["ticker"] == _ticker, "sector"].values
        _peer_med = _sector_medians.get(_sec_arr[0] if len(_sec_arr) else "Unknown")
        _s = multi_factor_score(_ticker, sector_ev_ebitda_median=_peer_med)
        all_warnings.extend(_s.get("warnings", []))
        _score_rows.append({
            "ticker": _s["ticker"],
            "sector": _s["sector"],
            "quality_score": _s["quality_score"],
            "value_score": _s["value_score"],
            "momentum_score": _s["momentum_score"],
            "growth_score": _s["growth_score"],
            "health_score": _s["health_score"],
            "total_score": _s["total_score"],
            "data_quality_score": _s["data_quality_score"],
            "score_breakdown": _s["score_breakdown"],
        })
    _status2.update(
        label=f"Stage 2 complete — scored {len(_score_rows)} ticker(s)",
        state="complete",
    )

_progress_bar.empty()

_scores_df = pd.DataFrame(_score_rows)
_merged = top_df.drop(columns=["sector"], errors="ignore").merge(
    _scores_df, on="ticker", how="inner"
)

# Factor score visualization
st.markdown("**Factor Score Breakdown (per ticker)**")

_factor_cols = ["quality_score", "value_score", "momentum_score", "growth_score", "health_score"]
_factor_labels = ["Quality\n(max 30)", "Value\n(max 25)", "Momentum\n(max 20)", "Growth\n(max 15)", "Health\n(max 10)"]
_factor_maxes = [30, 25, 20, 15, 10]

# Grouped bar chart — one group per ticker, bars = factor groups
_fig2 = go.Figure()
_colors = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2", "#D32F2F"]
for _j, (_col, _lbl, _mx, _color) in enumerate(
    zip(_factor_cols, _factor_labels, _factor_maxes, _colors)
):
    _fig2.add_trace(go.Bar(
        name=_lbl.replace("\n", " "),
        x=_scores_df["ticker"],
        y=_scores_df[_col],
        marker_color=_color,
        text=_scores_df[_col].round(1),
        textposition="inside",
        customdata=[[_mx]] * len(_scores_df),
        hovertemplate="%{x}: %{y:.1f} / %{customdata[0]}<extra></extra>",
    ))

_fig2.update_layout(
    barmode="stack",
    title="Stacked Factor Scores (max 100)",
    xaxis_title="Ticker",
    yaxis_title="Score",
    yaxis=dict(range=[0, 105]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=400,
)
# Add a horizontal reference line at 65 (Buy threshold) and 80 (Strong Buy)
_fig2.add_hline(y=80, line_dash="dot", line_color="green",
                annotation_text="Strong Buy (80)", annotation_position="top left")
_fig2.add_hline(y=65, line_dash="dot", line_color="blue",
                annotation_text="Buy (65)", annotation_position="top left")
_fig2.add_hline(y=50, line_dash="dot", line_color="orange",
                annotation_text="Hold (50)", annotation_position="top left")

st.plotly_chart(_fig2, width="stretch")

# Radar chart — show each ticker as a separate trace
if len(_scores_df) > 0:
    st.markdown("**Radar Chart — Factor Profile per Ticker**")
    _radar_cols = ["quality_score", "value_score", "momentum_score", "growth_score", "health_score"]
    _radar_max = [30, 25, 20, 15, 10]
    _radar_labels = ["Quality/30", "Value/25", "Momentum/20", "Growth/15", "Health/10"]

    _fig_radar = go.Figure()
    _palette = px.colors.qualitative.Set2
    for _idx, _srow in _scores_df.iterrows():
        _vals_norm = [
            _srow[c] / _mx * 100
            for c, _mx in zip(_radar_cols, _radar_max)
        ]
        _vals_norm.append(_vals_norm[0])  # close the polygon
        _fig_radar.add_trace(go.Scatterpolar(
            r=_vals_norm + [_vals_norm[0]],
            theta=_radar_labels + [_radar_labels[0]],
            fill="toself",
            name=_srow["ticker"],
            line_color=_palette[_idx % len(_palette)],
            opacity=0.6,
        ))
    _fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Factor Profiles (normalised to max per factor)",
        height=450,
    )
    st.plotly_chart(_fig_radar, width="stretch")

st.markdown("---")

# ── Stage 3 ──────────────────────────────────────────────────────────────────

st.subheader("Stage 3 — Reverse DCF Validation")

_dcf_tickers = _merged.loc[_merged["total_score"] >= dcf_score_threshold, "ticker"].tolist()

if not _dcf_tickers:
    st.info(
        f"No tickers reached the score threshold ({dcf_score_threshold}) for DCF analysis. "
        "Lower the threshold in Settings if you want to run DCF on all tickers."
    )
    _dcf_map: dict = {}
else:
    st.markdown(
        f"Running Reverse DCF for **{len(_dcf_tickers)}** ticker(s) "
        f"with score >= {dcf_score_threshold}: "
        + ", ".join(f"`{t}`" for t in _dcf_tickers)
    )

    _dcf_map = {}
    _dcf_progress = st.progress(0, text="Running Reverse DCF...")

    with st.status(
        f"Solving implied growth rates for {len(_dcf_tickers)} ticker(s)...",
        expanded=True,
    ) as _status3:
        for _i, _ticker in enumerate(_dcf_tickers, 1):
            st.write(f"DCF: **{_ticker}** ({_i}/{len(_dcf_tickers)}) at WACC={wacc:.1%}...")
            _dcf_progress.progress(
                _i / len(_dcf_tickers),
                text=f"DCF {_ticker} ({_i}/{len(_dcf_tickers)})",
            )
            _dcf = reverse_dcf(_ticker, wacc=wacc)
            _dcf_map[_ticker] = _dcf
            if _w := _dcf.get("warnings"):
                all_warnings.append(f"{_ticker}/DCF: {_w}")
        _status3.update(
            label=f"Stage 3 complete — {len(_dcf_map)} DCF(s) run",
            state="complete",
        )

    _dcf_progress.empty()

# Merge DCF results
_merged["implied_growth_rate"] = _merged["ticker"].map(
    lambda t: _dcf_map[t]["implied_growth_rate"] if t in _dcf_map else None
)
_merged["historical_fcf_cagr"] = _merged["ticker"].map(
    lambda t: _dcf_map[t].get("historical_fcf_cagr") if t in _dcf_map else None
)
_merged["growth_premium"] = _merged["ticker"].map(
    lambda t: _dcf_map[t]["growth_premium"] if t in _dcf_map else None
)
_merged["verdict"] = _merged["ticker"].map(
    lambda t: _dcf_map[t]["verdict"] if t in _dcf_map else None
)
_merged["rating"] = _merged["total_score"].map(_rating_label)

# Reorder columns for display
_ordered_cols = [
    "ticker", "combined_rank", "total_score",
    "quality_score", "value_score", "momentum_score",
    "growth_score", "health_score",
    "implied_growth_rate", "historical_fcf_cagr", "growth_premium",
    "verdict", "sector", "data_quality_score", "rating",
]
_final_cols = [c for c in _ordered_cols if c in _merged.columns]
_merged = (
    _merged[_final_cols]
    .sort_values("total_score", ascending=False)
    .reset_index(drop=True)
)

st.markdown("---")

# ── Summary results ───────────────────────────────────────────────────────────

st.subheader("Results Summary")

# Metrics row
_n_buy = (_merged["rating"].isin(["Strong Buy", "Buy"])).sum()
_n_hold = (_merged["rating"] == "Hold").sum()
_n_sell = (_merged["rating"].isin(["Underweight", "Avoid"])).sum()
_top_pick = _merged.iloc[0]["ticker"] if len(_merged) > 0 else "—"
_top_score = _merged.iloc[0]["total_score"] if len(_merged) > 0 else 0.0

_mc1, _mc2, _mc3, _mc4, _mc5 = st.columns(5)
_mc1.metric("Stocks Analysed", len(_merged))
_mc2.metric("Buy / Strong Buy", _n_buy)
_mc3.metric("Hold", _n_hold)
_mc4.metric("Underweight / Avoid", _n_sell)
_mc5.metric("Top Pick", f"{_top_pick} ({_top_score:.0f})")

st.markdown("---")

# Styled summary table
st.markdown("**Full Results Table**")

_RATING_BG = {
    "Strong Buy":  "background-color: #1B5E20; color: white",
    "Buy":         "background-color: #388E3C; color: white",
    "Hold":        "background-color: #F57F17; color: white",
    "Underweight": "background-color: #E65100; color: white",
    "Avoid":       "background-color: #B71C1C; color: white",
}
_VERDICT_BG = {
    "Reasonable":        "background-color: #C8E6C9",
    "Stretched":         "background-color: #FFF9C4",
    "Extreme":           "background-color: #FFCDD2",
    "Insufficient Data": "color: #888",
}


def _colour_row(row):
    styles = [""] * len(row)
    cols = list(row.index)
    if "rating" in cols:
        _ri = cols.index("rating")
        styles[_ri] = _RATING_BG.get(row["rating"], "")
    if "verdict" in cols:
        _vi = cols.index("verdict")
        styles[_vi] = _VERDICT_BG.get(row.get("verdict", ""), "")
    return styles


# Format display columns
_display = _merged.copy()
for _pct_col in ["implied_growth_rate", "historical_fcf_cagr", "growth_premium"]:
    if _pct_col in _display.columns:
        _display[_pct_col] = _display[_pct_col].apply(
            lambda v: f"{v*100:.1f}%" if v is not None and not (isinstance(v, float) and math.isnan(v)) else "N/A"
        )
if "data_quality_score" in _display.columns:
    _display["data_quality_score"] = _display["data_quality_score"].apply(
        lambda v: f"{v:.0f}%" if v is not None else "N/A"
    )
for _score_col in ["total_score", "quality_score", "value_score",
                   "momentum_score", "growth_score", "health_score"]:
    if _score_col in _display.columns:
        _display[_score_col] = _display[_score_col].round(1)

# Rename for readability
_display = _display.rename(columns={
    "combined_rank":      "Rank",
    "total_score":        "Score/100",
    "quality_score":      "Qual/30",
    "value_score":        "Val/25",
    "momentum_score":     "Mom/20",
    "growth_score":       "Grw/15",
    "health_score":       "Hlth/10",
    "implied_growth_rate": "Impl.Grw%",
    "historical_fcf_cagr": "Hist.FCF CAGR",
    "growth_premium":     "Growth Premium",
    "data_quality_score": "Data Quality",
})

st.dataframe(
    _display.style.apply(_colour_row, axis=1),
    width="stretch",
    height=min(600, 50 + 35 * len(_display)),
)

# Legend
st.markdown(
    "**Rating legend:**"
    "  <span style='background:#1B5E20;color:white;padding:2px 6px;border-radius:3px'>Strong Buy ≥80</span>"
    "  <span style='background:#388E3C;color:white;padding:2px 6px;border-radius:3px'>Buy ≥65</span>"
    "  <span style='background:#F57F17;color:white;padding:2px 6px;border-radius:3px'>Hold ≥50</span>"
    "  <span style='background:#E65100;color:white;padding:2px 6px;border-radius:3px'>Underweight ≥35</span>"
    "  <span style='background:#B71C1C;color:white;padding:2px 6px;border-radius:3px'>Avoid <35</span>",
    unsafe_allow_html=True,
)

st.markdown(
    "**Verdict legend:**"
    "  <span style='background:#C8E6C9;padding:2px 6px;border-radius:3px'>Reasonable</span>"
    "  <span style='background:#FFF9C4;padding:2px 6px;border-radius:3px'>Stretched</span>"
    "  <span style='background:#FFCDD2;padding:2px 6px;border-radius:3px'>Extreme</span>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ── DCF Details ───────────────────────────────────────────────────────────────

if _dcf_map:
    st.subheader("Reverse DCF Details")

    # Implied growth vs historical FCF CAGR bar chart
    _dcf_rows = []
    for _ticker, _dcf in _dcf_map.items():
        _ig = _dcf.get("implied_growth_rate")
        _hc = _dcf.get("historical_fcf_cagr")
        if _ig is not None:
            _dcf_rows.append({
                "ticker": _ticker,
                "Implied Growth %": round(_ig * 100, 2),
                "Historical FCF CAGR %": round(_hc * 100, 2) if _hc is not None else None,
                "verdict": _dcf.get("verdict", "N/A"),
            })

    if _dcf_rows:
        _dcf_chart_df = pd.DataFrame(_dcf_rows)
        _fig_dcf = go.Figure()
        _fig_dcf.add_trace(go.Bar(
            x=_dcf_chart_df["ticker"],
            y=_dcf_chart_df["Implied Growth %"],
            name="Implied Growth %",
            marker_color=[
                {"Reasonable": "#4CAF50", "Stretched": "#FFC107", "Extreme": "#F44336"}.get(
                    row["verdict"], "#9E9E9E"
                )
                for _, row in _dcf_chart_df.iterrows()
            ],
            text=_dcf_chart_df["Implied Growth %"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
        ))
        # Historical CAGR as scatter overlay
        _hist_valid = _dcf_chart_df["Historical FCF CAGR %"].notna()
        if _hist_valid.any():
            _fig_dcf.add_trace(go.Scatter(
                x=_dcf_chart_df.loc[_hist_valid, "ticker"],
                y=_dcf_chart_df.loc[_hist_valid, "Historical FCF CAGR %"],
                name="Historical FCF CAGR %",
                mode="markers",
                marker=dict(size=12, color="black", symbol="diamond"),
            ))
        _fig_dcf.update_layout(
            title="Implied Growth Rate vs Historical FCF CAGR (bar = Implied, diamond = Historical)",
            xaxis_title="Ticker",
            yaxis_title="Growth Rate (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400,
        )
        st.plotly_chart(_fig_dcf, width="stretch")

    # DCF table
    _dcf_display_rows = []
    for _ticker, _dcf in _dcf_map.items():
        def _fmt(v):
            if v is None:
                return "N/A"
            if isinstance(v, float) and math.isnan(v):
                return "N/A"
            return f"{v*100:.1f}%"

        _dcf_display_rows.append({
            "Ticker": _ticker,
            "Current FCF ($M)": (
                f"${_dcf['current_fcf']/1e6:,.0f}" if _dcf.get("current_fcf") else "N/A"
            ),
            "Market Cap ($M)": (
                f"${_dcf['market_cap']/1e6:,.0f}" if _dcf.get("market_cap") else "N/A"
            ),
            "Implied Growth": _fmt(_dcf.get("implied_growth_rate")),
            "Hist. FCF CAGR": _fmt(_dcf.get("historical_fcf_cagr")),
            "Growth Premium": _fmt(_dcf.get("growth_premium")),
            "Verdict": _dcf.get("verdict", "N/A"),
            "Notes": _dcf.get("warnings") or "",
        })

    _dcf_table = pd.DataFrame(_dcf_display_rows)

    def _style_verdict_col(val):
        return _VERDICT_BG.get(val, "")

    st.dataframe(
        _dcf_table.style.map(_style_verdict_col, subset=["Verdict"]),
        width="stretch",
    )

    st.markdown("---")

# ── Per-ticker score breakdown ────────────────────────────────────────────────

with st.expander("Score Breakdown per Sub-factor (detail)", expanded=False):
    _bd_rows = [r for r in _score_rows if r.get("score_breakdown")]
    if not _bd_rows:
        st.info("No breakdown data available.")
    else:
        _SUBF_LABELS = {
            "roic_pts":          "ROIC (Quality, /12)",
            "gm_trend_pts":      "Gross Margin Trend (Quality, /10)",
            "fcf_ni_pts":        "FCF/NI ratio (Quality, /8)",
            "ev_ebit_pts":       "EV/EBIT (Value, /10)",
            "fcf_yield_pts":     "FCF Yield (Value, /8)",
            "ev_ebitda_pts":     "EV/EBITDA vs peers (Value, /7)",
            "sma_200_pts":       "Price vs 200-SMA (Momentum, /8)",
            "momentum_12_1_pts": "12-1 Month Return (Momentum, /12)",
            "rev_cagr_pts":      "Revenue CAGR (Growth, /6)",
            "eps_leverage_pts":  "EPS Leverage (Growth, /5)",
            "sloan_pts":         "Sloan Accruals (Growth, /4)",
            "int_cov_pts":       "Interest Coverage (Health, /5)",
            "debt_ebitda_pts":   "Debt/EBITDA (Health, /5)",
        }

        _all_tickers_bd = [r["ticker"] for r in _bd_rows]
        _all_subf = list(_SUBF_LABELS.keys())

        _bd_data = {
            "Sub-factor": [_SUBF_LABELS.get(k, k) for k in _all_subf],
        }
        for _r in _bd_rows:
            _bd = _r["score_breakdown"]
            _bd_data[_r["ticker"]] = [
                _bd.get(k) for k in _all_subf
            ]

        _bd_df = pd.DataFrame(_bd_data).set_index("Sub-factor")

        def _highlight_none(val):
            if val is None:
                return "color: #aaa; font-style: italic"
            if val == 0:
                return "background-color: #FFCDD2"
            return ""

        st.dataframe(
            _bd_df.style.map(_highlight_none),
            width="stretch",
        )
        st.caption(
            "Gray/italic = data unavailable (score not counted). "
            "Red = scored 0 (data present but metric failed threshold). "
            "Blank sub-factors count toward overall data quality score."
        )

# ── Warnings ──────────────────────────────────────────────────────────────────

if all_warnings:
    with st.expander(f"Warnings & skipped metrics ({len(all_warnings)} total)", expanded=False):
        for _w in all_warnings:
            st.markdown(f"- {_w}")

# ── Store in session state ────────────────────────────────────────────────────

st.session_state["valuation_result"] = {
    "screen_df": screen_df,
    "scores_df": _scores_df,
    "merged": _merged,
    "dcf_map": _dcf_map,
    "all_warnings": all_warnings,
}
