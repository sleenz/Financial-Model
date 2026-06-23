"""
Factor Analysis Page

Analyze portfolio factor exposures and return attribution.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.factors.fama_french import FamaFrenchAnalyzer
from src.factors.attribution import SectorAttribution, BrinsonAttribution
from src.factors.style import StyleFactorAnalyzer
from src.factors.decomposition import FactorRiskDecomposition

st.set_page_config(page_title="Factor Analysis", page_icon=None, layout="wide")

st.title(" Factor Analysis")
st.markdown("Analyze what's driving your portfolio returns")

# Check for required session state
if 'portfolio_data' not in st.session_state or st.session_state.portfolio_data is None:
    st.warning("Please load portfolio data first on the Portfolio Input page.")
    st.stop()

data = st.session_state.portfolio_data
returns = data['returns']
prices = data.get('prices')

# Get weights based on what's available
if 'optimization_result' in st.session_state and st.session_state.optimization_result:
    weights = st.session_state.optimization_result['weights']
    st.info("Analyzing **optimized portfolio** factor exposures")
elif 'current_portfolio_weights' in st.session_state and st.session_state.current_portfolio_weights is not None:
    weights = st.session_state.current_portfolio_weights
    st.info("Analyzing **your current holdings** factor exposures")
else:
    weights = pd.Series(1/len(returns.columns), index=returns.columns)
    st.warning("Using equal weights. Enter holdings or run optimization for accurate analysis.")

# Tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs([
    "Fama-French Analysis",
    "Sector Attribution",
    "Style Factors",
    "Risk Decomposition"
])

# ============================================================================
# Tab 1: Fama-French Analysis
# ============================================================================
with tab1:
    st.header("Fama-French Factor Model")
    st.markdown("""
    Analyze factor exposures using the Fama-French model:
    - **Mkt-RF**: Market risk premium
    - **SMB**: Small minus Big (size)
    - **HML**: High minus Low (value)
    - **RMW**: Robust minus Weak (profitability)
    - **CMA**: Conservative minus Aggressive (investment)
    """)

    col1, col2 = st.columns(2)

    with col1:
        model_type = st.radio("Select Model", ["3-Factor", "5-Factor"], horizontal=True)
        model = '3' if model_type == "3-Factor" else '5'

    with col2:
        show_individual = st.checkbox("Show individual asset analysis", value=False)

    if st.button("Run Fama-French Analysis", type="primary"):
        with st.spinner("Analyzing factor exposures..."):
            try:
                analyzer = FamaFrenchAnalyzer(returns)

                # Portfolio analysis
                result = analyzer.analyze_portfolio(weights, model)

                # Display results
                col1, col2, col3 = st.columns(3)

                with col1:
                    alpha_ann = result['alpha_annualized'] * 100
                    st.metric(
                        "Alpha (Annualized)",
                        f"{alpha_ann:.2f}%",
                        delta="Significant" if result['alpha_p_value'] < 0.05 else "Not Significant"
                    )

                with col2:
                    st.metric("R-squared", f"{result['r_squared']:.3f}")

                with col3:
                    st.metric("Observations", result['n_observations'])

                # Factor betas chart
                st.subheader("Factor Exposures (Betas)")

                betas_df = pd.DataFrame({
                    'Factor': list(result['betas'].keys()),
                    'Beta': list(result['betas'].values()),
                    'T-Stat': [result['t_stats'][f] for f in result['betas'].keys()],
                    'P-Value': [result['p_values'][f] for f in result['betas'].keys()]
                })

                # Beta bar chart
                fig = px.bar(
                    betas_df,
                    x='Factor',
                    y='Beta',
                    color='Beta',
                    color_continuous_scale='RdYlGn',
                    title='Factor Betas'
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width="stretch")

                # Factor details table
                st.dataframe(
                    betas_df.style.format({
                        'Beta': '{:.3f}',
                        'T-Stat': '{:.2f}',
                        'P-Value': '{:.4f}'
                    }),
                    width="stretch"
                )

                # Factor contribution
                st.subheader("Return Contribution by Factor")
                contribution = analyzer.factor_contribution(weights, model)

                contrib_data = []
                for factor in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']:
                    if factor in contribution:
                        contrib_data.append({
                            'Factor': factor,
                            'Contribution': contribution[factor]['contribution'] * 100
                        })

                contrib_data.append({
                    'Factor': 'Alpha',
                    'Contribution': contribution['alpha'] * 100
                })

                contrib_df = pd.DataFrame(contrib_data)

                fig = px.pie(
                    contrib_df,
                    values='Contribution',
                    names='Factor',
                    title='Return Attribution'
                )
                st.plotly_chart(fig, width="stretch")

                # Individual asset analysis
                if show_individual:
                    st.subheader("Individual Asset Factor Exposures")
                    all_assets = analyzer.analyze_all_assets(model)
                    st.dataframe(
                        all_assets.style.format('{:.3f}'),
                        width="stretch"
                    )

                # Store for other tabs
                st.session_state['ff_analyzer'] = analyzer

            except Exception as e:
                st.error(f"Error in Fama-French analysis: {str(e)}")

# ============================================================================
# Tab 2: Sector Attribution
# ============================================================================
with tab2:
    st.header("Sector Attribution")
    st.markdown("Understand how sector allocation and selection drive returns")

    if st.button("Run Sector Attribution", type="primary"):
        with st.spinner("Analyzing sector attribution..."):
            try:
                # Sector analysis
                sector_attr = SectorAttribution(returns, weights)
                sector_weights = sector_attr.get_sector_weights()
                sector_contrib = sector_attr.sector_contribution()

                # Sector weights pie chart
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Sector Weights")
                    fig = px.pie(
                        values=sector_weights.values,
                        names=sector_weights.index,
                        title='Portfolio Sector Allocation'
                    )
                    st.plotly_chart(fig, width="stretch")

                with col2:
                    st.subheader("Sector Contribution to Return")
                    fig = px.bar(
                        x=sector_contrib.index,
                        y=sector_contrib['Contribution'] * 100,
                        title='Annual Return Contribution (%)',
                        labels={'x': 'Sector', 'y': 'Contribution (%)'}
                    )
                    fig.update_traces(marker_color=np.where(
                        sector_contrib['Contribution'] >= 0, 'green', 'red'
                    ))
                    st.plotly_chart(fig, width="stretch")

                # Sector details table
                st.subheader("Sector Performance Details")
                display_df = sector_contrib.copy()
                display_df['Weight'] = display_df['Weight'] * 100
                display_df['Return'] = display_df['Return'] * 100
                display_df['Volatility'] = display_df['Volatility'] * 100
                display_df['Contribution'] = display_df['Contribution'] * 100

                st.dataframe(
                    display_df.style.format({
                        'Weight': '{:.1f}%',
                        'Return': '{:.2f}%',
                        'Volatility': '{:.2f}%',
                        'Contribution': '{:.2f}%'
                    }),
                    width="stretch"
                )

                # Brinson Attribution
                st.subheader("Brinson Attribution Analysis")
                brinson = BrinsonAttribution(returns, weights)
                attr_result = brinson.calculate_attribution()

                summary = attr_result['summary']

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Allocation Effect",
                        f"{summary['allocation_effect']*100:.2f}%"
                    )

                with col2:
                    st.metric(
                        "Selection Effect",
                        f"{summary['selection_effect']*100:.2f}%"
                    )

                with col3:
                    st.metric(
                        "Interaction Effect",
                        f"{summary['interaction_effect']*100:.2f}%"
                    )

                with col4:
                    st.metric(
                        "Total Active Return",
                        f"{summary['total_active_return']*100:.2f}%"
                    )

                # Detailed Brinson table
                detailed = attr_result['detailed']
                display_detailed = detailed[['Allocation', 'Selection', 'Interaction', 'Total']].copy()
                display_detailed = display_detailed * 100

                st.dataframe(
                    display_detailed.style.format('{:.2f}%'),
                    width="stretch"
                )

                # Sector correlation
                st.subheader("Sector Correlation Matrix")
                sector_corr = sector_attr.sector_correlation()

                fig = px.imshow(
                    sector_corr,
                    color_continuous_scale='RdBu',
                    aspect='auto',
                    title='Sector Correlation'
                )
                st.plotly_chart(fig, width="stretch")

            except Exception as e:
                st.error(f"Error in sector attribution: {str(e)}")

# ============================================================================
# Tab 3: Style Factors
# ============================================================================
with tab3:
    st.header("Style Factor Analysis")
    st.markdown("""
    Analyze portfolio tilts toward style factors:
    - **Momentum**: Tendency toward recent winners
    - **Value**: Tendency toward cheap stocks
    - **Quality**: Tendency toward profitable, stable companies
    - **Low Volatility**: Tendency toward stable stocks
    - **Size**: Tendency toward small/large cap
    """)

    if st.button("Run Style Analysis", type="primary"):
        with st.spinner("Analyzing style factors..."):
            try:
                if prices is None:
                    st.error("Price data not available")
                    st.stop()

                analyzer = StyleFactorAnalyzer(prices)

                # Calculate all factors
                factors = analyzer.calculate_all_factors()
                exposures = analyzer.portfolio_factor_exposure(weights)
                tilts = analyzer.factor_tilt_analysis(weights)

                # Portfolio exposures
                st.subheader("Portfolio Factor Exposures")

                col1, col2 = st.columns(2)

                with col1:
                    # Exposure bar chart
                    exp_df = pd.DataFrame({
                        'Factor': list(exposures.keys()),
                        'Exposure': list(exposures.values())
                    })

                    fig = px.bar(
                        exp_df,
                        x='Factor',
                        y='Exposure',
                        color='Exposure',
                        color_continuous_scale='RdYlGn',
                        title='Portfolio Factor Exposures'
                    )
                    fig.add_hline(y=0, line_dash="dash")
                    st.plotly_chart(fig, width="stretch")

                with col2:
                    # Tilt comparison
                    tilt_df = tilts.reset_index()
                    tilt_df.columns = ['Factor', 'Portfolio', 'Equal Weight', 'Tilt']

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Portfolio',
                        x=tilt_df['Factor'],
                        y=tilt_df['Portfolio']
                    ))
                    fig.add_trace(go.Bar(
                        name='Equal Weight',
                        x=tilt_df['Factor'],
                        y=tilt_df['Equal Weight']
                    ))
                    fig.update_layout(
                        title='Portfolio vs Equal Weight',
                        barmode='group'
                    )
                    st.plotly_chart(fig, width="stretch")

                # Factor scores by asset
                st.subheader("Asset Factor Scores")
                st.dataframe(
                    factors.style.format('{:.3f}').background_gradient(cmap='RdYlGn', axis=0),
                    width="stretch"
                )

                # Return attribution
                st.subheader("Factor Return Attribution")
                attribution = analyzer.factor_return_attribution(weights)

                attr_data = []
                for factor, data in attribution.items():
                    if factor != 'total' and isinstance(data, dict):
                        attr_data.append({
                            'Factor': factor,
                            'Exposure': data['exposure'],
                            'Factor Return': data['factor_return'] * 100,
                            'Contribution': data['contribution'] * 100
                        })

                attr_df = pd.DataFrame(attr_data)

                fig = px.bar(
                    attr_df,
                    x='Factor',
                    y='Contribution',
                    title='Expected Return Contribution by Factor (%)'
                )
                st.plotly_chart(fig, width="stretch")

                st.metric(
                    "Total Factor-Based Expected Return",
                    f"{attribution['total']*100:.2f}%"
                )

                # Factor correlation
                st.subheader("Factor Correlation")
                factor_corr = analyzer.factor_correlation()

                fig = px.imshow(
                    factor_corr,
                    color_continuous_scale='RdBu',
                    aspect='auto',
                    title='Style Factor Correlations'
                )
                st.plotly_chart(fig, width="stretch")

                # Top stocks by factor
                st.subheader("Top Stocks by Factor")
                selected_factor = st.selectbox(
                    "Select Factor",
                    factors.columns.tolist()
                )

                top_stocks = analyzer.top_factor_stocks(selected_factor, n=5)
                st.dataframe(top_stocks, width="stretch")

            except Exception as e:
                st.error(f"Error in style analysis: {str(e)}")

# ============================================================================
# Tab 4: Risk Decomposition
# ============================================================================
with tab4:
    st.header("Factor Risk Decomposition")
    st.markdown("Understand the sources of portfolio risk")

    if st.button("Run Risk Decomposition", type="primary"):
        with st.spinner("Decomposing portfolio risk..."):
            try:
                decomp = FactorRiskDecomposition(returns)
                result = decomp.decompose_portfolio_risk(weights)

                # Risk summary
                st.subheader("Risk Breakdown")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Total Volatility",
                        f"{result['annualized_total_vol']*100:.2f}%"
                    )

                with col2:
                    st.metric(
                        "Systematic Risk",
                        f"{result['annualized_systematic_vol']*100:.2f}%",
                        delta=f"{result['systematic_pct']*100:.1f}% of total"
                    )

                with col3:
                    st.metric(
                        "Specific Risk",
                        f"{result['annualized_specific_vol']*100:.2f}%",
                        delta=f"{result['specific_pct']*100:.1f}% of total"
                    )

                # Pie chart of systematic vs specific
                col1, col2 = st.columns(2)

                with col1:
                    fig = px.pie(
                        values=[result['systematic_pct'], result['specific_pct']],
                        names=['Systematic', 'Specific'],
                        title='Risk Decomposition',
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
                    )
                    st.plotly_chart(fig, width="stretch")

                with col2:
                    # Factor betas
                    betas = result['portfolio_betas']
                    beta_df = pd.DataFrame({
                        'Factor': list(betas.keys()),
                        'Beta': list(betas.values())
                    })

                    fig = px.bar(
                        beta_df,
                        x='Factor',
                        y='Beta',
                        title='Portfolio Factor Betas',
                        color='Beta',
                        color_continuous_scale='RdYlGn'
                    )
                    fig.add_hline(y=0, line_dash="dash")
                    st.plotly_chart(fig, width="stretch")

                # Factor risk contributions
                st.subheader("Factor Risk Contributions")

                contrib_data = []
                for factor, data in result['factor_contributions'].items():
                    contrib_data.append({
                        'Factor': factor,
                        'Beta': data['beta'],
                        '% of Systematic': data['pct_of_systematic'] * 100
                    })

                contrib_df = pd.DataFrame(contrib_data)

                fig = px.bar(
                    contrib_df,
                    x='Factor',
                    y='% of Systematic',
                    title='Factor Contribution to Systematic Risk (%)'
                )
                st.plotly_chart(fig, width="stretch")

                # Factor stress scenarios
                st.subheader("Factor Stress Scenarios")
                stress_results = decomp.factor_stress_scenarios(weights)

                fig = px.bar(
                    stress_results.reset_index(),
                    x='Scenario',
                    y='Impact %',
                    title='Portfolio Impact Under Factor Stress (%)',
                    color='Impact %',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, width="stretch")

                st.dataframe(
                    stress_results.style.format({'Portfolio Impact': '{:.4f}', 'Impact %': '{:.2f}%'}),
                    width="stretch"
                )

                # Factor correlation
                st.subheader("Factor Correlation Matrix")
                factor_corr = decomp.factor_correlation_matrix()

                fig = px.imshow(
                    factor_corr,
                    color_continuous_scale='RdBu',
                    aspect='auto',
                    title='Factor Correlations'
                )
                st.plotly_chart(fig, width="stretch")

            except Exception as e:
                st.error(f"Error in risk decomposition: {str(e)}")

# Sidebar info
with st.sidebar:
    st.header("Factor Analysis Info")

    st.markdown("""
    ### Factor Models

    **Fama-French Factors:**
    - Market, Size, Value
    - Profitability, Investment

    **Style Factors:**
    - Momentum, Quality
    - Low Volatility

    ### Interpretation

    - **Positive beta**: Exposed to factor
    - **High R²**: Well-explained by factors
    - **Alpha**: Unexplained return
    """)

    if 'weights' in st.session_state:
        st.markdown("---")
        st.subheader("Current Portfolio")
        for ticker, weight in weights.items():
            st.write(f"{ticker}: {weight*100:.1f}%")
