"""
Factor Risk Decomposition

Decomposes portfolio risk into systematic and idiosyncratic components.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats

from ..utils.logger import get_logger
from .fama_french import FamaFrenchAnalyzer, get_factor_data

logger = get_logger(__name__)


class FactorRiskDecomposition:
    """
    Decomposes portfolio risk into factor and specific risk components.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        factor_data: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize factor risk decomposition.

        Args:
            returns: Asset returns DataFrame
            factor_data: Optional factor returns
            risk_free_rate: Annual risk-free rate
        """
        self.returns = returns

        if factor_data is not None:
            self.factor_data = factor_data
        else:
            start = returns.index[0].strftime('%Y-%m-%d')
            end = returns.index[-1].strftime('%Y-%m-%d')
            self.factor_data = get_factor_data(start, end)

        # Align dates
        common_dates = returns.index.intersection(self.factor_data.index)
        self.returns = returns.loc[common_dates]
        self.factor_data = self.factor_data.loc[common_dates]

        self.risk_free_rate = risk_free_rate
        self.daily_rf = (1 + risk_free_rate) ** (1/252) - 1

        # Store factor betas for each asset
        self._asset_betas = {}
        self._residuals = {}

        logger.info(f"FactorRiskDecomposition initialized with {len(common_dates)} observations")

    def _calculate_asset_betas(self, asset: str) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate factor betas for a single asset."""
        if asset in self._asset_betas:
            return self._asset_betas[asset], self._residuals[asset]

        # Get excess returns
        asset_returns = self.returns[asset]

        if 'RF' in self.factor_data.columns:
            excess_returns = asset_returns - self.factor_data['RF']
        else:
            excess_returns = asset_returns - self.daily_rf

        # Prepare factors
        factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        available_factors = [f for f in factors if f in self.factor_data.columns]
        X = self.factor_data[available_factors].values

        # Add constant
        X = np.column_stack([np.ones(len(X)), X])
        y = excess_returns.values

        # OLS
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except:
            beta = np.linalg.pinv(X) @ y

        # Residuals
        residuals = y - X @ beta

        self._asset_betas[asset] = beta
        self._residuals[asset] = residuals

        return beta, residuals

    def decompose_asset_variance(self, asset: str) -> Dict:
        """
        Decompose single asset variance into factor and specific risk.

        Args:
            asset: Asset ticker

        Returns:
            Dictionary with variance decomposition
        """
        beta, residuals = self._calculate_asset_betas(asset)

        # Total variance
        asset_returns = self.returns[asset]
        if 'RF' in self.factor_data.columns:
            excess_returns = asset_returns - self.factor_data['RF']
        else:
            excess_returns = asset_returns - self.daily_rf

        total_var = np.var(excess_returns)

        # Specific variance (from residuals)
        specific_var = np.var(residuals)

        # Systematic variance
        systematic_var = total_var - specific_var

        # Factor contributions
        factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        available_factors = [f for f in factors if f in self.factor_data.columns]

        factor_contributions = {}
        for i, factor in enumerate(available_factors):
            factor_var = np.var(self.factor_data[factor])
            contribution = (beta[i + 1] ** 2) * factor_var
            factor_contributions[factor] = contribution

        return {
            'asset': asset,
            'total_variance': total_var,
            'systematic_variance': systematic_var,
            'specific_variance': specific_var,
            'systematic_pct': systematic_var / total_var if total_var > 0 else 0,
            'specific_pct': specific_var / total_var if total_var > 0 else 0,
            'factor_contributions': factor_contributions,
            'annualized_total_vol': np.sqrt(total_var * 252),
            'annualized_specific_vol': np.sqrt(specific_var * 252)
        }

    def decompose_portfolio_risk(
        self,
        weights: pd.Series
    ) -> Dict:
        """
        Decompose portfolio risk into factor and specific components.

        Args:
            weights: Portfolio weights

        Returns:
            Dictionary with portfolio risk decomposition
        """
        aligned_weights = weights[weights.index.isin(self.returns.columns)]
        n_assets = len(aligned_weights)

        # Get betas for all assets
        factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        available_factors = [f for f in factors if f in self.factor_data.columns]
        n_factors = len(available_factors)

        # Build beta matrix (assets x factors)
        beta_matrix = np.zeros((n_assets, n_factors))
        specific_variances = np.zeros(n_assets)

        for i, asset in enumerate(aligned_weights.index):
            beta, residuals = self._calculate_asset_betas(asset)
            beta_matrix[i, :] = beta[1:n_factors+1]  # Skip intercept
            specific_variances[i] = np.var(residuals)

        # Portfolio factor betas
        w = aligned_weights.values
        portfolio_betas = w @ beta_matrix  # 1 x n_factors

        # Factor covariance matrix
        factor_cov = self.factor_data[available_factors].cov().values

        # Systematic variance = beta' * factor_cov * beta
        systematic_var = portfolio_betas @ factor_cov @ portfolio_betas.T

        # Specific variance (assuming uncorrelated specific risks)
        specific_var = np.sum((w ** 2) * specific_variances)

        # Total variance
        total_var = systematic_var + specific_var

        # Factor marginal contributions
        marginal_contributions = {}
        for i, factor in enumerate(available_factors):
            # Contribution = 2 * beta_i * (factor_cov[i,:] @ portfolio_betas)
            contribution = portfolio_betas[i] * (factor_cov[i, :] @ portfolio_betas.T)
            marginal_contributions[factor] = {
                'beta': portfolio_betas[i],
                'variance_contribution': contribution,
                'pct_of_systematic': contribution / systematic_var if systematic_var > 0 else 0
            }

        return {
            'total_variance': total_var,
            'systematic_variance': systematic_var,
            'specific_variance': specific_var,
            'systematic_pct': systematic_var / total_var if total_var > 0 else 0,
            'specific_pct': specific_var / total_var if total_var > 0 else 0,
            'annualized_total_vol': np.sqrt(total_var * 252),
            'annualized_systematic_vol': np.sqrt(systematic_var * 252),
            'annualized_specific_vol': np.sqrt(specific_var * 252),
            'portfolio_betas': dict(zip(available_factors, portfolio_betas)),
            'factor_contributions': marginal_contributions
        }

    def factor_correlation_matrix(self) -> pd.DataFrame:
        """Get factor correlation matrix."""
        factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        available_factors = [f for f in factors if f in self.factor_data.columns]
        return self.factor_data[available_factors].corr()

    def factor_stress_scenarios(
        self,
        weights: pd.Series,
        scenarios: Optional[Dict[str, Dict[str, float]]] = None
    ) -> pd.DataFrame:
        """
        Calculate portfolio impact under factor stress scenarios.

        Args:
            weights: Portfolio weights
            scenarios: Dict of scenario name -> factor shocks

        Returns:
            DataFrame with scenario impacts
        """
        default_scenarios = {
            'Market Crash': {'Mkt-RF': -0.20, 'SMB': -0.10, 'HML': 0.05},
            'Value Rally': {'Mkt-RF': 0.05, 'HML': 0.15, 'SMB': 0.05},
            'Quality Flight': {'Mkt-RF': -0.10, 'RMW': 0.10, 'CMA': 0.05},
            'Small Cap Surge': {'SMB': 0.15, 'Mkt-RF': 0.05},
            'Risk-Off': {'Mkt-RF': -0.15, 'SMB': -0.15, 'HML': -0.05}
        }

        scenarios = scenarios or default_scenarios

        decomp = self.decompose_portfolio_risk(weights)
        portfolio_betas = decomp['portfolio_betas']

        results = []

        for scenario_name, shocks in scenarios.items():
            impact = 0

            for factor, shock in shocks.items():
                if factor in portfolio_betas:
                    impact += portfolio_betas[factor] * shock

            results.append({
                'Scenario': scenario_name,
                'Portfolio Impact': impact,
                'Impact %': impact * 100
            })

        return pd.DataFrame(results).set_index('Scenario')

    def tracking_error_decomposition(
        self,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series
    ) -> Dict:
        """
        Decompose tracking error into factor and specific components.

        Args:
            portfolio_weights: Portfolio weights
            benchmark_weights: Benchmark weights

        Returns:
            Dictionary with tracking error decomposition
        """
        port_decomp = self.decompose_portfolio_risk(portfolio_weights)
        bench_decomp = self.decompose_portfolio_risk(benchmark_weights)

        # Active betas (difference in factor exposures)
        active_betas = {}
        factors = list(port_decomp['portfolio_betas'].keys())

        for factor in factors:
            active_betas[factor] = (
                port_decomp['portfolio_betas'][factor] -
                bench_decomp['portfolio_betas'][factor]
            )

        # Factor contribution to tracking error
        factor_cov = self.factor_correlation_matrix()

        active_beta_vec = np.array([active_betas[f] for f in factors])
        factor_te_var = active_beta_vec @ factor_cov.values @ active_beta_vec.T

        # Total tracking error variance (simplified)
        total_te_var = factor_te_var + 0.0001  # Add small specific component

        return {
            'tracking_error': np.sqrt(total_te_var * 252),
            'factor_te': np.sqrt(factor_te_var * 252),
            'active_betas': active_betas,
            'factor_contributions': {
                f: active_betas[f] ** 2 * factor_cov.loc[f, f]
                for f in factors
            }
        }

    def summary(self, weights: pd.Series) -> str:
        """
        Generate text summary of risk decomposition.

        Args:
            weights: Portfolio weights

        Returns:
            Formatted summary string
        """
        decomp = self.decompose_portfolio_risk(weights)

        lines = [
            "Factor Risk Decomposition",
            "=" * 40,
            "",
            "Risk Breakdown:",
            "-" * 30,
            f"  Total Volatility:      {decomp['annualized_total_vol']*100:.2f}%",
            f"  Systematic Volatility: {decomp['annualized_systematic_vol']*100:.2f}% ({decomp['systematic_pct']*100:.1f}%)",
            f"  Specific Volatility:   {decomp['annualized_specific_vol']*100:.2f}% ({decomp['specific_pct']*100:.1f}%)",
            "",
            "Factor Betas:",
            "-" * 30
        ]

        for factor, beta in decomp['portfolio_betas'].items():
            lines.append(f"  {factor}: {beta:.3f}")

        lines.extend([
            "",
            "Factor Risk Contributions:",
            "-" * 30
        ])

        for factor, contrib in decomp['factor_contributions'].items():
            pct = contrib['pct_of_systematic'] * 100
            lines.append(f"  {factor}: {pct:.1f}% of systematic risk")

        return "\n".join(lines)


def calculate_factor_risk(
    returns: pd.DataFrame,
    weights: pd.Series,
    factor_data: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Convenience function for factor risk decomposition.

    Args:
        returns: Asset returns
        weights: Portfolio weights
        factor_data: Optional factor data

    Returns:
        Risk decomposition dictionary
    """
    decomp = FactorRiskDecomposition(returns, factor_data)
    return decomp.decompose_portfolio_risk(weights)


def get_factor_stress_impact(
    returns: pd.DataFrame,
    weights: pd.Series,
    scenario: Dict[str, float]
) -> float:
    """
    Calculate portfolio impact from a factor stress scenario.

    Args:
        returns: Asset returns
        weights: Portfolio weights
        scenario: Dict of factor shocks

    Returns:
        Portfolio return impact
    """
    decomp = FactorRiskDecomposition(returns)
    result = decomp.decompose_portfolio_risk(weights)

    impact = 0
    for factor, shock in scenario.items():
        if factor in result['portfolio_betas']:
            impact += result['portfolio_betas'][factor] * shock

    return impact
