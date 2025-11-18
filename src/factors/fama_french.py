"""
Fama-French Factor Analysis

Implements 3-factor and 5-factor models for analyzing portfolio factor exposures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings

from ..utils.logger import get_logger

logger = get_logger(__name__)


def get_factor_data(
    start_date: str,
    end_date: str,
    model: str = '5'
) -> pd.DataFrame:
    """
    Fetch Fama-French factor data.

    Attempts to fetch from pandas_datareader, falls back to synthetic data.

    Args:
        start_date: Start date string
        end_date: End date string
        model: '3' for 3-factor, '5' for 5-factor

    Returns:
        DataFrame with factor returns
    """
    try:
        import pandas_datareader.data as web

        if model == '5':
            ff_data = web.DataReader(
                'F-F_Research_Data_5_Factors_2x3_daily',
                'famafrench',
                start=start_date,
                end=end_date
            )[0]
        else:
            ff_data = web.DataReader(
                'F-F_Research_Data_Factors_daily',
                'famafrench',
                start=start_date,
                end=end_date
            )[0]

        # Convert from percentage to decimal
        ff_data = ff_data / 100
        logger.info(f"Fetched Fama-French {model}-factor data")
        return ff_data

    except Exception as e:
        logger.warning(f"Could not fetch FF data: {e}. Using synthetic factors.")
        return _generate_synthetic_factors(start_date, end_date, model)


def _generate_synthetic_factors(
    start_date: str,
    end_date: str,
    model: str = '5'
) -> pd.DataFrame:
    """Generate synthetic factor data for demonstration."""
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n = len(dates)

    np.random.seed(42)

    # Generate correlated factors with realistic properties
    factors = {
        'Mkt-RF': np.random.normal(0.0004, 0.01, n),  # Market excess return
        'SMB': np.random.normal(0.0001, 0.005, n),     # Small minus Big
        'HML': np.random.normal(0.0001, 0.005, n),     # High minus Low (Value)
        'RF': np.full(n, 0.0001)                        # Risk-free rate
    }

    if model == '5':
        factors['RMW'] = np.random.normal(0.0001, 0.004, n)  # Robust minus Weak
        factors['CMA'] = np.random.normal(0.0001, 0.004, n)  # Conservative minus Aggressive

    df = pd.DataFrame(factors, index=dates)
    logger.info(f"Generated synthetic {model}-factor data")
    return df


class FamaFrenchAnalyzer:
    """
    Analyzes portfolio returns using Fama-French factor models.

    Supports both 3-factor and 5-factor models.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        factor_data: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize the analyzer.

        Args:
            returns: Asset returns DataFrame (dates x assets)
            factor_data: Optional pre-loaded factor data
            risk_free_rate: Annual risk-free rate
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.daily_rf = (1 + risk_free_rate) ** (1/252) - 1

        if factor_data is not None:
            self.factor_data = factor_data
        else:
            start = returns.index[0].strftime('%Y-%m-%d')
            end = returns.index[-1].strftime('%Y-%m-%d')
            self.factor_data = get_factor_data(start, end, model='5')

        # Align dates
        common_dates = returns.index.intersection(self.factor_data.index)
        self.returns = returns.loc[common_dates]
        self.factor_data = self.factor_data.loc[common_dates]

        logger.info(f"FamaFrenchAnalyzer initialized with {len(common_dates)} observations")

    def analyze_asset(
        self,
        asset: str,
        model: str = '5'
    ) -> Dict:
        """
        Analyze a single asset's factor exposures.

        Args:
            asset: Asset ticker
            model: '3' or '5' factor model

        Returns:
            Dictionary with regression results
        """
        if asset not in self.returns.columns:
            raise ValueError(f"Asset {asset} not found in returns")

        # Get excess returns
        asset_returns = self.returns[asset]

        if 'RF' in self.factor_data.columns:
            excess_returns = asset_returns - self.factor_data['RF']
        else:
            excess_returns = asset_returns - self.daily_rf

        # Prepare factors
        if model == '3':
            factors = ['Mkt-RF', 'SMB', 'HML']
        else:
            factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

        available_factors = [f for f in factors if f in self.factor_data.columns]
        X = self.factor_data[available_factors]

        # Add constant for intercept (alpha)
        X = pd.DataFrame(X)
        X.insert(0, 'const', 1.0)

        # Run regression
        y = excess_returns.values
        X_values = X.values

        # OLS regression
        try:
            beta = np.linalg.lstsq(X_values, y, rcond=None)[0]
        except:
            beta = np.linalg.pinv(X_values) @ y

        # Calculate statistics
        y_pred = X_values @ beta
        residuals = y - y_pred

        n = len(y)
        k = len(beta)

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Adjusted R-squared
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

        # Standard errors
        mse = ss_res / (n - k)
        var_beta = mse * np.linalg.inv(X_values.T @ X_values)
        std_errors = np.sqrt(np.diag(var_beta))

        # T-statistics and p-values
        t_stats = beta / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))

        # Build results
        result = {
            'asset': asset,
            'model': f'{model}-factor',
            'alpha': beta[0],
            'alpha_annualized': beta[0] * 252,
            'alpha_t_stat': t_stats[0],
            'alpha_p_value': p_values[0],
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'residual_std': np.std(residuals),
            'n_observations': n,
            'betas': {},
            't_stats': {},
            'p_values': {}
        }

        for i, factor in enumerate(available_factors):
            result['betas'][factor] = beta[i + 1]
            result['t_stats'][factor] = t_stats[i + 1]
            result['p_values'][factor] = p_values[i + 1]

        return result

    def analyze_portfolio(
        self,
        weights: pd.Series,
        model: str = '5'
    ) -> Dict:
        """
        Analyze portfolio factor exposures.

        Args:
            weights: Portfolio weights
            model: '3' or '5' factor model

        Returns:
            Dictionary with portfolio factor analysis
        """
        # Calculate portfolio returns
        aligned_weights = weights[weights.index.isin(self.returns.columns)]
        portfolio_returns = (self.returns[aligned_weights.index] * aligned_weights).sum(axis=1)

        # Create temporary analyzer for portfolio
        temp_returns = pd.DataFrame({'Portfolio': portfolio_returns})
        temp_analyzer = FamaFrenchAnalyzer(
            temp_returns,
            self.factor_data,
            self.risk_free_rate
        )

        result = temp_analyzer.analyze_asset('Portfolio', model)
        result['weights'] = aligned_weights.to_dict()

        return result

    def analyze_all_assets(self, model: str = '5') -> pd.DataFrame:
        """
        Analyze all assets in the portfolio.

        Args:
            model: '3' or '5' factor model

        Returns:
            DataFrame with factor exposures for all assets
        """
        results = []

        for asset in self.returns.columns:
            try:
                analysis = self.analyze_asset(asset, model)
                row = {
                    'Asset': asset,
                    'Alpha': analysis['alpha_annualized'],
                    'Alpha t-stat': analysis['alpha_t_stat'],
                    'R²': analysis['r_squared']
                }
                row.update(analysis['betas'])
                results.append(row)
            except Exception as e:
                logger.warning(f"Could not analyze {asset}: {e}")

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.set_index('Asset')

        return df

    def factor_contribution(
        self,
        weights: pd.Series,
        model: str = '5'
    ) -> Dict:
        """
        Calculate factor contribution to portfolio returns.

        Args:
            weights: Portfolio weights
            model: '3' or '5' factor model

        Returns:
            Dictionary with factor contributions
        """
        analysis = self.analyze_portfolio(weights, model)

        # Average factor returns
        if model == '3':
            factors = ['Mkt-RF', 'SMB', 'HML']
        else:
            factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

        contributions = {}
        total_explained = 0

        for factor in factors:
            if factor in analysis['betas'] and factor in self.factor_data.columns:
                avg_factor_return = self.factor_data[factor].mean() * 252
                contribution = analysis['betas'][factor] * avg_factor_return
                contributions[factor] = {
                    'beta': analysis['betas'][factor],
                    'factor_return': avg_factor_return,
                    'contribution': contribution
                }
                total_explained += contribution

        contributions['alpha'] = analysis['alpha_annualized']
        contributions['total_explained'] = total_explained + analysis['alpha_annualized']

        return contributions

    def rolling_betas(
        self,
        asset: str,
        window: int = 60,
        model: str = '5'
    ) -> pd.DataFrame:
        """
        Calculate rolling factor betas.

        Args:
            asset: Asset ticker
            window: Rolling window size
            model: '3' or '5' factor model

        Returns:
            DataFrame with rolling betas
        """
        if model == '3':
            factors = ['Mkt-RF', 'SMB', 'HML']
        else:
            factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

        available_factors = [f for f in factors if f in self.factor_data.columns]

        results = {factor: [] for factor in available_factors}
        results['alpha'] = []
        results['date'] = []

        asset_returns = self.returns[asset]

        if 'RF' in self.factor_data.columns:
            excess_returns = asset_returns - self.factor_data['RF']
        else:
            excess_returns = asset_returns - self.daily_rf

        for i in range(window, len(excess_returns)):
            y = excess_returns.iloc[i-window:i].values
            X = self.factor_data[available_factors].iloc[i-window:i].values
            X = np.column_stack([np.ones(window), X])

            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                results['alpha'].append(beta[0])
                for j, factor in enumerate(available_factors):
                    results[factor].append(beta[j + 1])
                results['date'].append(excess_returns.index[i])
            except:
                continue

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.set_index('date')

        return df

    def summary(self, weights: pd.Series, model: str = '5') -> str:
        """
        Generate a text summary of factor analysis.

        Args:
            weights: Portfolio weights
            model: '3' or '5' factor model

        Returns:
            Formatted summary string
        """
        analysis = self.analyze_portfolio(weights, model)
        contribution = self.factor_contribution(weights, model)

        lines = [
            f"Fama-French {model}-Factor Analysis",
            "=" * 40,
            "",
            f"Alpha (annualized): {analysis['alpha_annualized']:.4f} ({analysis['alpha_annualized']*100:.2f}%)",
            f"Alpha t-statistic: {analysis['alpha_t_stat']:.2f}",
            f"Alpha p-value: {analysis['alpha_p_value']:.4f}",
            f"R-squared: {analysis['r_squared']:.4f}",
            "",
            "Factor Exposures (Betas):",
            "-" * 30
        ]

        for factor, beta in analysis['betas'].items():
            t_stat = analysis['t_stats'].get(factor, 0)
            p_val = analysis['p_values'].get(factor, 1)
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            lines.append(f"  {factor}: {beta:.3f} (t={t_stat:.2f}) {sig}")

        lines.extend([
            "",
            "Return Contribution:",
            "-" * 30
        ])

        for factor in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']:
            if factor in contribution:
                cont = contribution[factor]['contribution']
                lines.append(f"  {factor}: {cont*100:.2f}%")

        lines.append(f"  Alpha: {contribution['alpha']*100:.2f}%")
        lines.append(f"  Total: {contribution['total_explained']*100:.2f}%")

        return "\n".join(lines)
