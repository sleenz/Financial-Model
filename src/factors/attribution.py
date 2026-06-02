"""
Sector and Industry Attribution Analysis

Implements Brinson attribution model for decomposing portfolio returns.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Tuple

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Runtime cache: populated from SECTOR_MAPPINGS below plus live yfinance lookups.
# Keyed by upper-case ticker symbol.
_SECTOR_CACHE: Dict[str, str] = {}


def _fetch_sector(ticker: str) -> str:
    """
    Return the GICS sector for a ticker.

    Checks the in-process cache first (seeded from SECTOR_MAPPINGS on import).
    On a cache miss, queries yfinance once and stores the result so subsequent
    calls for the same ticker are free.  Falls back to 'Unknown' if yfinance
    returns nothing useful.
    """
    key = ticker.upper()
    if key in _SECTOR_CACHE:
        return _SECTOR_CACHE[key]
    try:
        info = yf.Ticker(key).info or {}
        sector = info.get("sector") or "Unknown"
    except Exception:
        sector = "Unknown"
    _SECTOR_CACHE[key] = sector
    logger.debug("Fetched sector for %s from yfinance: %s", key, sector)
    return sector


# GICS Sector mappings for common stocks
SECTOR_MAPPINGS = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'GOOG': 'Technology',
    'META': 'Technology', 'NVDA': 'Technology', 'AMD': 'Technology', 'INTC': 'Technology',
    'CRM': 'Technology', 'ADBE': 'Technology', 'CSCO': 'Technology', 'ORCL': 'Technology',
    'IBM': 'Technology', 'NOW': 'Technology', 'QCOM': 'Technology', 'TXN': 'Technology',
    'AVGO': 'Technology', 'MU': 'Technology', 'AMAT': 'Technology', 'LRCX': 'Technology',

    # Healthcare
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare', 'ABBV': 'Healthcare',
    'MRK': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare', 'DHR': 'Healthcare',
    'LLY': 'Healthcare', 'BMY': 'Healthcare', 'AMGN': 'Healthcare', 'GILD': 'Healthcare',
    'MDT': 'Healthcare', 'ISRG': 'Healthcare', 'CVS': 'Healthcare', 'CI': 'Healthcare',

    # Financials
    'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials',
    'MS': 'Financials', 'C': 'Financials', 'BLK': 'Financials', 'SCHW': 'Financials',
    'AXP': 'Financials', 'USB': 'Financials', 'PNC': 'Financials', 'TFC': 'Financials',
    'BK': 'Financials', 'COF': 'Financials', 'CME': 'Financials', 'ICE': 'Financials',
    'V': 'Financials', 'MA': 'Financials', 'PYPL': 'Financials',

    # Consumer Discretionary
    'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary', 'HD': 'Consumer Discretionary',
    'NKE': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary', 'SBUX': 'Consumer Discretionary',
    'LOW': 'Consumer Discretionary', 'TJX': 'Consumer Discretionary', 'BKNG': 'Consumer Discretionary',
    'TGT': 'Consumer Discretionary', 'F': 'Consumer Discretionary', 'GM': 'Consumer Discretionary',

    # Consumer Staples
    'PG': 'Consumer Staples', 'KO': 'Consumer Staples', 'PEP': 'Consumer Staples',
    'WMT': 'Consumer Staples', 'COST': 'Consumer Staples', 'PM': 'Consumer Staples',
    'MO': 'Consumer Staples', 'CL': 'Consumer Staples', 'MDLZ': 'Consumer Staples',
    'KHC': 'Consumer Staples', 'GIS': 'Consumer Staples', 'K': 'Consumer Staples',

    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
    'EOG': 'Energy', 'MPC': 'Energy', 'PSX': 'Energy', 'VLO': 'Energy',
    'OXY': 'Energy', 'KMI': 'Energy', 'WMB': 'Energy', 'HAL': 'Energy',

    # Industrials
    'UPS': 'Industrials', 'UNP': 'Industrials', 'HON': 'Industrials', 'BA': 'Industrials',
    'CAT': 'Industrials', 'GE': 'Industrials', 'MMM': 'Industrials', 'LMT': 'Industrials',
    'RTX': 'Industrials', 'DE': 'Industrials', 'FDX': 'Industrials', 'NSC': 'Industrials',

    # Materials
    'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials', 'FCX': 'Materials',
    'NEM': 'Materials', 'ECL': 'Materials', 'DD': 'Materials', 'NUE': 'Materials',

    # Utilities
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities',
    'AEP': 'Utilities', 'EXC': 'Utilities', 'SRE': 'Utilities', 'XEL': 'Utilities',

    # Real Estate
    'AMT': 'Real Estate', 'PLD': 'Real Estate', 'CCI': 'Real Estate', 'EQIX': 'Real Estate',
    'PSA': 'Real Estate', 'SPG': 'Real Estate', 'O': 'Real Estate', 'WELL': 'Real Estate',

    # Communication Services
    'DIS': 'Communication Services', 'NFLX': 'Communication Services', 'CMCSA': 'Communication Services',
    'VZ': 'Communication Services', 'T': 'Communication Services', 'TMUS': 'Communication Services',
    'CHTR': 'Communication Services', 'ATVI': 'Communication Services'
}

# Seed the runtime cache from the static map so common tickers never hit yfinance.
_SECTOR_CACHE.update({k.upper(): v for k, v in SECTOR_MAPPINGS.items()})

# Default benchmark sector weights (approximate S&P 500)
DEFAULT_BENCHMARK_WEIGHTS = {
    'Technology': 0.28,
    'Healthcare': 0.13,
    'Financials': 0.12,
    'Consumer Discretionary': 0.11,
    'Communication Services': 0.09,
    'Industrials': 0.08,
    'Consumer Staples': 0.06,
    'Energy': 0.04,
    'Utilities': 0.03,
    'Real Estate': 0.03,
    'Materials': 0.03
}


class SectorAttribution:
    """
    Analyzes portfolio performance by sector.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        weights: pd.Series,
        sector_map: Optional[Dict[str, str]] = None
    ):
        """
        Initialize sector attribution.

        Args:
            returns: Asset returns DataFrame
            weights: Portfolio weights
            sector_map: Custom sector mappings (ticker -> sector)
        """
        self.returns = returns
        self.weights = weights
        self.sector_map = sector_map or {}

        # Map assets to sectors: explicit override → static cache → live yfinance lookup
        self.asset_sectors = {}
        for asset in weights.index:
            if asset in self.sector_map:
                self.asset_sectors[asset] = self.sector_map[asset]
            else:
                self.asset_sectors[asset] = _fetch_sector(asset)

        logger.info(f"SectorAttribution initialized with {len(weights)} assets")

    def get_sector_weights(self) -> pd.Series:
        """Calculate portfolio weights by sector."""
        sector_weights = {}

        for asset, weight in self.weights.items():
            sector = self.asset_sectors.get(asset, 'Other')
            sector_weights[sector] = sector_weights.get(sector, 0) + weight

        return pd.Series(sector_weights).sort_values(ascending=False)

    def get_sector_returns(self) -> pd.DataFrame:
        """Calculate returns by sector."""
        sectors = list(set(self.asset_sectors.values()))
        sector_returns = pd.DataFrame(index=self.returns.index)

        for sector in sectors:
            # Get assets in this sector
            sector_assets = [a for a, s in self.asset_sectors.items()
                           if s == sector and a in self.returns.columns]

            if sector_assets:
                # Weight-averaged return within sector
                sector_weights = self.weights[sector_assets]
                if sector_weights.sum() > 0:
                    normalized_weights = sector_weights / sector_weights.sum()
                    sector_returns[sector] = (
                        self.returns[sector_assets] * normalized_weights
                    ).sum(axis=1)
                else:
                    sector_returns[sector] = self.returns[sector_assets].mean(axis=1)

        return sector_returns

    def sector_contribution(self) -> pd.DataFrame:
        """
        Calculate each sector's contribution to total return.

        Returns:
            DataFrame with sector contributions
        """
        sector_weights = self.get_sector_weights()
        sector_returns = self.get_sector_returns()

        results = []

        for sector in sector_weights.index:
            if sector in sector_returns.columns:
                weight = sector_weights[sector]
                avg_return = sector_returns[sector].mean() * 252  # Annualized
                vol = sector_returns[sector].std() * np.sqrt(252)
                contribution = weight * avg_return

                results.append({
                    'Sector': sector,
                    'Weight': weight,
                    'Return': avg_return,
                    'Volatility': vol,
                    'Contribution': contribution
                })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.set_index('Sector')
            df = df.sort_values('Contribution', ascending=False)

        return df

    def sector_correlation(self) -> pd.DataFrame:
        """Calculate correlation matrix between sectors."""
        sector_returns = self.get_sector_returns()
        return sector_returns.corr()


class BrinsonAttribution:
    """
    Implements Brinson-Fachler attribution model.

    Decomposes portfolio active return into:
    - Allocation Effect: Sector weight differences
    - Selection Effect: Stock selection within sectors
    - Interaction Effect: Combined impact
    """

    def __init__(
        self,
        portfolio_returns: pd.DataFrame,
        portfolio_weights: pd.Series,
        benchmark_returns: Optional[pd.DataFrame] = None,
        benchmark_weights: Optional[Dict[str, float]] = None,
        sector_map: Optional[Dict[str, str]] = None
    ):
        """
        Initialize Brinson attribution.

        Args:
            portfolio_returns: Portfolio asset returns
            portfolio_weights: Portfolio weights
            benchmark_returns: Benchmark sector returns (optional)
            benchmark_weights: Benchmark sector weights (optional)
            sector_map: Custom sector mappings
        """
        self.portfolio_returns = portfolio_returns
        self.portfolio_weights = portfolio_weights
        self.sector_map = sector_map or {}
        self.benchmark_weights = benchmark_weights or DEFAULT_BENCHMARK_WEIGHTS

        # Get sector attribution helper
        self.sector_attr = SectorAttribution(
            portfolio_returns, portfolio_weights, sector_map
        )

        # Generate benchmark returns if not provided
        if benchmark_returns is not None:
            self.benchmark_returns = benchmark_returns
        else:
            self.benchmark_returns = self._generate_benchmark_returns()

        logger.info("BrinsonAttribution initialized")

    def _generate_benchmark_returns(self) -> pd.DataFrame:
        """Generate synthetic benchmark returns based on portfolio sectors."""
        sector_returns = self.sector_attr.get_sector_returns()

        # Add small noise to simulate benchmark
        np.random.seed(42)
        noise = pd.DataFrame(
            np.random.normal(0, 0.001, sector_returns.shape),
            index=sector_returns.index,
            columns=sector_returns.columns
        )

        return sector_returns + noise

    def calculate_attribution(self) -> Dict:
        """
        Calculate Brinson attribution effects.

        Returns:
            Dictionary with attribution breakdown
        """
        # Get portfolio sector weights and returns
        port_sector_weights = self.sector_attr.get_sector_weights()
        port_sector_returns = self.sector_attr.get_sector_returns()

        # Get benchmark sector weights and returns
        bench_weights = pd.Series(self.benchmark_weights)
        bench_returns = self.benchmark_returns

        # Align sectors
        all_sectors = list(set(port_sector_weights.index) | set(bench_weights.index))

        results = []
        total_allocation = 0
        total_selection = 0
        total_interaction = 0

        for sector in all_sectors:
            # Portfolio values
            wp = port_sector_weights.get(sector, 0)
            if sector in port_sector_returns.columns:
                rp = port_sector_returns[sector].mean() * 252
            else:
                rp = 0

            # Benchmark values
            wb = bench_weights.get(sector, 0)
            if sector in bench_returns.columns:
                rb = bench_returns[sector].mean() * 252
            else:
                rb = 0

            # Brinson effects
            allocation = (wp - wb) * rb
            selection = wb * (rp - rb)
            interaction = (wp - wb) * (rp - rb)

            total_allocation += allocation
            total_selection += selection
            total_interaction += interaction

            results.append({
                'Sector': sector,
                'Port Weight': wp,
                'Bench Weight': wb,
                'Weight Diff': wp - wb,
                'Port Return': rp,
                'Bench Return': rb,
                'Allocation': allocation,
                'Selection': selection,
                'Interaction': interaction,
                'Total': allocation + selection + interaction
            })

        df = pd.DataFrame(results).set_index('Sector')

        return {
            'detailed': df.sort_values('Total', ascending=False),
            'summary': {
                'allocation_effect': total_allocation,
                'selection_effect': total_selection,
                'interaction_effect': total_interaction,
                'total_active_return': total_allocation + total_selection + total_interaction
            }
        }

    def rolling_attribution(self, window: int = 60) -> pd.DataFrame:
        """
        Calculate rolling attribution effects.

        Args:
            window: Rolling window size

        Returns:
            DataFrame with rolling attribution
        """
        port_sector_returns = self.sector_attr.get_sector_returns()
        port_sector_weights = self.sector_attr.get_sector_weights()
        bench_weights = pd.Series(self.benchmark_weights)

        results = []

        for i in range(window, len(port_sector_returns)):
            period_returns = port_sector_returns.iloc[i-window:i]

            allocation = 0
            selection = 0
            interaction = 0

            for sector in port_sector_weights.index:
                wp = port_sector_weights.get(sector, 0)
                wb = bench_weights.get(sector, 0)

                if sector in period_returns.columns:
                    rp = period_returns[sector].mean() * 252
                    rb = self.benchmark_returns[sector].iloc[i-window:i].mean() * 252 \
                         if sector in self.benchmark_returns.columns else rp * 0.95
                else:
                    continue

                allocation += (wp - wb) * rb
                selection += wb * (rp - rb)
                interaction += (wp - wb) * (rp - rb)

            results.append({
                'date': port_sector_returns.index[i],
                'allocation': allocation,
                'selection': selection,
                'interaction': interaction,
                'total': allocation + selection + interaction
            })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.set_index('date')

        return df

    def summary(self) -> str:
        """Generate text summary of attribution."""
        attr = self.calculate_attribution()
        summary = attr['summary']

        lines = [
            "Brinson Attribution Analysis",
            "=" * 40,
            "",
            f"Allocation Effect:  {summary['allocation_effect']*100:+.2f}%",
            f"Selection Effect:   {summary['selection_effect']*100:+.2f}%",
            f"Interaction Effect: {summary['interaction_effect']*100:+.2f}%",
            "-" * 40,
            f"Total Active Return: {summary['total_active_return']*100:+.2f}%",
            "",
            "Top Contributing Sectors:",
            "-" * 30
        ]

        detailed = attr['detailed']
        for sector in detailed.head(5).index:
            total = detailed.loc[sector, 'Total']
            lines.append(f"  {sector}: {total*100:+.2f}%")

        return "\n".join(lines)


def calculate_sector_attribution(
    returns: pd.DataFrame,
    weights: pd.Series,
    sector_map: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Convenience function for sector attribution.

    Args:
        returns: Asset returns
        weights: Portfolio weights
        sector_map: Optional sector mappings

    Returns:
        DataFrame with sector contributions
    """
    attr = SectorAttribution(returns, weights, sector_map)
    return attr.sector_contribution()


def run_brinson_attribution(
    returns: pd.DataFrame,
    weights: pd.Series,
    benchmark_weights: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Convenience function for Brinson attribution.

    Args:
        returns: Asset returns
        weights: Portfolio weights
        benchmark_weights: Optional benchmark weights

    Returns:
        Attribution results dictionary
    """
    attr = BrinsonAttribution(returns, weights, benchmark_weights=benchmark_weights)
    return attr.calculate_attribution()
