"""Historical stress scenarios and custom stress testing."""

from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import numpy as np
import pandas as pd

from ..utils.logger import get_logger
from src.simulation.historical_scenarios import (
    HistoricalStressor,
    HistoricalStressorConfig,
    HISTORICAL_SCENARIOS as NEW_HISTORICAL_SCENARIOS,
    HistoricalScenarioResult,
)

logger = get_logger(__name__)


# Historical stress scenarios with date ranges and characteristics
HISTORICAL_SCENARIOS = {
    '2008_financial_crisis': {
        'name': '2008 Financial Crisis',
        'start_date': '2008-09-01',
        'end_date': '2009-03-31',
        'description': 'Global financial crisis triggered by subprime mortgage collapse',
        'characteristics': {
            'equity_drop': -0.50,
            'volatility_spike': 3.0,
            'correlation_spike': 0.90,
        }
    },
    'covid_crash': {
        'name': 'COVID-19 Crash',
        'start_date': '2020-02-19',
        'end_date': '2020-03-23',
        'description': 'Market crash due to COVID-19 pandemic',
        'characteristics': {
            'equity_drop': -0.34,
            'volatility_spike': 4.0,
            'correlation_spike': 0.85,
        }
    },
    'dotcom_bubble': {
        'name': 'Dot-com Bubble',
        'start_date': '2000-03-10',
        'end_date': '2002-10-09',
        'description': 'Tech bubble burst and subsequent bear market',
        'characteristics': {
            'equity_drop': -0.49,
            'volatility_spike': 2.0,
            'correlation_spike': 0.70,
        }
    },
    'flash_crash': {
        'name': 'Flash Crash',
        'start_date': '2010-05-06',
        'end_date': '2010-05-06',
        'description': 'Sudden intraday market crash',
        'characteristics': {
            'equity_drop': -0.09,
            'volatility_spike': 5.0,
            'correlation_spike': 0.95,
        }
    },
    'taper_tantrum': {
        'name': 'Taper Tantrum',
        'start_date': '2013-05-22',
        'end_date': '2013-06-24',
        'description': 'Market reaction to Fed tapering announcement',
        'characteristics': {
            'equity_drop': -0.06,
            'volatility_spike': 1.5,
            'correlation_spike': 0.65,
        }
    },
    'china_slowdown': {
        'name': 'China Slowdown',
        'start_date': '2015-08-11',
        'end_date': '2015-08-25',
        'description': 'Market turmoil from China growth concerns',
        'characteristics': {
            'equity_drop': -0.11,
            'volatility_spike': 2.5,
            'correlation_spike': 0.80,
        }
    },
    '2022_bear_market': {
        'name': '2022 Bear Market',
        'start_date': '2022-01-03',
        'end_date': '2022-10-12',
        'description': 'Bear market due to inflation and rate hikes',
        'characteristics': {
            'equity_drop': -0.25,
            'volatility_spike': 1.8,
            'correlation_spike': 0.75,
        }
    },
    'black_monday': {
        'name': 'Black Monday 1987',
        'start_date': '1987-10-14',
        'end_date': '1987-10-19',
        'description': 'Single largest one-day percentage decline',
        'characteristics': {
            'equity_drop': -0.22,
            'volatility_spike': 6.0,
            'correlation_spike': 0.95,
        }
    },
}


class StressTestScenario:
    """
    Represents a stress test scenario.
    """

    def __init__(
        self,
        name: str,
        shocks: Dict[str, float],
        description: str = '',
        correlation_adjustment: float = None,
        volatility_multiplier: float = None,
    ):
        """
        Initialize stress scenario.

        Args:
            name: Scenario name
            shocks: Dictionary of asset/sector shocks
            description: Scenario description
            correlation_adjustment: Correlation multiplier during stress
            volatility_multiplier: Volatility multiplier during stress
        """
        self.name = name
        self.shocks = shocks
        self.description = description
        self.correlation_adjustment = correlation_adjustment
        self.volatility_multiplier = volatility_multiplier

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'shocks': self.shocks,
            'description': self.description,
            'correlation_adjustment': self.correlation_adjustment,
            'volatility_multiplier': self.volatility_multiplier,
        }


class StressTester:
    """
    Perform stress testing on portfolios.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray = None,
        portfolio_value: float = 10000,
    ):
        """
        Initialize stress tester.

        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights
            portfolio_value: Total portfolio value
        """
        self.returns = returns
        self.tickers = list(returns.columns)
        self.n_assets = len(self.tickers)
        self.weights = weights if weights is not None else np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = portfolio_value

        logger.info(f"StressTester initialized with {self.n_assets} assets")

    def historical_scenario(
        self,
        scenario_key: str,
        historical_data: pd.DataFrame = None,
    ) -> Dict:
        """
        Apply historical stress scenario.

        Args:
            scenario_key: Key from HISTORICAL_SCENARIOS
            historical_data: Historical price/return data covering scenario period

        Returns:
            Stress test results
        """
        if scenario_key not in HISTORICAL_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_key}")

        scenario = HISTORICAL_SCENARIOS[scenario_key]

        if historical_data is not None:
            # Use actual historical data
            start = pd.to_datetime(scenario['start_date'])
            end = pd.to_datetime(scenario['end_date'])

            mask = (historical_data.index >= start) & (historical_data.index <= end)
            scenario_data = historical_data[mask]

            if len(scenario_data) > 0:
                scenario_returns = scenario_data.pct_change().dropna()
                cumulative_return = (1 + scenario_returns).prod() - 1
                portfolio_return = (cumulative_return * self.weights).sum()
            else:
                # Fall back to characteristics
                portfolio_return = scenario['characteristics']['equity_drop']
        else:
            # Use scenario characteristics
            portfolio_return = scenario['characteristics']['equity_drop']

        # Calculate impact
        portfolio_loss = self.portfolio_value * portfolio_return

        return {
            'scenario': scenario['name'],
            'description': scenario['description'],
            'period': f"{scenario['start_date']} to {scenario['end_date']}",
            'portfolio_return': portfolio_return,
            'portfolio_loss': portfolio_loss,
            'ending_value': self.portfolio_value + portfolio_loss,
            'characteristics': scenario['characteristics'],
        }

    def custom_scenario(
        self,
        scenario: Union[StressTestScenario, Dict],
        asset_mapping: Dict[str, str] = None,
    ) -> Dict:
        """
        Apply custom stress scenario.

        Args:
            scenario: StressTestScenario or dictionary with shocks
            asset_mapping: Map assets to shock categories (e.g., {'AAPL': 'tech'})

        Returns:
            Stress test results
        """
        if isinstance(scenario, dict):
            scenario = StressTestScenario(**scenario)

        # Calculate asset-level impacts
        asset_impacts = {}

        for ticker in self.tickers:
            # Check if ticker is directly in shocks
            if ticker in scenario.shocks:
                asset_impacts[ticker] = scenario.shocks[ticker]
            # Check asset mapping
            elif asset_mapping and ticker in asset_mapping:
                category = asset_mapping[ticker]
                if category in scenario.shocks:
                    asset_impacts[ticker] = scenario.shocks[category]
                else:
                    asset_impacts[ticker] = 0
            # Default to market shock if available
            elif 'market' in scenario.shocks:
                asset_impacts[ticker] = scenario.shocks['market']
            elif 'equity' in scenario.shocks:
                asset_impacts[ticker] = scenario.shocks['equity']
            else:
                asset_impacts[ticker] = 0

        # Calculate portfolio impact
        impacts = np.array([asset_impacts[t] for t in self.tickers])
        portfolio_return = np.dot(self.weights, impacts)
        portfolio_loss = self.portfolio_value * portfolio_return

        return {
            'scenario': scenario.name,
            'description': scenario.description,
            'asset_impacts': asset_impacts,
            'portfolio_return': portfolio_return,
            'portfolio_loss': portfolio_loss,
            'ending_value': self.portfolio_value + portfolio_loss,
        }

    def run_all_historical(
        self,
        historical_data: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Run all historical scenarios.

        Args:
            historical_data: Historical price data

        Returns:
            DataFrame with all scenario results
        """
        results = []

        for scenario_key in HISTORICAL_SCENARIOS.keys():
            try:
                result = self.historical_scenario(scenario_key, historical_data)
                results.append({
                    'Scenario': result['scenario'],
                    'Period': result['period'],
                    'Portfolio Return': result['portfolio_return'],
                    'Portfolio Loss': result['portfolio_loss'],
                    'Ending Value': result['ending_value'],
                })
            except Exception as e:
                logger.warning(f"Failed to run scenario {scenario_key}: {e}")

        return pd.DataFrame(results)

    def run_historical_actual(
        self,
        tickers: list,
        weights: pd.Series,
        portfolio_value: float,
        scenarios: list = None,
        config: HistoricalStressorConfig = None,
    ) -> dict:
        """
        Run historical stress scenarios using actual per-stock returns
        instead of uniform hardcoded shocks.

        Parameters
        ----------
        tickers : list[str]
        weights : pd.Series
            Index = tickers, values = decimal weights summing to 1.
        portfolio_value : float
        scenarios : list[HistoricalScenario], optional
            If None, uses HISTORICAL_SCENARIOS from historical_scenarios.py.
        config : HistoricalStressorConfig, optional
            If None, uses defaults.

        Returns
        -------
        dict[str, HistoricalScenarioResult]
            Keyed by scenario name.
        """
        stressor = HistoricalStressor(config or HistoricalStressorConfig())
        return stressor.run_all(tickers, weights, portfolio_value, scenarios)

    def parametric_stress(
        self,
        equity_shock: float = -0.20,
        volatility_multiplier: float = 2.0,
        correlation_adjustment: float = 0.90,
    ) -> Dict:
        """
        Apply parametric stress test.

        Args:
            equity_shock: Percentage shock to apply
            volatility_multiplier: Multiply volatility by this factor
            correlation_adjustment: Set all correlations to this level

        Returns:
            Stress test results
        """
        # Simple implementation - apply uniform shock
        portfolio_return = equity_shock
        portfolio_loss = self.portfolio_value * portfolio_return

        # Calculate stressed volatility
        base_vol = self.returns.std() * np.sqrt(252)
        stressed_vol = base_vol * volatility_multiplier

        # Calculate stressed correlation
        n = self.n_assets
        stressed_corr = np.ones((n, n)) * correlation_adjustment
        np.fill_diagonal(stressed_corr, 1.0)

        # Stressed covariance
        stressed_cov = np.outer(stressed_vol, stressed_vol) * stressed_corr
        stressed_port_vol = np.sqrt(np.dot(self.weights.T, np.dot(stressed_cov, self.weights)))

        return {
            'scenario': 'Parametric Stress',
            'equity_shock': equity_shock,
            'portfolio_return': portfolio_return,
            'portfolio_loss': portfolio_loss,
            'ending_value': self.portfolio_value + portfolio_loss,
            'base_volatility': float(np.dot(self.weights, base_vol)),
            'stressed_volatility': float(stressed_port_vol),
            'volatility_multiplier': volatility_multiplier,
            'correlation_adjustment': correlation_adjustment,
        }

    def sensitivity_analysis(
        self,
        shock_range: List[float] = None,
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis across shock ranges.

        Args:
            shock_range: List of shock levels to test

        Returns:
            DataFrame with sensitivity results
        """
        if shock_range is None:
            shock_range = [-0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.40, -0.50]

        results = []

        for shock in shock_range:
            portfolio_return = shock
            portfolio_loss = self.portfolio_value * portfolio_return

            results.append({
                'Shock': shock,
                'Portfolio Return': portfolio_return,
                'Portfolio Loss': portfolio_loss,
                'Ending Value': self.portfolio_value + portfolio_loss,
            })

        return pd.DataFrame(results)

    def reverse_stress_test(
        self,
        target_loss: float,
        asset_betas: Dict[str, float] = None,
    ) -> Dict:
        """
        Find the market shock that would cause target loss.

        Args:
            target_loss: Target portfolio loss (negative)
            asset_betas: Beta of each asset to market

        Returns:
            Required market shock
        """
        if asset_betas is None:
            # Assume beta of 1 for all assets
            asset_betas = {t: 1.0 for t in self.tickers}

        # Calculate portfolio beta
        betas = np.array([asset_betas.get(t, 1.0) for t in self.tickers])
        portfolio_beta = np.dot(self.weights, betas)

        # Required return to hit target loss
        required_return = target_loss / self.portfolio_value

        # Required market shock
        if portfolio_beta != 0:
            market_shock = required_return / portfolio_beta
        else:
            market_shock = required_return

        return {
            'target_loss': target_loss,
            'required_return': required_return,
            'portfolio_beta': portfolio_beta,
            'required_market_shock': market_shock,
        }


# Predefined custom scenarios
CUSTOM_SCENARIOS = {
    'market_crash': StressTestScenario(
        name='Market Crash',
        shocks={
            'equity': -0.30,
            'tech': -0.35,
            'finance': -0.40,
            'healthcare': -0.20,
            'utilities': -0.15,
        },
        description='Severe market crash with sector differentiation',
        correlation_adjustment=0.90,
        volatility_multiplier=3.0,
    ),
    'interest_rate_shock': StressTestScenario(
        name='Interest Rate Shock',
        shocks={
            'equity': -0.15,
            'tech': -0.20,
            'finance': -0.10,
            'utilities': -0.25,
            'real_estate': -0.30,
        },
        description='200 bps interest rate increase',
        correlation_adjustment=0.70,
        volatility_multiplier=1.8,
    ),
    'sector_rotation': StressTestScenario(
        name='Sector Rotation',
        shocks={
            'tech': -0.25,
            'growth': -0.20,
            'value': 0.10,
            'energy': 0.20,
            'finance': -0.10,
        },
        description='Rotation from growth to value stocks',
        correlation_adjustment=0.50,
        volatility_multiplier=1.5,
    ),
    'stagflation': StressTestScenario(
        name='Stagflation',
        shocks={
            'equity': -0.20,
            'tech': -0.25,
            'consumer': -0.30,
            'energy': 0.15,
            'materials': 0.10,
        },
        description='High inflation with low growth',
        correlation_adjustment=0.75,
        volatility_multiplier=2.0,
    ),
    'liquidity_crisis': StressTestScenario(
        name='Liquidity Crisis',
        shocks={
            'equity': -0.25,
            'small_cap': -0.35,
            'large_cap': -0.20,
            'finance': -0.40,
        },
        description='Credit markets freeze',
        correlation_adjustment=0.95,
        volatility_multiplier=4.0,
    ),
}


def get_scenario(name: str) -> StressTestScenario:
    """
    Get a predefined custom scenario.

    Args:
        name: Scenario name

    Returns:
        StressTestScenario object
    """
    if name in CUSTOM_SCENARIOS:
        return CUSTOM_SCENARIOS[name]
    raise ValueError(f"Unknown scenario: {name}. Available: {list(CUSTOM_SCENARIOS.keys())}")


def list_scenarios() -> Dict[str, str]:
    """
    List all available scenarios.

    Returns:
        Dictionary of scenario names and descriptions
    """
    scenarios = {}

    # Historical
    for key, scenario in HISTORICAL_SCENARIOS.items():
        scenarios[key] = f"[Historical] {scenario['description']}"

    # Custom
    for key, scenario in CUSTOM_SCENARIOS.items():
        scenarios[key] = f"[Custom] {scenario.description}"

    return scenarios
