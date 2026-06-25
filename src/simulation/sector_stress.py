"""
Sector stress testing engine.

Orchestrates SectorBetaAnalyzer, DCCGARCHModel, StudentTCopula, and
MarketRegimeDetector to produce per-holding stress P&L estimates under
user-defined (or pre-defined) sector shock scenarios.

Typical usage::

    engine = SectorStressEngine()
    engine.fit(returns_df, sector_map)
    result = engine.run_stress(DEFAULT_SCENARIOS[0], holdings)
    print(result.to_dataframe())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.risk.copula import CopulaConfig, CopulaResult, StudentTCopula
from src.risk.dcc_garch import DCCGARCHConfig, DCCGARCHResult, DCCGARCHModel
from src.risk.regime_detection import RegimeConfig, RegimeResult, MarketRegimeDetector
from src.risk.sector_beta import (
    SectorBetaConfig,
    SectorBetaResult,
    SectorBetaAnalyzer,
)
from src.risk.stock_sector_beta import (
    StockBetaResult,
    compute_all_stock_betas,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Scenario definitions
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SectorStressScenario:
    """Definition of a sector-level stress scenario."""

    name: str
    description: str
    shocked_sectors: dict  # {sector_label: shock_return (decimal, e.g. -0.20)}
    use_copula: bool = field(default=True)
    use_dcc: bool = field(default=True)
    use_regime: bool = field(default=True)
    use_regime_correlation: bool = field(default=True)
    copula_shock_quantile: float = field(default=0.05)
    # copula_shock_quantile: CDF percentile used as the conditioning anchor.
    # Values ≤ 0.5 represent negative (loss) tails; > 0.5 represent positive.
    beta_window: Optional[str] = field(default=None)
    # beta_window: "short" | "long" | "average" | None (None = use SectorStressConfig default)


DEFAULT_SCENARIOS: list[SectorStressScenario] = [
    SectorStressScenario(
        name="Tech Selloff",
        description=(
            "Sharp correction in Technology driven by rate re-pricing or earnings miss, "
            "dragging Telecommunications along with it."
        ),
        shocked_sectors={"Technology": -0.20, "Telecommunications Services": -0.10},
        copula_shock_quantile=0.05,
    ),
    SectorStressScenario(
        name="Commodity Crash",
        description=(
            "Global demand shock collapses commodity prices, "
            "hitting Energy and Basic Materials hardest."
        ),
        shocked_sectors={"Energy": -0.25, "Basic Materials": -0.20},
        copula_shock_quantile=0.05,
    ),
    SectorStressScenario(
        name="Rate Spike",
        description=(
            "Sudden 150 bps rate rise compresses bank net-interest margins "
            "and reprices commercial real estate."
        ),
        shocked_sectors={"Financials": -0.15, "Real Estate": -0.20},
        copula_shock_quantile=0.05,
    ),
    SectorStressScenario(
        name="EM Risk-Off",
        description=(
            "Capital flight from emerging markets drives currency weakness "
            "and broad equity sell-off across cyclical sectors."
        ),
        shocked_sectors={
            "Financials": -0.12,
            "Consumer Discretionary": -0.10,
            "Industrials": -0.08,
        },
        copula_shock_quantile=0.05,
    ),
    SectorStressScenario(
        name="Commodity Boom",
        description=(
            "Supply disruption triggers a commodity super-cycle, "
            "strongly benefiting Energy and Basic Materials exporters."
        ),
        shocked_sectors={"Energy": 0.20, "Basic Materials": 0.15},
        use_copula=True,
        copula_shock_quantile=0.95,  # upper tail for positive shocks
    ),
    SectorStressScenario(
        name="Financial Crisis",
        description=(
            "Systemic credit event triggers cascading failures across banking, "
            "property, and consumer-credit sectors."
        ),
        shocked_sectors={
            "Financials": -0.30,
            "Real Estate": -0.25,
            "Consumer Discretionary": -0.20,
        },
        copula_shock_quantile=0.02,
    ),
    SectorStressScenario(
        name="Full Market Crash",
        description=(
            "Simultaneous global macro shock affecting all major sectors "
            "(COVID-2020 style, or GFC-2008 style)."
        ),
        shocked_sectors={
            "Technology": -0.35,
            "Financials": -0.35,
            "Energy": -0.30,
            "Basic Materials": -0.30,
            "Industrials": -0.28,
            "Consumer Discretionary": -0.28,
            "Healthcare": -0.15,
            "Consumer Staples": -0.12,
        },
        copula_shock_quantile=0.01,
    ),
    SectorStressScenario(
        name="Healthcare Policy Shock",
        shocked_sectors={"Healthcare": -0.20},
        description=(
            "Drug pricing regulation, patent cliff, or pipeline failure. "
            "Defensive sector but acutely vulnerable to policy risk."
        ),
        use_copula=True,
        use_regime_correlation=True,
        beta_window="short"
    ),
    SectorStressScenario(
        name="Telco Margin Compression",
        shocked_sectors={"Telecommunication Services": -0.15},
        description=(
            "Spectrum auction cost spike, competitive tariff war, "
            "or infrastructure capex overrun. High IDX relevance: TLKM."
        ),
        use_copula=True,
        use_regime_correlation=True,
        beta_window="short"
    ),
    SectorStressScenario(
        name="Utility Rate Risk",
        shocked_sectors={"Utilities": -0.20},
        description=(
            "Rate spike reprices utilities as bond proxies. "
            "High IDX relevance: PGEO, PGAS, BREN. "
            "Typically a natural hedge vs. financials."
        ),
        use_copula=True,
        use_regime_correlation=True,
        beta_window="short"
    ),
    SectorStressScenario(
        name="FMCG Margin Squeeze",
        shocked_sectors={"Consumer Non-Cyclicals": -0.15},
        description=(
            "Input cost inflation erodes FMCG margins. "
            "IDX-specific driver: CPO price surge hits food manufacturers. "
            "Counter-intuitively negative for 'defensive' Indonesian consumer names."
        ),
        use_copula=True,
        use_regime_correlation=True,
        beta_window="short"
    ),
    SectorStressScenario(
        name="IDX Commodity + Currency Double Shock",
        shocked_sectors={
            "Energy": -0.20,
            "Basic Materials": -0.20,
            "Financials": -0.10,
            "Consumer Cyclicals": -0.08,
        },
        description=(
            "Rupiah weakens sharply while commodity prices fall — "
            "the worst-case IDX macro scenario. Coal, nickel, CPO all drop "
            "simultaneously as EM capital outflows pressure the currency. "
            "Consumer Cyclicals dragged down by weakening domestic purchasing power."
        ),
        use_copula=True,
        use_regime_correlation=True,
        beta_window="long"
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SectorStressConfig:
    """Top-level configuration for SectorStressEngine."""

    beta_config: SectorBetaConfig = field(default_factory=SectorBetaConfig)
    dcc_config: DCCGARCHConfig = field(default_factory=DCCGARCHConfig)
    copula_config: CopulaConfig = field(default_factory=CopulaConfig)
    regime_config: RegimeConfig = field(default_factory=RegimeConfig)
    portfolio_value: float = field(default=1_000_000.0)
    min_weight_threshold: float = field(default=0.001)
    # Tickers with |weight| below this are skipped in per-holding output.
    dcc_fail_action: str = field(default="warn")    # "warn" | "raise"
    copula_fail_action: str = field(default="warn") # "warn" | "raise"
    regime_fail_action: str = field(default="warn") # "warn" | "raise"
    beta_window: str = field(default="average")     # "short" | "long" | "average"


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HoldingStressResult:
    """Stress-test output for a single portfolio holding."""

    ticker: str
    sector: str
    weight: float
    beta_implied_return: float
    # Return implied by beta-matrix propagation of the sector shocks.
    copula_median_return: float
    # Median of the copula conditional simulation for this sector.
    copula_var_return: float
    # Quantile of the copula simulation at scenario.copula_shock_quantile.
    pnl_contribution_beta: float
    # weight × beta_implied_return × portfolio_value
    pnl_contribution_copula: float
    # weight × copula_var_return × portfolio_value
    role: str
    # "shocked"    — ticker's sector is directly in the scenario shock dict.
    # "propagated" — sector loses via beta (beta_implied_return < -0.5%).
    # "hedged"     — sector gains via beta (beta_implied_return > +0.5%).
    # "neutral"    — change below ±0.5% threshold.
    beta_stability: str
    # "stable" | "unstable" | "unknown"
    stock_beta: float = field(default=1.0)
    sector_etf: str = field(default="unknown")


@dataclass
class SectorStressResult:
    """Full output from SectorStressEngine.run_stress()."""

    scenario: SectorStressScenario
    holdings_results: list
    total_beta_pnl: float
    total_copula_pnl: float
    regime_at_shock: str
    regime_probability: float
    correlation_used: pd.DataFrame
    warnings: list
    computation_time_seconds: float
    beta_result: Optional[SectorBetaResult]
    dcc_result: Optional[DCCGARCHResult]
    copula_result: Optional[CopulaResult]
    regime_result: Optional[RegimeResult]

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert holdings_results to a tidy DataFrame.

        Sorted by absolute beta P&L contribution descending (largest movers first).
        """
        if not self.holdings_results:
            return pd.DataFrame(columns=[
                "ticker", "sector", "weight",
                "beta_implied_return", "copula_median_return", "copula_var_return",
                "pnl_contribution_beta", "pnl_contribution_copula",
                "role", "beta_stability", "stock_beta", "sector_etf",
            ])
        rows = [
            {
                "ticker": h.ticker,
                "sector": h.sector,
                "weight": h.weight,
                "beta_implied_return": h.beta_implied_return,
                "copula_median_return": h.copula_median_return,
                "copula_var_return": h.copula_var_return,
                "pnl_contribution_beta": h.pnl_contribution_beta,
                "pnl_contribution_copula": h.pnl_contribution_copula,
                "role": h.role,
                "beta_stability": h.beta_stability,
                "stock_beta": h.stock_beta,
                "sector_etf": h.sector_etf,
            }
            for h in self.holdings_results
        ]
        df = pd.DataFrame(rows)
        df = df.sort_values(
            "pnl_contribution_beta", key=abs, ascending=False
        ).reset_index(drop=True)
        return df


# ──────────────────────────────────────────────────────────────────────────────
# Main engine
# ──────────────────────────────────────────────────────────────────────────────

class SectorStressEngine:
    """
    Sector stress testing engine.

    Orchestrates four sub-models:

    * :class:`~src.risk.sector_beta.SectorBetaAnalyzer` — cross-sector beta
      propagation, dual-window stability.
    * :class:`~src.risk.dcc_garch.DCCGARCHModel` — time-varying correlation
      and conditional volatilities.
    * :class:`~src.risk.copula.StudentTCopula` — joint tail dependence via
      Student-t copula.
    * :class:`~src.risk.regime_detection.MarketRegimeDetector` — HMM market
      regime for regime-conditioned correlation selection.

    Sub-model failures during ``fit()`` are isolated: a failing sub-model
    emits a warning and downstream analysis falls back gracefully.

    Parameters
    ----------
    config : SectorStressConfig
        Nested configuration for all sub-models and engine-level parameters.
    """

    def __init__(self, config: SectorStressConfig = SectorStressConfig()) -> None:
        self._config = config
        self._beta_analyzer = SectorBetaAnalyzer(config.beta_config)
        self._dcc_model = DCCGARCHModel(config.dcc_config)
        self._copula_model = StudentTCopula(config.copula_config)
        self._regime_detector = MarketRegimeDetector(config.regime_config)

        self._beta_result: Optional[SectorBetaResult] = None
        self._dcc_result: Optional[DCCGARCHResult] = None
        self._copula_result: Optional[CopulaResult] = None
        self._regime_result: Optional[RegimeResult] = None
        self._sector_returns: Optional[pd.DataFrame] = None
        self._sector_map: dict[str, str] = {}
        self._fit_warnings: list[str] = []
        self._is_fitted: bool = False
        self._stock_betas: Optional[StockBetaResult] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        returns: pd.DataFrame,
        sector_map: dict[str, str],
        market_caps: Optional[dict[str, float]] = None,
    ) -> "SectorStressEngine":
        """
        Fit all four sub-models on historical return data.

        Sub-model failures are isolated: if a model raises, a warning is
        appended and fitting continues with the remaining models.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns, shape (T, N_tickers). Columns are ticker symbols.
        sector_map : dict[str, str]
            Mapping {ticker: sector_label}.
        market_caps : dict[str, float], optional
            {ticker: market_cap} for value-weighted sector aggregation.
            Required only when ``beta_config.aggregation_method == "value_weight"``.

        Returns
        -------
        SectorStressEngine
            Self, for method chaining.
        """
        t0 = time.perf_counter()
        self._fit_warnings = []
        self._sector_map = dict(sector_map)

        first_date = (
            returns.index[0].date()
            if hasattr(returns.index[0], "date")
            else str(returns.index[0])
        )
        last_date = (
            returns.index[-1].date()
            if hasattr(returns.index[-1], "date")
            else str(returns.index[-1])
        )
        logger.info(
            f"SectorStressEngine.fit(): tickers={len(returns.columns)}, "
            f"date_range={first_date}→{last_date}"
        )

        # ── Step 1: Sector beta ──────────────────────────────────────────────
        try:
            self._sector_returns = self._beta_analyzer.build_sector_returns(
                returns, sector_map, market_caps
            )
            self._beta_result = self._beta_analyzer.compute(
                returns, sector_map, market_caps
            )
            logger.info(
                f"  [1/4] Beta: {len(self._beta_result.sectors)} sectors, "
                f"{self._beta_result.n_unstable_pairs} unstable pairs"
            )
        except Exception as exc:
            msg = f"SectorBetaAnalyzer.compute() failed: {exc}"
            logger.error(msg)
            self._fit_warnings.append(msg)
            # Attempt to build sector_returns even if full beta compute failed
            if self._sector_returns is None:
                try:
                    self._sector_returns = self._beta_analyzer.build_sector_returns(
                        returns, sector_map, market_caps
                    )
                except Exception as inner_exc:
                    inner_msg = f"build_sector_returns() also failed: {inner_exc}"
                    logger.error(inner_msg)
                    self._fit_warnings.append(inner_msg)

        if self._sector_returns is None or self._sector_returns.empty:
            msg = "Could not build sector returns — stress testing unavailable."
            logger.error(msg)
            self._fit_warnings.append(msg)
            return self

        sr = self._sector_returns

        # ── Step 1b: Per-stock sector-relative betas (ETF OLS with circularity fix)
        # Isolated in its own try/except so a failure never kills steps 2-4.
        try:
            import traceback as _tb
            _start = (
                str(returns.index[0].date())
                if hasattr(returns.index[0], "date")
                else str(returns.index[0])
            )
            _end = (
                str(returns.index[-1].date())
                if hasattr(returns.index[-1], "date")
                else str(returns.index[-1])
            )
            logger.debug(
                f"  [1b] compute_all_stock_betas: "
                f"tickers={list(returns.columns)}, "
                f"date_range={_start}→{_end}, "
                f"sector_map={self._sector_map}"
            )
            self._stock_betas = compute_all_stock_betas(
                tickers=list(returns.columns),
                sector_map=self._sector_map,
                stock_returns=returns,
                start_date=_start,
                end_date=_end,
                min_observations=self._config.beta_config.min_observations,
            )
            _etf_estimated = sum(
                1 for e in self._stock_betas.entries.values()
                if e.source in ("sector_etf", "etf_ex_stock", "market_proxy")
            )
            logger.info(
                f"  [1b] Stock betas: {len(self._stock_betas.entries)} tickers, "
                f"{_etf_estimated} estimated from sector ETFs, "
                f"{self._stock_betas.n_fallbacks} fallbacks"
            )
        except Exception as _exc:
            _trace = _tb.format_exc()
            msg = f"compute_all_stock_betas() failed — all betas default to 1.0: {_exc}"
            logger.error(f"{msg}\n{_trace}")
            self._fit_warnings.append(msg)

        # ── Step 2: DCC-GARCH ────────────────────────────────────────────────
        try:
            self._dcc_result = self._dcc_model.fit(sr)
            logger.info(
                f"  [2/4] DCC-GARCH: alpha={self._dcc_result.dcc_alpha:.4f}, "
                f"beta={self._dcc_result.dcc_beta:.4f}, "
                f"converged={self._dcc_result.convergence_status.get('converged', False)}"
            )
        except Exception as exc:
            msg = f"DCCGARCHModel.fit() failed: {exc}"
            logger.error(msg)
            self._fit_warnings.append(msg)
            if self._config.dcc_fail_action == "raise":
                raise

        # ── Step 3: Student-t copula ─────────────────────────────────────────
        try:
            # Seed the copula correlation with DCC stress correlation if available
            corr_override = (
                self._dcc_result.stress_correlation
                if self._dcc_result is not None
                else None
            )
            self._copula_result = self._copula_model.fit(sr, correlation_matrix=corr_override)
            logger.info(
                f"  [3/4] Copula: type={self._copula_result.copula_type}, "
                f"df={self._copula_result.degrees_of_freedom:.2f}"
            )
        except Exception as exc:
            msg = f"StudentTCopula.fit() failed: {exc}"
            logger.error(msg)
            self._fit_warnings.append(msg)
            if self._config.copula_fail_action == "raise":
                raise

        # ── Step 4: HMM regime detection ─────────────────────────────────────
        try:
            self._regime_result = self._regime_detector.fit(sr)
            logger.info(
                f"  [4/4] Regime: current={self._regime_result.current_state_label} "
                f"(p={self._regime_result.current_state_probability * 100:.1f}%), "
                f"converged={self._regime_result.convergence_achieved}"
            )
        except Exception as exc:
            msg = f"MarketRegimeDetector.fit() failed: {exc}"
            logger.error(msg)
            self._fit_warnings.append(msg)
            if self._config.regime_fail_action == "raise":
                raise

        self._is_fitted = True
        elapsed = time.perf_counter() - t0
        logger.info(
            f"SectorStressEngine.fit() done in {elapsed:.2f}s — "
            f"{len(self._fit_warnings)} sub-model warnings"
        )
        return self

    def run_stress(
        self,
        scenario: SectorStressScenario,
        holdings: dict[str, float],
    ) -> SectorStressResult:
        """
        Run a single stress scenario against a portfolio.

        Parameters
        ----------
        scenario : SectorStressScenario
            Sector shock definition.
        holdings : dict[str, float]
            {ticker: portfolio_weight}. Weights should sum to ~1.0.

        Returns
        -------
        SectorStressResult
        """
        if not self._is_fitted:
            raise RuntimeError(
                "SectorStressEngine.run_stress(): call fit() before run_stress()."
            )

        t0 = time.perf_counter()
        run_warnings: list[str] = list(self._fit_warnings)

        logger.info(
            f"SectorStressEngine.run_stress(): scenario='{scenario.name}', "
            f"holdings={len(holdings)}"
        )

        # ── Determine which correlation matrix to surface in results ─────────
        correlation_used = self._select_correlation(scenario, run_warnings)

        # ── Current regime info ───────────────────────────────────────────────
        regime_label, regime_prob = self._get_current_regime()

        # ── Resolve scenario shocks against fitted sectors ────────────────────
        # When sector-to-sector OLS succeeded, use its sector list; otherwise
        # fall back to unique sectors from the holdings map so per-stock betas
        # still get a valid implied return even if contagion estimation failed.
        if self._beta_result is not None:
            sectors = self._beta_result.sectors
        else:
            sectors = list(set(self._sector_map.values()))
        matched_shocks: dict[str, float] = {
            sec: shock
            for sec, shock in scenario.shocked_sectors.items()
            if sec in sectors
        }
        unmatched = [s for s in scenario.shocked_sectors if s not in sectors]
        if unmatched:
            run_warnings.append(
                f"Scenario '{scenario.name}': sectors not found in fitted data "
                f"and skipped: {unmatched}. Fitted sectors: {sectors}."
            )
        if not matched_shocks and scenario.shocked_sectors:
            run_warnings.append(
                f"No shocked sectors matched — beta P&L will be zero for '{scenario.name}'."
            )

        # ── Beta-implied sector returns ───────────────────────────────────────
        _effective_beta_window = scenario.beta_window or self._config.beta_window
        implied_sector_returns = self._compute_beta_implied(matched_shocks, _effective_beta_window)

        # ── Unstable pair lookup (for per-holding stability flag) ─────────────
        unstable_pairs: set[frozenset] = set()
        if self._beta_result:
            for flag in self._beta_result.stability_flags:
                if flag.is_unstable:
                    unstable_pairs.add(frozenset({flag.sector_driver, flag.sector_responder}))

        # ── Copula conditional simulation ─────────────────────────────────────
        copula_sector_sim: Optional[pd.DataFrame] = None
        if scenario.use_copula and self._copula_result is not None and matched_shocks:
            copula_sector_sim = self._run_copula_simulation(scenario, matched_shocks, run_warnings)

        # ── Per-holding stress P&L ────────────────────────────────────────────
        holdings_results: list[HoldingStressResult] = []

        for ticker, weight in holdings.items():
            if abs(weight) < self._config.min_weight_threshold:
                continue

            ticker_sector = self._sector_map.get(ticker, "Unknown")

            sector_return = float(implied_sector_returns.get(ticker_sector, 0.0))
            _entry = (
                self._stock_betas.entries.get(ticker)
                if self._stock_betas is not None
                else None
            )
            if _entry is None:
                logger.warning(f"[STRESS] {ticker}: no beta entry found, using 1.0 fallback")
                stock_beta = 1.0
            else:
                stock_beta = _entry.beta
                if _entry.source == "fallback":
                    logger.warning(f"[STRESS] {ticker}: using fallback beta=1.0")
            beta_ret = stock_beta * sector_return
            logger.debug(
                f"[STRESS] {ticker}: sector_shock={sector_return:.4f}, "
                f"beta={stock_beta:.4f}, implied_return={beta_ret:.4f}, "
                f"source={_entry.source if _entry else 'none'}"
            )

            # Copula returns for this sector
            if (
                copula_sector_sim is not None
                and ticker_sector in copula_sector_sim.columns
            ):
                sim_col = copula_sector_sim[ticker_sector].dropna().values
                if len(sim_col) >= 30:
                    cop_median = float(np.median(sim_col))
                    cop_var = float(np.quantile(sim_col, scenario.copula_shock_quantile))
                else:
                    cop_median = 0.0
                    cop_var = 0.0
            else:
                cop_median = 0.0
                cop_var = 0.0

            pnl_beta = weight * beta_ret * self._config.portfolio_value
            pnl_copula = weight * cop_var * self._config.portfolio_value

            role = self._assign_role(ticker_sector, beta_ret, matched_shocks)
            beta_stability = self._get_stability(ticker_sector, matched_shocks, unstable_pairs)

            holdings_results.append(HoldingStressResult(
                ticker=ticker,
                sector=ticker_sector,
                weight=weight,
                beta_implied_return=beta_ret,
                copula_median_return=cop_median,
                copula_var_return=cop_var,
                pnl_contribution_beta=pnl_beta,
                pnl_contribution_copula=pnl_copula,
                role=role,
                beta_stability=beta_stability,
                stock_beta=stock_beta,
                sector_etf=(_entry.etf_proxy if _entry is not None else "unknown"),
            ))

        total_beta_pnl = sum(h.pnl_contribution_beta for h in holdings_results)
        total_copula_pnl = sum(h.pnl_contribution_copula for h in holdings_results)

        elapsed = time.perf_counter() - t0
        logger.info(
            f"run_stress('{scenario.name}') done in {elapsed:.3f}s — "
            f"beta_pnl={total_beta_pnl:.0f}, copula_pnl={total_copula_pnl:.0f}, "
            f"holdings={len(holdings_results)}, warnings={len(run_warnings)}"
        )

        return SectorStressResult(
            scenario=scenario,
            holdings_results=holdings_results,
            total_beta_pnl=total_beta_pnl,
            total_copula_pnl=total_copula_pnl,
            regime_at_shock=regime_label,
            regime_probability=regime_prob,
            correlation_used=correlation_used,
            warnings=run_warnings,
            computation_time_seconds=elapsed,
            beta_result=self._beta_result,
            dcc_result=self._dcc_result,
            copula_result=self._copula_result,
            regime_result=self._regime_result,
        )

    def run_all_scenarios(
        self,
        holdings: dict[str, float],
        scenarios: Optional[list[SectorStressScenario]] = None,
    ) -> list[SectorStressResult]:
        """
        Run all scenarios (default: DEFAULT_SCENARIOS) against a portfolio.

        Failed scenarios are skipped with an error log; they do not raise.

        Parameters
        ----------
        holdings : dict[str, float]
            {ticker: weight} mapping.
        scenarios : list[SectorStressScenario], optional
            Override the scenario list. Defaults to ``DEFAULT_SCENARIOS``.

        Returns
        -------
        list[SectorStressResult]
            One result per successfully completed scenario, in input order.
        """
        if scenarios is None:
            scenarios = DEFAULT_SCENARIOS

        results: list[SectorStressResult] = []
        for scenario in scenarios:
            try:
                results.append(self.run_stress(scenario, holdings))
            except Exception as exc:
                logger.error(
                    f"run_all_scenarios: scenario='{scenario.name}' failed "
                    f"and was skipped: {exc}"
                )
        return results

    def get_hedge_candidates(
        self,
        result: SectorStressResult,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Return holdings with the largest positive P&L contribution.

        These are natural hedges — they gain when the shock hits the rest of
        the portfolio.

        Parameters
        ----------
        result : SectorStressResult
        top_n : int
            Maximum number of holdings to return.

        Returns
        -------
        pd.DataFrame
            Columns: ticker, sector, weight, pnl_contribution_beta, role.
        """
        df = result.to_dataframe()
        if df.empty:
            return pd.DataFrame(
                columns=["ticker", "sector", "weight", "pnl_contribution_beta", "role"]
            )
        candidates = df[df["pnl_contribution_beta"] > 0].nlargest(
            top_n, "pnl_contribution_beta"
        )
        return candidates[
            ["ticker", "sector", "weight", "pnl_contribution_beta", "role"]
        ].reset_index(drop=True)

    def get_most_exposed(
        self,
        result: SectorStressResult,
        top_n: int = 5,
        use_copula: bool = False,
    ) -> pd.DataFrame:
        """
        Return holdings with the largest loss (most negative P&L contribution).

        Parameters
        ----------
        result : SectorStressResult
        top_n : int
            Maximum number of holdings to return.
        use_copula : bool
            If True, rank by copula VaR P&L; otherwise by beta P&L.

        Returns
        -------
        pd.DataFrame
            Columns: ticker, sector, weight, pnl, role.
        """
        df = result.to_dataframe()
        if df.empty:
            return pd.DataFrame(columns=["ticker", "sector", "weight", "pnl", "role"])

        pnl_col = "pnl_contribution_copula" if use_copula else "pnl_contribution_beta"
        losers = df[df[pnl_col] < 0].nsmallest(top_n, pnl_col)
        out = losers[["ticker", "sector", "weight", pnl_col, "role"]].copy()
        out = out.rename(columns={pnl_col: "pnl"}).reset_index(drop=True)
        return out

    def get_fit_summary(self) -> dict:
        """
        Return a dict summarising the fit status of all sub-models.

        Keys
        ----
        is_fitted, beta, dcc, copula, regime,
        n_sectors, current_regime, regime_probability, warnings.
        """
        return {
            "is_fitted": self._is_fitted,
            "beta": self._beta_result is not None,
            "dcc": self._dcc_result is not None,
            "copula": self._copula_result is not None,
            "regime": self._regime_result is not None,
            "n_sectors": (
                len(self._beta_result.sectors) if self._beta_result else 0
            ),
            "current_regime": (
                self._regime_result.current_state_label
                if self._regime_result
                else "unknown"
            ),
            "regime_probability": (
                self._regime_result.current_state_probability
                if self._regime_result
                else 0.0
            ),
            "warnings": list(self._fit_warnings),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _select_correlation(
        self,
        scenario: SectorStressScenario,
        warnings: list[str],
    ) -> pd.DataFrame:
        """
        Choose the correlation matrix surfaced in SectorStressResult.

        Priority:
        1. Regime-conditioned DCC (if both DCC and regime are fitted).
        2. DCC stress or calm correlation (depending on shock direction).
        3. Copula correlation matrix.
        4. Identity matrix (fallback).
        """
        if scenario.use_dcc and self._dcc_result is not None:
            if (scenario.use_regime or scenario.use_regime_correlation) and self._regime_result is not None:
                try:
                    return self._regime_detector.get_current_regime_correlation(
                        self._dcc_result, self._regime_result
                    )
                except Exception as exc:
                    warnings.append(
                        f"Regime-conditioned DCC correlation failed: {exc}. "
                        "Falling back to DCC stress/calm correlation."
                    )
            # Positive shock → calm-end correlation; negative → stress
            if scenario.copula_shock_quantile > 0.5:
                return self._dcc_result.calm_correlation
            return self._dcc_result.stress_correlation

        if scenario.use_copula and self._copula_result is not None:
            return self._copula_result.correlation_matrix

        if self._beta_result is not None:
            n = len(self._beta_result.sectors)
            return pd.DataFrame(
                np.eye(n),
                index=self._beta_result.sectors,
                columns=self._beta_result.sectors,
            )

        return pd.DataFrame()

    def _get_current_regime(self) -> tuple[str, float]:
        """Return (regime_label, probability) for the current market state."""
        if self._regime_result is not None:
            return (
                self._regime_result.current_state_label,
                self._regime_result.current_state_probability,
            )
        return ("unknown", 0.0)

    def _compute_beta_implied(
        self,
        matched_shocks: dict[str, float],
        beta_window: Optional[str] = None,
    ) -> pd.Series:
        """
        Propagate sector shocks via beta matrix.

        Uses ``SectorBetaAnalyzer.get_implied_returns`` with the window
        specified by ``beta_window`` (falls back to ``config.beta_window``).
        Returns an empty Series if beta_result is unavailable.
        """
        if self._beta_result is None:
            # No contagion matrix — use the direct sector shocks as implied returns.
            return pd.Series(matched_shocks) if matched_shocks else pd.Series(dtype=float)

        if not matched_shocks:
            return pd.Series(0.0, index=self._beta_result.sectors)

        use_window = beta_window if beta_window else self._config.beta_window
        return self._beta_analyzer.get_implied_returns(
            self._beta_result,
            shocked_sectors=matched_shocks,
            use_window=use_window,
        )

    def _run_copula_simulation(
        self,
        scenario: SectorStressScenario,
        matched_shocks: dict[str, float],
        warnings: list[str],
    ) -> Optional[pd.DataFrame]:
        """
        Run copula conditional simulation anchored on the most extreme shocked sector.

        Returns a DataFrame of conditional return draws (n_accepted_paths, N_sectors),
        or None if the simulation fails.
        """
        if self._copula_result is None or not matched_shocks:
            return None

        # Pick the most severely shocked sector as the conditioning anchor.
        # For loss scenarios (quantile ≤ 0.5) that means the most negative shock.
        # For gain scenarios (quantile > 0.5) it means the most positive shock.
        if scenario.copula_shock_quantile <= 0.5:
            anchor_sector = min(matched_shocks, key=lambda s: matched_shocks[s])
        else:
            anchor_sector = max(matched_shocks, key=lambda s: matched_shocks[s])

        try:
            sim_returns = self._copula_model.simulate_conditional(
                self._copula_result,
                shocked_sector=anchor_sector,
                shock_quantile=scenario.copula_shock_quantile,
            )
            return sim_returns
        except ValueError as exc:
            msg = f"Copula conditional simulation failed for '{anchor_sector}': {exc}"
            logger.warning(msg)
            warnings.append(msg)
            return None
        except Exception as exc:
            msg = f"Copula simulation unexpected error: {exc}"
            logger.error(msg)
            warnings.append(msg)
            return None

    def _assign_role(
        self,
        sector: str,
        beta_ret: float,
        matched_shocks: dict[str, float],
    ) -> str:
        """Assign a descriptive role to a holding under the scenario."""
        if sector in matched_shocks:
            return "shocked"
        if beta_ret < -0.005:
            return "propagated"
        if beta_ret > 0.005:
            return "hedged"
        return "neutral"

    def _get_stability(
        self,
        sector: str,
        matched_shocks: dict[str, float],
        unstable_pairs: set[frozenset],
    ) -> str:
        """Return beta stability status for a holding's sector."""
        if self._beta_result is None:
            return "unknown"
        for shocked_sec in matched_shocks:
            if frozenset({shocked_sec, sector}) in unstable_pairs:
                return "unstable"
        return "stable"


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test (run as: python src/simulation/sector_stress.py)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from src.utils.logger import setup_logger

    setup_logger()

    rng = np.random.default_rng(0)
    T, N_TICKERS = 700, 15
    dates = pd.date_range("2021-01-04", periods=T, freq="B")
    tickers = [f"T{i:02d}" for i in range(N_TICKERS)]
    raw_returns = pd.DataFrame(
        rng.normal(0.0, 0.01, (T, N_TICKERS)),
        index=dates,
        columns=tickers,
    )

    # 5 synthetic sectors, 3 tickers each
    sectors_list = ["Technology", "Financials", "Energy", "Basic Materials", "Industrials"]
    sector_map: dict[str, str] = {}
    for i, ticker in enumerate(tickers):
        sector_map[ticker] = sectors_list[i % len(sectors_list)]

    # Equal-weight portfolio across all tickers
    holdings = {t: 1.0 / N_TICKERS for t in tickers}

    engine = SectorStressEngine()
    engine.fit(raw_returns, sector_map)

    summary = engine.get_fit_summary()
    assert summary["is_fitted"], "Engine should be fitted"
    assert summary["beta"], "Beta model should be fitted"
    print(f"  fit_summary: {summary}")

    # ── run_stress ────────────────────────────────────────────────────────────
    tech_scenario = SectorStressScenario(
        name="Test Tech Selloff",
        description="Synthetic test",
        shocked_sectors={"Technology": -0.20, "Financials": -0.10},
        copula_shock_quantile=0.05,
    )
    result = engine.run_stress(tech_scenario, holdings)
    assert len(result.holdings_results) > 0, "Should have holding results"
    assert isinstance(result.total_beta_pnl, float), "total_beta_pnl must be float"
    print(f"  total_beta_pnl: {result.total_beta_pnl:,.0f}")
    print(f"  total_copula_pnl: {result.total_copula_pnl:,.0f}")
    print(f"  regime_at_shock: {result.regime_at_shock} (p={result.regime_probability:.1%})")
    print(f"  warnings: {len(result.warnings)}")

    # ── to_dataframe ─────────────────────────────────────────────────────────
    df = result.to_dataframe()
    assert not df.empty, "DataFrame should not be empty"
    assert list(df.columns) == [
        "ticker", "sector", "weight",
        "beta_implied_return", "copula_median_return", "copula_var_return",
        "pnl_contribution_beta", "pnl_contribution_copula",
        "role", "beta_stability", "stock_beta", "sector_etf",
    ], "DataFrame columns mismatch"
    role_values = set(df["role"].unique())
    assert role_values <= {"shocked", "propagated", "hedged", "neutral"}, \
        f"Unexpected roles: {role_values}"
    print(f"  to_dataframe(): shape={df.shape}, roles={role_values} ✓")

    # ── run_all_scenarios ─────────────────────────────────────────────────────
    all_results = engine.run_all_scenarios(holdings)
    assert len(all_results) == len(DEFAULT_SCENARIOS), \
        f"Expected {len(DEFAULT_SCENARIOS)} results, got {len(all_results)}"
    print(f"  run_all_scenarios(): {len(all_results)} scenarios completed ✓")

    # ── get_most_exposed / get_hedge_candidates ───────────────────────────────
    exposed = engine.get_most_exposed(result, top_n=3)
    assert "pnl" in exposed.columns, "get_most_exposed() should have 'pnl' column"
    if not exposed.empty:
        assert all(exposed["pnl"] < 0), "All exposed holdings should have negative PnL"
    print(f"  get_most_exposed(): {len(exposed)} rows ✓")

    hedges = engine.get_hedge_candidates(result, top_n=3)
    print(f"  get_hedge_candidates(): {len(hedges)} rows ✓")

    # ── DEFAULT_SCENARIOS sanity check ────────────────────────────────────────
    assert len(DEFAULT_SCENARIOS) == 12, f"Expected 12 default scenarios, got {len(DEFAULT_SCENARIOS)}"
    names = [s.name for s in DEFAULT_SCENARIOS]
    assert "Tech Selloff" in names
    assert "Full Market Crash" in names
    assert "Healthcare Policy Shock" in names
    assert "IDX Commodity + Currency Double Shock" in names
    print(f"  DEFAULT_SCENARIOS: {names} ✓")

    # ── Commodity Boom (positive shock, upper tail) ────────────────────────────
    boom = next(s for s in DEFAULT_SCENARIOS if s.name == "Commodity Boom")
    assert boom.copula_shock_quantile == 0.95, "Commodity Boom should use upper tail"
    boom_result = engine.run_stress(boom, holdings)
    print(f"  Commodity Boom beta_pnl={boom_result.total_beta_pnl:,.0f} ✓")

    # ── Unfit engine raises ───────────────────────────────────────────────────
    fresh_engine = SectorStressEngine()
    try:
        fresh_engine.run_stress(tech_scenario, holdings)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        print("  Unfit engine raises RuntimeError ✓")

    print("\n✓ [SectorStressEngine] smoke test passed\n")
    sys.exit(0)
