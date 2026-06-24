"""
Macro stress engine for the Leontief contagion model.

Orchestrates MacroDataFetcher, MacroSensitivityEstimator,
LeontifContagionEngine, and SectorBetaAnalyzer into a single
callable interface for the Macro Contagion tab in Page 4.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.data.macro_data import (
    MacroDataConfig,
    MacroDataFetcher,
    MacroDataResult,
    MacroVariableConfig,
    DEFAULT_MACRO_VARIABLES,
)
from src.risk.contagion import (
    ContagionConfig,
    ContagionResult,
    IDRFeedbackParams,
    LeontifContagionEngine,
)
from src.risk.macro_sensitivity import (
    MacroSensitivityConfig,
    MacroSensitivityEstimator,
    SensitivityResult,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Optional dependency guards ────────────────────────────────────────────────

try:
    import joblib as _joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _joblib = None
    _JOBLIB_AVAILABLE = False

try:
    import networkx as _nx
    _NETWORKX_AVAILABLE = True
except ImportError:
    _nx = None
    _NETWORKX_AVAILABLE = False

try:
    import sqlite3 as _sqlite3
    _SQLITE3_AVAILABLE = True
except ImportError:
    _sqlite3 = None
    _SQLITE3_AVAILABLE = False

try:
    import fredapi as _fredapi
    _FREDAPI_AVAILABLE = True
except ImportError:
    _fredapi = None
    _FREDAPI_AVAILABLE = False


# ── Shock and scenario definitions ────────────────────────────────────────────

# Mapping from MacroShock field names to variable names
_SHOCK_FIELD_MAP = {
    "DXY": "dxy_pct",
    "VIX": "vix_delta",
    "US_10Y": "us_10y_bps",
    "BI_RATE": "bi_rate_bps",
    "IDR_USD": "idr_usd_pct",
    "CHINA_PMI": "china_pmi_delta",
    "CPO": "cpo_pct",
    "COAL": "coal_pct",
    "NICKEL": "nickel_pct",
}


@dataclass
class MacroShock:
    """
    Macro shock vector for the contagion model.

    All values in the same units as the transformed macro variables:
    - Rate variables (US_10Y, BI_RATE, VIX): point/bps-equivalent change.
    - FX/commodity (DXY, IDR_USD, CPO, COAL, NICKEL): decimal pct change.
    - CHINA_PMI: month-over-month PMI point change (e.g., -3.0 = PMI falls 3 pts).

    Default of 0.0 for all = no shock (baseline).
    """

    dxy_pct: float = field(default=0.0)
    vix_delta: float = field(default=0.0)
    us_10y_bps: float = field(default=0.0)
    bi_rate_bps: float = field(default=0.0)
    idr_usd_pct: float = field(default=0.0)
    china_pmi_delta: float = field(default=0.0)
    cpo_pct: float = field(default=0.0)
    coal_pct: float = field(default=0.0)
    nickel_pct: float = field(default=0.0)

    def to_series(self, variable_names: list[str]) -> pd.Series:
        """
        Convert to pd.Series aligned to variable_names order.

        Parameters
        ----------
        variable_names : list[str]
            Ordered list of macro variable names (e.g. ["DXY", "VIX", ...]).
            Variables not in this dataclass receive 0.0.

        Returns
        -------
        pd.Series
        """
        values = {}
        for var in variable_names:
            attr = _SHOCK_FIELD_MAP.get(var)
            if attr is not None:
                values[var] = float(getattr(self, attr, 0.0))
            else:
                values[var] = 0.0
        return pd.Series(values)


@dataclass
class MacroStressScenario:
    """
    A named macro stress scenario.

    Parameters
    ----------
    name : str
        Short identifier.
    shock : MacroShock
        Shock magnitudes for each macro variable.
    description : str
        Human-readable description.
    tags : list[str]
        Classification tags (e.g. ["rates", "FX", "idxrelevant"]).
    historical_reference : str
        Reference period (documentation only, not used computationally).
    """

    name: str
    shock: MacroShock
    description: str
    tags: list[str] = field(default_factory=list)
    historical_reference: str = field(default="")


DEFAULT_MACRO_SCENARIOS: list[MacroStressScenario] = [
    MacroStressScenario(
        name="BI Tightening Cycle",
        shock=MacroShock(bi_rate_bps=50, idr_usd_pct=0.02),
        description="BI hikes 50bps, mild IDR adjustment",
        tags=["rates", "domestic"],
        historical_reference="2022-2023 BI hiking cycle",
    ),
    MacroStressScenario(
        name="Taper Tantrum Replay",
        shock=MacroShock(
            us_10y_bps=100,
            bi_rate_bps=175,
            idr_usd_pct=0.18,
            dxy_pct=0.05,
            vix_delta=15,
        ),
        description="Fed taper signal: BI forced to hike hard to defend IDR",
        tags=["rates", "FX", "EM", "idxrelevant"],
        historical_reference="2013 Taper Tantrum",
    ),
    MacroStressScenario(
        name="Dollar Surge + Commodity Crash",
        shock=MacroShock(
            dxy_pct=0.08,
            idr_usd_pct=0.12,
            cpo_pct=-0.25,
            coal_pct=-0.30,
            nickel_pct=-0.20,
            vix_delta=20,
        ),
        description="Strong USD crushes commodity prices — worst-case IDX scenario",
        tags=["FX", "commodity", "EM", "idxrelevant"],
        historical_reference="2015 commodity supercycle end",
    ),
    MacroStressScenario(
        name="China Demand Shock",
        shock=MacroShock(china_pmi_delta=-3.0, cpo_pct=-0.15, nickel_pct=-0.25),
        description="Chinese factory activity contracts — commodity demand collapse",
        tags=["commodity", "China", "idxrelevant"],
        historical_reference="2015 China slowdown",
    ),
    MacroStressScenario(
        name="Global Risk-Off",
        shock=MacroShock(
            vix_delta=30,
            dxy_pct=0.06,
            us_10y_bps=-50,
            idr_usd_pct=0.10,
        ),
        description="VIX spike, flight to dollar safety — EM outflows",
        tags=["risk-off", "EM"],
        historical_reference="COVID-19 March 2020",
    ),
    MacroStressScenario(
        name="Commodity Boom",
        shock=MacroShock(cpo_pct=0.30, coal_pct=0.25, nickel_pct=0.40),
        description="Commodity supercycle — upside scenario for IDX",
        tags=["commodity", "upside", "idxrelevant"],
    ),
]


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class MacroStressConfig:
    """
    Top-level configuration for MacroStressEngine.

    Parameters
    ----------
    macro_data_config : MacroDataConfig
    sensitivity_config : MacroSensitivityConfig
    contagion_config : ContagionConfig
    reestimate_if_older_days : int
        Re-estimate if cached models are older than this. Default 30.
    show_cascade_warnings : bool
        Surface cascade warnings in results. Default True.
    min_sector_distress_to_report : float
        Minimum |impact| to report for a sector. Default 0.005 (0.5%).
    portfolio_value : float
        Portfolio value in USD for P&L calculations. Default 1,000,000.
    """

    macro_data_config: MacroDataConfig = field(default_factory=MacroDataConfig)
    sensitivity_config: MacroSensitivityConfig = field(
        default_factory=MacroSensitivityConfig
    )
    contagion_config: ContagionConfig = field(default_factory=ContagionConfig)
    reestimate_if_older_days: int = field(default=30)
    show_cascade_warnings: bool = field(default=True)
    min_sector_distress_to_report: float = field(default=0.005)
    portfolio_value: float = field(default=1_000_000.0)


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class HoldingContagionResult:
    """
    Contagion stress result for a single portfolio holding.

    Attributes
    ----------
    ticker : str
    sector : str
    weight : float
    direct_return : float
        First-order macro impact = S · shock for this sector (signed decimal).
    total_return : float
        Leontief-amplified total impact (I-W)^{-1} · h_initial for this sector.
    pnl_direct : float
        weight × direct_return × portfolio_value.
    pnl_total : float
        weight × total_return × portfolio_value.
    amplification : float
        total_return / direct_return. 1.0 if direct_return ≈ 0.
    """

    ticker: str
    sector: str
    weight: float
    direct_return: float
    total_return: float
    pnl_direct: float
    pnl_total: float
    amplification: float


@dataclass
class MacroStressResult:
    """
    Full output of MacroStressEngine.run_stress().

    Attributes
    ----------
    scenario : MacroStressScenario
    h_initial : pd.Series
        First-order macro impacts per sector.
    contagion : ContagionResult
        Full contagion propagation result.
    holding_results : list[HoldingContagionResult]
    total_pnl_direct : float
    total_pnl_total : float
    cascade_risk : str
    spectral_radius : float
    multiplier_table : pd.DataFrame
    systemic_importance : pd.Series
    warnings : list[str]
    """

    scenario: MacroStressScenario
    h_initial: pd.Series
    contagion: ContagionResult
    holding_results: list[HoldingContagionResult]
    total_pnl_direct: float
    total_pnl_total: float
    cascade_risk: str
    spectral_radius: float
    multiplier_table: pd.DataFrame
    systemic_importance: pd.Series
    warnings: list[str]

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flatten holding_results to a display-ready DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ticker, sector, weight, direct_return, total_return,
            pnl_direct, pnl_total, amplification.
            Sorted by |pnl_total| descending.
        """
        if not self.holding_results:
            return pd.DataFrame(
                columns=[
                    "ticker", "sector", "weight",
                    "direct_return", "total_return",
                    "pnl_direct", "pnl_total", "amplification",
                ]
            )
        rows = [
            {
                "ticker": h.ticker,
                "sector": h.sector,
                "weight": h.weight,
                "direct_return": h.direct_return,
                "total_return": h.total_return,
                "pnl_direct": h.pnl_direct,
                "pnl_total": h.pnl_total,
                "amplification": h.amplification,
            }
            for h in self.holding_results
        ]
        df = pd.DataFrame(rows)
        df = df.sort_values("pnl_total", key=abs, ascending=False).reset_index(drop=True)
        return df


# ── Main engine ───────────────────────────────────────────────────────────────

class MacroStressEngine:
    """
    End-to-end macro stress engine for the Leontief contagion model.

    Orchestrates data fetching, macro sensitivity estimation, beta
    matrix construction, and contagion propagation.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns, shape (T, N_tickers). Columns are ticker symbols.
    sector_map : dict[str, str]
        {ticker: sector_label}.
    config : MacroStressConfig
        Nested configuration for all sub-components.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        sector_map: dict[str, str],
        config: MacroStressConfig = MacroStressConfig(),
    ) -> None:
        self._returns = returns
        self._sector_map = dict(sector_map)
        self._config = config

        self._contagion_engine = LeontifContagionEngine(config.contagion_config)
        self._sensitivity_estimator = MacroSensitivityEstimator(config.sensitivity_config)

        self._macro_data: Optional[MacroDataResult] = None
        self._sensitivity: Optional[SensitivityResult] = None
        self._W: Optional[pd.DataFrame] = None
        self._idr_params: Optional[IDRFeedbackParams] = None
        self._is_fitted: bool = False
        self._fit_date: str = ""

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, force_reestimate: bool = False) -> None:
        """
        Fit all sub-components. Must be called before run_stress().

        Steps
        -----
        1. Check if fitted models exist in cache and are fresh enough.
           If fresh (< config.reestimate_if_older_days): load and return.
        2. Fetch macro data via MacroDataFetcher.
        3. Build sector return series via SectorBetaAnalyzer.
        4. Estimate sensitivity matrix S via MacroSensitivityEstimator
           (tries SQLite cache first).
        5. Fetch IDR return series from macro data.
        6. Build beta matrix via SectorBetaAnalyzer.compute().
        7. Build and normalize W via LeontifContagionEngine.build_weight_matrix().
        8. Estimate IDR feedback via LeontifContagionEngine.estimate_idr_feedback().
        9. Serialize everything via engine.save_fitted().
        10. Log total fit time and spectral_radius.

        Parameters
        ----------
        force_reestimate : bool
            If True, bypass all caches and re-estimate from scratch.
        """
        t0 = time.time()

        # Step 1: Try loading from cache
        if not force_reestimate:
            cache_path = os.path.join(
                self._config.contagion_config.cache_dir, "contagion_fitted.joblib"
            )
            if os.path.exists(cache_path):
                try:
                    mtime = os.path.getmtime(cache_path)
                    age_days = (time.time() - mtime) / 86400.0
                    if age_days <= self._config.reestimate_if_older_days:
                        cached = self._contagion_engine.load_fitted()
                        s_cached = self._sensitivity_estimator.load_from_db(
                            max_age_days=self._config.reestimate_if_older_days
                        )
                        if cached is not None and s_cached is not None:
                            self._W, self._idr_params = cached
                            self._sensitivity = s_cached
                            self._is_fitted = True
                            self._fit_date = datetime.now().isoformat()
                            logger.info(
                                f"MacroStressEngine: loaded from cache "
                                f"({age_days:.1f} days old)"
                            )
                            return
                except Exception as exc:
                    logger.debug(f"Cache load failed, re-estimating: {exc}")

        # Step 2: Fetch macro data
        logger.info("MacroStressEngine.fit(): fetching macro data")
        fetcher = MacroDataFetcher(self._config.macro_data_config)
        try:
            self._macro_data = fetcher.fetch()
        except Exception as exc:
            logger.error(f"MacroDataFetcher.fetch() failed: {exc}")
            raise RuntimeError(f"MacroStressEngine.fit() failed at macro data fetch: {exc}") from exc

        # Step 3: Build sector returns
        from src.risk.sector_beta import SectorBetaAnalyzer, SectorBetaConfig
        beta_analyzer = SectorBetaAnalyzer(SectorBetaConfig())
        try:
            sector_returns = beta_analyzer.build_sector_returns(
                self._returns, self._sector_map
            )
        except Exception as exc:
            logger.error(f"build_sector_returns() failed: {exc}")
            raise RuntimeError(f"MacroStressEngine.fit() failed at sector returns: {exc}") from exc

        if sector_returns.empty:
            raise RuntimeError(
                "MacroStressEngine.fit(): sector_returns is empty. "
                "Check that returns and sector_map share common tickers."
            )

        # Step 4: Estimate S matrix (try SQLite cache first)
        macro_weekly = self._macro_data.aligned_weekly
        if not force_reestimate:
            s_cached = self._sensitivity_estimator.load_from_db(
                max_age_days=self._config.reestimate_if_older_days
            )
        else:
            s_cached = None

        if s_cached is not None:
            self._sensitivity = s_cached
            logger.info("MacroStressEngine.fit(): loaded S matrix from SQLite cache")
        else:
            logger.info("MacroStressEngine.fit(): estimating S matrix")
            try:
                self._sensitivity = self._sensitivity_estimator.estimate(
                    sector_returns, macro_weekly
                )
            except Exception as exc:
                logger.error(f"MacroSensitivityEstimator.estimate() failed: {exc}")
                raise RuntimeError(
                    f"MacroStressEngine.fit() failed at S estimation: {exc}"
                ) from exc

        # Step 5: IDR return series
        idr_col = "IDR_USD"
        if idr_col in macro_weekly.columns:
            idr_returns = macro_weekly[idr_col].dropna()
        else:
            logger.warning("IDR_USD not found in macro data — IDR feedback disabled")
            idr_returns = pd.Series(dtype=float)

        # Step 6: Build sector beta matrix
        logger.info("MacroStressEngine.fit(): computing sector beta matrix")
        try:
            beta_result = beta_analyzer.compute(self._returns, self._sector_map)
        except Exception as exc:
            logger.error(f"SectorBetaAnalyzer.compute() failed: {exc}")
            raise RuntimeError(
                f"MacroStressEngine.fit() failed at beta computation: {exc}"
            ) from exc

        # Step 7: Build W
        try:
            self._W, spectral_radius = self._contagion_engine.build_weight_matrix(
                beta_result.beta_matrix_short
            )
        except Exception as exc:
            logger.error(f"build_weight_matrix() failed: {exc}")
            raise RuntimeError(
                f"MacroStressEngine.fit() failed at W construction: {exc}"
            ) from exc

        # Step 8: IDR feedback
        idr_sensitivity = self._sensitivity.S.get(
            idr_col,
            pd.Series(0.0, index=self._sensitivity.sector_names),
        )
        try:
            self._idr_params = self._contagion_engine.estimate_idr_feedback(
                sector_returns, idr_returns, idr_sensitivity
            )
        except Exception as exc:
            logger.warning(f"estimate_idr_feedback() failed: {exc} — using zero coefficients")
            self._idr_params = IDRFeedbackParams(
                equity_to_idr_coef=0.0,
                idr_to_sector=idr_sensitivity,
                r_squared=0.0,
                n_observations=0,
                estimation_date=datetime.now().isoformat(),
            )

        # Step 9: Serialize
        try:
            self._contagion_engine.save_fitted(self._W, self._idr_params)
        except Exception as exc:
            logger.warning(f"save_fitted() failed (non-fatal): {exc}")

        self._is_fitted = True
        self._fit_date = datetime.now().isoformat()
        elapsed = time.time() - t0
        logger.info(
            f"MacroStressEngine.fit() done in {elapsed:.2f}s — "
            f"spectral_radius={spectral_radius:.4f}, "
            f"sectors={len(self._sensitivity.sector_names)}, "
            f"macro_vars={len(self._sensitivity.macro_variable_names)}"
        )

    def run_stress(
        self,
        scenario: MacroStressScenario,
        weights: pd.Series,
    ) -> MacroStressResult:
        """
        Run one macro stress scenario end-to-end.

        Steps
        -----
        1. Validate self._is_fitted.
        2. Convert scenario.shock to pd.Series aligned to macro variables.
        3. Compute h_initial = S · shock_vector.
        4. Run Leontief propagation.
        5. Map sector-level results to individual holdings via sector_map.
        6. Compute per-holding direct and total P&L.
        7. Collect cascade warnings if result.cascade_risk != "low".

        Parameters
        ----------
        scenario : MacroStressScenario
        weights : pd.Series
            {ticker: portfolio_weight}. Should sum to ~1.0.

        Returns
        -------
        MacroStressResult
        """
        if not self._is_fitted:
            raise RuntimeError(
                "MacroStressEngine.run_stress(): call fit() before run_stress()."
            )

        t0 = time.time()
        warnings_out: list[str] = []

        # Step 2: Convert shock to series
        variable_names = self._sensitivity.macro_variable_names
        shock_vector = scenario.shock.to_series(variable_names)

        # Step 3: Initial distress
        h_initial = self._sensitivity_estimator.get_initial_distress(
            self._sensitivity, shock_vector
        )

        # Aggregate ticker weights to sector weights
        sector_weights = pd.Series(0.0, index=self._sensitivity.sector_names)
        for ticker, w in weights.items():
            sector = self._sector_map.get(str(ticker))
            if sector and sector in sector_weights.index:
                sector_weights[sector] += float(w)

        # Reindex h_initial to known sectors
        known_sectors = list(self._W.index)
        h_initial_aligned = h_initial.reindex(known_sectors).fillna(0.0)

        # Step 4: Propagate
        contagion = self._contagion_engine.propagate(
            h_initial_aligned, self._W, self._idr_params, sector_weights
        )

        # Step 5-6: Map to holdings
        pv = self._config.portfolio_value
        holding_results: list[HoldingContagionResult] = []

        for ticker, weight in weights.items():
            w = float(weight)
            if abs(w) < 1e-9:
                continue
            sector = self._sector_map.get(str(ticker), "Unknown")
            direct = float(contagion.h_initial.get(sector, 0.0))
            total = float(contagion.leontief_total.get(sector, direct))
            pnl_d = w * direct * pv
            pnl_t = w * total * pv
            with np.errstate(divide="ignore", invalid="ignore"):
                amp = float(total / direct) if abs(direct) > 1e-10 else 1.0

            holding_results.append(HoldingContagionResult(
                ticker=str(ticker),
                sector=sector,
                weight=w,
                direct_return=direct,
                total_return=total,
                pnl_direct=pnl_d,
                pnl_total=pnl_t,
                amplification=amp,
            ))

        total_pnl_direct = sum(h.pnl_direct for h in holding_results)
        total_pnl_total = sum(h.pnl_total for h in holding_results)

        # Step 7: Cascade warnings
        if contagion.cascade_risk != "low" and self._config.show_cascade_warnings:
            warnings_out.extend(contagion.warnings)

        elapsed = time.time() - t0
        logger.info(
            f"run_stress('{scenario.name}'): direct_pnl={total_pnl_direct:.0f}, "
            f"total_pnl={total_pnl_total:.0f}, "
            f"cascade={contagion.cascade_risk}, elapsed={elapsed:.3f}s"
        )

        return MacroStressResult(
            scenario=scenario,
            h_initial=contagion.h_initial,
            contagion=contagion,
            holding_results=holding_results,
            total_pnl_direct=total_pnl_direct,
            total_pnl_total=total_pnl_total,
            cascade_risk=contagion.cascade_risk,
            spectral_radius=contagion.spectral_radius,
            multiplier_table=contagion.multiplier_table,
            systemic_importance=contagion.systemic_importance,
            warnings=warnings_out,
        )

    def run_all_scenarios(
        self,
        weights: pd.Series,
        scenarios: list[MacroStressScenario] = None,
    ) -> dict[str, "MacroStressResult"]:
        """
        Run all scenarios. If None, uses DEFAULT_MACRO_SCENARIOS.

        Failed scenarios are logged and skipped without raising.

        Parameters
        ----------
        weights : pd.Series
            {ticker: portfolio_weight}.
        scenarios : list[MacroStressScenario], optional

        Returns
        -------
        dict[str, MacroStressResult]
            Keys are scenario names.
        """
        if scenarios is None:
            scenarios = DEFAULT_MACRO_SCENARIOS
        results: dict[str, MacroStressResult] = {}
        for scenario in scenarios:
            try:
                results[scenario.name] = self.run_stress(scenario, weights)
            except Exception as exc:
                logger.error(
                    f"run_all_scenarios: scenario='{scenario.name}' "
                    f"failed and was skipped: {exc}"
                )
        return results

    def get_fit_summary(self) -> pd.DataFrame:
        """
        Return DataFrame summarising fit status of all sub-components.

        Returns
        -------
        pd.DataFrame
            Columns: Component, Status, Details.
            Rows: MacroData, Sensitivity, ContagionNetwork, IDRFeedback.
        """
        rows = []
        # MacroData
        if self._macro_data is not None:
            missing = self._macro_data.missing_variables
            n_vars = len(self._macro_data.aligned_weekly.columns)
            status = "OK" if not missing else f"PARTIAL ({len(missing)} missing)"
            detail = (
                f"{n_vars} variables, "
                f"date range {self._macro_data.date_range[0]} to "
                f"{self._macro_data.date_range[1]}"
            )
        else:
            status = "NOT FITTED"
            detail = ""
        rows.append({"Component": "MacroData", "Status": status, "Details": detail})

        # Sensitivity
        if self._sensitivity is not None:
            r2_mean = float(self._sensitivity.r_squared["r_squared"].mean())
            status = "OK"
            detail = (
                f"{len(self._sensitivity.sector_names)} sectors, "
                f"{len(self._sensitivity.macro_variable_names)} macro vars, "
                f"mean R²={r2_mean:.3f}"
            )
        else:
            status = "NOT FITTED"
            detail = ""
        rows.append({"Component": "Sensitivity", "Status": status, "Details": detail})

        # ContagionNetwork
        if self._W is not None:
            try:
                eigvals = np.linalg.eigvals(self._W.values)
                sr = float(np.max(np.abs(eigvals)))
            except Exception:
                sr = float("nan")
            cascade = self._contagion_engine.get_cascade_risk_label(sr)
            status = "OK"
            detail = (
                f"{self._W.shape[0]} sectors, "
                f"spectral_radius={sr:.4f}, cascade_risk={cascade}"
            )
        else:
            status = "NOT FITTED"
            detail = ""
        rows.append({"Component": "ContagionNetwork", "Status": status, "Details": detail})

        # IDRFeedback
        if self._idr_params is not None:
            status = "OK"
            detail = (
                f"equity_to_idr_coef={self._idr_params.equity_to_idr_coef:.4f}, "
                f"R²={self._idr_params.r_squared:.3f}, "
                f"n_obs={self._idr_params.n_observations}"
            )
        else:
            status = "NOT FITTED"
            detail = ""
        rows.append({"Component": "IDRFeedback", "Status": status, "Details": detail})

        return pd.DataFrame(rows)

    def get_sensitivity_heatmap_data(self) -> pd.DataFrame:
        """
        Return the S matrix formatted for plotly heatmap display.

        Rows = sectors, columns = macro variables.

        Returns
        -------
        pd.DataFrame
            The S matrix from SensitivityResult. Empty DataFrame if not fitted.
        """
        if self._sensitivity is None:
            return pd.DataFrame()
        return self._sensitivity.S.copy()


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    def _smoke_test() -> None:
        rng = np.random.default_rng(1)
        T = 600

        dates = pd.date_range("2019-01-07", periods=T, freq="B")
        tickers = ["AAPL", "JPM", "XOM"]
        sector_map = {
            "AAPL": "Technology",
            "JPM": "Financials",
            "XOM": "Energy",
        }
        returns = pd.DataFrame(
            rng.normal(0, 0.01, (T, 3)), index=dates, columns=tickers
        )

        # MacroShock.to_series()
        shock = MacroShock(dxy_pct=0.05, vix_delta=10.0, us_10y_bps=50.0)
        var_names = ["DXY", "VIX", "US_10Y", "BI_RATE", "IDR_USD"]
        vec = shock.to_series(var_names)
        assert vec["DXY"] == 0.05
        assert vec["VIX"] == 10.0
        assert vec["BI_RATE"] == 0.0
        print(f"  MacroShock.to_series(): {vec.to_dict()}")

        # DEFAULT_MACRO_SCENARIOS count
        assert len(DEFAULT_MACRO_SCENARIOS) == 6
        names = [s.name for s in DEFAULT_MACRO_SCENARIOS]
        assert "Commodity Boom" in names
        assert "Global Risk-Off" in names
        print(f"  DEFAULT_MACRO_SCENARIOS: {names}")

        # MacroStressConfig default construction
        cfg = MacroStressConfig()
        assert cfg.reestimate_if_older_days == 30
        assert cfg.portfolio_value == 1_000_000.0
        print(f"  MacroStressConfig defaults OK")

        # MacroStressEngine construction (no fit)
        engine = MacroStressEngine(returns, sector_map, cfg)
        assert not engine._is_fitted
        try:
            engine.run_stress(DEFAULT_MACRO_SCENARIOS[0], pd.Series(1 / 3, index=tickers))
            assert False, "Should raise RuntimeError"
        except RuntimeError as exc:
            assert "fit()" in str(exc)
            print(f"  Unfit engine raises RuntimeError: OK")

        # get_fit_summary before fit
        summary_df = engine.get_fit_summary()
        assert len(summary_df) == 4
        assert all(summary_df["Status"] == "NOT FITTED")
        print(f"  get_fit_summary() before fit: {summary_df['Component'].tolist()}")

        print(
            "\nmacro_stress smoke test PASSED (network fetch skipped — "
            "run with live data to test full fit)"
        )

    _smoke_test()
