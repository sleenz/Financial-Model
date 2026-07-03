"""Portfolio constraint handling system."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ConstraintError(Exception):
    """Custom exception for constraint violations."""
    pass


class PortfolioConstraints:
    """
    Manages portfolio optimization constraints.

    Supports position limits, sector limits, turnover constraints,
    and various risk constraints.
    """

    def __init__(
        self,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        min_position_size: float = 0.0,
        max_position_size: float = 1.0,
        sector_limits: Dict[str, float] = None,
        max_turnover: float = None,
        target_volatility: float = None,
        max_volatility: float = None,
        min_return: float = None,
        max_drawdown: float = None,
        long_only: bool = True,
        turnover_enabled: bool = False,
        reduction_pct: float = 0.50,
        increase_pct: float = 0.30,
        allow_full_exit: bool = True,
        current_weights: Optional[Any] = None,
    ):
        """
        Initialize constraints.

        Args:
            min_weight: Minimum weight for any position (can be negative for short)
            max_weight: Maximum weight for any position
            min_position_size: Minimum non-zero position size (positions below this become 0)
            max_position_size: Maximum position size for concentration limit
            sector_limits: Dict mapping sector names to max allocation
            max_turnover: Maximum portfolio turnover (0-1)
            target_volatility: Target portfolio volatility
            max_volatility: Maximum portfolio volatility
            min_return: Minimum expected return
            max_drawdown: Maximum acceptable drawdown
            long_only: Whether to enforce long-only constraint
            turnover_enabled: When True, each position is bounded within a trading band
                around its current weight. When False (default), optimizer ignores
                current_weights entirely and uses standard min/max bounds.
            reduction_pct: Maximum allowed reduction from current weight (0.50 = 50%).
                Range [0.0, 1.0]. 1.0 = can reduce to zero (same as allow_full_exit=True).
            increase_pct: Maximum allowed increase from current weight (0.30 = 30%).
                Range [0.0, inf). max_weight still applies as an upper cap.
            allow_full_exit: If True, any position can be reduced to 0 regardless of
                reduction_pct. Applies only when turnover_enabled=True.
            current_weights: pd.Series with index=tickers, values=current decimal weights.
                Required when turnover_enabled=True. If None with turnover_enabled=True,
                optimizer logs a warning and falls back to standard bounds.
        """
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size
        self.sector_limits = sector_limits or {}
        self.max_turnover = max_turnover
        self.target_volatility = target_volatility
        self.max_volatility = max_volatility
        self.min_return = min_return
        self.max_drawdown = max_drawdown
        self.long_only = long_only
        self.turnover_enabled = turnover_enabled
        self.reduction_pct = reduction_pct
        self.increase_pct = increase_pct
        self.allow_full_exit = allow_full_exit
        self.current_weights = current_weights

        # Validate constraints
        self._validate()

    def _validate(self):
        """Validate constraint parameters."""
        if self.min_weight > self.max_weight:
            raise ConstraintError("min_weight cannot exceed max_weight")

        if self.min_position_size > self.max_position_size:
            raise ConstraintError("min_position_size cannot exceed max_position_size")

        if self.long_only and self.min_weight < 0:
            logger.warning("long_only=True but min_weight<0, setting min_weight=0")
            self.min_weight = 0.0

        for sector, limit in self.sector_limits.items():
            if limit < 0 or limit > 1:
                raise ConstraintError(f"Invalid sector limit for {sector}: {limit}")

    def get_bounds(self, n_assets: int) -> List[Tuple[float, float]]:
        """
        Get weight bounds for each asset.

        Args:
            n_assets: Number of assets

        Returns:
            List of (min, max) tuples for each asset
        """
        return [(self.min_weight, self.max_weight) for _ in range(n_assets)]

    def compute_bounds(self, tickers: List[str]) -> List[Tuple[float, float]]:
        """
        Compute per-asset (lower, upper) weight bounds, honoring the position
        reduction (turnover) trading band when enabled. Shared by every
        optimizer backend (PortfolioOptimizer, BlackLittermanModel, ...) so
        the position reduction constraint applies consistently regardless of
        which optimization method is selected.

        When turnover_enabled=False (or current_weights is None): all assets
        get the standard (min_weight, max_weight) bounds.

        When turnover_enabled=True: each held asset is bounded within a
        trading band around its current weight — [current * (1 - reduction_pct),
        current * (1 + increase_pct)], additionally clamped to [min_weight,
        max_weight]. New positions (not in current_weights) get the standard
        (min_weight, max_weight) bounds since there's no "current" position to
        band around.

        Parameters
        ----------
        tickers : list of str
            Asset names, in the same order the caller wants bounds returned.

        Returns
        -------
        list of (float, float)
            One (lb, ub) per ticker, in the same order as `tickers`.

        Raises
        ------
        ValueError
            If the turnover-constrained bounds are infeasible (e.g. minimum
            weights already sum to over 100%).
        """
        if not self.turnover_enabled or self.current_weights is None:
            if self.turnover_enabled and self.current_weights is None:
                logger.warning(
                    "PortfolioConstraints: turnover_enabled=True but "
                    "current_weights is None — falling back to standard bounds"
                )
            lower_bounds = [self.min_weight for _ in tickers]
            upper_bounds = [self.max_weight for _ in tickers]
            self._check_bounds_feasibility(lower_bounds, upper_bounds)
            return list(zip(lower_bounds, upper_bounds))

        lower_bounds = []
        upper_bounds = []

        for ticker in tickers:
            if ticker in self.current_weights.index:
                current_w = float(self.current_weights[ticker])

                if self.allow_full_exit:
                    lb = 0.0
                else:
                    lb = max(self.min_weight, current_w * (1.0 - self.reduction_pct))

                ub = min(self.max_weight, current_w * (1.0 + self.increase_pct))
                ub = max(ub, lb)  # safety: ensure ub >= lb
            else:
                # New position: no turnover restriction
                lb = self.min_weight
                ub = self.max_weight

            lower_bounds.append(lb)
            upper_bounds.append(ub)

        self._check_bounds_feasibility(lower_bounds, upper_bounds)

        return list(zip(lower_bounds, upper_bounds))

    def project_to_bounds(self, weights: np.ndarray, tickers: List[str]) -> np.ndarray:
        """
        Project a weight vector onto this constraint set's per-asset bounds
        (position limits and, if enabled, the position reduction/turnover
        band) while keeping the weights summing to 1.

        This exists because some optimization methods — Hierarchical Risk
        Parity and Equal Weight — compute weights purely from the
        covariance/correlation structure (or a flat 1/n split) and never
        look at `constraints` at all. Without this step, changing Position
        Limits or the Position Reduction band and re-running with one of
        those methods silently produces the exact same weights every time.
        For methods that already solve with `compute_bounds()` as scipy
        bounds (max_sharpe, min_volatility, ...), the result is already
        feasible, so this is a no-op safety net for them.

        Uses iterative water-filling: clip to bounds, then redistribute the
        remaining budget across assets not yet pinned to a bound, repeating
        until every asset is either pinned or the budget is exhausted.

        Parameters
        ----------
        weights : np.ndarray
            Raw weights to project, in the same order as `tickers`.
        tickers : list of str
            Asset names, in the same order as `weights`.

        Returns
        -------
        np.ndarray
            Weights within [lb, ub] per asset, summing to 1.
        """
        bounds = self.compute_bounds(tickers)
        lb = np.array([b[0] for b in bounds], dtype=float)
        ub = np.array([b[1] for b in bounds], dtype=float)

        w = np.clip(np.asarray(weights, dtype=float), lb, ub)
        active = np.ones(len(w), dtype=bool)

        for _ in range(len(w)):
            if not active.any():
                break

            remaining = 1.0 - w[~active].sum()
            active_sum = w[active].sum()

            if active_sum <= 1e-12:
                # Nothing left to scale proportionally — split what's left
                # equally among the still-active assets instead.
                w[active] = remaining / active.sum()
            else:
                w[active] *= remaining / active_sum

            newly_pinned = active & ((w > ub + 1e-9) | (w < lb - 1e-9))
            if not newly_pinned.any():
                break
            w[newly_pinned] = np.clip(w[newly_pinned], lb[newly_pinned], ub[newly_pinned])
            active &= ~newly_pinned

        return w

    def _check_bounds_feasibility(
        self,
        lower_bounds: List[float],
        upper_bounds: List[float],
    ) -> None:
        """
        Verify that per-asset bounds are feasible before optimization —
        i.e. that 100% allocation is actually reachable within them.
        Applies to both the plain Position Limits bounds and the
        turnover-constrained trading band; the error message is tailored
        to whichever is actually active so it points at the right slider.

        Raises
        ------
        ValueError
            With a human-readable message explaining which direction is
            infeasible and what the user should do to fix it.
        """
        sum_lower = sum(lower_bounds)
        sum_upper = sum(upper_bounds)
        turnover_active = self.turnover_enabled and self.current_weights is not None

        # Catch both truly infeasible (sum > 1.0) and degenerate (sum == 1.0, zero
        # optimization freedom) cases. Using 1e-6 tolerance captures floating-point
        # noise and the degenerate edge case where every weight is locked at its floor.
        if sum_lower > 1.0 - 1e-6:
            if turnover_active:
                raise ValueError(
                    f"Position reduction constraint infeasible: "
                    f"sum of minimum weights = {sum_lower:.3f} > 1.0. "
                    f"Increase the maximum reduction percentage "
                    f"(currently allowing only {(1 - self.reduction_pct)*100:.0f}% "
                    f"of each position to be retained as minimum) "
                    f"or enable 'Allow full exit'."
                )
            raise ValueError(
                f"Position limits infeasible: sum of minimum weights = "
                f"{sum_lower:.3f} > 1.0. {len(lower_bounds)} assets at a "
                f"{self.min_weight*100:.1f}% minimum each already exceed 100% — "
                f"lower the Minimum Position Size."
            )
        if sum_upper < 1.0 - 1e-6:
            if turnover_active:
                raise ValueError(
                    f"Position increase constraint infeasible: "
                    f"sum of maximum weights = {sum_upper:.3f} < 1.0. "
                    f"Increase the maximum increase percentage "
                    f"(currently allowing only {self.increase_pct*100:.0f}% "
                    f"increase per position)."
                )
            raise ValueError(
                f"Position limits infeasible: sum of maximum weights = "
                f"{sum_upper:.3f} < 1.0. {len(upper_bounds)} assets at a "
                f"{self.max_weight*100:.1f}% maximum each cannot reach 100% — "
                f"raise the Maximum Position Size."
            )

    def get_sector_constraints(
        self,
        tickers: List[str],
        sector_map: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """
        Generate sector constraint matrices.

        Args:
            tickers: List of ticker symbols
            sector_map: Dict mapping ticker to sector

        Returns:
            List of constraint dictionaries for scipy.optimize
        """
        if not self.sector_limits:
            return []

        constraints = []

        for sector, limit in self.sector_limits.items():
            # Find indices of assets in this sector
            indices = [
                i for i, ticker in enumerate(tickers)
                if sector_map.get(ticker, "Unknown") == sector
            ]

            if not indices:
                continue

            # Create constraint: sum of weights in sector <= limit
            def sector_constraint(weights, idx=indices, lim=limit):
                return lim - sum(weights[i] for i in idx)

            constraints.append({
                'type': 'ineq',
                'fun': sector_constraint,
            })

        return constraints

    def get_turnover_constraint(
        self,
        current_weights: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Generate turnover constraint.

        Args:
            current_weights: Current portfolio weights

        Returns:
            Constraint dictionary for scipy.optimize
        """
        if self.max_turnover is None:
            return None

        def turnover_constraint(weights):
            return self.max_turnover - np.sum(np.abs(weights - current_weights))

        return {
            'type': 'ineq',
            'fun': turnover_constraint,
        }

    def apply_minimum_position(self, weights: np.ndarray) -> np.ndarray:
        """
        Apply minimum position size constraint.

        Positions below min_position_size are set to zero and weights are
        renormalized.  The loop repeats until no position lies between 0 and
        the threshold (renormalisation can push a borderline weight above or
        below the limit).  If every position would be eliminated (e.g. equal-
        weight with more assets than 1/min_position_size), the top-k assets
        that can satisfy the constraint receive equal weight as a fallback.

        Args:
            weights: Portfolio weights

        Returns:
            Adjusted weights with all non-zero positions >= min_position_size
        """
        if self.min_position_size <= 0:
            return weights

        adjusted = weights.copy()

        for _ in range(len(weights) + 1):
            below = (adjusted > 0) & (np.abs(adjusted) < self.min_position_size)
            if not below.any():
                break

            adjusted[below] = 0
            total = np.sum(adjusted)

            if total <= 0:
                # All positions were eliminated by the threshold.
                # Fallback: keep only the top-k largest (by original weight)
                # where k = floor(1 / min_position_size) so equal weight >= threshold.
                # Use `continue` (not `break`) so the next loop iteration can
                # re-filter any fallback weights that are still below the threshold.
                max_k = max(1, int(1.0 / self.min_position_size))
                top_idx = np.argsort(weights)[::-1][:max_k]
                adjusted = np.zeros_like(weights)
                top_w = weights[top_idx]
                w_sum = top_w.sum()
                adjusted[top_idx] = (top_w / w_sum) if w_sum > 0 else np.full(max_k, 1.0 / max_k)
                continue

            adjusted = adjusted / total

        return adjusted

    def check_constraints(
        self,
        weights: np.ndarray,
        tickers: List[str] = None,
        sector_map: Dict[str, str] = None,
        expected_return: float = None,
        volatility: float = None,
        current_weights: np.ndarray = None,
    ) -> Tuple[bool, List[str]]:
        """
        Check if weights satisfy all constraints.

        Args:
            weights: Portfolio weights
            tickers: List of ticker symbols
            sector_map: Dict mapping ticker to sector
            expected_return: Expected portfolio return
            volatility: Portfolio volatility
            current_weights: Current weights for turnover check

        Returns:
            Tuple of (is_valid, list of violations)
        """
        violations = []

        # Check weight bounds
        for i, w in enumerate(weights):
            if w < self.min_weight - 1e-6:
                ticker = tickers[i] if tickers else f"Asset {i}"
                violations.append(f"{ticker} weight {w:.4f} below min {self.min_weight}")
            if w > self.max_weight + 1e-6:
                ticker = tickers[i] if tickers else f"Asset {i}"
                violations.append(f"{ticker} weight {w:.4f} above max {self.max_weight}")

        # Check max position size
        max_pos = np.max(np.abs(weights))
        if max_pos > self.max_position_size + 1e-6:
            violations.append(f"Max position {max_pos:.4f} exceeds limit {self.max_position_size}")

        # Check sector limits
        if tickers and sector_map and self.sector_limits:
            for sector, limit in self.sector_limits.items():
                sector_weight = sum(
                    weights[i] for i, t in enumerate(tickers)
                    if sector_map.get(t, "Unknown") == sector
                )
                if sector_weight > limit + 1e-6:
                    violations.append(
                        f"Sector {sector} weight {sector_weight:.4f} exceeds limit {limit}"
                    )

        # Check turnover
        if current_weights is not None and self.max_turnover is not None:
            turnover = np.sum(np.abs(weights - current_weights))
            if turnover > self.max_turnover + 1e-6:
                violations.append(
                    f"Turnover {turnover:.4f} exceeds limit {self.max_turnover}"
                )

        # Check volatility
        if volatility is not None:
            if self.max_volatility and volatility > self.max_volatility + 1e-6:
                violations.append(
                    f"Volatility {volatility:.4f} exceeds max {self.max_volatility}"
                )
            if self.target_volatility and abs(volatility - self.target_volatility) > 0.01:
                violations.append(
                    f"Volatility {volatility:.4f} differs from target {self.target_volatility}"
                )

        # Check minimum return
        if expected_return is not None and self.min_return is not None:
            if expected_return < self.min_return - 1e-6:
                violations.append(
                    f"Expected return {expected_return:.4f} below min {self.min_return}"
                )

        # Check weights sum to 1
        weight_sum = np.sum(weights)
        if abs(weight_sum - 1.0) > 1e-4:
            violations.append(f"Weights sum to {weight_sum:.4f}, not 1.0")

        return len(violations) == 0, violations

    def to_dict(self) -> Dict[str, Any]:
        """Convert constraints to dictionary."""
        return {
            'min_weight': self.min_weight,
            'max_weight': self.max_weight,
            'min_position_size': self.min_position_size,
            'max_position_size': self.max_position_size,
            'sector_limits': self.sector_limits,
            'max_turnover': self.max_turnover,
            'target_volatility': self.target_volatility,
            'max_volatility': self.max_volatility,
            'min_return': self.min_return,
            'max_drawdown': self.max_drawdown,
            'long_only': self.long_only,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PortfolioConstraints':
        """Create constraints from dictionary."""
        return cls(**d)

    def __repr__(self) -> str:
        return (
            f"PortfolioConstraints(min_weight={self.min_weight}, "
            f"max_weight={self.max_weight}, "
            f"max_position_size={self.max_position_size}, "
            f"long_only={self.long_only})"
        )


def default_constraints(conservative: bool = False) -> PortfolioConstraints:
    """
    Get default constraint settings.

    Args:
        conservative: If True, use more restrictive constraints

    Returns:
        PortfolioConstraints instance
    """
    if conservative:
        return PortfolioConstraints(
            min_weight=0.0,
            max_weight=0.20,  # Max 20% per position
            min_position_size=0.02,  # Min 2% or nothing
            max_position_size=0.20,
            long_only=True,
        )
    else:
        return PortfolioConstraints(
            min_weight=0.0,
            max_weight=0.40,  # Max 40% per position
            min_position_size=0.01,  # Min 1% or nothing
            max_position_size=0.40,
            long_only=True,
        )
