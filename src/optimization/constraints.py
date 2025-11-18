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

        Positions below min_position_size are set to zero and
        weights are renormalized.

        Args:
            weights: Portfolio weights

        Returns:
            Adjusted weights
        """
        if self.min_position_size <= 0:
            return weights

        adjusted = weights.copy()
        adjusted[np.abs(adjusted) < self.min_position_size] = 0

        # Renormalize
        total = np.sum(adjusted)
        if total > 0:
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
