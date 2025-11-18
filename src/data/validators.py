"""Data validation and quality checks."""

from typing import List, Tuple, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class DataValidator:
    """
    Validates financial data for quality and completeness.

    Checks for missing values, outliers, stale data, and other issues.
    """

    def __init__(
        self,
        max_missing_pct: float = 0.05,
        max_zero_pct: float = 0.05,
        max_constant_days: int = 5,
        outlier_std: float = 5.0,
        min_data_points: int = 30,
    ):
        """
        Initialize the validator.

        Args:
            max_missing_pct: Maximum allowed percentage of missing values
            max_zero_pct: Maximum allowed percentage of zero values
            max_constant_days: Maximum consecutive days with same value
            outlier_std: Number of standard deviations for outlier detection
            min_data_points: Minimum required data points
        """
        self.max_missing_pct = max_missing_pct
        self.max_zero_pct = max_zero_pct
        self.max_constant_days = max_constant_days
        self.outlier_std = outlier_std
        self.min_data_points = min_data_points

    def validate_price_data(
        self,
        data: pd.DataFrame,
        tickers: List[str] = None,
        raise_on_error: bool = False,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Validate price data for quality issues.

        Args:
            data: DataFrame with price data (columns = tickers)
            tickers: List of expected tickers
            raise_on_error: Whether to raise exception on critical errors

        Returns:
            Tuple of (cleaned_data, validation_report)
        """
        report = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "stats": {},
            "removed_tickers": [],
            "filled_missing": {},
        }

        if data is None or data.empty:
            report["valid"] = False
            report["errors"].append("No data provided")
            if raise_on_error:
                raise DataValidationError("No data provided")
            return data, report

        # Check for expected tickers
        if tickers:
            missing_tickers = set(tickers) - set(data.columns)
            if missing_tickers:
                report["warnings"].append(f"Missing tickers: {missing_tickers}")

        # Validate each column
        cleaned_data = data.copy()

        for col in data.columns:
            col_data = data[col]
            col_report = self._validate_series(col_data, col)

            # Aggregate issues
            report["stats"][col] = col_report

            if col_report.get("critical"):
                report["errors"].append(f"{col}: {col_report['critical']}")
                report["removed_tickers"].append(col)
                cleaned_data = cleaned_data.drop(columns=[col])
            else:
                if col_report.get("warnings"):
                    for warning in col_report["warnings"]:
                        report["warnings"].append(f"{col}: {warning}")

                # Apply cleaning
                if col_report.get("missing_filled"):
                    report["filled_missing"][col] = col_report["missing_filled"]
                    cleaned_data[col] = col_report["cleaned_series"]

        # Check overall data quality
        if len(cleaned_data.columns) == 0:
            report["valid"] = False
            report["errors"].append("No valid tickers remaining after validation")
            if raise_on_error:
                raise DataValidationError("No valid tickers remaining")
        elif report["errors"]:
            report["valid"] = False
            if raise_on_error:
                raise DataValidationError(f"Validation errors: {report['errors']}")

        # Log results
        if report["errors"]:
            logger.error(f"Validation errors: {report['errors']}")
        if report["warnings"]:
            logger.warning(f"Validation warnings: {len(report['warnings'])} issues found")

        return cleaned_data, report

    def _validate_series(self, series: pd.Series, name: str) -> dict:
        """
        Validate a single data series.

        Args:
            series: Data series to validate
            name: Name of the series (ticker)

        Returns:
            Validation report for the series
        """
        report = {
            "warnings": [],
            "critical": None,
            "missing_pct": 0,
            "zero_pct": 0,
            "outliers_pct": 0,
            "missing_filled": 0,
            "cleaned_series": series.copy(),
        }

        # Check data length
        if len(series) < self.min_data_points:
            report["critical"] = f"Insufficient data points: {len(series)} < {self.min_data_points}"
            return report

        # Check for missing values
        missing_pct = series.isna().sum() / len(series)
        report["missing_pct"] = round(missing_pct, 4)

        if missing_pct > self.max_missing_pct:
            report["critical"] = f"Too many missing values: {missing_pct:.1%}"
            return report
        elif missing_pct > 0:
            # Fill missing values with forward fill then backward fill
            filled_series = series.ffill().bfill()
            report["missing_filled"] = series.isna().sum()
            report["cleaned_series"] = filled_series
            report["warnings"].append(f"Filled {report['missing_filled']} missing values")

        # Check for zero values (suspicious for price data)
        clean_series = report["cleaned_series"]
        zero_pct = (clean_series == 0).sum() / len(clean_series)
        report["zero_pct"] = round(zero_pct, 4)

        if zero_pct > self.max_zero_pct:
            report["warnings"].append(f"High zero value percentage: {zero_pct:.1%}")

        # Check for constant values (stale data)
        consecutive_same = self._max_consecutive_same(clean_series)
        if consecutive_same > self.max_constant_days:
            report["warnings"].append(
                f"Possible stale data: {consecutive_same} consecutive identical values"
            )

        # Check for outliers
        returns = clean_series.pct_change().dropna()
        if len(returns) > 0:
            mean_return = returns.mean()
            std_return = returns.std()

            if std_return > 0:
                outliers = abs(returns - mean_return) > (self.outlier_std * std_return)
                outlier_pct = outliers.sum() / len(returns)
                report["outliers_pct"] = round(outlier_pct, 4)

                if outlier_pct > 0.01:  # More than 1% outliers
                    report["warnings"].append(f"Detected {outliers.sum()} potential outliers")

        # Check for negative prices (should not happen)
        negative_count = (clean_series < 0).sum()
        if negative_count > 0:
            report["critical"] = f"Negative prices detected: {negative_count}"
            return report

        return report

    def _max_consecutive_same(self, series: pd.Series) -> int:
        """Find the maximum consecutive identical values in a series."""
        if len(series) == 0:
            return 0

        max_consecutive = 1
        current_consecutive = 1

        for i in range(1, len(series)):
            if series.iloc[i] == series.iloc[i - 1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1

        return max_consecutive

    def check_date_range(
        self,
        data: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        min_coverage: float = 0.8,
    ) -> Tuple[bool, str]:
        """
        Check if data covers the requested date range.

        Args:
            data: DataFrame with datetime index
            start_date: Requested start date
            end_date: Requested end date
            min_coverage: Minimum required coverage ratio

        Returns:
            Tuple of (is_valid, message)
        """
        if data is None or data.empty:
            return False, "No data available"

        data_start = data.index.min()
        data_end = data.index.max()

        # Convert to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # Check coverage
        requested_days = (end_date - start_date).days
        actual_days = (min(data_end, end_date) - max(data_start, start_date)).days
        coverage = actual_days / requested_days if requested_days > 0 else 0

        if coverage < min_coverage:
            return False, (
                f"Insufficient date coverage: {coverage:.1%} "
                f"(data: {data_start.date()} to {data_end.date()})"
            )

        return True, f"Date coverage: {coverage:.1%}"

    def validate_returns(
        self,
        returns: pd.DataFrame,
        max_daily_return: float = 1.0,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Validate return data for reasonableness.

        Args:
            returns: DataFrame of returns
            max_daily_return: Maximum reasonable daily return (100% = 1.0)

        Returns:
            Tuple of (cleaned_returns, validation_report)
        """
        report = {
            "valid": True,
            "warnings": [],
            "clipped_values": {},
        }

        if returns is None or returns.empty:
            report["valid"] = False
            report["warnings"].append("No return data provided")
            return returns, report

        cleaned = returns.copy()

        for col in returns.columns:
            # Check for extreme returns
            extreme_mask = abs(returns[col]) > max_daily_return
            extreme_count = extreme_mask.sum()

            if extreme_count > 0:
                report["warnings"].append(
                    f"{col}: {extreme_count} extreme returns (>{max_daily_return:.0%})"
                )
                report["clipped_values"][col] = extreme_count

                # Clip extreme values
                cleaned[col] = returns[col].clip(-max_daily_return, max_daily_return)

        return cleaned, report


def quick_validate(data: pd.DataFrame, tickers: List[str] = None) -> pd.DataFrame:
    """
    Quick validation function for simple use cases.

    Args:
        data: Price data DataFrame
        tickers: Expected tickers

    Returns:
        Cleaned data
    """
    validator = DataValidator()
    cleaned, report = validator.validate_price_data(data, tickers)

    if not report["valid"]:
        logger.error(f"Validation failed: {report['errors']}")

    return cleaned
