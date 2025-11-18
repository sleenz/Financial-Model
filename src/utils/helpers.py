"""Helper utility functions."""

import re
from typing import List, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def validate_tickers(tickers: Union[str, List[str]]) -> List[str]:
    """
    Validate and normalize ticker symbols.

    Args:
        tickers: Single ticker string or list of tickers

    Returns:
        List of validated, uppercase ticker symbols

    Raises:
        ValueError: If tickers are invalid
    """
    if isinstance(tickers, str):
        # Split by comma, semicolon, or whitespace
        tickers = re.split(r'[,;\s]+', tickers)

    # Clean and validate each ticker
    validated = []
    for ticker in tickers:
        ticker = ticker.strip().upper()
        if not ticker:
            continue

        # Basic validation: alphanumeric with possible dots (e.g., BRK.B)
        if not re.match(r'^[A-Z0-9.-]+$', ticker):
            raise ValueError(f"Invalid ticker symbol: {ticker}")

        # Length check (typically 1-5 characters)
        if len(ticker) > 10:
            raise ValueError(f"Ticker symbol too long: {ticker}")

        validated.append(ticker)

    if not validated:
        raise ValueError("No valid ticker symbols provided")

    return validated


def format_currency(value: float, currency: str = "$", decimals: int = 2) -> str:
    """
    Format a number as currency.

    Args:
        value: Numeric value to format
        currency: Currency symbol
        decimals: Number of decimal places

    Returns:
        Formatted currency string
    """
    if pd.isna(value):
        return "N/A"

    if value < 0:
        return f"-{currency}{abs(value):,.{decimals}f}"
    return f"{currency}{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a number as percentage.

    Args:
        value: Numeric value (0.15 = 15%)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    if pd.isna(value):
        return "N/A"

    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """
    Format a number with thousand separators.

    Args:
        value: Numeric value to format
        decimals: Number of decimal places

    Returns:
        Formatted number string
    """
    if pd.isna(value):
        return "N/A"

    return f"{value:,.{decimals}f}"


def calculate_date_range(
    period: str = "1y",
    end_date: datetime = None
) -> tuple:
    """
    Calculate start and end dates based on a period string.

    Args:
        period: Period string (e.g., "1m", "3m", "6m", "1y", "2y", "5y", "10y", "max")
        end_date: End date (defaults to today)

    Returns:
        Tuple of (start_date, end_date) as datetime objects
    """
    if end_date is None:
        end_date = datetime.now()

    period_map = {
        "1m": timedelta(days=30),
        "3m": timedelta(days=90),
        "6m": timedelta(days=180),
        "1y": timedelta(days=365),
        "2y": timedelta(days=730),
        "3y": timedelta(days=1095),
        "5y": timedelta(days=1825),
        "10y": timedelta(days=3650),
        "max": timedelta(days=36500),  # ~100 years
    }

    if period.lower() not in period_map:
        raise ValueError(f"Invalid period: {period}. Use one of: {list(period_map.keys())}")

    start_date = end_date - period_map[period.lower()]
    return start_date, end_date


def annualize_returns(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualize periodic returns.

    Args:
        returns: Series of periodic returns
        periods_per_year: Number of periods in a year (252 for daily)

    Returns:
        Annualized return
    """
    if len(returns) == 0:
        return 0.0

    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    annualized = (1 + total_return) ** (periods_per_year / n_periods) - 1
    return annualized


def annualize_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualize periodic volatility.

    Args:
        returns: Series of periodic returns
        periods_per_year: Number of periods in a year (252 for daily)

    Returns:
        Annualized volatility
    """
    return returns.std() * np.sqrt(periods_per_year)


def calculate_returns(prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """
    Calculate returns from price data.

    Args:
        prices: DataFrame of asset prices
        method: "simple" for arithmetic returns, "log" for logarithmic returns

    Returns:
        DataFrame of returns
    """
    if method == "simple":
        return prices.pct_change().dropna()
    elif method == "log":
        return np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError(f"Invalid method: {method}. Use 'simple' or 'log'")


def get_trading_days(
    start_date: datetime,
    end_date: datetime,
    market: str = "NYSE"
) -> int:
    """
    Estimate the number of trading days between two dates.

    Args:
        start_date: Start date
        end_date: End date
        market: Market for calendar (currently estimates based on 252 days/year)

    Returns:
        Estimated number of trading days
    """
    # Simple estimation: 252 trading days per year
    total_days = (end_date - start_date).days
    return int(total_days * 252 / 365)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a string to be used as a filename.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    return sanitized


def chunk_list(lst: list, chunk_size: int) -> list:
    """
    Split a list into chunks of specified size.

    Args:
        lst: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dataframes(
    dataframes: List[pd.DataFrame],
    how: str = "outer"
) -> pd.DataFrame:
    """
    Merge multiple DataFrames with the same index.

    Args:
        dataframes: List of DataFrames to merge
        how: Join method ('outer', 'inner', 'left', 'right')

    Returns:
        Merged DataFrame
    """
    if not dataframes:
        return pd.DataFrame()

    result = dataframes[0]
    for df in dataframes[1:]:
        result = result.join(df, how=how)

    return result
