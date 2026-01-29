"""
Test script for holdings tracker with Indonesian stocks.

This tests the full workflow:
1. Enter holdings (BBCA.JK, ANTM.JK)
2. Fetch data
3. Calculate diversity
4. Verify prices are fetched correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.data_manager import DataManager
from src.portfolio.holdings import HoldingsTracker

print("=" * 80)
print("TESTING HOLDINGS TRACKER WITH INDONESIAN STOCKS")
print("=" * 80)

# Test holdings
holdings = {
    'BBCA.JK': 100,  # Bank Central Asia (Indonesia)
    'ANTM.JK': 50,   # Aneka Tambang (Indonesia)
}

print(f"\nTest Holdings: {holdings}")

# Dates
end_date = datetime.now()
start_date = end_date - timedelta(days=365*3)

print(f"\nDate Range: {start_date.date()} to {end_date.date()}")

# Test 1: Fetch data
print("\n" + "=" * 80)
print("TEST 1: Fetching Historical Data")
print("=" * 80)

try:
    dm = DataManager(show_progress=False)
    tickers_list = list(holdings.keys())

    print(f"Fetching data for: {tickers_list}")

    prices = dm.get_price_data(
        tickers_list,
        start_date,
        end_date
    )

    print(f"✓ Successfully fetched data")
    print(f"  - Tickers received: {list(prices.columns)}")
    print(f"  - Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"  - Total rows: {len(prices)}")
    print(f"  - Shape: {prices.shape}")

    # Show sample prices
    print(f"\n  Recent Prices (last 5 days):")
    print(prices.tail())

    # Get current prices
    current_prices = prices.iloc[-1]
    print(f"\n  Current Prices (most recent):")
    for ticker in current_prices.index:
        print(f"    - {ticker}: {current_prices[ticker]:.2f}")

except Exception as e:
    print(f"✗ FAILED to fetch data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Create Holdings Tracker
print("\n" + "=" * 80)
print("TEST 2: Creating Holdings Tracker")
print("=" * 80)

try:
    tracker = HoldingsTracker(holdings, current_prices)
    print("✓ Holdings tracker created")

    # Get holdings dataframe
    holdings_df = tracker.get_holdings_dataframe()
    print("\nHoldings DataFrame:")
    print(holdings_df)

    # Check prices
    print("\nPrice Verification:")
    for ticker in holdings_df.index:
        row = holdings_df.loc[ticker]
        print(f"  {ticker}:")
        print(f"    - Shares: {row['Shares']:.2f}")
        print(f"    - Price: {row['Price']:.2f}")
        print(f"    - Value: {row['Value']:.2f}")
        print(f"    - Weight: {row['Weight']*100:.1f}%")

    total_value = tracker.calculate_total_value()
    print(f"\nTotal Portfolio Value: {total_value:,.2f}")

    if total_value == 0:
        print("\n⚠️ WARNING: Total value is 0! Prices were not fetched correctly.")
        sys.exit(1)
    else:
        print("✓ Prices fetched correctly")

except Exception as e:
    print(f"✗ FAILED to create tracker: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Diversity Analysis
print("\n" + "=" * 80)
print("TEST 3: Diversity Analysis")
print("=" * 80)

try:
    metrics = tracker.calculate_diversity_metrics()
    rating = tracker.get_diversity_rating()
    recommendations = tracker.get_diversity_recommendations()

    print(f"Diversity Rating: {rating}")
    print(f"\nMetrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.4f}")
        else:
            print(f"  - {key}: {value}")

    print(f"\nRecommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

    print("\n✓ Diversity analysis completed")

except Exception as e:
    print(f"✗ FAILED diversity analysis: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Generate Full Report
print("\n" + "=" * 80)
print("TEST 4: Full Holdings Report")
print("=" * 80)

try:
    report = tracker.format_holdings_report()
    print(report)
    print("\n✓ Report generated successfully")

except Exception as e:
    print(f"✗ FAILED to generate report: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
print("\nThe holdings tracker is working correctly with Indonesian stocks!")
print(f"Your portfolio of BBCA.JK and ANTM.JK is valued at {total_value:,.2f}")
print(f"Diversity Rating: {rating}")
