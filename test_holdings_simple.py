"""Simple test script for holdings tracker functionality - minimal dependencies."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np

# Import directly from the module file
from src.portfolio.holdings import HoldingsTracker

# Example holdings from the user's request
holdings = {
    'NVDA': 9,
    'AAPL': 3,
}

# Mock prices
prices = pd.Series({
    'NVDA': 500.00,
    'AAPL': 180.00,
})

print("=" * 70)
print("TESTING HOLDINGS TRACKER")
print("=" * 70)

# Test 1: Create tracker
print("\nTest 1: Creating HoldingsTracker...")
try:
    tracker = HoldingsTracker(holdings, prices)
    print("✓ Tracker created successfully")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 2: Get holdings dataframe
print("\nTest 2: Get holdings dataframe...")
try:
    df = tracker.get_holdings_dataframe()
    print(df)
    print("✓ DataFrame generated successfully")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 3: Calculate total value
print("\nTest 3: Calculate total value...")
try:
    total_value = tracker.calculate_total_value()
    print(f"Total Portfolio Value: ${total_value:,.2f}")
    expected = 9 * 500 + 3 * 180
    assert abs(total_value - expected) < 0.01, f"Expected ${expected:,.2f}"
    print("✓ Total value calculated correctly")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 4: Calculate diversity metrics
print("\nTest 4: Calculate diversity metrics...")
try:
    metrics = tracker.calculate_diversity_metrics()
    print("\nMetrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Verify metrics
    assert metrics['num_holdings'] == 2, "Should have 2 holdings"
    assert 0 <= metrics['herfindahl_index'] <= 1, "HHI should be between 0 and 1"
    print("✓ Metrics calculated successfully")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Get diversity rating
print("\nTest 5: Get diversity rating...")
try:
    rating = tracker.get_diversity_rating()
    print(f"Diversity Rating: {rating}")
    assert rating in ["Highly Concentrated", "Concentrated", "Moderately Diversified",
                      "Well Diversified", "Highly Diversified"], "Invalid rating"
    print("✓ Rating generated successfully")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 6: Get recommendations
print("\nTest 6: Get recommendations...")
try:
    recommendations = tracker.get_diversity_recommendations()
    print(f"\nRecommendations ({len(recommendations)}):")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    assert len(recommendations) > 0, "Should have at least one recommendation"
    print("✓ Recommendations generated successfully")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 7: Test with equal-weighted portfolio
print("\n\nTest 7: Testing with equal-weighted portfolio...")
try:
    equal_holdings = {
        'STOCK1': 10,
        'STOCK2': 10,
        'STOCK3': 10,
        'STOCK4': 10,
        'STOCK5': 10,
    }
    equal_prices = pd.Series({
        'STOCK1': 100.00,
        'STOCK2': 100.00,
        'STOCK3': 100.00,
        'STOCK4': 100.00,
        'STOCK5': 100.00,
    })

    equal_tracker = HoldingsTracker(equal_holdings, equal_prices)
    equal_metrics = equal_tracker.calculate_diversity_metrics()
    equal_rating = equal_tracker.get_diversity_rating()

    print(f"Diversity Rating: {equal_rating}")
    print(f"HHI: {equal_metrics['herfindahl_index']:.4f}")
    print(f"Effective Stocks: {equal_metrics['effective_stocks']:.2f}")

    # Equal weights should have HHI = 1/N = 0.2
    expected_hhi = 1.0 / 5
    assert abs(equal_metrics['herfindahl_index'] - expected_hhi) < 0.001, \
        f"Equal weights should have HHI={expected_hhi:.3f}, got {equal_metrics['herfindahl_index']:.3f}"
    print("✓ Equal portfolio test passed")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Test without prices
print("\nTest 8: Test without prices (should still work for share counts)...")
try:
    no_price_tracker = HoldingsTracker(holdings)
    metrics_no_price = no_price_tracker.calculate_diversity_metrics()
    # Without prices, values should be 0, so metrics should handle this gracefully
    print(f"Metrics calculated without prices: {metrics_no_price['num_holdings']} holdings")
    print("✓ No-price test passed")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
print("\nThe Holdings Tracker module is working correctly!")
print("\nYou can now:")
print("1. Run the Streamlit app with: streamlit run app/Home.py")
print("2. Go to the 'Portfolio Input' page")
print("3. Enter your holdings (e.g., NVDA: 9 shares, AAPL: 3 shares)")
print("4. Click 'Analyze Portfolio Diversity' to see your diversity metrics")
