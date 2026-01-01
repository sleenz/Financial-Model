"""Quick test script for holdings tracker functionality."""

import pandas as pd
from src.portfolio.holdings import HoldingsTracker, analyze_portfolio_diversity

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
tracker = HoldingsTracker(holdings, prices)
print("✓ Tracker created successfully")

# Test 2: Get holdings dataframe
print("\nTest 2: Get holdings dataframe...")
df = tracker.get_holdings_dataframe()
print(df)
print("✓ DataFrame generated successfully")

# Test 3: Calculate diversity metrics
print("\nTest 3: Calculate diversity metrics...")
metrics = tracker.calculate_diversity_metrics()
print("\nMetrics:")
for key, value in metrics.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")
print("✓ Metrics calculated successfully")

# Test 4: Get diversity rating
print("\nTest 4: Get diversity rating...")
rating = tracker.get_diversity_rating()
print(f"Diversity Rating: {rating}")
print("✓ Rating generated successfully")

# Test 5: Get recommendations
print("\nTest 5: Get recommendations...")
recommendations = tracker.get_diversity_recommendations()
print("\nRecommendations:")
for i, rec in enumerate(recommendations, 1):
    print(f"  {i}. {rec}")
print("✓ Recommendations generated successfully")

# Test 6: Format report
print("\nTest 6: Generate formatted report...")
report = tracker.format_holdings_report()
print("\n" + report)
print("✓ Report generated successfully")

# Test 7: Test with more diverse portfolio
print("\n\nTest 7: Testing with more diverse portfolio...")
diverse_holdings = {
    'NVDA': 5,
    'AAPL': 5,
    'MSFT': 5,
    'GOOGL': 5,
    'AMZN': 5,
}
diverse_prices = pd.Series({
    'NVDA': 500.00,
    'AAPL': 180.00,
    'MSFT': 380.00,
    'GOOGL': 140.00,
    'AMZN': 175.00,
})

diverse_tracker = HoldingsTracker(diverse_holdings, diverse_prices)
diverse_metrics = diverse_tracker.calculate_diversity_metrics()
diverse_rating = diverse_tracker.get_diversity_rating()

print(f"Diversity Rating: {diverse_rating}")
print(f"HHI: {diverse_metrics['herfindahl_index']:.4f}")
print(f"Effective Stocks: {diverse_metrics['effective_stocks']:.2f}")
print("✓ Diverse portfolio test passed")

# Test 8: Test convenience function
print("\nTest 8: Test convenience function...")
analysis = analyze_portfolio_diversity(holdings, prices)
print(f"Analysis completed: {len(analysis)} metrics calculated")
print("✓ Convenience function works")

print("\n" + "=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
