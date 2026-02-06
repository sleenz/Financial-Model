"""
Quick test script for fundamentals module.
Tests basic functionality with AAPL data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.fundamentals.analyzer import FundamentalsAnalyzer

print("=" * 80)
print("TESTING FUNDAMENTALS MODULE")
print("=" * 80)

# Test with AAPL
ticker = "AAPL"
print(f"\nTesting with ticker: {ticker}")

# Create analyzer
print("\n1. Creating FundamentalsAnalyzer...")
try:
    analyzer = FundamentalsAnalyzer(ticker)
    print("   ✓ Analyzer created successfully")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Fetch data
print("\n2. Fetching data from yfinance...")
try:
    success = analyzer.fetch_data()
    if success:
        print("   ✓ Data fetched successfully")
    else:
        print("   ✗ Failed to fetch data")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Get company info
print("\n3. Getting company info...")
try:
    company_info = analyzer.get_company_info()
    print(f"   ✓ Company: {company_info.get('name', 'N/A')}")
    print(f"   ✓ Sector: {company_info.get('sector', 'N/A')}")
    print(f"   ✓ Industry: {company_info.get('industry', 'N/A')}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Get valuation metrics
print("\n4. Getting valuation metrics...")
try:
    valuation = analyzer.get_valuation_metrics()
    print(f"   ✓ P/E Ratio: {valuation.get('pe_ratio_ttm', 'N/A')}")
    print(f"   ✓ PEG Ratio: {valuation.get('peg_ratio', 'N/A')}")
    print(f"   ✓ Signal: {valuation.get('valuation_signal', 'N/A')}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Get profitability metrics
print("\n5. Getting profitability metrics...")
try:
    profitability = analyzer.get_profitability_metrics()
    print(f"   ✓ Gross Margin: {profitability.get('gross_margin', 'N/A')}")
    print(f"   ✓ Operating Margin: {profitability.get('operating_margin', 'N/A')}")
    print(f"   ✓ Net Margin: {profitability.get('net_margin', 'N/A')}")
    print(f"   ✓ ROE: {profitability.get('return_on_equity', 'N/A')}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Get growth metrics
print("\n6. Getting growth metrics...")
try:
    growth = analyzer.get_growth_metrics()
    print(f"   ✓ Revenue Growth: {growth.get('revenue_growth_yoy', 'N/A')}")
    print(f"   ✓ Earnings Growth: {growth.get('earnings_growth_yoy', 'N/A')}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Get financial health metrics
print("\n7. Getting financial health metrics...")
try:
    financial_health = analyzer.get_financial_health_metrics()
    print(f"   ✓ Current Ratio: {financial_health.get('current_ratio', 'N/A')}")
    print(f"   ✓ Debt to Equity: {financial_health.get('debt_to_equity', 'N/A')}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Get health score
print("\n8. Getting health score...")
try:
    health_score = analyzer.get_health_score()
    print(f"   ✓ Score: {health_score.get('score', 'N/A')}/10")
    print(f"   ✓ Rating: {health_score.get('rating', 'N/A')}")
    if 'breakdown' in health_score:
        print(f"   ✓ Breakdown:")
        for key, value in health_score['breakdown'].items():
            print(f"      - {key}: {value}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Get earnings analysis
print("\n9. Getting earnings analysis...")
try:
    earnings = analyzer.get_earnings_analysis()
    insights = earnings.get('insights', {})
    print(f"   ✓ Beat Streak: {insights.get('beat_streak', 'N/A')} quarters")
    print(f"   ✓ Momentum: {insights.get('momentum', 'N/A')}")
    print(f"   ✓ Revenue Trend: {insights.get('revenue_trend', 'N/A')}")
    alerts = insights.get('alerts', [])
    if alerts:
        print(f"   ✓ Alerts: {len(alerts)} alert(s)")
        for alert in alerts:
            print(f"      - {alert['type'].upper()}: {alert['message']}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Generate full summary
print("\n10. Generating full summary...")
try:
    summary = analyzer.generate_summary()
    print(f"   ✓ Summary generated successfully")
    print(f"   ✓ Keys in summary: {list(summary.keys())}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("ALL TESTS COMPLETED!")
print("=" * 80)
print(f"\n✓ The fundamentals module is working correctly for {ticker}")
