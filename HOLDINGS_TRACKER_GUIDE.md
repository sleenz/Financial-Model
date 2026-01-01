# Stock Holdings Tracker & Portfolio Rebalancing Guide

## Overview

The Portfolio Optimizer now includes a comprehensive holdings tracker that allows you to:
1. Enter your current stock holdings (ticker + number of shares)
2. Analyze portfolio diversity and concentration
3. Fetch historical data automatically
4. Get rebalancing recommendations to optimize your existing portfolio

## How to Use

### Step 1: Enter Your Current Holdings

1. Run the app: `streamlit run app/Home.py`
2. Go to **Portfolio Input** page
3. Select the **Time Period** for analysis (default: 3 years)
4. Click on **"Option 1: My Current Holdings"** tab
5. Enter each stock:
   - **Ticker**: e.g., NVDA
   - **Shares**: e.g., 9
   - Click **"Add Holding"**
6. Repeat for all your stocks

**Example:**
- NVDA: 9 shares
- AAPL: 3 shares

### Step 2: Analyze Your Portfolio

1. Click **"📊 Analyze My Portfolio & Fetch Data"**
2. The system will:
   - Fetch current prices for all your stocks
   - Calculate your total portfolio value
   - Calculate current allocation percentages
   - Fetch 3 years of historical data for optimization
   - Analyze portfolio diversity

### Step 3: Review Diversity Analysis

After analysis, you'll see:

**Key Metrics:**
- Number of stocks you own
- Diversity Rating (Highly Concentrated → Highly Diversified)
- Effective number of stocks
- Top 3 concentration percentage

**Detailed Metrics:**
- **Herfindahl Index (HHI)**: Measures concentration (0-1, lower is better)
  - < 0.15: Well diversified
  - 0.15-0.25: Moderately diversified
  - > 0.25: Concentrated
- **Gini Coefficient**: Measures inequality in position sizes
- **Top 3/5 Concentration**: % of portfolio in largest positions
- **Diversification Ratio**: How spread out your positions are

**Recommendations:**
The system provides personalized suggestions like:
- "Add more holdings" (if you have < 5 stocks)
- "Reduce top 3 concentration" (if > 70%)
- "Reduce largest position" (if > 40%)

### Step 4: Optimize & Get Rebalancing Recommendations

1. Go to **Optimization** page
2. Select your preferred optimization method:
   - Maximum Sharpe Ratio (best risk-adjusted returns)
   - Minimum Volatility (lowest risk)
   - Risk Parity (equal risk contribution)
   - HRP (Hierarchical Risk Parity)
   - Maximum Diversification
   - Equal Weight
3. Click **"Run Optimization"**

### Step 5: View Rebalancing Recommendations

The optimization page will show:

**🔄 Rebalancing Recommendations Section:**
- **Current vs Target Allocation** chart
  - Blue bars = your current weights
  - Dark blue bars = optimal target weights
- **Required Changes Table**
  - Shows current weight, target weight, and difference for each stock
  - Shows how many shares to buy or sell
- **Trading Actions**
  - Clear list of what to BUY and SELL
  - Exact share counts for each trade

**Example Output:**
```
📈 Buy:
- MSFT: Buy 5 shares
- GOOGL: Buy 3 shares

📉 Sell:
- NVDA: Sell 4 shares
```

## Example Workflow

### Scenario: You own NVDA (9 shares) and AAPL (3 shares)

**Current State (before analysis):**
- You don't know your exact allocation
- You don't know if you're diversified enough
- You don't know if rebalancing would help

**After Holdings Input:**
- NVDA: $4,500 (89.3% of portfolio)
- AAPL: $540 (10.7% of portfolio)
- Total Value: $5,040

**Diversity Analysis Shows:**
- Rating: **Highly Concentrated**
- HHI: 0.8087 (very high, bad)
- Effective Stocks: 1.24 (you basically own 1 stock)
- Recommendations:
  1. Add more holdings (you have only 2 stocks)
  2. Top position is 89% - reduce to 25-30%
  3. Consider rebalancing

**After Optimization (Max Sharpe):**
The system might recommend:
- NVDA: 30% (sell 4 shares)
- AAPL: 25% (keep as is)
- MSFT: 25% (buy 3 shares)
- GOOGL: 20% (buy 2 shares)

This gives you:
- Better diversification (4 stocks instead of 2)
- Lower concentration risk (largest position 30% vs 89%)
- Higher expected risk-adjusted returns
- HHI drops to ~0.25 (much better)

## Diversity Metrics Explained

### Herfindahl-Hirschman Index (HHI)
- **Formula**: Sum of squared weights
- **Range**: 1/N (perfectly diversified) to 1 (single stock)
- **Interpretation**:
  - 0.10-0.15: Highly diversified
  - 0.15-0.25: Moderately diversified
  - 0.25-0.40: Concentrated
  - > 0.40: Highly concentrated

### Effective Number of Stocks
- **Formula**: 1 / HHI
- **Interpretation**: How many equal-weighted stocks your portfolio "acts like"
- **Example**: 5 stocks with HHI=0.2 → Effective N = 5 (perfectly balanced)

### Gini Coefficient
- **Range**: 0 (perfect equality) to 1 (maximum inequality)
- **Interpretation**: How unequal your position sizes are
- **Example**:
  - All positions equal → Gini ≈ 0
  - One huge position, rest tiny → Gini → 1

## Tips for Best Results

1. **Enter ALL your holdings** - even small positions matter for diversity analysis
2. **Use at least 2-3 years** of historical data for reliable optimization
3. **Review diversity first** before optimizing - understand your current state
4. **Consider your risk tolerance** when choosing optimization method:
   - Conservative → Minimum Volatility
   - Balanced → Maximum Sharpe or Risk Parity
   - Aggressive → Maximum Diversification
5. **Don't over-diversify** - more than 20-30 stocks shows diminishing returns
6. **Rebalance periodically** - check quarterly or when positions drift > 5%

## Alternative: Manual Ticker Entry

If you **don't own stocks yet** or want to explore other combinations:
1. Click **"Option 2: Manual Ticker Entry"** tab
2. Enter tickers (comma-separated): `AAPL, MSFT, GOOGL, AMZN, NVDA`
3. Click **"Fetch Data & Continue"**
4. System suggests optimal allocation from scratch

## Technical Details

**Data Sources**: yfinance, Alpha Vantage, Twelve Data (automatic fallback)

**Optimization Algorithms**:
- Mean-Variance Optimization (Markowitz)
- Hierarchical Risk Parity (machine learning-based)
- Risk Parity (equal risk contribution)
- Maximum Diversification Ratio

**Constraints**:
- Maximum position size (default: 40%)
- Minimum position size (default: 0%)
- Configurable in settings

## Frequently Asked Questions

**Q: What if I have fractional shares?**
A: The system supports whole shares by default, but you can enter fractional shares (e.g., 9.5).

**Q: Can I input holdings from my CSV/Excel?**
A: Not yet for holdings, but you can upload tickers in Option 2.

**Q: Will this work with ETFs and mutual funds?**
A: Yes! Any ticker symbol that has price data will work.

**Q: How often should I rebalance?**
A: Generally quarterly, or when positions drift > 5% from targets.

**Q: What if the optimizer suggests selling at a loss?**
A: The optimizer focuses on future risk/return. Consider tax implications separately.

**Q: Can I save my holdings?**
A: Currently session-based. Bookmark the page or use the session state (future feature).

## File Structure

- `/src/portfolio/holdings.py` - Holdings tracker and diversity analysis
- `/app/pages/1_Portfolio_Input.py` - Input interface
- `/app/pages/2_Optimization.py` - Optimization and rebalancing
- `/tests/test_holdings.py` - Test suite

## Support

For issues or questions, check the README or create an issue on GitHub.
