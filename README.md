# Portfolio Optimizer - Professional Financial Analysis Platform

A comprehensive portfolio optimization and analysis system with advanced risk analytics, fundamental analysis, and professional reporting capabilities.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.31+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

This platform provides institutional-grade portfolio analysis tools accessible through an intuitive web interface. Built for investors, financial advisors, and portfolio managers who need comprehensive analytics without the complexity of enterprise systems.

**Not financial advice.** This tool is designed to assist with portfolio analysis and calculations. Always conduct your own research (DYOR) and consult with qualified financial advisors before making investment decisions.

## ✨ Key Features

### 📊 Portfolio Management
- **Holdings Tracker**: Input your current positions with share quantities
- **Multi-Source Data Pipeline**: Automatic fallback across yfinance, Alpha Vantage, Twelve Data, and FMP
- **Real-Time Price Fetching**: Support for US and international stocks (e.g., Indonesian stocks .JK)
- **Diversity Metrics**: HHI, Gini coefficient, effective stock count, concentration ratios

### 🎯 Portfolio Optimization
- **Mean-Variance Optimization**: Classic Markowitz framework with modern solvers
- **Hierarchical Risk Parity (HRP)**: Diversification through clustering
- **Black-Litterman**: Incorporate market views into optimization
- **Risk Parity**: Equal risk contribution across assets
- **Minimum Volatility**: Conservative portfolio construction
- **Rebalancing Recommendations**: Exact share quantities to buy/sell

### 📈 Risk Analytics
- **Value at Risk (VaR)**: 90%, 95%, 99% confidence levels
- **Conditional VaR (CVaR)**: Expected shortfall analysis
- **GARCH Volatility**: Time-varying volatility forecasting
- **Maximum Drawdown**: Historical peak-to-trough analysis
- **Beta & Correlation**: Individual asset risk metrics

### 🎲 Stress Testing & Simulation
- **Historical Scenarios**: 2008 Crisis, COVID-19, Dot-com Bubble, etc.
- **Monte Carlo Simulation**: 10,000+ path simulations
- **Custom Scenarios**: Define your own market shock scenarios
- **Portfolio Impact Analysis**: Scenario-specific loss projections

### 💼 Fundamental Analysis
- **Valuation Metrics**: P/E, PEG, P/B, EV/EBITDA with signals
- **Profitability Analysis**: Margins, ROE, ROA
- **Growth Metrics**: Revenue and earnings growth YoY
- **Financial Health**: Current ratio, debt/equity, interest coverage
- **Earnings Intelligence**: Beat/miss tracking, momentum indicators
- **Health Score**: Transparent 0-10 scoring system

### 📄 Professional Reporting
- **Comprehensive Professional Report**: Executive summary, fundamentals, risk analysis, recommendations
- **Portfolio Summary**: Holdings, performance, and key metrics
- **Performance Review**: Period-specific returns and attribution
- **Risk Dashboard**: VaR, correlations, and stress test results
- **PDF Export**: Professional-grade reports for clients

### 🔍 Factor Analysis
- **Fama-French 3-Factor Model**: Market, size, and value factors
- **Factor Exposure**: Understand systematic risk sources
- **Attribution Analysis**: Decompose returns by factor

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/PortfolioOptimizer.git
cd PortfolioOptimizer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure API keys** (optional, for enhanced data sources)

Create a `.env` file in the project root:
```env
ALPHA_VANTAGE_KEY=your_key_here
TWELVE_DATA_KEY=your_key_here
FMP_KEY=your_key_here
```

*Note: yfinance works without API keys and is sufficient for most use cases.*

### Launch Web Interface

```bash
streamlit run app/Home.py
```

Navigate to `http://localhost:8501` in your browser.

### Python API Usage

```python
from src.data.data_manager import DataManager
from src.optimization.optimizers import PortfolioOptimizer
from src.risk.risk_metrics import RiskAnalyzer

# Fetch price data
dm = DataManager()
prices = dm.get_price_data(
    tickers=['AAPL', 'GOOGL', 'MSFT', 'NVDA'],
    start_date='2020-01-01',
    end_date='2024-01-01'
)

# Calculate returns
returns = prices.pct_change().dropna()

# Optimize portfolio for maximum Sharpe ratio
optimizer = PortfolioOptimizer(returns)
result = optimizer.optimize(method='max_sharpe')

print(f"Optimal Weights: {result['weights']}")
print(f"Expected Return: {result['expected_return']*100:.2f}%")
print(f"Volatility: {result['volatility']*100:.2f}%")
print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")

# Analyze risk
risk_analyzer = RiskAnalyzer(returns)
portfolio_returns = (returns * result['weights']).sum(axis=1)
var_95 = risk_analyzer.calculate_var(portfolio_returns, confidence=0.95)
print(f"VaR (95%): {var_95*100:.2f}%")
```

## 📖 Usage Guide

### 1. Input Your Holdings

Navigate to **Portfolio Input** and choose:
- **My Current Holdings**: Enter tickers and share quantities (e.g., NVDA: 10 shares)
- **Manual Ticker Entry**: Comma-separated list of tickers for exploration

Click **Analyze My Portfolio** to fetch data and calculate diversity metrics.

### 2. Optimize Your Portfolio

Go to **Optimization** and select:
- **Optimization Method**: Max Sharpe, Min Volatility, HRP, etc.
- **Constraints**: Min/max weights per asset, sector limits
- **Risk-Free Rate**: For Sharpe ratio calculations

View **Rebalancing Recommendations** with exact share quantities to trade.

### 3. Analyze Risk

Visit **Risk Analytics** for:
- VaR/CVaR calculations
- GARCH volatility forecasts
- Correlation matrices
- Individual asset risk metrics

### 4. Run Stress Tests

In **Stress Testing**:
- Select historical scenarios or run Monte Carlo
- View portfolio impact across scenarios
- Analyze worst-case outcomes

### 5. Review Fundamentals

Navigate to **Fundamentals**:
- Select stocks from your holdings or enter manually
- View health scores, valuation signals, earnings momentum
- Identify stocks with strong/weak fundamentals

### 6. Generate Reports

Go to **Reports** and:
- Select **Comprehensive Professional Report** for full analysis
- Choose charts to include
- Download PDF for sharing or recordkeeping

## 📁 Project Structure

```
PortfolioOptimizer/
├── app/
│   ├── Home.py                      # Main entry point
│   └── pages/
│       ├── 1_Portfolio_Input.py     # Data fetching & holdings input
│       ├── 2_Optimization.py        # Portfolio optimization
│       ├── 3_Risk_Analytics.py      # Risk metrics & VaR
│       ├── 4_Stress_Testing.py      # Scenarios & simulations
│       ├── 5_Monitoring.py          # Performance tracking
│       ├── 6_Factor_Analysis.py     # Factor models
│       ├── 7_Reports.py             # PDF report generation
│       └── 8_Fundamentals.py        # Fundamental analysis
├── src/
│   ├── data/                        # Data fetching & caching
│   │   ├── data_manager.py
│   │   ├── sources.py               # yfinance, Alpha Vantage, etc.
│   │   └── cache.py
│   ├── optimization/                # Optimization algorithms
│   │   ├── optimizers.py
│   │   ├── constraints.py
│   │   └── backtesting.py
│   ├── risk/                        # Risk analytics
│   │   ├── risk_metrics.py
│   │   ├── var_models.py
│   │   └── garch.py
│   ├── simulation/                  # Stress testing & Monte Carlo
│   │   ├── monte_carlo.py
│   │   └── scenarios.py
│   ├── portfolio/                   # Portfolio management
│   │   ├── holdings.py              # Holdings tracker
│   │   ├── rebalancing.py
│   │   └── position_sizing.py
│   ├── fundamentals/                # Fundamental analysis
│   │   ├── analyzer.py
│   │   ├── metrics.py
│   │   ├── earnings.py
│   │   └── health_score.py
│   ├── reports/                     # PDF report generation
│   │   ├── generator.py
│   │   ├── templates.py
│   │   └── charts.py
│   └── utils/                       # Utilities
│       ├── logger.py
│       └── helpers.py
└── requirements.txt                 # Python dependencies
```

## 🛠️ Technical Stack

- **Frontend**: Streamlit (interactive web UI)
- **Data Processing**: pandas, NumPy
- **Optimization**: SciPy, cvxpy, PyPortfolioOpt
- **Risk Analytics**: arch (GARCH), empyrical, quantstats
- **Financial Data**: yfinance, alpha_vantage, pandas_datareader
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Reporting**: ReportLab (PDF generation)
- **Machine Learning**: scikit-learn, statsmodels

## 📊 Supported Optimization Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **Max Sharpe** | Maximize risk-adjusted returns | Balanced growth |
| **Min Volatility** | Minimize portfolio variance | Conservative investors |
| **Max Return** | Maximize expected returns | Aggressive growth |
| **Risk Parity** | Equal risk contribution | Diversification |
| **HRP** | Hierarchical clustering | Complex portfolios |
| **Black-Litterman** | Bayesian framework with views | Active management |

## 🔐 Data Privacy

- All calculations run **locally** on your machine
- No portfolio data is transmitted to external servers (except for fetching public market data)
- API keys stored in `.env` are never logged or shared
- Session data cleared when browser is closed

## ⚠️ Important Disclaimers

1. **Not Financial Advice**: This tool is for educational and analytical purposes only
2. **Past Performance**: Historical returns do not guarantee future results
3. **Model Risk**: All models are simplifications and may not capture all risks
4. **Data Quality**: Results depend on accuracy of third-party data sources
5. **No Warranties**: Provided "as-is" without guarantees of accuracy or suitability

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Modern Portfolio Theory (Markowitz, 1952)
- Fama-French Factor Models
- Black-Litterman Model
- Hierarchical Risk Parity (López de Prado)
- Open-source libraries: yfinance, PyPortfolioOpt, Streamlit, and many others

## 📧 Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Check existing documentation
- Review example notebooks

---

**Built with ❤️ for better investing decisions**

*Last Updated: February 2026*
