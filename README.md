# Advanced Portfolio Optimization System
Nothing too fancy, just a portfolio optimizer and risk calculation model. This project aims to provide help for financial analysis but not intended to be used as the main tool for financial analysis. DYOR and used this as an additional calculation assistant program.  

## Features

- **Multi-Source Data Pipeline**: Intelligent fallback between yfinance, Alpha Vantage, Twelve Data, and FMP
- **Optimization**: Mean-Variance, HRP, Black-Litterman, Risk Parity, and more
- **Risk Analytics**: VaR, CVaR, GARCH volatility, drawdown metrics
- **Stress Testing**: Historical scenarios and Monte Carlo simulations
- **Position Sizing**: Actionable investment recommendations with share counts
  
## Configuration

Edit `.env` with your API keys:

```env
ALPHA_VANTAGE_KEY=your_key_here
TWELVE_DATA_KEY=your_key_here
FMP_KEY=your_key_here
```

### Web Interface

```bash
streamlit run app/Home.py
```

### Python API

```python
from src.data.data_manager import DataManager
from src.optimization.optimizers import PortfolioOptimizer

# Fetch data
dm = DataManager()
prices = dm.get_price_data(['AAPL', 'GOOGL', 'MSFT'], '2020-01-01', '2024-01-01')

# Optimize portfolio
optimizer = PortfolioOptimizer(prices)
weights = optimizer.optimize(method='max_sharpe')
```

## Project Structure

```
portfolio-optimizer/
├── src/
│   ├── data/           # Data fetching and caching
│   ├── optimization/   # Portfolio optimization algorithms
│   ├── risk/           # Risk metrics and GARCH models
│   ├── simulation/     # Monte Carlo and stress testing
│   ├── portfolio/      # Position sizing and rebalancing
│   └── utils/          # Logging and helper functions
├── app/                # Streamlit web interface
├── tests/              # Unit tests
├── examples/           # Example scripts
```
