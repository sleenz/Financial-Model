# Advanced Portfolio Optimization System

A production-ready portfolio optimization and risk management system with a Streamlit web interface, implementing cutting-edge quantitative finance techniques.

## Features

- **Multi-Source Data Pipeline**: Intelligent fallback between yfinance, Alpha Vantage, Twelve Data, and FMP
- **Advanced Optimization**: Mean-Variance, HRP, Black-Litterman, Risk Parity, and more
- **Comprehensive Risk Analytics**: VaR, CVaR, GARCH volatility, drawdown metrics
- **Stress Testing**: Historical scenarios and Monte Carlo simulations
- **Position Sizing**: Actionable investment recommendations with share counts

## Installation

```bash
# Clone the repository
git clone https://github.com/example/portfolio-optimizer.git
cd portfolio-optimizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file and add your API keys
cp .env.example .env
```

## Configuration

Edit `.env` with your API keys:

```env
ALPHA_VANTAGE_KEY=your_key_here
TWELVE_DATA_KEY=your_key_here
FMP_KEY=your_key_here
```

## Usage

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
└── docs/               # Documentation
```

## License

MIT License
