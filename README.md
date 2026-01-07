# High-Frequency Trading Simulation

A comprehensive, research-ready framework for simulating high-frequency trading strategies with proper backtesting, risk management, and performance evaluation.

## DISCLAIMER

**This is a research and educational demonstration only. This simulation is NOT investment advice and may contain inaccurate assumptions. Past performance does not guarantee future results. Backtests are hypothetical and do not reflect actual trading conditions. Use at your own risk.**

## Overview

This project provides a modern, well-structured framework for simulating high-frequency trading strategies. It includes:

- **Multiple Trading Strategies**: Moving average crossover, momentum, mean reversion, and ensemble strategies
- **Comprehensive Feature Engineering**: Technical indicators, volatility measures, volume analysis
- **Realistic Backtesting**: Transaction costs, slippage, position sizing, and risk management
- **Performance Metrics**: Sharpe ratio, Sortino ratio, Calmar ratio, maximum drawdown, and more
- **Interactive Demo**: Streamlit-based web interface for strategy exploration
- **Production-Ready Structure**: Modular design with proper configuration management

## Features

### Trading Strategies
- **Moving Average Crossover**: Classic trend-following strategy
- **Momentum Strategy**: RSI, ROC, and momentum-based signals
- **Mean Reversion**: Bollinger Bands and Z-score based strategies
- **Ensemble Strategy**: Combines multiple approaches with weighted voting

### Technical Indicators
- Simple and Exponential Moving Averages
- Relative Strength Index (RSI)
- Bollinger Bands
- Rate of Change (ROC)
- On-Balance Volume (OBV)
- Price-Volume Trend (PVT)
- Volatility measures

### Risk Management
- Position sizing controls
- Transaction cost modeling
- Slippage simulation
- Drawdown monitoring
- Stop-loss and take-profit (configurable)

### Performance Evaluation
- Total and annualized returns
- Risk-adjusted metrics (Sharpe, Sortino, Calmar ratios)
- Maximum drawdown analysis
- Trade statistics (win rate, profit factor)
- Portfolio value tracking

## Installation

### Prerequisites
- Python 3.10+
- pip or conda

### Setup

1. Clone or download the project:
```bash
git clone https://github.com/kryptologyst/High-Frequency-Trading-Simulation.git
cd High-Frequency-Trading-Simulation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
mkdir -p data/{raw,processed,external} assets logs
```

## Quick Start

### Command Line Usage

Run the simulation with default parameters:
```bash
python src/main.py
```

### Interactive Demo

Launch the Streamlit demo:
```bash
streamlit run demo/app.py
```

The demo provides an interactive interface to:
- Configure simulation parameters
- Select different trading strategies
- Visualize results with interactive charts
- Analyze performance metrics
- Review trade details

## Configuration

The simulation can be configured through YAML files in the `configs/` directory:

- `default.yaml`: Default configuration
- Custom configurations can be created for different scenarios

### Key Configuration Parameters

**Data Parameters:**
- `n_periods`: Number of simulation periods (default: 288 for 1 day of 5-minute data)
- `initial_price`: Starting price (default: $100)
- `volatility`: Price volatility (default: 0.02)
- `drift`: Price drift (default: 0.0)

**Model Parameters:**
- `strategy_type`: Trading strategy ("moving_average", "momentum", "mean_reversion", "ensemble")
- `short_window`: Short moving average window (default: 10)
- `long_window`: Long moving average window (default: 50)

**Backtest Parameters:**
- `initial_capital`: Starting capital (default: $10,000)
- `commission`: Transaction cost percentage (default: 0.1%)
- `slippage`: Slippage percentage (default: 0.05%)

## Project Structure

```
├── src/                    # Source code
│   ├── data/              # Data generation and loading
│   ├── features/          # Feature engineering
│   ├── models/            # Trading strategies
│   ├── backtest/          # Backtesting engine
│   ├── risk/              # Risk management
│   └── utils/             # Utilities and configuration
├── configs/               # Configuration files
├── data/                  # Data storage
│   ├── raw/              # Raw market data
│   └── processed/        # Processed features
├── assets/               # Output files and results
├── demo/                 # Streamlit demo
├── tests/                # Unit tests
├── notebooks/            # Jupyter notebooks
└── scripts/              # Utility scripts
```

## Usage Examples

### Basic Simulation

```python
from src.utils.config import Config
from src.main import HFTSimulationPipeline

# Load configuration
config = Config()

# Create and run pipeline
pipeline = HFTSimulationPipeline(config)
results = pipeline.run_simulation()

# Access results
print(f"Total Return: {results['backtest_results'].total_return:.2%}")
print(f"Sharpe Ratio: {results['backtest_results'].sharpe_ratio:.2f}")
```

### Custom Strategy

```python
from src.models.strategies import MomentumStrategy
from src.utils.config import ModelConfig

# Create custom configuration
config = ModelConfig()
config.strategy_type = "momentum"
config.momentum_window = 15

# Use custom strategy
strategy = MomentumStrategy(config)
signals = strategy.generate_signals(data_with_features)
```

### Advanced Backtesting

```python
from src.backtest.engine import BacktestEngine, BacktestConfig

# Configure backtesting
backtest_config = BacktestConfig()
backtest_config.commission = 0.002  # 0.2% commission
backtest_config.slippage = 0.001    # 0.1% slippage

# Run backtest
engine = BacktestEngine(backtest_config)
results = engine.run_backtest(data_with_signals)
```

## Performance Metrics

The framework calculates comprehensive performance metrics:

### Return Metrics
- **Total Return**: Overall percentage return
- **Annualized Return**: Return adjusted for time period
- **Volatility**: Standard deviation of returns

### Risk-Adjusted Metrics
- **Sharpe Ratio**: Return per unit of risk
- **Sortino Ratio**: Return per unit of downside risk
- **Calmar Ratio**: Return relative to maximum drawdown

### Risk Metrics
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Drawdown Duration**: Time spent in drawdown

### Trade Statistics
- **Total Trades**: Number of executed trades
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Average Win/Loss**: Average profit/loss per trade

## Data Schema

### Market Data Format
```csv
timestamp,open,high,low,close,volume
2023-01-01 00:00:00,100.0,101.2,99.8,100.5,1000000
2023-01-01 00:05:00,100.5,100.8,100.2,100.3,950000
...
```

### Generated Features
The framework automatically generates technical indicators:
- Moving averages (SMA, EMA)
- Momentum indicators (RSI, ROC)
- Volatility measures (Bollinger Bands)
- Volume indicators (OBV, PVT)
- Mean reversion signals

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
ruff src/
```

### Adding New Strategies

1. Create a new strategy class inheriting from `BaseStrategy`
2. Implement the `generate_signals` method
3. Add the strategy to `StrategyFactory`
4. Update configuration options

### Adding New Features

1. Add feature calculation methods to `FeatureEngineer`
2. Update the `engineer_all_features` method
3. Ensure proper handling of NaN values and edge cases

## Limitations and Considerations

### Data Limitations
- Uses synthetic data for demonstration
- Real market data requires proper licensing
- Historical data may not reflect current market conditions

### Model Limitations
- Strategies are simplified for educational purposes
- Real trading involves additional complexities (liquidity, market impact, etc.)
- Past performance does not guarantee future results

### Risk Considerations
- All simulations are hypothetical
- Transaction costs and slippage are simplified
- Market conditions can change rapidly
- High-frequency trading requires sophisticated infrastructure

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure code passes linting
6. Submit a pull request

## License

This project is for educational and research purposes only. See the disclaimer above for important limitations.

## Support

For questions or issues:
1. Check the documentation
2. Review the example notebooks
3. Open an issue on the repository

## Acknowledgments

This project is inspired by modern quantitative finance practices and educational materials. It is designed for learning and research purposes only.
# High-Frequency-Trading-Simulation
