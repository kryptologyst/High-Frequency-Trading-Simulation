"""Test script for HFT simulation."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import Config, DataConfig, ModelConfig, BacktestConfig
from data.generator import MarketDataGenerator
from features.engineer import FeatureEngineer
from models.strategies import StrategyFactory, MovingAverageStrategy
from backtest.engine import BacktestEngine


class TestConfig:
    """Test configuration management."""
    
    def test_config_creation(self):
        """Test configuration object creation."""
        config = Config()
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.backtest, BacktestConfig)
        assert config.random_seed == 42
    
    def test_config_from_yaml(self):
        """Test loading configuration from YAML."""
        config_path = "configs/default.yaml"
        if Path(config_path).exists():
            config = Config.from_yaml(config_path)
            assert config.data.n_periods == 288
            assert config.model.strategy_type == "moving_average"


class TestDataGenerator:
    """Test data generation."""
    
    def test_data_generation(self):
        """Test synthetic data generation."""
        config = DataConfig(n_periods=100, seed=42)
        generator = MarketDataGenerator(config)
        
        data = generator.generate_price_data()
        
        assert len(data) == 100
        assert 'open' in data.columns
        assert 'high' in data.columns
        assert 'low' in data.columns
        assert 'close' in data.columns
        assert 'volume' in data.columns
        
        # Check OHLC relationships
        assert (data['high'] >= data['low']).all()
        assert (data['high'] >= data['open']).all()
        assert (data['high'] >= data['close']).all()
        assert (data['low'] <= data['open']).all()
        assert (data['low'] <= data['close']).all()


class TestFeatureEngineer:
    """Test feature engineering."""
    
    def test_moving_averages(self):
        """Test moving average calculation."""
        config = ModelConfig(short_window=5, long_window=10)
        engineer = FeatureEngineer(config)
        
        # Create sample data
        data = pd.DataFrame({
            'close': np.random.randn(20).cumsum() + 100
        })
        
        result = engineer.add_moving_averages(data)
        
        assert 'sma_short' in result.columns
        assert 'sma_long' in result.columns
        assert 'ema_short' in result.columns
        assert 'ema_long' in result.columns
        assert 'ma_crossover' in result.columns
        
        # Check that moving averages are calculated correctly
        assert not result['sma_short'].isna().all()
        assert not result['sma_long'].isna().all()
    
    def test_momentum_features(self):
        """Test momentum feature calculation."""
        config = ModelConfig(momentum_window=10, rsi_window=14)
        engineer = FeatureEngineer(config)
        
        # Create sample data
        data = pd.DataFrame({
            'close': np.random.randn(50).cumsum() + 100
        })
        
        result = engineer.add_momentum_features(data)
        
        assert 'momentum' in result.columns
        assert 'roc' in result.columns
        assert 'rsi' in result.columns
        assert 'rsi_oversold' in result.columns
        assert 'rsi_overbought' in result.columns
        
        # Check RSI bounds
        assert (result['rsi'] >= 0).all()
        assert (result['rsi'] <= 100).all()


class TestStrategies:
    """Test trading strategies."""
    
    def test_moving_average_strategy(self):
        """Test moving average strategy."""
        config = ModelConfig(short_window=5, long_window=10)
        strategy = MovingAverageStrategy(config)
        
        # Create sample data with features
        data = pd.DataFrame({
            'close': np.random.randn(50).cumsum() + 100,
            'sma_short': np.random.randn(50).cumsum() + 100,
            'sma_long': np.random.randn(50).cumsum() + 100
        })
        
        result = strategy.generate_signals(data)
        
        assert 'signal' in result.columns
        assert 'position' in result.columns
        
        # Check signal values
        assert result['signal'].isin([0, 1]).all()
        assert result['position'].isin([-1, 0, 1]).all()
    
    def test_strategy_factory(self):
        """Test strategy factory."""
        config = ModelConfig()
        
        # Test all strategy types
        for strategy_type in ['moving_average', 'momentum', 'mean_reversion', 'ensemble']:
            strategy = StrategyFactory.create_strategy(strategy_type, config)
            assert strategy is not None
            assert hasattr(strategy, 'generate_signals')


class TestBacktestEngine:
    """Test backtesting engine."""
    
    def test_backtest_execution(self):
        """Test backtest execution."""
        config = BacktestConfig(initial_capital=10000)
        engine = BacktestEngine(config)
        
        # Create sample data with signals
        data = pd.DataFrame({
            'close': np.random.randn(50).cumsum() + 100,
            'position': np.random.choice([-1, 0, 1], 50)
        })
        data.index = pd.date_range('2023-01-01', periods=50, freq='5T')
        
        results = engine.run_backtest(data)
        
        assert results.total_return is not None
        assert results.annualized_return is not None
        assert results.volatility is not None
        assert results.sharpe_ratio is not None
        assert results.max_drawdown is not None
        assert results.total_trades >= 0
        assert results.win_rate >= 0
        assert results.win_rate <= 1
        
        # Check portfolio values
        assert len(results.portfolio_values) == len(data)
        assert results.portfolio_values.iloc[0] == config.initial_capital


if __name__ == "__main__":
    pytest.main([__file__])
