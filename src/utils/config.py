"""Configuration management for HFT simulation."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import yaml
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data generation and loading."""
    
    # Data generation parameters
    n_periods: int = 288  # Number of 5-minute periods (1 day)
    initial_price: float = 100.0
    volatility: float = 0.02
    drift: float = 0.0
    seed: int = 42
    
    # Data paths
    data_dir: str = "data"
    raw_data_path: str = "data/raw/market_data.csv"
    processed_data_path: str = "data/processed/features.csv"


@dataclass
class ModelConfig:
    """Configuration for trading models."""
    
    # Moving average parameters
    short_window: int = 10
    long_window: int = 50
    
    # Advanced strategy parameters
    momentum_window: int = 20
    volatility_window: int = 20
    rsi_window: int = 14
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    
    # Model selection
    strategy_type: str = "moving_average"  # moving_average, momentum, mean_reversion, ensemble


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    
    # Initial capital
    initial_capital: float = 10000.0
    
    # Transaction costs
    commission: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    
    # Risk management
    max_position_size: float = 1.0  # Maximum position as fraction of capital
    stop_loss: Optional[float] = None  # Stop loss percentage
    take_profit: Optional[float] = None  # Take profit percentage


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    
    # Benchmark settings
    benchmark_symbol: str = "SPY"
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    
    # Evaluation periods
    evaluation_frequency: str = "daily"  # daily, weekly, monthly
    
    # Metrics to compute
    compute_sharpe: bool = True
    compute_sortino: bool = True
    compute_calmar: bool = True
    compute_max_drawdown: bool = True


@dataclass
class Config:
    """Main configuration class."""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # General settings
    random_seed: int = 42
    device: str = "auto"  # auto, cpu, cuda, mps
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create nested config objects
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        backtest_config = BacktestConfig(**config_dict.get('backtest', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        
        return cls(
            data=data_config,
            model=model_config,
            backtest=backtest_config,
            evaluation=evaluation_config,
            random_seed=config_dict.get('random_seed', 42),
            device=config_dict.get('device', 'auto')
        )
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'backtest': self.backtest.__dict__,
            'evaluation': self.evaluation.__dict__,
            'random_seed': self.random_seed,
            'device': self.device
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
