"""Data generation and loading utilities for HFT simulation."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from pathlib import Path
import logging

from utils.config import DataConfig


class MarketDataGenerator:
    """Generate synthetic market data for HFT simulation."""
    
    def __init__(self, config: DataConfig):
        """Initialize the data generator.
        
        Args:
            config: Data configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def generate_price_data(self) -> pd.DataFrame:
        """Generate synthetic price data using geometric Brownian motion.
        
        Returns:
            DataFrame with OHLCV data
        """
        np.random.seed(self.config.seed)
        
        # Generate timestamps (5-minute intervals)
        timestamps = pd.date_range(
            "2023-01-01", 
            periods=self.config.n_periods, 
            freq="5min"
        )
        
        # Generate price returns using GBM
        dt = 1 / (252 * 24 * 12)  # 5-minute intervals in years
        returns = np.random.normal(
            self.config.drift * dt,
            self.config.volatility * np.sqrt(dt),
            self.config.n_periods
        )
        
        # Generate price levels
        log_prices = np.log(self.config.initial_price) + np.cumsum(returns)
        prices = np.exp(log_prices)
        
        # Generate OHLCV data
        data = []
        for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
            # Add some intraday volatility for OHLC
            volatility_factor = np.random.uniform(0.8, 1.2)
            high = price * volatility_factor
            low = price / volatility_factor
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            
            # Generate volume (higher volume during volatile periods)
            base_volume = 1000000
            volume_factor = 1 + abs(returns[i]) * 10
            volume = int(base_volume * volume_factor * np.random.uniform(0.5, 1.5))
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        self.logger.info(f"Generated {len(df)} data points from {df.index[0]} to {df.index[-1]}")
        return df
    
    def save_data(self, df: pd.DataFrame, filepath: str) -> None:
        """Save data to CSV file.
        
        Args:
            df: DataFrame to save
            filepath: Path to save the file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath)
        self.logger.info(f"Data saved to {filepath}")
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file.
        
        Args:
            filepath: Path to the data file
            
        Returns:
            Loaded DataFrame
        """
        df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        self.logger.info(f"Data loaded from {filepath}: {len(df)} rows")
        return df


def generate_sample_data(config: DataConfig) -> pd.DataFrame:
    """Generate sample market data for demonstration.
    
    Args:
        config: Data configuration
        
    Returns:
        Generated market data DataFrame
    """
    generator = MarketDataGenerator(config)
    data = generator.generate_price_data()
    
    # Save the data
    generator.save_data(data, config.raw_data_path)
    
    return data
