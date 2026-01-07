"""Feature engineering for HFT trading strategies."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

from utils.config import ModelConfig


class FeatureEngineer:
    """Engineer technical indicators and features for trading strategies."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the feature engineer.
        
        Args:
            config: Model configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving average features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with moving average features added
        """
        df = df.copy()
        
        # Simple moving averages
        df['sma_short'] = df['close'].rolling(
            window=self.config.short_window, 
            min_periods=1
        ).mean()
        
        df['sma_long'] = df['close'].rolling(
            window=self.config.long_window, 
            min_periods=1
        ).mean()
        
        # Exponential moving averages
        df['ema_short'] = df['close'].ewm(
            span=self.config.short_window, 
            adjust=False
        ).mean()
        
        df['ema_long'] = df['close'].ewm(
            span=self.config.long_window, 
            adjust=False
        ).mean()
        
        # Moving average crossover signals
        df['ma_crossover'] = np.where(
            df['sma_short'] > df['sma_long'], 1, -1
        )
        
        self.logger.info("Added moving average features")
        return df
    
    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with momentum features added
        """
        df = df.copy()
        
        # Price momentum
        df['momentum'] = df['close'].pct_change(self.config.momentum_window)
        
        # Rate of Change (ROC)
        df['roc'] = df['close'].pct_change(self.config.momentum_window) * 100
        
        # Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(
            window=self.config.rsi_window
        ).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(
            window=self.config.rsi_window
        ).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI signals
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        self.logger.info("Added momentum features")
        return df
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volatility features added
        """
        df = df.copy()
        
        # Rolling volatility
        df['volatility'] = df['close'].pct_change().rolling(
            window=self.config.volatility_window
        ).std() * np.sqrt(252 * 24 * 12)  # Annualized
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(
            window=self.config.bollinger_window
        ).mean()
        
        bb_std = df['close'].rolling(
            window=self.config.bollinger_window
        ).std()
        
        df['bb_upper'] = df['bb_middle'] + (
            bb_std * self.config.bollinger_std
        )
        df['bb_lower'] = df['bb_middle'] - (
            bb_std * self.config.bollinger_std
        )
        
        # Bollinger Band position
        df['bb_position'] = (
            (df['close'] - df['bb_lower']) / 
            (df['bb_upper'] - df['bb_lower'])
        )
        
        # Bollinger Band signals
        df['bb_squeeze'] = (
            (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] < 0.1
        ).astype(int)
        
        self.logger.info("Added volatility features")
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume features added
        """
        df = df.copy()
        
        # Volume moving average
        df['volume_sma'] = df['volume'].rolling(
            window=self.config.short_window
        ).mean()
        
        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price-volume trend
        df['pvt'] = (df['close'].pct_change() * df['volume']).cumsum()
        
        # On-Balance Volume (OBV)
        df['obv'] = np.where(
            df['close'] > df['close'].shift(1),
            df['volume'],
            np.where(
                df['close'] < df['close'].shift(1),
                -df['volume'],
                0
            )
        ).cumsum()
        
        self.logger.info("Added volume features")
        return df
    
    def add_mean_reversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add mean reversion features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with mean reversion features added
        """
        df = df.copy()
        
        # Price deviation from moving average
        df['price_deviation'] = (
            df['close'] - df['sma_long']
        ) / df['sma_long']
        
        # Z-score of price
        df['price_zscore'] = (
            df['close'] - df['close'].rolling(50).mean()
        ) / df['close'].rolling(50).std()
        
        # Mean reversion signals
        df['mean_reversion_long'] = (df['price_zscore'] < -2).astype(int)
        df['mean_reversion_short'] = (df['price_zscore'] > 2).astype(int)
        
        self.logger.info("Added mean reversion features")
        return df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer all available features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all features added
        """
        df = df.copy()
        
        # Add all feature categories
        df = self.add_moving_averages(df)
        df = self.add_momentum_features(df)
        df = self.add_volatility_features(df)
        df = self.add_volume_features(df)
        df = self.add_mean_reversion_features(df)
        
        # Add additional derived features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            df[f'price_change_lag_{lag}'] = df['price_change'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        self.logger.info(f"Engineered {len(df.columns)} total features")
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of engineered feature names.
        
        Args:
            df: DataFrame with features
            
        Returns:
            List of feature column names
        """
        # Exclude OHLCV and timestamp columns
        exclude_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
