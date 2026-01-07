"""Backtesting engine for HFT trading strategies."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

from utils.config import BacktestConfig


@dataclass
class Trade:
    """Represents a single trade."""
    
    timestamp: pd.Timestamp
    action: str  # 'buy' or 'sell'
    price: float
    quantity: float
    value: float
    commission: float
    slippage: float


@dataclass
class BacktestResults:
    """Results from backtesting."""
    
    # Performance metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Portfolio values
    portfolio_values: pd.Series
    returns: pd.Series
    drawdowns: pd.Series
    
    # Trade history
    trades: List[Trade]
    
    # Final portfolio state
    final_capital: float
    final_position: float


class BacktestEngine:
    """Engine for backtesting trading strategies."""
    
    def __init__(self, config: BacktestConfig):
        """Initialize the backtesting engine.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def run_backtest(self, df: pd.DataFrame) -> BacktestResults:
        """Run backtest on the given data.
        
        Args:
            df: DataFrame with signals and prices
            
        Returns:
            BacktestResults object with performance metrics
        """
        self.logger.info("Starting backtest")
        
        # Initialize portfolio
        capital = self.config.initial_capital
        position = 0.0
        portfolio_values = []
        trades = []
        
        # Track portfolio state
        for timestamp, row in df.iterrows():
            current_price = row['close']
            
            # Execute trades based on position changes
            if row['position'] != 0:
                trade = self._execute_trade(
                    timestamp, row['position'], current_price, capital, position
                )
                
                if trade:
                    trades.append(trade)
                    capital += trade.value - trade.commission - trade.slippage
                    position += trade.quantity
            
            # Calculate portfolio value
            portfolio_value = capital + (position * current_price)
            portfolio_values.append(portfolio_value)
        
        # Create results
        portfolio_series = pd.Series(portfolio_values, index=df.index)
        returns = portfolio_series.pct_change().dropna()
        
        results = self._calculate_metrics(
            portfolio_series, returns, trades, capital, position
        )
        
        self.logger.info(f"Backtest completed: {results.total_return:.2%} total return")
        return results
    
    def _execute_trade(self, timestamp: pd.Timestamp, position_change: float, 
                      price: float, capital: float, current_position: float) -> Optional[Trade]:
        """Execute a trade based on position change.
        
        Args:
            timestamp: Trade timestamp
            position_change: Change in position (positive = buy, negative = sell)
            price: Current price
            capital: Available capital
            current_position: Current position
            
        Returns:
            Trade object if trade is executed, None otherwise
        """
        if position_change == 0:
            return None
        
        # Calculate trade quantity
        if position_change > 0:  # Buy
            # Use available capital
            max_quantity = capital / price
            quantity = min(position_change, max_quantity)
            action = 'buy'
        else:  # Sell
            # Use current position
            quantity = min(abs(position_change), current_position)
            action = 'sell'
        
        if quantity <= 0:
            return None
        
        # Calculate trade value and costs
        value = quantity * price
        commission = value * self.config.commission
        slippage = value * self.config.slippage
        
        return Trade(
            timestamp=timestamp,
            action=action,
            price=price,
            quantity=quantity,
            value=value,
            commission=commission,
            slippage=slippage
        )
    
    def _calculate_metrics(self, portfolio_values: pd.Series, returns: pd.Series,
                          trades: List[Trade], final_capital: float, 
                          final_position: float) -> BacktestResults:
        """Calculate performance metrics.
        
        Args:
            portfolio_values: Series of portfolio values
            returns: Series of returns
            trades: List of executed trades
            final_capital: Final capital amount
            final_position: Final position
            
        Returns:
            BacktestResults object
        """
        # Basic metrics
        total_return = (portfolio_values.iloc[-1] - portfolio_values.iloc[0]) / portfolio_values.iloc[0]
        
        # Annualized metrics (assuming 5-minute intervals)
        periods_per_year = 252 * 24 * 12  # 5-minute periods in a year
        # For short periods, use simple scaling instead of compound annualization
        if len(portfolio_values) < periods_per_year:
            annualized_return = total_return * (periods_per_year / len(portfolio_values))
        else:
            annualized_return = (1 + total_return) ** (periods_per_year / len(portfolio_values)) - 1
        volatility = returns.std() * np.sqrt(periods_per_year)
        
        # Risk metrics
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(periods_per_year)
        sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0
        
        # Drawdown calculation
        rolling_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        total_trades = len(trades)
        if total_trades > 0:
            trade_pnl = []
            for i in range(0, len(trades), 2):
                if i + 1 < len(trades):
                    buy_trade = trades[i] if trades[i].action == 'buy' else trades[i + 1]
                    sell_trade = trades[i + 1] if trades[i].action == 'buy' else trades[i]
                    
                    pnl = (sell_trade.price - buy_trade.price) * buy_trade.quantity
                    trade_pnl.append(pnl)
            
            if trade_pnl:
                winning_trades = sum(1 for pnl in trade_pnl if pnl > 0)
                losing_trades = sum(1 for pnl in trade_pnl if pnl < 0)
                win_rate = winning_trades / len(trade_pnl)
                
                wins = [pnl for pnl in trade_pnl if pnl > 0]
                losses = [pnl for pnl in trade_pnl if pnl < 0]
                
                avg_win = np.mean(wins) if wins else 0
                avg_loss = np.mean(losses) if losses else 0
                
                profit_factor = abs(sum(wins) / sum(losses)) if losses else float('inf')
            else:
                winning_trades = losing_trades = win_rate = avg_win = avg_loss = profit_factor = 0
        else:
            winning_trades = losing_trades = win_rate = avg_win = avg_loss = profit_factor = 0
        
        return BacktestResults(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            portfolio_values=portfolio_values,
            returns=returns,
            drawdowns=drawdowns,
            trades=trades,
            final_capital=final_capital,
            final_position=final_position
        )
