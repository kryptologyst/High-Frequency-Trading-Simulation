"""Streamlit demo for HFT simulation."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import Config
from main import HFTSimulationPipeline


def main():
    """Main Streamlit app."""
    
    # Page configuration
    st.set_page_config(
        page_title="HFT Simulation Demo",
        page_icon="📈",
        layout="wide"
    )
    
    # Title and disclaimer
    st.title("High-Frequency Trading Simulation Demo")
    
    st.warning("""
    **DISCLAIMER**: This is a research and educational demonstration only. 
    This simulation is NOT investment advice and may contain inaccurate assumptions. 
    Past performance does not guarantee future results. Backtests are hypothetical 
    and do not reflect actual trading conditions.
    """)
    
    # Sidebar controls
    st.sidebar.header("Simulation Parameters")
    
    # Strategy selection
    strategy_type = st.sidebar.selectbox(
        "Trading Strategy",
        ["moving_average", "momentum", "mean_reversion", "ensemble"],
        index=0
    )
    
    # Data parameters
    st.sidebar.subheader("Data Parameters")
    n_periods = st.sidebar.slider("Number of Periods", 100, 1000, 288)
    initial_price = st.sidebar.slider("Initial Price", 50.0, 200.0, 100.0)
    volatility = st.sidebar.slider("Volatility", 0.01, 0.05, 0.02)
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    short_window = st.sidebar.slider("Short Window", 5, 50, 10)
    long_window = st.sidebar.slider("Long Window", 20, 100, 50)
    
    # Backtest parameters
    st.sidebar.subheader("Backtest Parameters")
    initial_capital = st.sidebar.slider("Initial Capital", 5000, 50000, 10000)
    commission = st.sidebar.slider("Commission (%)", 0.0, 0.5, 0.1) / 100
    slippage = st.sidebar.slider("Slippage (%)", 0.0, 0.2, 0.05) / 100
    
    # Run simulation button
    if st.sidebar.button("Run Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            # Create configuration
            config = Config()
            config.data.n_periods = n_periods
            config.data.initial_price = initial_price
            config.data.volatility = volatility
            config.model.strategy_type = strategy_type
            config.model.short_window = short_window
            config.model.long_window = long_window
            config.backtest.initial_capital = initial_capital
            config.backtest.commission = commission
            config.backtest.slippage = slippage
            
            # Run simulation
            pipeline = HFTSimulationPipeline(config)
            results = pipeline.run_simulation()
            
            # Store results in session state
            st.session_state.results = results
    
    # Display results if available
    if 'results' in st.session_state:
        results = st.session_state.results
        backtest_results = results['backtest_results']
        data = results['data']
        
        # Performance metrics
        st.header("Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return",
                f"{backtest_results.total_return:.2%}",
                delta=None
            )
            st.metric(
                "Annualized Return",
                f"{backtest_results.annualized_return:.2%}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Sharpe Ratio",
                f"{backtest_results.sharpe_ratio:.2f}",
                delta=None
            )
            st.metric(
                "Sortino Ratio",
                f"{backtest_results.sortino_ratio:.2f}",
                delta=None
            )
        
        with col3:
            st.metric(
                "Max Drawdown",
                f"{backtest_results.max_drawdown:.2%}",
                delta=None
            )
            st.metric(
                "Calmar Ratio",
                f"{backtest_results.calmar_ratio:.2f}",
                delta=None
            )
        
        with col4:
            st.metric(
                "Total Trades",
                f"{backtest_results.total_trades}",
                delta=None
            )
            st.metric(
                "Win Rate",
                f"{backtest_results.win_rate:.2%}",
                delta=None
            )
        
        # Charts
        st.header("Performance Charts")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Price and Signals", "Portfolio Value", "Drawdown"),
            vertical_spacing=0.08,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Price and signals
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['close'],
                name='Price',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Moving averages
        if 'sma_short' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['sma_short'],
                    name='Short MA',
                    line=dict(color='orange', dash='dash')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['sma_long'],
                    name='Long MA',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=1
            )
        
        # Buy/sell signals
        buy_signals = data[data['position'] > 0]
        sell_signals = data[data['position'] < 0]
        
        if len(buy_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['close'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(color='green', symbol='triangle-up', size=8)
                ),
                row=1, col=1
            )
        
        if len(sell_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['close'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(color='red', symbol='triangle-down', size=8)
                ),
                row=1, col=1
            )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=backtest_results.portfolio_values.index,
                y=backtest_results.portfolio_values.values,
                name='Portfolio Value',
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=backtest_results.drawdowns.index,
                y=backtest_results.drawdowns.values,
                name='Drawdown',
                fill='tonexty',
                line=dict(color='red')
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"HFT Simulation Results - {strategy_type.replace('_', ' ').title()} Strategy"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade details
        st.header("Trade Details")
        
        if backtest_results.trades:
            trades_df = pd.DataFrame([
                {
                    'Timestamp': trade.timestamp,
                    'Action': trade.action,
                    'Price': trade.price,
                    'Quantity': trade.quantity,
                    'Value': trade.value,
                    'Commission': trade.commission,
                    'Slippage': trade.slippage
                }
                for trade in backtest_results.trades
            ])
            
            st.dataframe(trades_df, use_container_width=True)
        else:
            st.info("No trades were executed during the simulation.")
        
        # Feature importance (if available)
        if 'rsi' in data.columns:
            st.header("Technical Indicators")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("RSI")
                fig_rsi = go.Figure()
                fig_rsi.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['rsi'],
                        name='RSI',
                        line=dict(color='purple')
                    )
                )
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig_rsi.update_layout(height=300)
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            with col2:
                st.subheader("Bollinger Bands")
                fig_bb = go.Figure()
                fig_bb.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['close'],
                        name='Price',
                        line=dict(color='blue')
                    )
                )
                if 'bb_upper' in data.columns:
                    fig_bb.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['bb_upper'],
                            name='Upper Band',
                            line=dict(color='red', dash='dash')
                        )
                    )
                    fig_bb.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['bb_lower'],
                            name='Lower Band',
                            line=dict(color='green', dash='dash')
                        )
                    )
                fig_bb.update_layout(height=300)
                st.plotly_chart(fig_bb, use_container_width=True)
    
    else:
        st.info("Configure parameters in the sidebar and click 'Run Simulation' to see results.")


if __name__ == "__main__":
    main()
