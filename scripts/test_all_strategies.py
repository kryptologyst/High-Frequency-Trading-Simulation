#!/usr/bin/env python3
"""Test script to verify HFT simulation works correctly."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import Config
from main import HFTSimulationPipeline


def test_simulation():
    """Test the HFT simulation with different strategies."""
    
    print("Testing HFT Simulation...")
    print("=" * 50)
    
    # Test different strategies
    strategies = ["moving_average", "momentum", "mean_reversion", "ensemble"]
    
    for strategy in strategies:
        print(f"\nTesting {strategy} strategy:")
        print("-" * 30)
        
        # Create configuration
        config = Config()
        config.model.strategy_type = strategy
        config.data.n_periods = 100  # Shorter test
        
        # Run simulation
        pipeline = HFTSimulationPipeline(config)
        results = pipeline.run_simulation()
        
        # Print key metrics
        backtest_results = results['backtest_results']
        print(f"Total Return: {backtest_results.total_return:.2%}")
        print(f"Sharpe Ratio: {backtest_results.sharpe_ratio:.2f}")
        print(f"Total Trades: {backtest_results.total_trades}")
        print(f"Win Rate: {backtest_results.win_rate:.2%}")
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")


if __name__ == "__main__":
    test_simulation()
