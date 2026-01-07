"""Main pipeline for HFT simulation."""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional

from utils.config import Config
from utils.device import set_seed, get_device
from data.generator import MarketDataGenerator
from features.engineer import FeatureEngineer
from models.strategies import StrategyFactory
from backtest.engine import BacktestEngine


class HFTSimulationPipeline:
    """Main pipeline for High-Frequency Trading simulation."""
    
    def __init__(self, config: Config):
        """Initialize the pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Set random seeds
        set_seed(config.random_seed)
        
        # Initialize components
        self.data_generator = MarketDataGenerator(config.data)
        self.feature_engineer = FeatureEngineer(config.model)
        self.strategy = StrategyFactory.create_strategy(
            config.model.strategy_type, config.model
        )
        self.backtest_engine = BacktestEngine(config.backtest)
        
        # Set device
        self.device = get_device(config.device)
        self.logger.info(f"Using device: {self.device}")
    
    def run_simulation(self, data_path: Optional[str] = None) -> dict:
        """Run the complete HFT simulation pipeline.
        
        Args:
            data_path: Optional path to existing data file
            
        Returns:
            Dictionary containing results and metrics
        """
        self.logger.info("Starting HFT simulation pipeline")
        
        # Step 1: Generate or load data
        if data_path and Path(data_path).exists():
            self.logger.info(f"Loading data from {data_path}")
            df = self.data_generator.load_data(data_path)
        else:
            self.logger.info("Generating synthetic market data")
            df = self.data_generator.generate_price_data()
            
            # Save generated data
            self.data_generator.save_data(df, self.config.data.raw_data_path)
        
        # Step 2: Engineer features
        self.logger.info("Engineering features")
        df_with_features = self.feature_engineer.engineer_all_features(df)
        
        # Step 3: Generate trading signals
        self.logger.info("Generating trading signals")
        df_with_signals = self.strategy.generate_signals(df_with_features)
        
        # Step 4: Run backtest
        self.logger.info("Running backtest")
        backtest_results = self.backtest_engine.run_backtest(df_with_signals)
        
        # Step 5: Compile results
        results = {
            'data': df_with_signals,
            'backtest_results': backtest_results,
            'config': self.config,
            'strategy_type': self.config.model.strategy_type
        }
        
        self.logger.info("HFT simulation pipeline completed")
        return results
    
    def save_results(self, results: dict, output_dir: str = "assets") -> None:
        """Save simulation results to files.
        
        Args:
            results: Results dictionary from run_simulation
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save data with signals
        results['data'].to_csv(output_path / "simulation_data.csv")
        
        # Save backtest results summary
        summary = {
            'strategy_type': results['strategy_type'],
            'total_return': results['backtest_results'].total_return,
            'annualized_return': results['backtest_results'].annualized_return,
            'volatility': results['backtest_results'].volatility,
            'sharpe_ratio': results['backtest_results'].sharpe_ratio,
            'sortino_ratio': results['backtest_results'].sortino_ratio,
            'calmar_ratio': results['backtest_results'].calmar_ratio,
            'max_drawdown': results['backtest_results'].max_drawdown,
            'total_trades': results['backtest_results'].total_trades,
            'win_rate': results['backtest_results'].win_rate,
            'profit_factor': results['backtest_results'].profit_factor
        }
        
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(output_path / "backtest_summary.csv", index=False)
        
        self.logger.info(f"Results saved to {output_path}")


def main():
    """Main function to run the HFT simulation."""
    # Load configuration
    config = Config()  # Uses default configuration
    
    # Create and run pipeline
    pipeline = HFTSimulationPipeline(config)
    results = pipeline.run_simulation()
    
    # Save results
    pipeline.save_results(results)
    
    # Print summary
    print("\n" + "="*50)
    print("HFT SIMULATION RESULTS")
    print("="*50)
    print(f"Strategy: {results['strategy_type']}")
    print(f"Total Return: {results['backtest_results'].total_return:.2%}")
    print(f"Annualized Return: {results['backtest_results'].annualized_return:.2%}")
    print(f"Volatility: {results['backtest_results'].volatility:.2%}")
    print(f"Sharpe Ratio: {results['backtest_results'].sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {results['backtest_results'].sortino_ratio:.2f}")
    print(f"Calmar Ratio: {results['backtest_results'].calmar_ratio:.2f}")
    print(f"Max Drawdown: {results['backtest_results'].max_drawdown:.2%}")
    print(f"Total Trades: {results['backtest_results'].total_trades}")
    print(f"Win Rate: {results['backtest_results'].win_rate:.2%}")
    print(f"Profit Factor: {results['backtest_results'].profit_factor:.2f}")
    print("="*50)


if __name__ == "__main__":
    main()
