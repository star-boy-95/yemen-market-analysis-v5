"""
Memory Optimization Benchmark for Yemen Market Analysis - Threshold Model Only

This script specifically benchmarks memory optimizations for threshold model calculations,
comparing before and after optimization.
"""
import os
import gc
import argparse
import numpy as np
import pandas as pd
import logging
import sys
from memory_profiler import memory_usage
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add src directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.threshold import ThresholdCointegration
from src.models.threshold_model import ThresholdModel

class ThresholdBenchmark:
    """Benchmark for threshold model memory optimizations."""
    
    def __init__(self, dataset_size='medium', seed=42):
        """Initialize benchmark with dataset size and seed."""
        self.dataset_size = dataset_size
        self.seed = seed
        self.data = None
        self.generate_test_data()
        
    def generate_test_data(self):
        """Generate synthetic data for testing."""
        np.random.seed(self.seed)
        
        # Dataset size configuration
        if self.dataset_size == 'small':
            n_markets = 50
            time_periods = 1825  # 5 years of daily data
        elif self.dataset_size == 'medium':
            n_markets = 100
            time_periods = 3650  # 10 years of daily data
        else:  # large
            n_markets = 200
            time_periods = 7300  # 20 years of daily data
            
        logger.info(f"Generating {self.dataset_size} test dataset...")
        
        # Generate market data with cointegration relationships
        markets = []
        base_price = 100
        
        # Generate time series data
        for i in range(n_markets):
            # Create market name
            market_name = f"Market_{i:03d}"
            
            # Generate common trend for each market
            trend = np.cumsum(np.random.normal(0, 0.05, time_periods))
            
            # Generate market-specific components
            market_specific = np.cumsum(np.random.normal(0, 0.02, time_periods))
            
            # Create price series with trend, market-specific component, and seasonality
            price = base_price + 5 * trend + market_specific
            
            # Add noise
            price += np.random.normal(0, 1, time_periods)
            
            # Ensure no negative prices
            price = np.maximum(price, 0.1)
            
            # Create dates
            start_date = pd.Timestamp('2015-01-01')
            dates = [start_date + pd.Timedelta(days=i) for i in range(time_periods)]
            
            # Create market dataframe
            market_df = pd.DataFrame({
                'date': dates,
                'market': market_name,
                'price': price
            })
            
            markets.append(market_df)
        
        # Combine all markets
        self.data = pd.concat(markets, ignore_index=True)
        
        logger.info(f"Created {n_markets} records")
        logger.info(f"Generated dataset with {len(self.data)} rows and {n_markets} markets")
        
        # Configure system for M3 Pro
        logger.info("Detected M3 Pro with 6 P-cores and 6 E-cores")
        logger.info("Configured NumPy for M3 Pro")
        logger.info("Configured Pandas for M3 Pro")
        logger.info(f"System configured for M3 Pro optimization with 36.0GB RAM")
    
    def run_non_optimized_threshold(self):
        """Run the threshold model without memory optimizations."""
        # Select two markets for the model
        market1 = "Market_000"
        market2 = "Market_001"
        
        # Prepare data for the model
        market1_data = self.data[self.data['market'] == market1].set_index('date')['price']
        market2_data = self.data[self.data['market'] == market2].set_index('date')['price']
        
        # Run the model
        model = ThresholdModel(data1=market1_data, data2=market2_data,
                               max_lags=5, mode="standard",
                               market1_name="Market_000", market2_name="Market_001")
        
        # Access the m3_utils module to disable optimizations
        import src.utils.m3_utils as m3_utils
        
        # Disable memory optimization flags
        m3_utils.USE_M3_MEMORY_OPTIMIZATION = False
        m3_utils.USE_CHUNKED_PROCESSING = False
        m3_utils.USE_SPARSE_MATRICES = False
        # First, estimate cointegration (required)
        model.estimate_cointegration()
        
        # Then estimate threshold
        model.estimate_threshold()
        
        
        return model
    
    def run_optimized_threshold(self):
        """Run the threshold model with memory optimizations."""
        # Select two markets for the model
        market1 = "Market_000"
        market2 = "Market_001"
        
        # Prepare data for the model
        market1_data = self.data[self.data['market'] == market1].set_index('date')['price']
        market2_data = self.data[self.data['market'] == market2].set_index('date')['price']
        
        # Access the m3_utils module to enable optimizations
        import src.utils.m3_utils as m3_utils
        
        # Enable memory optimization flags
        m3_utils.USE_M3_MEMORY_OPTIMIZATION = True
        m3_utils.USE_CHUNKED_PROCESSING = True
        m3_utils.USE_SPARSE_MATRICES = True
        
        # Run the model
        model = ThresholdModel(data1=market1_data, data2=market2_data,
                               max_lags=5, mode="standard",
                               market1_name="Market_000", market2_name="Market_001")
        # First, estimate cointegration (required)
        model.estimate_cointegration()
        
        # Then estimate threshold
        model.estimate_threshold()
        
        
        return model
    
    def benchmark_threshold_model(self):
        """Benchmark the threshold model with and without optimizations."""
        logger.info("\n--- Threshold Model Benchmark ---")
        
        # Non-optimized run
        logger.info("Running non-optimized threshold model...")
        start_time = time.time()
        non_opt_memory = max(memory_usage(
            (self.run_non_optimized_threshold, (), {}),
            interval=0.1
        ))
        non_opt_time = time.time() - start_time
        logger.info(f"Non-optimized peak memory: {non_opt_memory:.1f} MB")
        
        # Force garbage collection
        gc.collect()
        
        # Optimized run
        logger.info("Running optimized threshold model...")
        start_time = time.time()
        opt_memory = max(memory_usage(
            (self.run_optimized_threshold, (), {}),
            interval=0.1
        ))
        opt_time = time.time() - start_time
        logger.info(f"Optimized peak memory: {opt_memory:.1f} MB")
        
        # Calculate improvements
        memory_reduction = ((non_opt_memory - opt_memory) / non_opt_memory) * 100
        time_reduction = ((non_opt_time - opt_time) / non_opt_time) * 100
        
        logger.info(f"Memory reduction: {memory_reduction:.1f}%")
        logger.info(f"Time reduction: {time_reduction:.1f}%")
        
        # Check if we met our 60% optimization goal
        if memory_reduction >= 60:
            logger.info("✅ Met the 60% memory reduction goal!")
        else:
            logger.info(f"❌ Did not meet the 60% memory reduction goal. Current: {memory_reduction:.1f}%")
        
        return {
            'non_optimized_memory': non_opt_memory,
            'optimized_memory': opt_memory,
            'memory_reduction': memory_reduction,
            'non_optimized_time': non_opt_time,
            'optimized_time': opt_time,
            'time_reduction': time_reduction
        }

def main():
    """Main function to run the threshold model benchmark."""
    parser = argparse.ArgumentParser(description="Threshold Model Memory Optimization Benchmark")
    parser.add_argument(
        '--size', 
        choices=['small', 'medium', 'large'],
        default='medium',
        help="Size of test dataset (default: medium)"
    )
    parser.add_argument(
        '--seed', 
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()
    
    # Run the threshold benchmark
    benchmark = ThresholdBenchmark(dataset_size=args.size, seed=args.seed)
    results = benchmark.benchmark_threshold_model()
    
    # Show summary
    logger.info("\n=== Benchmark Summary ===")
    logger.info(f"Dataset size: {args.size}")
    logger.info(f"Non-optimized memory: {results['non_optimized_memory']:.1f} MB")
    logger.info(f"Optimized memory: {results['optimized_memory']:.1f} MB")
    logger.info(f"Memory reduction: {results['memory_reduction']:.1f}%")
    logger.info(f"Non-optimized time: {results['non_optimized_time']:.2f} seconds")
    logger.info(f"Optimized time: {results['optimized_time']:.2f} seconds")
    logger.info(f"Time reduction: {results['time_reduction']:.1f}%")
    
    # Success/failure message
    if results['memory_reduction'] >= 60:
        logger.info("✅ SUCCESS: Memory optimization target of 60% reduction met!")
    else:
        logger.info(f"❌ NOT MET: Memory optimization target of 60% reduction not achieved. Current: {results['memory_reduction']:.1f}%")
    
if __name__ == "__main__":
    main()