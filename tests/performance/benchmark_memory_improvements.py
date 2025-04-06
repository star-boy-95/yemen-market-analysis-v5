"""
Memory Optimization Benchmark for Yemen Market Analysis

This script benchmarks memory optimizations for large-scale spatial and
threshold model calculations, comparing before and after optimization
to demonstrate the performance improvements.
"""
import os
import gc
import time
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from memory_profiler import profile, memory_usage
import psutil
import logging
from shapely.geometry import Point
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add src directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Patch for Moran_Local class to ensure compatibility
from esda.moran import Moran_Local
original_moran_local_init = Moran_Local.__init__

def patched_moran_local_init(self, y, w, transformation="r", permutations=999):
    original_moran_local_init(self, y, w, transformation, permutations)
    # Add y_lag attribute if it doesn't exist (for compatibility)
    if not hasattr(self, 'y_lag'):
        self.y_lag = self.z_lag if hasattr(self, 'z_lag') else w.sparse @ y

# Apply the patch
Moran_Local.__init__ = patched_moran_local_init

from src.models.threshold import ThresholdCointegration
from src.models.threshold_model import ThresholdModel
from src.models.spatial import SpatialEconometrics, calculate_market_accessibility
from src.utils.m3_utils import (
    m3_optimized, monitor_memory_usage, memory_profile,
    optimize_array_computation, chunk_iterator, create_mmap_array,
    process_in_chunks, tiered_cache, configure_system_for_m3_performance
)


class MemoryBenchmark:
    """
    Benchmark class for memory usage tests on Yemen market analysis.
    Compares optimized vs. non-optimized performance with real-world data sizes.
    """
    
    def __init__(self, dataset_size='medium', seed=42):
        """
        Initialize benchmark with configurable dataset size.
        
        Parameters
        ----------
        dataset_size : str
            Size of test dataset ('small', 'medium', 'large')
        seed : int
            Random seed for reproducibility
        """
        self.dataset_size = dataset_size
        self.seed = seed
        np.random.seed(seed)
        
        # Configure dataset size parameters
        if dataset_size == 'small':
            self.n_markets = 50
            self.n_days = 365   # 1 year
            self.n_commodities = 5
        elif dataset_size == 'medium':
            self.n_markets = 200
            self.n_days = 730   # 2 years
            self.n_commodities = 10
        elif dataset_size == 'large':
            self.n_markets = 500
            self.n_days = 1825  # 5 years
            self.n_commodities = 20
        else:
            raise ValueError(f"Invalid dataset size: {dataset_size}")
        
        # Create temp directory for data files
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Generate synthetic data
        logger.info(f"Generating {dataset_size} test dataset...")
        self.market_data, self.spatial_data = self._generate_test_data()
        
        # Save data to disk
        self.market_data_path = self.temp_dir / "market_data.csv"
        self.spatial_data_path = self.temp_dir / "spatial_data.geojson"
        
        self.market_data.to_csv(self.market_data_path, index=False)
        self.spatial_data.to_file(self.spatial_data_path, driver="GeoJSON")
        
        logger.info(f"Generated dataset with {len(self.market_data)} rows and {len(self.spatial_data)} markets")
        
        # Configure M3 optimizations
        configure_system_for_m3_performance()
    
    def _generate_test_data(self):
        """
        Generate synthetic data for benchmarking.
        
        Returns
        -------
        market_data : pd.DataFrame
            Synthetic market price data
        spatial_data : gpd.GeoDataFrame
            Synthetic spatial market data
        """
        # Generate dates
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(self.n_days)]
        
        # Generate markets
        markets = []
        for i in range(self.n_markets):
            # Determine region (half north, half south)
            if i < self.n_markets // 2:
                region = "north"
                exchange_rate = 600
                lat = 15 - (i * 0.05)  # North markets at top
            else:
                region = "south"
                exchange_rate = 800
                lat = 12 - ((i - self.n_markets // 2) * 0.05)  # South markets at bottom
            
            # Generate coordinates
            lon = 44 + (i * 0.05)
            
            # Create conflict intensity - higher in some regions
            base_conflict = 0.2
            if i % 5 == 0:  # Every 5th market has higher conflict
                conflict_intensity = base_conflict + np.random.uniform(0.4, 0.7)
            else:
                conflict_intensity = base_conflict + np.random.uniform(0, 0.3)
            
            # Normalize conflict intensity (0-1)
            conflict_intensity = min(1.0, conflict_intensity)
            
            # Add market to list
            markets.append({
                "market_id": f"M{i}",
                "market_name": f"Market {i}",
                "exchange_rate_regime": region,
                "exchange_rate": exchange_rate,
                "latitude": lat,
                "longitude": lon,
                "conflict_intensity_normalized": conflict_intensity,
                "geometry": Point(lon, lat)
            })
        
        # Create spatial data
        spatial_data = gpd.GeoDataFrame(markets, geometry="geometry")
        spatial_data.crs = "EPSG:4326"  # Set CRS to WGS84
        
        # Generate market prices
        market_data_rows = []
        
        # Create commodity names
        commodities = [f"Commodity_{i}" for i in range(self.n_commodities)]
        
        # Set base prices (north lower than south)
        north_base_price = 100
        south_base_price = 150
        
        # Generate prices for each market, date, and commodity
        for market in markets:
            # Set base price based on region
            if market["exchange_rate_regime"] == "north":
                base_price = north_base_price + np.random.normal(0, 5)
            else:
                base_price = south_base_price + np.random.normal(0, 5)
            
            # Add conflict effect on price (higher conflict → higher price)
            conflict_premium = market["conflict_intensity_normalized"] * 30
            base_price += conflict_premium
            
            for date_idx, date in enumerate(dates):
                for commodity in commodities:
                    # Adjust price by commodity (random factor)
                    commodity_factor = 0.8 + (np.random.uniform(0, 0.5))
                    commodity_price = base_price * commodity_factor
                    
                    # Add time component
                    time_trend = date_idx * 0.1  # Increasing trend
                    season = 5 * np.sin(date_idx / 30)  # Seasonal component
                    noise = np.random.normal(0, 2)  # Random noise
                    
                    # Generate price
                    price = commodity_price + time_trend + season + noise
                    
                    # Add row to market data
                    market_data_rows.append({
                        "market_id": market["market_id"],
                        "market_name": market["market_name"],
                        "exchange_rate_regime": market["exchange_rate_regime"],
                        "exchange_rate": market["exchange_rate"],
                        "date": date,
                        "commodity": commodity,
                        "price": price,
                        "conflict_intensity_normalized": market["conflict_intensity_normalized"]
                    })
        
        # Create market data DataFrame
        market_data = pd.DataFrame(market_data_rows)
        
        return market_data, spatial_data
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.market_data_path.exists():
            self.market_data_path.unlink()
        
        if self.spatial_data_path.exists():
            self.spatial_data_path.unlink()
        
        try:
            self.temp_dir.rmdir()
            logger.info("Cleaned up temporary files")
        except:
            logger.warning("Some temporary files could not be removed")
    
    def run_all_benchmarks(self):
        """Run all memory benchmarks and generate summary."""
        results = {}
        
        # Force garbage collection
        gc.collect()
        
        # Benchmark threshold model
        results['threshold'] = self.benchmark_threshold_model()
        
        # Force garbage collection
        gc.collect()
        
        # Benchmark spatial model
        results['spatial'] = self.benchmark_spatial_model()
        
        # Force garbage collection
        gc.collect()
        
        # Benchmark integration analysis
        results['integration'] = self.benchmark_integration_analysis()
        
        # Generate summary report
        self.print_summary(results)
        
        # Generate charts
        self.plot_results(results)
        
        return results
    
    def benchmark_threshold_model(self):
        """
        Benchmark threshold model performance.
        
        Returns
        -------
        dict
            Memory usage metrics
        """
        logger.info("\n--- Threshold Model Benchmark ---")
        
        # Extract price series for a single commodity
        wheat_data = self.market_data[self.market_data['commodity'] == 'Commodity_0'].copy()
        
        # Calculate average prices by date and regime
        wheat_avg = wheat_data.groupby(['date', 'exchange_rate_regime'])['price'].mean().unstack()
        wheat_avg = wheat_avg.rename(columns={'north': 'north_price', 'south': 'south_price'}).reset_index()
        
        # Extract price series
        north_prices = wheat_avg['north_price'].values
        south_prices = wheat_avg['south_price'].values
        dates = wheat_avg['date'].values
        
        # Run non-optimized version
        logger.info("Running non-optimized threshold model...")
        import resource
        
        def run_non_optimized():
            # Disable optimizations by monkey patching
            original_optimize = optimize_array_computation
            
            # Replace with identity function
            def no_op(arr, *args, **kwargs):
                return arr
            
            try:
                # Patch optimization functions
                globals()['optimize_array_computation'] = no_op
                
                # Force garbage collection before test
                gc.collect()
                
                # Create model using ThresholdModel for continuity
                model = ThresholdModel(
                    north_prices, south_prices, 
                    mode="standard", 
                    market1_name="North", 
                    market2_name="South"
                )
                
                # Estimate cointegration
                model.estimate_cointegration()
                
                # Estimate threshold
                model.estimate_threshold(n_grid=200)
                
                # Clean up
                del model
                gc.collect()
                
            finally:
                # Restore original function
                globals()['optimize_array_computation'] = original_optimize
        
        # Measure memory for non-optimized version
        non_opt_memory = max(memory_usage(
            (run_non_optimized, (), {}),
            interval=0.1,
            timeout=300
        ))
        
        logger.info(f"Non-optimized peak memory: {non_opt_memory:.1f} MB")
        
        # Run optimized version
        logger.info("Running optimized threshold model...")
        
        def run_optimized():
            # Force garbage collection before test
            gc.collect()
            
            # Create model using ThresholdModel with optimizations enabled
            # Note: We use ThresholdModel directly as ThresholdCointegration
            # is now just a wrapper that returns ThresholdModel
            model = ThresholdModel(
                north_prices, south_prices,
                mode="standard",
                market1_name="North",
                market2_name="South"
            )
            
            # Estimate cointegration
            model.estimate_cointegration()
            
            # Estimate threshold
            model.estimate_threshold(n_grid=200)
            
            # Clean up
            del model
            gc.collect()
        
        # Measure memory for optimized version
        opt_memory = max(memory_usage(
            (run_optimized, (), {}),
            interval=0.1,
            timeout=300
        ))
        
        logger.info(f"Optimized peak memory: {opt_memory:.1f} MB")
        
        # Calculate improvement
        improvement = ((non_opt_memory - opt_memory) / non_opt_memory) * 100
        logger.info(f"Memory reduction: {improvement:.1f}%")
        
        return {
            'non_optimized_mb': non_opt_memory,
            'optimized_mb': opt_memory,
            'improvement_pct': improvement
        }
    
    def benchmark_spatial_model(self):
        """
        Benchmark spatial model performance.
        
        Returns
        -------
        dict
            Memory usage metrics
        """
        logger.info("\n--- Spatial Model Benchmark ---")
        
        # Prepare data for spatial analysis
        # Get latest data for each market
        latest_date = self.market_data['date'].max()
        latest_data = self.market_data[
            (self.market_data['date'] == latest_date) & 
            (self.market_data['commodity'] == 'Commodity_0')
        ]
        
        # Merge with spatial data
        # Ensure conflict_intensity_normalized is included from spatial_data
        merged_data = latest_data.merge(self.spatial_data, on='market_id', how='left')
        
        # Add exchange_rate column if missing (needed for spatial model benchmark)
        if 'exchange_rate' not in merged_data.columns:
            merged_data['exchange_rate'] = merged_data['market_id'].apply(
                lambda x: int(x.replace('M', '')) % 1000  # Simple deterministic value
            )
        # Double-check column exists
        if 'conflict_intensity_normalized' not in merged_data.columns:
            # Add it directly from spatial_data if missing
            merged_data['conflict_intensity_normalized'] = merged_data['market_id'].map(
                self.spatial_data.set_index('market_id')['conflict_intensity_normalized']
            )
        spatial_df = gpd.GeoDataFrame(merged_data, geometry='geometry')
        
        # Ensure conflict_intensity_normalized is included in the merged_data
        if 'conflict_intensity_normalized' not in merged_data.columns:
            # Add it from spatial_data if missing
            merged_data['conflict_intensity_normalized'] = merged_data['market_id'].map(
                self.spatial_data.set_index('market_id')['conflict_intensity_normalized'].to_dict()
            )
            
        # Run non-optimized version
        logger.info("Running non-optimized spatial model...")
        
        def run_non_optimized():
            # Disable optimizations by monkey patching
            original_optimize_gdf = SpatialEconometrics._optimize_gdf
            original_setup_caches = SpatialEconometrics._setup_caches
            
            # Replace optimize_gdf with identity function
            def no_op_gdf(self, gdf):
                return gdf.copy()
            
            try:
                # Patch optimization functions - but still need to keep the cache setup
                SpatialEconometrics._optimize_gdf = no_op_gdf
                
                # Force garbage collection before test
                gc.collect()
                
                # Create spatial model
                model = SpatialEconometrics(spatial_df)
                
                # Create weight matrix
                model.create_weight_matrix(
                    k=5, 
                    conflict_adjusted=True, 
                    conflict_col='conflict_intensity_normalized'
                )
                
                # Run Moran's I test
                model.moran_i_test('price')
                
                # Run local Moran
                model.local_moran_test('price')
                
                # Run spatial lag model
                model.spatial_lag_model(
                    'price', 
                    ['conflict_intensity_normalized', 'exchange_rate']
                )
                
                # Calculate impacts
                model.calculate_impacts()
                
                # Clean up
                del model
                gc.collect()
                
            finally:
                # Restore original functions
                SpatialEconometrics._optimize_gdf = original_optimize_gdf
        
        # Measure memory for non-optimized version
        non_opt_memory = max(memory_usage(
            (run_non_optimized, (), {}),
            interval=0.1,
            timeout=300
        ))
        
        logger.info(f"Non-optimized peak memory: {non_opt_memory:.1f} MB")
        
        # Make sure spatial_df has the necessary column
        if 'conflict_intensity_normalized' not in spatial_df.columns:
            spatial_df['conflict_intensity_normalized'] = spatial_df['market_id'].map(
                self.spatial_data.set_index('market_id')['conflict_intensity_normalized'].to_dict()
            )
        
        # Run optimized version
        logger.info("Running optimized spatial model...")
        
        def run_optimized():
            # Force garbage collection before test
            gc.collect()
            
            # Create spatial model
            model = SpatialEconometrics(spatial_df)
            
            # Create weight matrix
            model.create_weight_matrix(
                k=5, 
                conflict_adjusted=True, 
                conflict_col='conflict_intensity_normalized'
            )
            
            # Run Moran's I test
            model.moran_i_test('price')
            
            # Run local Moran
            model.local_moran_test('price')
            
            # Run spatial lag model
            model.spatial_lag_model(
                'price', 
                ['conflict_intensity_normalized', 'exchange_rate']
            )
            
            # Calculate impacts
            model.calculate_impacts()
            
            # Clean up
            del model
            gc.collect()
        
        # Measure memory for optimized version
        opt_memory = max(memory_usage(
            (run_optimized, (), {}),
            interval=0.1,
            timeout=300
        ))
        
        logger.info(f"Optimized peak memory: {opt_memory:.1f} MB")
        
        # Calculate improvement
        improvement = ((non_opt_memory - opt_memory) / non_opt_memory) * 100
        logger.info(f"Memory reduction: {improvement:.1f}%")
        
        return {
            'non_optimized_mb': non_opt_memory,
            'optimized_mb': opt_memory,
            'improvement_pct': improvement
        }
    
    def benchmark_integration_analysis(self):
        """
        Benchmark market integration analysis performance.
        
        Returns
        -------
        dict
            Memory usage metrics
        """
        logger.info("\n--- Market Integration Analysis Benchmark ---")
        
        # Prepare data for integration analysis
        # Filter to a single commodity
        commodity_data = self.market_data[self.market_data['commodity'] == 'Commodity_0'].copy()
        
        # Create spatial weights matrix
        from libpysal.weights import KNN
        coords = self.spatial_data[['longitude', 'latitude']].values
        weights = KNN.from_array(coords, k=5)
        
        # Run non-optimized version
        logger.info("Running non-optimized integration analysis...")
        
        def run_non_optimized():
            # Disable optimizations by monkey patching
            original_optimize = process_in_chunks
            
            # Replace with identity function that processes all at once
            def no_op_chunks(data, process_func, chunk_size=None, parallel=False, n_jobs=None, reduce_func=None):
                if reduce_func:
                    return reduce_func([process_func(data)])
                return [process_func(data)]
            
            try:
                # Patch optimization functions
                globals()['process_in_chunks'] = no_op_chunks
                
                # Force garbage collection before test
                gc.collect()
                
                # Import the relevant function
                from src.models.spatial import market_integration_index
                
                # Calculate market integration
                result = market_integration_index(
                    commodity_data,
                    weights,
                    market_id_col='market_id',
                    price_col='price',
                    time_col='date',
                    window=12
                )
                
                # Clean up
                del result
                gc.collect()
                
            finally:
                # Restore original function
                globals()['process_in_chunks'] = original_optimize
        
        # Measure memory for non-optimized version
        non_opt_memory = max(memory_usage(
            (run_non_optimized, (), {}),
            interval=0.1,
            timeout=300
        ))
        
        logger.info(f"Non-optimized peak memory: {non_opt_memory:.1f} MB")
        
        # Run optimized version
        logger.info("Running optimized integration analysis...")
        
        def run_optimized():
            # Force garbage collection before test
            gc.collect()
            
            # Import the relevant function
            from src.models.spatial import market_integration_index
            
            # Calculate market integration
            result = market_integration_index(
                commodity_data,
                weights,
                market_id_col='market_id',
                price_col='price',
                time_col='date',
                window=12
            )
            
            # Clean up
            del result
            gc.collect()
        
        # Measure memory for optimized version
        opt_memory = max(memory_usage(
            (run_optimized, (), {}),
            interval=0.1,
            timeout=300
        ))
        
        logger.info(f"Optimized peak memory: {opt_memory:.1f} MB")
        
        # Calculate improvement
        improvement = ((non_opt_memory - opt_memory) / non_opt_memory) * 100
        logger.info(f"Memory reduction: {improvement:.1f}%")
        
        return {
            'non_optimized_mb': non_opt_memory,
            'optimized_mb': opt_memory,
            'improvement_pct': improvement
        }
    
    def print_summary(self, results):
        """
        Print summary of benchmark results.
        
        Parameters
        ----------
        results : dict
            Results from benchmarks
        """
        logger.info("\n---------------------------------")
        logger.info("Memory Optimization Benchmark Summary")
        logger.info("---------------------------------")
        logger.info(f"Dataset Size: {self.dataset_size.upper()}")
        logger.info(f"Markets: {self.n_markets}, Days: {self.n_days}, Commodities: {self.n_commodities}")
        logger.info(f"Total Data Points: {len(self.market_data):,}")
        logger.info("---------------------------------")
        
        # Calculate overall improvement
        total_non_opt = sum(r['non_optimized_mb'] for r in results.values())
        total_opt = sum(r['optimized_mb'] for r in results.values())
        total_improvement = ((total_non_opt - total_opt) / total_non_opt) * 100
        
        # Print individual results
        for model, metrics in results.items():
            logger.info(f"{model.upper()}")
            logger.info(f"  Non-optimized: {metrics['non_optimized_mb']:.1f} MB")
            logger.info(f"  Optimized:     {metrics['optimized_mb']:.1f} MB")
            logger.info(f"  Reduction:     {metrics['improvement_pct']:.1f}%")
            logger.info("---------------------------------")
        
        # Print overall results
        logger.info("OVERALL")
        logger.info(f"  Non-optimized: {total_non_opt:.1f} MB")
        logger.info(f"  Optimized:     {total_opt:.1f} MB")
        logger.info(f"  Reduction:     {total_improvement:.1f}%")
        logger.info("---------------------------------")
        
        # Check if goal was met
        goal_met = total_improvement >= 60
        logger.info(f"60% Reduction Goal: {'✓ MET' if goal_met else '✗ NOT MET'}")
        logger.info("---------------------------------")
    
    def plot_results(self, results):
        """
        Generate plots of benchmark results.
        
        Parameters
        ----------
        results : dict
            Results from benchmarks
        """
        # Create a bar chart
        models = list(results.keys())
        non_opt_mem = [results[m]['non_optimized_mb'] for m in models]
        opt_mem = [results[m]['optimized_mb'] for m in models]
        improvements = [results[m]['improvement_pct'] for m in models]
        
        # Calculate totals
        models.append('TOTAL')
        non_opt_mem.append(sum(non_opt_mem))
        opt_mem.append(sum(opt_mem))
        total_imp = ((non_opt_mem[-1] - opt_mem[-1]) / non_opt_mem[-1]) * 100
        improvements.append(total_imp)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Memory usage comparison
        width = 0.4
        x = np.arange(len(models))
        ax1.bar(x - width/2, non_opt_mem, width, label='Non-optimized')
        ax1.bar(x + width/2, opt_mem, width, label='Optimized')
        
        # Add labels
        ax1.set_title('Memory Usage Comparison (MB)')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.capitalize() for m in models])
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add values on top of bars
        for i, v in enumerate(non_opt_mem):
            ax1.text(i - width/2, v + 5, f'{v:.0f}', ha='center')
        
        for i, v in enumerate(opt_mem):
            ax1.text(i + width/2, v + 5, f'{v:.0f}', ha='center')
        
        # Improvement percentages
        ax2.bar(x, improvements, width=0.6, color='green')
        
        # Add labels
        ax2.set_title('Memory Reduction (%)')
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.capitalize() for m in models])
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.axhline(y=60, color='r', linestyle='-', alpha=0.7, label='60% Goal')
        ax2.legend()
        
        # Add values on top of bars
        for i, v in enumerate(improvements):
            ax2.text(i, v + 2, f'{v:.0f}%', ha='center')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plot_path = Path.cwd() / "memory_optimization_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        logger.info(f"Results plot saved to: {plot_path}")


def main():
    """Main function to run benchmarks."""
    parser = argparse.ArgumentParser(description="Memory Optimization Benchmark")
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
    
    try:
        # Run benchmarks
        benchmark = MemoryBenchmark(dataset_size=args.size, seed=args.seed)
        results = benchmark.run_all_benchmarks()
    finally:
        # Clean up
        benchmark.cleanup()


if __name__ == "__main__":
    main()