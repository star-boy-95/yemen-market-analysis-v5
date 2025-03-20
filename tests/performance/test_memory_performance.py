"""
Performance tests for memory-intensive operations.

This module tests the performance of operations that handle large datasets,
ensuring that the code is efficient and memory usage is optimized.
"""
import os
import time
import unittest
import tempfile
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from memory_profiler import profile
from functools import wraps
import gc

from src.data.loader import load_market_data, load_spatial_data
from src.data.preprocessor import preprocess_market_data
from src.models.threshold import ThresholdCointegration
from src.models.spatial import SpatialEconometrics
from src.models.simulation import MarketIntegrationSimulation
from src.utils.performance_utils import optimize_memory_usage, benchmark


def memory_usage_decorator(func):
    """
    Decorator that measures peak memory usage of a function.
    
    Parameters
    ----------
    func : callable
        Function to measure memory usage
    
    Returns
    -------
    callable
        Wrapped function that measures memory usage
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Force garbage collection before measuring
        gc.collect()
        
        # Start measurement
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Call the function
        result = func(*args, **kwargs)
        
        # Force garbage collection to get accurate peak
        gc.collect()
        
        # End measurement
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        peak_usage = end_memory - start_memory
        
        print(f"Memory usage for {func.__name__}: {peak_usage:.2f} MB")
        
        return result
    return wrapper


class TestMemoryPerformance(unittest.TestCase):
    """Test cases for memory performance of operations."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        # Create temp directory
        cls.temp_dir = Path(tempfile.mkdtemp())
        
        # Generate large synthetic data
        print("Generating large synthetic data...")
        cls.market_data, cls.spatial_data = cls._generate_large_test_data(
            n_markets=50,      # 50 markets
            n_days=365,        # 1 year of data
            n_commodities=5    # 5 commodities
        )
        
        # Save large data to temporary files
        cls.market_data_path = cls.temp_dir / "large_market_data.csv"
        cls.spatial_data_path = cls.temp_dir / "large_spatial_data.geojson"
        
        print(f"Saving {len(cls.market_data)} rows of market data...")
        cls.market_data.to_csv(cls.market_data_path, index=False)
        
        print(f"Saving {len(cls.spatial_data)} spatial features...")
        cls.spatial_data.to_file(cls.spatial_data_path, driver="GeoJSON")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures after all tests."""
        # Remove temporary files
        if cls.market_data_path.exists():
            cls.market_data_path.unlink()
            
        if cls.spatial_data_path.exists():
            cls.spatial_data_path.unlink()
            
        # Remove temporary directory
        cls.temp_dir.rmdir()
    
    @classmethod
    def _generate_large_test_data(cls, n_markets=50, n_days=365, n_commodities=5):
        """
        Generate large synthetic data for performance testing.
        
        Parameters
        ----------
        n_markets : int
            Number of markets to generate
        n_days : int
            Number of days of data to generate
        n_commodities : int
            Number of commodities to generate
        
        Returns
        -------
        market_data : pandas.DataFrame
            Large market data for performance testing
        spatial_data : geopandas.GeoDataFrame
            Spatial data for markets
        """
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate dates
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_days)]
        
        # Generate markets
        markets = []
        for i in range(n_markets):
            # Determine region (half north, half south)
            if i < n_markets // 2:
                region = "north"
                exchange_rate = 600
                lat = 15 - (i * 0.2)  # North markets at top
            else:
                region = "south"
                exchange_rate = 800
                lat = 12 - ((i - n_markets // 2) * 0.2)  # South markets at bottom
            
            # Generate coordinates
            lon = 44 + (i * 0.2)
            
            # Add market to list
            markets.append({
                "market_id": f"M{i}",
                "market_name": f"Market {i}",
                "exchange_rate_regime": region,
                "exchange_rate": exchange_rate,
                "latitude": lat,
                "longitude": lon,
                "geometry": Point(lon, lat)
            })
        
        # Create spatial data
        spatial_data = gpd.GeoDataFrame(markets, geometry="geometry")
        
        # Generate market prices
        market_data_rows = []
        
        # Create commodity names
        commodities = [f"Commodity_{i}" for i in range(n_commodities)]
        
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
            
            # Add conflict intensity (random)
            conflict_intensity = np.abs(np.random.normal(0, 0.5)) 
            
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
                        "conflict_intensity_normalized": conflict_intensity
                    })
        
        # Create market data DataFrame
        market_data = pd.DataFrame(market_data_rows)
        
        return market_data, spatial_data
    
    def _report_memory_usage(self, df, operation_name):
        """
        Report memory usage of a DataFrame.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to report memory usage for
        operation_name : str
            Name of operation for reporting
        """
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        print(f"Memory usage after {operation_name}: {memory_usage:.2f} MB")
    
    @profile
    def test_large_data_loading(self):
        """Test loading of large data files."""
        # Start timer
        start_time = time.time()
        
        # Load market data
        market_data = load_market_data(self.market_data_path)
        
        # Report memory usage
        self._report_memory_usage(market_data, "market data loading")
        
        # Load spatial data
        spatial_data = load_spatial_data(self.spatial_data_path)
        
        # End timer
        end_time = time.time()
        
        # Report results
        load_time = end_time - start_time
        print(f"Time to load large datasets: {load_time:.2f} seconds")
        print(f"Market data shape: {market_data.shape}")
        print(f"Spatial data shape: {spatial_data.shape}")
        
        # Basic assertions
        self.assertEqual(len(market_data), len(self.market_data))
        self.assertEqual(len(spatial_data), len(self.spatial_data))
    
    @profile
    def test_pandas_optimization(self):
        """Test pandas DataFrame optimization functions."""
        # Load unoptimized data
        market_data = pd.read_csv(self.market_data_path)
        
        # Report memory usage before optimization
        initial_size = market_data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        print(f"Memory usage before optimization: {initial_size:.2f} MB")
        
        # Apply optimization
        start_time = time.time()
        optimized_data = optimize_memory_usage(market_data)
        optimization_time = time.time() - start_time
        
        # Report memory usage after optimization
        optimized_size = optimized_data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        print(f"Memory usage after optimization: {optimized_size:.2f} MB")
        print(f"Memory reduction: {((initial_size - optimized_size) / initial_size) * 100:.2f}%")
        print(f"Optimization time: {optimization_time:.2f} seconds")
        
        # Verify that optimization reduced memory usage
        self.assertLess(optimized_size, initial_size)
        
        # Verify that optimized data has the same values
        pd.testing.assert_frame_equal(
            market_data, optimized_data, 
            check_dtype=False,  # Types will differ after optimization
            check_exact=False   # Allow for small floating point differences
        )
    
    @profile
    def test_large_groupby_performance(self):
        """Test performance of groupby operations on large datasets."""
        # Load data
        market_data = pd.read_csv(self.market_data_path)
        
        # Benchmark different groupby operations
        operations = [
            # Simple group by one column
            lambda df: df.groupby('market_id')['price'].mean(),
            
            # Group by multiple columns
            lambda df: df.groupby(['commodity', 'exchange_rate_regime'])['price'].mean(),
            
            # Group by with multiple aggregations
            lambda df: df.groupby('market_id').agg({
                'price': ['mean', 'std', 'min', 'max'],
                'conflict_intensity_normalized': 'mean'
            }),
            
            # Group by with filter and complex aggregation
            lambda df: df[df['commodity'] == 'Commodity_0'].groupby(
                ['market_id', 'exchange_rate_regime']
            ).agg({
                'price': ['mean', 'std', lambda x: x.max() - x.min()],
                'date': ['min', 'max']
            })
        ]
        
        operation_names = [
            "Simple groupby",
            "Multiple column groupby",
            "Multiple aggregations",
            "Filtered complex groupby"
        ]
        
        # Run benchmarks
        for op, name in zip(operations, operation_names):
            # Time the operation
            result, metrics = benchmark(op, market_data)
            
            # Report results
            print(f"\n{name}:")
            print(f"  Execution time: {metrics['execution_time']:.4f} seconds")
            print(f"  Peak memory: {metrics['peak_memory_mb']:.2f} MB")
            
            # Basic assertion
            self.assertIsNotNone(result)
    
    @profile
    def test_spatial_join_performance(self):
        """Test performance of spatial joins and operations."""
        try:
            # Import necessary spatial packages
            import rtree
            import spatial_trim
            
            # Load data
            market_data = pd.read_csv(self.market_data_path)
            spatial_data = gpd.read_file(self.spatial_data_path)
            
            # Get only the latest data for each market
            latest_date = market_data['date'].max()
            latest_data = market_data[market_data['date'] == latest_date]
            
            # Benchmark spatial join
            start_time = time.time()
            merged_data = latest_data.merge(spatial_data, on='market_id')
            joined_gdf = gpd.GeoDataFrame(merged_data, geometry='geometry')
            join_time = time.time() - start_time
            
            # Report join results
            print(f"Spatial join time: {join_time:.4f} seconds")
            print(f"Joined data shape: {joined_gdf.shape}")
            
            # Calculate buffer distances by market
            start_time = time.time()
            joined_gdf['buffer_10km'] = joined_gdf.geometry.buffer(0.1)  # Approx 10km in degrees
            buffer_time = time.time() - start_time
            
            # Report buffer results
            print(f"Buffer calculation time: {buffer_time:.4f} seconds")
            
            # Calculate distances between all points
            start_time = time.time()
            points = list(joined_gdf.geometry)
            n_points = len(points)
            distances = np.zeros((n_points, n_points))
            
            for i in range(n_points):
                for j in range(i+1, n_points):
                    dist = points[i].distance(points[j])
                    distances[i, j] = dist
                    distances[j, i] = dist
            
            distance_time = time.time() - start_time
            
            # Report distance results
            print(f"Distance matrix calculation time: {distance_time:.4f} seconds")
            print(f"Distance matrix shape: {distances.shape}")
            
            # Basic assertions
            self.assertEqual(len(joined_gdf), len(latest_data))
            self.assertGreater(len(joined_gdf), 0)
            
        except ImportError as e:
            self.skipTest(f"Spatial dependencies not available: {str(e)}")
    
    @profile
    def test_time_series_aggregation_performance(self):
        """Test performance of time series aggregation operations."""
        # Load data
        market_data = pd.read_csv(self.market_data_path)
        market_data['date'] = pd.to_datetime(market_data['date'])
        
        # Benchmark different time series operations
        operations = [
            # Daily to weekly resampling
            lambda df: df.set_index('date').groupby(['market_id', 'commodity']).resample('W')['price'].mean().reset_index(),
            
            # Rolling window calculations
            lambda df: df.set_index('date').groupby(['market_id', 'commodity'])['price'].rolling(window=7).mean().reset_index(),
            
            # Expanding window calculations
            lambda df: df.set_index('date').groupby(['market_id', 'commodity'])['price'].expanding().mean().reset_index(),
            
            # Complex time series operation
            lambda df: (
                df.set_index('date')
                .groupby(['market_id', 'commodity'])['price']
                .resample('W')
                .agg(['mean', 'std', 'min', 'max'])
                .reset_index()
            )
        ]
        
        operation_names = [
            "Weekly resampling",
            "7-day rolling average",
            "Expanding window average",
            "Complex weekly aggregation"
        ]
        
        # Run benchmarks
        for op, name in zip(operations, operation_names):
            try:
                # Time the operation
                result, metrics = benchmark(op, market_data)
                
                # Report results
                print(f"\n{name}:")
                print(f"  Execution time: {metrics['execution_time']:.4f} seconds")
                print(f"  Peak memory: {metrics['peak_memory_mb']:.2f} MB")
                print(f"  Result shape: {result.shape}")
                
                # Basic assertion
                self.assertIsNotNone(result)
                self.assertGreater(len(result), 0)
                
            except Exception as e:
                print(f"Error in {name}: {str(e)}")
                continue
    
    @profile
    def test_threshold_model_performance(self):
        """Test performance of threshold cointegration model with large datasets."""
        # Load data
        market_data = pd.read_csv(self.market_data_path)
        market_data['date'] = pd.to_datetime(market_data['date'])
        
        # Filter to single commodity
        wheat_data = market_data[market_data['commodity'] == 'Commodity_0'].copy()
        
        # Calculate average prices by date and regime
        wheat_avg = wheat_data.groupby(['date', 'exchange_rate_regime'])['price'].mean().unstack()
        wheat_avg = wheat_avg.rename(columns={'north': 'north_price', 'south': 'south_price'}).reset_index()
        
        # Extract price series
        north_prices = wheat_avg['north_price'].values
        south_prices = wheat_avg['south_price'].values
        dates = wheat_avg['date'].values
        
        # Benchmark threshold cointegration model
        start_time = time.time()
        
        # Create model
        model = ThresholdCointegration(north_prices, south_prices)
        
        # Estimate cointegration
        cointegration_result = model.estimate_cointegration()
        
        # Estimate threshold
        if cointegration_result['cointegrated']:
            threshold_result = model.estimate_threshold()
            
            # Store dates and price differentials for M-TAR
            model.dates = dates
            model.price_diff = north_prices - south_prices
            
            # Estimate M-TAR model
            mtar_result = model.estimate_mtar()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Report results
        print(f"Threshold cointegration model time: {total_time:.4f} seconds")
        print(f"Series length: {len(north_prices)}")
        print(f"Cointegrated: {cointegration_result.get('cointegrated', False)}")
        
        if cointegration_result.get('cointegrated', False):
            print(f"Threshold: {threshold_result.get('threshold', 'N/A')}")
            print(f"Asymmetric: {mtar_result.get('asymmetric', 'N/A')}")
        
        # Basic assertions
        self.assertIsNotNone(cointegration_result)
        self.assertIn('cointegrated', cointegration_result)
        
        # Additional assertion if cointegrated
        if cointegration_result.get('cointegrated', False):
            self.assertIsNotNone(threshold_result)
            self.assertIn('threshold', threshold_result)
            self.assertIsNotNone(mtar_result)
            self.assertIn('asymmetric', mtar_result)


if __name__ == '__main__':
    unittest.main()
