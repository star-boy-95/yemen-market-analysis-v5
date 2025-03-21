"""
Integration tests for the Yemen Market Analysis project.

This module provides tests for the complete analysis pipeline, ensuring that
all components work together correctly: data preprocessing, unit root testing, 
cointegration, threshold estimation, diagnostics, and simulation.
"""
import unittest
import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
import gc
import sys
import tempfile
from datetime import datetime, timedelta

# Import project components using consistent package imports
from yemen_market_integration.models.unit_root import UnitRootTester
from yemen_market_integration.models.threshold import ThresholdCointegration
from yemen_market_integration.models.diagnostics import ModelDiagnostics
from yemen_market_integration.models.simulation import MarketIntegrationSimulation
from yemen_market_integration.models.spatial import SpatialEconometrics

# Import performance utilities
from yemen_market_integration.utils.performance_utils import timer, memory_usage_decorator, parallelize_dataframe
from yemen_market_integration.utils.error_handler import handle_errors, ModelError, DataError, capture_error
from yemen_market_integration.utils.config import config
from yemen_market_integration.utils.performance_utils import configure_system_for_performance

# Configure system for optimal performance
configure_system_for_performance()

# Set up logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('integration_tests.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


class MarketAnalysisPipelineTest(unittest.TestCase):
    """Test for the complete market analysis pipeline."""
    
    @classmethod
    @timer
    @handle_errors(logger=logger)
    def setUpClass(cls):
        """Set up test fixtures once for all test methods."""
        logger.info("Setting up integration test environment")
        
        # Create temporary directory for test outputs
        cls.temp_dir = tempfile.TemporaryDirectory()
        
        # Generate sample data with known properties
        cls._generate_sample_data()
        
        # Initialize testers and models
        cls.unit_root_tester = UnitRootTester()
        
        # Save memory usage baseline
        cls.initial_memory = cls._get_memory_usage()
        logger.info(f"Initial memory usage: {cls.initial_memory:.2f} MB")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests have run."""
        # Clean up temporary directory
        cls.temp_dir.cleanup()
        
        # Force garbage collection
        gc.collect()
        
        # Report final memory usage
        final_memory = cls._get_memory_usage()
        logger.info(f"Final memory usage: {final_memory:.2f} MB")
        logger.info(f"Memory difference: {final_memory - cls.initial_memory:.2f} MB")
    
    @classmethod
    def _get_memory_usage(cls):
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # MB
    
    @classmethod
    def _generate_sample_data(cls):
        """Generate sample data for integration testing."""
        logger.info("Generating sample data for integration tests")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Define parameters
        n_samples = 200  # Number of time periods
        known_threshold = 0.5  # Known threshold value
        
        # Create date range
        start_date = datetime(2018, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_samples)]
        
        # Create synthetic cointegrated series with threshold effects
        common_trend = np.cumsum(np.random.normal(0, 1, n_samples))
        
        # Create equilibrium errors with threshold effect
        eq_error = np.zeros(n_samples)
        eq_error[0] = np.random.normal(0, 1)
        
        # Generate equilibrium errors with different adjustment speeds
        # based on whether they're above or below threshold
        for t in range(1, n_samples):
            if eq_error[t-1] <= known_threshold:  # Below threshold - slow adjustment
                eq_error[t] = 0.9 * eq_error[t-1] + np.random.normal(0, 0.5)
            else:  # Above threshold - faster adjustment
                eq_error[t] = 0.4 * eq_error[t-1] + np.random.normal(0, 0.5)
        
        # Create price series with equilibrium relationship and threshold effect
        series1 = common_trend + eq_error
        series2 = 0.5 * common_trend + np.random.normal(0, 0.3, n_samples)
        
        # Create stationary series for comparison
        stationary_series = np.random.normal(0, 1, n_samples)
        
        # Create non-cointegrated series (independent random walks)
        non_coint_series1 = np.cumsum(np.random.normal(0, 1, n_samples))
        non_coint_series2 = np.cumsum(np.random.normal(0, 1, n_samples))
        
        # Create spatial data (mock market locations)
        n_markets = 10
        market_ids = [f"M{i}" for i in range(1, n_markets+1)]
        lats = np.random.uniform(12.5, 17.5, n_markets)  # Yemen latitudes
        lons = np.random.uniform(43.0, 49.0, n_markets)  # Yemen longitudes
        
        # Store data as class attributes
        cls.dates = dates
        cls.date_index = pd.DatetimeIndex(dates)
        cls.series1 = pd.Series(series1, index=cls.date_index, name="Price1")
        cls.series2 = pd.Series(series2, index=cls.date_index, name="Price2")
        cls.stationary_series = pd.Series(stationary_series, index=cls.date_index, name="Stationary")
        cls.non_coint_series1 = pd.Series(non_coint_series1, index=cls.date_index, name="NonCoint1")
        cls.non_coint_series2 = pd.Series(non_coint_series2, index=cls.date_index, name="NonCoint2")
        cls.known_threshold = known_threshold
        cls.eq_error = eq_error
        
        # Create GeoDataFrame for spatial tests
        market_data = {
            'market_id': market_ids,
            'latitude': lats,
            'longitude': lons,
            'market_name': [f"Market {i}" for i in range(1, n_markets+1)]
        }
        
        # Add price columns for each market
        for i, market_id in enumerate(market_ids):
            # Different markets have correlated but slightly different prices
            market_effect = np.random.normal(0, 0.5)
            for t in range(n_samples):
                col_name = f"{market_id}_price"
                if not col_name in market_data:
                    market_data[col_name] = []
                market_data[col_name].append(series1[t] + market_effect + np.random.normal(0, 0.3))
        
        # Convert to DataFrame
        cls.market_df = pd.DataFrame(market_data)
        
        # Create north/south price series for simulation tests
        north_prices = series1 + 2.0  # Higher price level in north
        south_prices = series2
        
        cls.north_prices = pd.Series(north_prices, index=cls.date_index, name="North")
        cls.south_prices = pd.Series(south_prices, index=cls.date_index, name="South")
        
        logger.info("Sample data generation complete")

    @timer
    @handle_errors(logger=logger)
    def test_full_pipeline(self):
        """Test the complete analysis pipeline from data preprocessing to simulation."""
        logger.info("Running full pipeline integration test")
        
        # STEP 1: Unit root testing
        logger.info("Step 1: Running unit root tests")
        
        # Test I(1) series
        series1_results = self.unit_root_tester.run_all_tests(self.series1)
        series2_results = self.unit_root_tester.run_all_tests(self.series2)
        
        # Verify results
        self.assertFalse(series1_results['adf']['stationary'], 
                        "Series1 should be non-stationary in levels")
        self.assertFalse(series2_results['adf']['stationary'], 
                        "Series2 should be non-stationary in levels")
        
        # Test differenced series for stationarity
        diff_series1 = self.series1.diff().dropna()
        diff_series2 = self.series2.diff().dropna()
        
        diff1_results = self.unit_root_tester.test_adf(diff_series1)
        diff2_results = self.unit_root_tester.test_adf(diff_series2)
        
        self.assertTrue(diff1_results['stationary'], 
                       "Differenced series1 should be stationary")
        self.assertTrue(diff2_results['stationary'], 
                       "Differenced series2 should be stationary")
        
        # Determine integration order
        order1 = self.unit_root_tester.determine_integration_order(self.series1)
        order2 = self.unit_root_tester.determine_integration_order(self.series2)
        
        self.assertEqual(order1, 1, "Series1 should be I(1)")
        self.assertEqual(order2, 1, "Series2 should be I(1)")
        
        # STEP 2: Threshold cointegration analysis
        logger.info("Step 2: Running threshold cointegration analysis")
        
        # Create and estimate threshold cointegration model
        threshold_model = ThresholdCointegration(
            self.series1.values, 
            self.series2.values,
            market1_name="Market1", 
            market2_name="Market2"
        )
        
        # Test cointegration
        coint_results = threshold_model.estimate_cointegration()
        self.assertTrue(coint_results['cointegrated'], 
                       "Series should be cointegrated")
        
        # Test threshold estimation
        threshold_results = threshold_model.estimate_threshold(n_grid=100)
        self.assertIsNotNone(threshold_results['threshold'], 
                            "Threshold should be estimated")
        self.assertAlmostEqual(threshold_results['threshold'], self.known_threshold, 
                              delta=0.3, msg="Estimated threshold should be close to true value")
        
        # Test TVECM estimation
        tvecm_results = threshold_model.estimate_tvecm()
        
        # Check adjustment coefficients
        self.assertLess(tvecm_results['adjustment_above_1'], 0, 
                       "Above-threshold adjustment should be negative")
        self.assertLess(abs(tvecm_results['adjustment_below_1']), abs(tvecm_results['adjustment_above_1']), 
                       "Adjustment should be faster (more negative) above threshold")
        
        # STEP 3: Model diagnostics
        logger.info("Step 3: Running model diagnostics")
        
        # Run diagnostics
        diagnostic_results = threshold_model.run_diagnostics()
        
        # Check if diagnostics results are returned properly
        self.assertIn('residual_tests', diagnostic_results, 
                     "Diagnostics should include residual tests")
        self.assertIn('asymmetric_adjustment', diagnostic_results, 
                     "Diagnostics should include asymmetric adjustment test")
        
        # STEP 4: Simulation
        logger.info("Step 4: Running policy simulation")
        
        # Create a simplified simulation
        simulation = MarketIntegrationSimulation(
            data=None,  # We'll use series directly
            threshold_model=threshold_model
        )
        
        # Access the internal methods directly for testing
        # Normally this would run through simulate_policy but we'll test components
        
        # Test reestimation with simulated data
        # Since we don't have full data, we'll just verify the method works
        # without raising exceptions
        try:
            # Create simple simulated data - exchange rate unity
            simulated_data = {
                'north_price': self.north_prices,
                'south_price': self.south_prices,
            }
            
            # Test internal simulation components
            structural_break_results = simulation._test_structural_breaks(
                original_data=simulated_data, 
                simulated_data=simulated_data
            )
            
            # Verify results contain expected keys
            self.assertIn('break_detected', structural_break_results,
                         "Structural break test results should contain 'break_detected' key")
            
        except Exception as e:
            self.fail(f"Simulation test failed with exception: {str(e)}")
        
        # Clean up
        del threshold_model
        del simulation
        gc.collect()
        
        logger.info("Full pipeline integration test completed successfully")

    @timer
    @handle_errors(logger=logger)
    def test_threshold_model_accuracy(self):
        """Test threshold model accuracy on data with known threshold value."""
        logger.info("Testing threshold model accuracy")
        
        # Create threshold model
        threshold_model = ThresholdCointegration(
            self.series1.values, 
            self.series2.values,
            market1_name="Market1", 
            market2_name="Market2",
            max_lags=2  # Use small lags for test efficiency
        )
        
        # Estimate the model
        threshold_model.estimate_cointegration()
        threshold_results = threshold_model.estimate_threshold(n_grid=200)
        tvecm_results = threshold_model.estimate_tvecm()
        
        # Test threshold estimation accuracy
        self.assertAlmostEqual(
            threshold_results['threshold'], 
            self.known_threshold,
            delta=0.3,  # Allow some margin of error
            msg="Threshold estimation should recover the true threshold within 0.3 units"
        )
        
        # Test adjustment parameters - should show asymmetry
        self.assertLess(tvecm_results['adjustment_above_1'], tvecm_results['adjustment_below_1'],
                       "Adjustment should be faster (more negative) above threshold")
        
        # Test half-lives - should be shorter above threshold
        if 'asymmetric_adjustment' in tvecm_results:
            asymm = tvecm_results['asymmetric_adjustment']
            self.assertLess(asymm['half_life_above_1'], asymm['half_life_below_1'],
                           "Half-life should be shorter above threshold")
        
        # Test threshold significance
        significance_results = threshold_model.test_threshold_significance(n_bootstrap=50)  # Lower for speed
        
        self.assertIn('significant', significance_results,
                     "Threshold significance test should provide 'significant' result")
        
        # Run MTAR test
        mtar_results = threshold_model.estimate_mtar()
        self.assertIn('asymmetric', mtar_results, "MTAR results should indicate asymmetry")
        
        # Clean up
        del threshold_model
        gc.collect()
        
        logger.info("Threshold model accuracy test completed")

    @timer
    @handle_errors(logger=logger)
    @memory_usage_decorator
    def test_market_integration_simulation(self):
        """Test market integration simulation with controlled inputs."""
        logger.info("Testing market integration simulation")
        
        # Create threshold model for simulation
        threshold_model = ThresholdCointegration(
            self.north_prices.values, 
            self.south_prices.values,
            market1_name="North", 
            market2_name="South",
            max_lags=2  # Use small lags for test efficiency
        )
        
        # Run analysis first
        threshold_model.run_full_analysis()
        
        # Create simulation with threshold model
        simulation = MarketIntegrationSimulation(
            threshold_model=threshold_model,
            data=None  # We'll work with simplified data
        )
        
        # Prepare data for testing
        original_data = pd.DataFrame({
            'north_price': self.north_prices,
            'south_price': self.south_prices
        })
        
        # Simulate policy effect: reduce the price differential by 25%
        simulated_data = pd.DataFrame({
            'north_price': self.north_prices * 0.9,  # Reduce north price
            'south_price': self.south_prices * 1.1   # Increase south price 
        })
        
        # Test robustness
        try:
            # Instead of running full simulation, we'll test key internal methods
            
            # Test structural break detection
            break_results = simulation._test_structural_breaks(
                original_data=original_data,
                simulated_data=simulated_data
            )
            
            self.assertIn('break_detected', break_results, 
                         "Structural break test should provide 'break_detected' result")
            
            # Test diagnostic comparison
            diagnostic_results = simulation._test_residual_diagnostics({
                'original_residuals': np.random.normal(0, 1, 100),  # Mock residuals
                'simulated_residuals': np.random.normal(0, 1, 100)   # Mock residuals
            })
            
            self.assertIsInstance(diagnostic_results, dict,
                                "Diagnostic results should be a dictionary")
            
        except Exception as e:
            self.fail(f"Simulation test failed with exception: {str(e)}")
        
        # Clean up
        del threshold_model
        del simulation
        gc.collect()
        
        logger.info("Market integration simulation test completed")

    @timer
    @handle_errors(logger=logger)
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery in the integration pipeline."""
        logger.info("Testing error handling and recovery")
        
        # Test error handling with invalid inputs
        
        # Test with too short series
        short_series = pd.Series(np.random.normal(0, 1, 5))
        
        with self.assertRaises(Exception):
            self.unit_root_tester.test_adf(short_series)
        
        # Test with invalid series (containing NaN)
        invalid_series = self.series1.copy()
        invalid_series[10:15] = np.nan
        
        with self.assertRaises(Exception):
            threshold_model = ThresholdCointegration(
                invalid_series.values, 
                self.series2.values
            )
            threshold_model.estimate_cointegration()
        
        # Test recovery - ensure we can still run new analyses after errors
        try:
            # This should work despite previous errors
            valid_result = self.unit_root_tester.test_adf(self.stationary_series)
            self.assertTrue(valid_result['stationary'], 
                           "Stationary series should be identified as stationary")
            
            logger.info("Successfully recovered from errors")
        except Exception as e:
            self.fail(f"Failed to recover from errors: {str(e)}")
        
        logger.info("Error handling and recovery test completed")


if __name__ == "__main__":
    unittest.main()
