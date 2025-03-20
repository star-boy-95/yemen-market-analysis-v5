"""
Test module for the MarketIntegrationSimulation class.

This module provides comprehensive tests for the simulation capabilities
implemented in the Yemen Market Integration project.
"""
import unittest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import tempfile
import os
import sys
import logging
from unittest.mock import patch, MagicMock

# Prevent logging during tests
logging.disable(logging.CRITICAL)

from src.models.simulation import MarketIntegrationSimulation
from src.models.threshold import ThresholdCointegration

class TestMarketIntegrationSimulation(unittest.TestCase):
    """Test cases for the MarketIntegrationSimulation class."""
    
    def setUp(self):
        """Set up test fixtures, creating sample data."""
        # Create sample market data with spatial component
        np.random.seed(42)  # For reproducibility
        
        # Generate dates
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        # Generate market data for north and south
        n_markets = 10
        markets = []
        
        # Market IDs and names
        market_ids = [f'M{i}' for i in range(n_markets)]
        market_names = [f'Market {i}' for i in range(n_markets)]
        
        # Regions (5 north, 5 south)
        regions = ['north'] * 5 + ['south'] * 5
        
        # Exchange rates (different for north and south)
        north_rate = 600
        south_rate = 800
        exchange_rates = [north_rate] * 5 + [south_rate] * 5
        
        # Coordinates (north markets at top, south at bottom)
        lats = np.linspace(15, 13, 5).tolist() + np.linspace(12, 10, 5).tolist()
        lons = np.linspace(44, 45, n_markets)
        
        # Base prices with noise (north generally lower than south)
        base_prices_north = np.random.normal(100, 10, 5)
        base_prices_south = np.random.normal(150, 15, 5)
        base_prices = np.concatenate([base_prices_north, base_prices_south])
        
        # Conflict intensity (higher in some areas)
        conflict_raw = np.abs(np.random.normal(0, 1, n_markets))
        conflict_normalized = conflict_raw / conflict_raw.max()
        
        # Create data for each market and date
        data = []
        
        for i, (market_id, market_name, region, rate, lat, lon, base_price, conflict) in enumerate(
            zip(market_ids, market_names, regions, exchange_rates, lats, lons, base_prices, conflict_normalized)
        ):
            # Add time component with trend and noise
            for j, date in enumerate(dates):
                # Time trend
                trend = j * 0.5
                # Seasonal component
                season = 5 * np.sin(j / 10)
                # Random noise
                noise = np.random.normal(0, 5)
                
                # Final price
                price = base_price + trend + season + noise
                
                # Add row to data
                data.append({
                    'market_id': market_id,
                    'market_name': market_name,
                    'exchange_rate_regime': region,
                    'exchange_rate': rate,
                    'date': date,
                    'price': price,
                    'conflict_intensity_normalized': conflict,
                    'geometry': Point(lon, lat)
                })
        
        # Convert to GeoDataFrame
        self.market_data = gpd.GeoDataFrame(data, geometry='geometry')
        
        # Create mock ThresholdCointegration model
        self.mock_threshold_model = self._create_mock_threshold_model()
        
        # Create mock SpatialEconometrics model
        self.mock_spatial_model = self._create_mock_spatial_model()
        
    def _create_mock_threshold_model(self):
        """Create a mock threshold model for testing."""
        # Get north and south data for the latest date
        latest_date = self.market_data['date'].max()
        latest_data = self.market_data[self.market_data['date'] == latest_date]
        
        north_price = latest_data[latest_data['exchange_rate_regime'] == 'north']['price'].values
        south_price = latest_data[latest_data['exchange_rate_regime'] == 'south']['price'].values
        
        # Create a simple threshold model (not fully estimated)
        model = ThresholdCointegration(
            north_price, 
            south_price,
            market1_name="North",
            market2_name="South"
        )
        
        # Set attributes that would be created during estimation
        model.threshold = 20.0
        model.price_diff = north_price - south_price
        model.dates = latest_data['date'].unique()
        
        # Create results dictionary
        model.results = {
            'adjustment_below_1': -0.3,
            'adjustment_middle_1': -0.1,
            'adjustment_above_1': -0.2,
            'threshold': 20.0,
        }
        
        # Mock residuals
        model.eq_errors = np.random.normal(0, 1, len(north_price))
        
        return model
    
    def _create_mock_spatial_model(self):
        """Create a mock spatial model for testing."""
        # Simple mock object for the spatial model
        mock_spatial = MagicMock()
        
        # Mock the weights attribute
        mock_spatial.weights = np.random.rand(10, 10)
        
        return mock_spatial
    
    def test_init(self):
        """Test initialization of the simulation class."""
        sim = MarketIntegrationSimulation(self.market_data)
        
        # Check that data was correctly stored
        self.assertIsInstance(sim.data, gpd.GeoDataFrame)
        self.assertIsInstance(sim.original_data, gpd.GeoDataFrame)
        self.assertEqual(len(sim.data), len(self.market_data))
        
        # Test initialization with models
        sim_with_models = MarketIntegrationSimulation(
            self.market_data,
            threshold_model=self.mock_threshold_model,
            spatial_model=self.mock_spatial_model
        )
        
        self.assertEqual(sim_with_models.threshold_model, self.mock_threshold_model)
        self.assertEqual(sim_with_models.spatial_model, self.mock_spatial_model)
    
    def test_validate_input_data(self):
        """Test input data validation."""
        # Valid data should not raise errors
        sim = MarketIntegrationSimulation(self.market_data)
        
        # Test with missing required columns
        bad_data = self.market_data.drop(columns=['exchange_rate'])
        with self.assertRaises(Exception):
            sim = MarketIntegrationSimulation(bad_data)
        
        # Test with invalid exchange rate regime values
        bad_regime_data = self.market_data.copy()
        bad_regime_data.loc[0, 'exchange_rate_regime'] = 'invalid'
        with self.assertRaises(Exception):
            sim = MarketIntegrationSimulation(bad_regime_data)
    
    def test_convert_to_usd(self):
        """Test USD conversion functionality."""
        sim = MarketIntegrationSimulation(self.market_data)
        
        # Make a copy of data to modify
        test_data = self.market_data.copy()
        
        # Apply conversion
        sim._convert_to_usd(test_data)
        
        # Check that USD price column was created
        self.assertIn('usd_price', test_data.columns)
        
        # Verify conversion for a north market
        north_data = test_data[test_data['exchange_rate_regime'] == 'north'].iloc[0]
        expected_usd = north_data['price'] / north_data['exchange_rate']
        self.assertAlmostEqual(north_data['usd_price'], expected_usd)
        
        # Verify conversion for a south market
        south_data = test_data[test_data['exchange_rate_regime'] == 'south'].iloc[0]
        expected_usd = south_data['price'] / south_data['exchange_rate']
        self.assertAlmostEqual(south_data['usd_price'], expected_usd)
    
    def test_determine_unified_rate(self):
        """Test determination of unified exchange rate."""
        sim = MarketIntegrationSimulation(self.market_data)
        
        # Test with numeric value
        numeric_rate = sim._determine_unified_rate("700")
        self.assertEqual(numeric_rate, 700.0)
        
        # Test with 'official' method (North rate)
        official_rate = sim._determine_unified_rate("official")
        north_mean = self.market_data[self.market_data['exchange_rate_regime'] == 'north']['exchange_rate'].mean()
        self.assertAlmostEqual(official_rate, north_mean)
        
        # Test with 'market' method (South rate)
        market_rate = sim._determine_unified_rate("market")
        south_mean = self.market_data[self.market_data['exchange_rate_regime'] == 'south']['exchange_rate'].mean()
        self.assertAlmostEqual(market_rate, south_mean)
        
        # Test with 'average' method
        avg_rate = sim._determine_unified_rate("average")
        expected_avg = (north_mean + south_mean) / 2
        self.assertAlmostEqual(avg_rate, expected_avg)
        
        # Test with reference date
        latest_date = self.market_data['date'].max()
        date_rate = sim._determine_unified_rate("official", reference_date=str(latest_date.date()))
        latest_north = self.market_data[
            (self.market_data['exchange_rate_regime'] == 'north') & 
            (self.market_data['date'] == latest_date)
        ]['exchange_rate'].mean()
        self.assertAlmostEqual(date_rate, latest_north)
    
    def test_calculate_price_changes(self):
        """Test calculation of price changes."""
        sim = MarketIntegrationSimulation(self.market_data)
        
        # Create sample original and simulated prices
        original_prices = pd.Series([100, 110, 120, 130, 140])
        simulated_prices = pd.Series([110, 115, 125, 128, 135])
        
        # Calculate changes without grouping
        changes = sim._calculate_price_changes(original_prices, simulated_prices)
        
        # Verify calculated values
        self.assertIn('abs_change', changes.columns)
        self.assertIn('pct_change', changes.columns)
        
        # Check a specific value
        self.assertAlmostEqual(changes.loc[0, 'abs_change'], 10.0)
        self.assertAlmostEqual(changes.loc[0, 'pct_change'], 10.0)  # 10% increase
    
    @patch('src.models.simulation.estimate_threshold_model')
    def test_reestimate_threshold_model(self, mock_estimate):
        """Test re-estimation of threshold model."""
        # Set up the mock to return a simple model
        mock_model = MagicMock()
        mock_model.threshold = 15.0
        mock_model.results = {'threshold': 15.0}
        mock_estimate.return_value = mock_model
        
        # Create simulation with mock threshold model
        sim = MarketIntegrationSimulation(
            self.market_data, 
            threshold_model=self.mock_threshold_model
        )
        
        # Generate simulated data
        sim_data = self.market_data.copy()
        sim_data['simulated_price'] = sim_data['price'] * 1.1  # 10% increase
        
        # Re-estimate model
        reestimated = sim._reestimate_threshold_model(sim_data)
        
        # Verify mock was called
        mock_estimate.assert_called_once()
        
        # Verify returned model
        self.assertEqual(reestimated, mock_model)
    
    def test_simulate_exchange_rate_unification(self):
        """Test exchange rate unification simulation."""
        # Create simulation with mock models
        sim = MarketIntegrationSimulation(
            self.market_data, 
            threshold_model=self.mock_threshold_model
        )
        
        # Run simulation
        with patch('src.models.simulation.estimate_threshold_model') as mock_estimate:
            # Set up the mock to return a simple model
            mock_model = MagicMock()
            mock_model.threshold = 15.0
            mock_model.results = {'threshold': 15.0}
            mock_estimate.return_value = mock_model
            
            # Run simulation
            result = sim.simulate_exchange_rate_unification(target_rate='official')
        
        # Verify results structure
        self.assertIn('simulated_data', result)
        self.assertIn('unified_rate', result)
        self.assertIn('price_changes', result)
        self.assertIn('threshold_model', result)
        
        # Check simulated data
        self.assertIn('simulated_price', result['simulated_data'].columns)
        self.assertIn('usd_price', result['simulated_data'].columns)
        
        # Verify results are stored in instance
        self.assertIn('exchange_rate_unification', sim.results)
    
    def test_simulate_improved_connectivity(self):
        """Test improved connectivity simulation."""
        # Create simulation with mock models
        sim = MarketIntegrationSimulation(
            self.market_data, 
            spatial_model=self.mock_spatial_model
        )
        
        # Run simulation
        result = sim.simulate_improved_connectivity(reduction_factor=0.5)
        
        # Verify results structure
        self.assertIn('simulated_data', result)
        self.assertIn('reduction_factor', result)
        self.assertIn('spatial_weights', result)
        
        # Check conflict reduction
        original_conflict = self.market_data['conflict_intensity_normalized'].mean()
        reduced_conflict = result['simulated_data']['conflict_intensity_normalized'].mean()
        
        # Should be reduced by approximately half
        self.assertAlmostEqual(reduced_conflict, original_conflict * 0.5, places=5)
        
        # Verify results are stored in instance
        self.assertIn('improved_connectivity', sim.results)
    
    def test_calculate_welfare_effects(self):
        """Test welfare effects calculation."""
        # Create simulation with mock models
        sim = MarketIntegrationSimulation(
            self.market_data, 
            threshold_model=self.mock_threshold_model
        )
        
        # First run a simulation
        with patch('src.models.simulation.estimate_threshold_model') as mock_estimate:
            # Set up the mock to return a simple model
            mock_model = MagicMock()
            mock_model.threshold = 15.0
            mock_model.results = {'threshold': 15.0}
            mock_estimate.return_value = mock_model
            
            # Run simulation
            sim.simulate_exchange_rate_unification(target_rate='official')
        
        # Calculate welfare effects
        welfare = sim.calculate_welfare_effects('exchange_rate_unification')
        
        # Verify results structure
        self.assertIn('regional_metrics', welfare)
        self.assertIn('price_convergence', welfare)
        
        # Verify price convergence calculation
        self.assertIn('original_difference', welfare['price_convergence'])
        self.assertIn('simulated_difference', welfare['price_convergence'])
    
    @patch('src.models.simulation.create_conflict_adjusted_weights')
    def test_combined_policy(self, mock_weights):
        """Test combined policy simulation."""
        # Mock the weights function
        mock_weights.return_value = np.random.rand(10, 10)
        
        # Create simulation with mock models
        sim = MarketIntegrationSimulation(
            self.market_data, 
            threshold_model=self.mock_threshold_model,
            spatial_model=self.mock_spatial_model
        )
        
        # Run combined simulation
        with patch('src.models.simulation.estimate_threshold_model') as mock_estimate:
            # Set up the mock to return a simple model
            mock_model = MagicMock()
            mock_model.threshold = 15.0
            mock_model.results = {'threshold': 15.0}
            mock_estimate.return_value = mock_model
            
            # Run simulation
            result = sim.simulate_combined_policy(
                exchange_rate_target='official',
                conflict_reduction=0.5
            )
        
        # Verify results structure
        self.assertIn('simulated_data', result)
        self.assertIn('unified_rate', result)
        self.assertIn('reduction_factor', result)
        self.assertIn('price_changes', result)
        self.assertIn('spatial_weights', result)
        
        # Verify both exchange rate and conflict effects
        self.assertIn('simulated_price', result['simulated_data'].columns)
        
        # Conflict should be reduced
        original_conflict = self.market_data['conflict_intensity_normalized'].mean()
        reduced_conflict = result['simulated_data']['conflict_intensity_normalized'].mean()
        self.assertLess(reduced_conflict, original_conflict)
        
        # Verify results are stored in instance
        self.assertIn('combined_policy', sim.results)

if __name__ == '__main__':
    unittest.main()
